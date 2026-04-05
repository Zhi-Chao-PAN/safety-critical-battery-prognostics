"""
Online Continual Learning Module for Battery PINN Model

This module implements an online calibration system that enables the PINN model
to adapt to battery aging mechanisms in real-time without catastrophic forgetting.

Key Features:
1. Few-shot Fine-tuning: Minimal data required for adaptation
2. Selective Parameter Updates: Only last two NN layers + SPM physics parameters
3. Replay Buffer: Prevents catastrophic forgetting with minimal memory footprint
4. Adaptive Learning Rate: Extremely small LR for stable online updates
5. Memory-efficient: Designed for edge deployment on RTX 4060

Architecture:
    - OnlineCalibrator wraps existing PINNModel (no modifications required)
    - Receives streaming battery cycle data (e.g., every 10 cycles)
    - Performs few-shot fine-tuning with on-device data
    - Maintains replay buffer for stability
"""

import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pinn_model import PINNModel

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Minimal replay buffer to prevent catastrophic forgetting.
    
    Stores a small subset of historical data to maintain model stability
    during online fine-tuning. Uses FIFO policy with reservoir sampling
    for representative data retention.
    """
    
    def __init__(self, max_size: int = 100, random_sample: bool = True):
        """
        Initialize replay buffer.
        
        Args:
            max_size: Maximum number of samples to store
            random_sample: Whether to use reservoir sampling for diversity
        """
        self.max_size = max_size
        self.random_sample = random_sample
        self.buffer_X: List[np.ndarray] = []
        self.buffer_y: List[np.ndarray] = []
        self.buffer_cycles: List[float] = []
        self._count = 0
        
        logger.info(f"ReplayBuffer initialized: max_size={max_size}")
    
    def add(self, X: np.ndarray, y: np.ndarray, cycles: np.ndarray):
        """
        Add new data to buffer with reservoir sampling.
        
        Args:
            X: Input features [n_samples, n_features]
            y: Target values [n_samples]
            cycles: Cycle numbers [n_samples]
        """
        n_samples = len(X)
        
        for i in range(n_samples):
            self._count += 1
            
            if len(self.buffer_X) < self.max_size:
                self.buffer_X.append(X[i:i+1])
                self.buffer_y.append(y[i:i+1])
                self.buffer_cycles.append(cycles[i])
            elif self.random_sample and np.random.random() < self.max_size / self._count:
                idx = np.random.randint(0, self.max_size)
                self.buffer_X[idx] = X[i:i+1]
                self.buffer_y[idx] = y[i:i+1]
                self.buffer_cycles[idx] = cycles[i]
    
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get batch from buffer.
        
        Args:
            batch_size: Number of samples to return (None = all)
            
        Returns:
            Tuple of (X, y, cycles)
        """
        if len(self.buffer_X) == 0:
            return np.array([]), np.array([]), np.array([])
        
        if batch_size is None or batch_size >= len(self.buffer_X):
            X = np.vstack(self.buffer_X)
            y = np.hstack(self.buffer_y)
            cycles = np.array(self.buffer_cycles)
        else:
            indices = np.random.choice(len(self.buffer_X), batch_size, replace=False)
            X = np.vstack([self.buffer_X[i] for i in indices])
            y = np.hstack([self.buffer_y[i] for i in indices])
            cycles = np.array([self.buffer_cycles[i] for i in indices])
        
        return X, y, cycles
    
    def __len__(self) -> int:
        return len(self.buffer_X)
    
    def clear(self):
        """Clear buffer."""
        self.buffer_X.clear()
        self.buffer_y.clear()
        self.buffer_cycles.clear()
        self._count = 0


class OnlineCalibrator:
    """
    Online continual learning adapter for PINN battery model.
    
    This class enables real-time model adaptation to battery aging without
    modifying the original PINNModel. It implements:
    
    1. Few-shot fine-tuning with selective parameter updates
    2. Replay buffer for catastrophic forgetting prevention
    3. Adaptive learning rate scheduling
    4. Memory-efficient operations for edge deployment
    
    Usage:
        calibrator = OnlineCalibrator(pinn_model)
        
        # Receive new data every 10 cycles
        calibrator.update(new_X, new_y)
        
        # Get calibrated predictions
        mean, lower, upper = calibrator.predict(X)
    """
    
    def __init__(
        self,
        pinn_model: PINNModel,
        replay_buffer_size: int = 100,
        online_lr: float = 1e-5,
        online_epochs: int = 5,
        replay_ratio: float = 0.3,
        freeze_ratio: float = 0.7,
        enable_spm_tuning: bool = True,
        device: str = "cuda",
        verbose: bool = True
    ):
        """
        Initialize online calibrator.
        
        Args:
            pinn_model: Pre-trained PINNModel instance
            replay_buffer_size: Size of replay buffer for forgetting prevention
            online_lr: Learning rate for online fine-tuning (very small)
            online_epochs: Number of epochs per online update
            replay_ratio: Ratio of replay data in each update batch
            freeze_ratio: Ratio of NN layers to freeze (0.7 = freeze first 70%)
            enable_spm_tuning: Whether to fine-tune SPM physics parameters
            device: Device for computation
            verbose: Enable verbose logging
        """
        self.pinn_model = pinn_model
        self.replay_buffer_size = replay_buffer_size
        self.online_lr = online_lr
        self.online_epochs = online_epochs
        self.replay_ratio = replay_ratio
        self.freeze_ratio = freeze_ratio
        self.enable_spm_tuning = enable_spm_tuning
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        
        # Track update statistics
        self.update_count = 0
        self.total_samples_seen = 0
        
        # Setup selective fine-tuning
        self._setup_selective_fine_tuning()
        
        # Initialize optimizer for online learning
        self._setup_optimizer()
        
        logger.info(f"OnlineCalibrator initialized: lr={online_lr}, "
                   f"replay_size={replay_buffer_size}, "
                   f"spm_tuning={enable_spm_tuning}")
    
    def _setup_selective_fine_tuning(self):
        """
        Setup selective parameter freezing for few-shot fine-tuning.
        
        Strategy:
        1. Freeze early layers of neural network (preserve learned features)
        2. Unfreeze last two layers (adapt to new data)
        3. Optionally unfreeze SPM physics parameters (D_s coefficients)
        """
        if self.pinn_model.model is None:
            raise RuntimeError("PINN model must be fitted before online calibration")
        
        # Get all layers
        layers = list(self.pinn_model.model.layers.children())
        num_layers = len(layers)
        
        # Freeze early layers
        num_freeze = int(num_layers * self.freeze_ratio)
        for i in range(num_freeze):
            for param in layers[i].parameters():
                param.requires_grad = False
        
        # Unfreeze last layers
        for i in range(num_freeze, num_layers):
            for param in layers[i].parameters():
                param.requires_grad = True
        
        if self.verbose:
            frozen_count = sum(1 for p in self.pinn_model.model.parameters() if not p.requires_grad)
            trainable_count = sum(1 for p in self.pinn_model.model.parameters() if p.requires_grad)
            logger.info(f"Selective fine-tuning setup: {frozen_count} frozen, "
                      f"{trainable_count} trainable NN parameters")
        
        # SPM parameter tuning (if available)
        self.spm_params = []
        if self.enable_spm_tuning:
            if hasattr(self.pinn_model, 'constraint_manager'):
                for name, constraint in self.pinn_model.constraint_manager.constraints.items():
                    if hasattr(constraint, 'spm'):
                        spm = constraint.spm
                        if hasattr(spm, 'log_D_s_a'):
                            spm.log_D_s_a.requires_grad = True
                            self.spm_params.append(spm.log_D_s_a)
                        if hasattr(spm, 'log_D_s_c'):
                            spm.log_D_s_c.requires_grad = True
                            self.spm_params.append(spm.log_D_s_c)
        
        if self.verbose and self.spm_params:
            logger.info(f"SPM physics parameters enabled for tuning: {len(self.spm_params)} params")
    
    def _setup_optimizer(self):
        """Setup optimizer for online fine-tuning."""
        trainable_params = []
        
        # Add trainable NN parameters
        for param in self.pinn_model.model.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        # Add SPM parameters
        trainable_params.extend(self.spm_params)
        
        if len(trainable_params) == 0:
            logger.warning("No trainable parameters found for online fine-tuning")
        
        # Use AdamW with very small learning rate
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.online_lr,
            weight_decay=1e-6
        )
        
        # Learning rate scheduler for gradual decay
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.99
        )
    
    def update(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        cycles_new: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Update model with new streaming data.
        
        This method performs few-shot fine-tuning using:
        1. New data from current cycle
        2. Replay buffer data (for stability)
        3. Selective parameter updates
        
        Args:
            X_new: New input features [n_samples, n_features]
            y_new: New target values [n_samples]
            cycles_new: Cycle numbers (extracted from X if None)
            
        Returns:
            Dictionary with update statistics
        """
        if self.pinn_model.model is None:
            raise RuntimeError("PINN model must be fitted before online calibration")
        
        # Extract cycles if not provided
        if cycles_new is None:
            cycles_new = X_new[:, 0] if X_new.ndim > 1 else X_new
        
        # Add new data to replay buffer
        self.replay_buffer.add(X_new, y_new, cycles_new)
        
        # Get replay data
        X_replay, y_replay, cycles_replay = self.replay_buffer.get_batch()
        
        # Combine new and replay data
        if len(X_replay) > 0:
            X_combined = np.vstack([X_new, X_replay])
            y_combined = np.hstack([y_new, y_replay])
            cycles_combined = np.hstack([cycles_new, cycles_replay])
        else:
            X_combined = X_new
            y_combined = y_new
            cycles_combined = cycles_new
        
        # Perform few-shot fine-tuning
        stats = self._fine_tune(X_combined, y_combined, cycles_combined)
        
        # Update statistics
        self.update_count += 1
        self.total_samples_seen += len(X_new)
        
        stats.update({
            'update_count': self.update_count,
            'total_samples_seen': self.total_samples_seen,
            'replay_buffer_size': len(self.replay_buffer)
        })
        
        if self.verbose:
            logger.info(f"Online update #{self.update_count}: "
                      f"new_samples={len(X_new)}, "
                      f"replay_samples={len(X_replay)}, "
                      f"loss={stats['final_loss']:.6f}")
        
        return stats
    
    def _fine_tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cycles: np.ndarray
    ) -> Dict[str, Any]:
        """
        Fine-tune model on combined data.
        
        Args:
            X: Input features [n_samples, n_features]
            y: Target values [n_samples]
            cycles: Cycle numbers [n_samples]
            
        Returns:
            Dictionary with training statistics
        """
        # Prepare tensors
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)
        cycles_t = torch.tensor(cycles, dtype=torch.float32).to(self.device)
        
        # Compute physics baseline
        if self.pinn_model._physics_params:
            physics_pred = self.pinn_model.physics.predict(cycles, "train")
            y_residual = y - physics_pred
            y_t = torch.tensor(y_residual, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Training loop
        losses = []
        self.pinn_model.model.train()
        
        for epoch in range(self.online_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.pinn_model.model(X_t)
            
            # Data loss
            data_loss = F.mse_loss(predictions, y_t)
            
            # Physics constraints (if available)
            constraint_loss = torch.tensor(0.0, device=self.device)
            if self.pinn_model.constraint_manager is not None:
                constraint_inputs = {
                    "cycles": cycles_t.unsqueeze(1) if cycles_t.dim() == 1 else cycles_t,
                    "features": X_t
                }
                if self.pinn_model._physics_params:
                    physics_t = torch.tensor(physics_pred, dtype=torch.float32).unsqueeze(1).to(self.device)
                    constraint_inputs["physics_baseline"] = physics_t
                
                constraint_loss, _ = self.pinn_model.constraint_manager.compute_total_loss(
                    predictions, constraint_inputs, cycles_t, self.pinn_model._max_cycle
                )
            
            # Total loss
            total_loss = data_loss + constraint_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.pinn_model.model.parameters() if p.requires_grad] + self.spm_params,
                max_norm=0.1
            )
            
            self.optimizer.step()
            
            losses.append(total_loss.item())
        
        # Update learning rate
        self.scheduler.step()
        
        # Reset model to eval mode
        self.pinn_model.model.eval()
        
        return {
            'final_loss': losses[-1],
            'mean_loss': np.mean(losses),
            'losses': losses,
            'current_lr': self.optimizer.param_groups[0]['lr']
        }
    
    def predict(
        self,
        X: np.ndarray,
        **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with the calibrated model.
        
        Args:
            X: Input features [n_samples, n_features]
            **kwargs: Additional prediction arguments
            
        Returns:
            Tuple of (mean, lower, upper) predictions
        """
        return self.pinn_model.predict(X, **kwargs)
    
    def predict_single(
        self,
        X: np.ndarray,
        mc_samples: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with detailed uncertainty quantification.
        
        Args:
            X: Input features
            mc_samples: Override default MC samples
            
        Returns:
            Dictionary with prediction statistics
        """
        return self.pinn_model.predict_single(X, mc_samples)
    
    def get_parameter_updates(self) -> Dict[str, Any]:
        """
        Get current parameter values for monitoring.
        
        Returns:
            Dictionary with current parameter values
        """
        params = {
            'update_count': self.update_count,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'replay_buffer_size': len(self.replay_buffer)
        }
        
        # Get SPM parameters
        if self.spm_params:
            for i, param in enumerate(self.spm_params):
                params[f'spm_param_{i}'] = param.item()
        
        return params
    
    def save_state(self, path: str | Path) -> None:
        """
        Save calibrator state.
        
        Args:
            path: Path to save state
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'update_count': self.update_count,
            'total_samples_seen': self.total_samples_seen,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'replay_buffer': {
                'X': self.replay_buffer.buffer_X,
                'y': self.replay_buffer.buffer_y,
                'cycles': self.replay_buffer.buffer_cycles,
                'count': self.replay_buffer._count
            }
        }
        
        torch.save(state, path)
        logger.info(f"Calibrator state saved to {path}")
    
    def load_state(self, path: str | Path) -> None:
        """
        Load calibrator state.
        
        Args:
            path: Path to load state from
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        
        self.update_count = state['update_count']
        self.total_samples_seen = state['total_samples_seen']
        
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.scheduler.load_state_dict(state['scheduler_state'])
        
        # Restore replay buffer
        rb_state = state['replay_buffer']
        self.replay_buffer.buffer_X = rb_state['X']
        self.replay_buffer.buffer_y = rb_state['y']
        self.replay_buffer.buffer_cycles = rb_state['cycles']
        self.replay_buffer._count = rb_state['count']
        
        logger.info(f"Calibrator state loaded from {path}")
    
    def reset(self):
        """Reset calibrator to initial state."""
        self.replay_buffer.clear()
        self.update_count = 0
        self.total_samples_seen = 0
        
        # Reset optimizer
        self._setup_optimizer()
        
        logger.info("Calibrator reset to initial state")


def create_online_calibrator(
    pinn_model: PINNModel,
    config: Optional[Dict[str, Any]] = None
) -> OnlineCalibrator:
    """
    Factory function to create online calibrator with configuration.
    
    Args:
        pinn_model: Pre-trained PINNModel instance
        config: Configuration dictionary (optional)
        
    Returns:
        OnlineCalibrator instance
    """
    default_config = {
        'replay_buffer_size': 100,
        'online_lr': 1e-5,
        'online_epochs': 5,
        'replay_ratio': 0.3,
        'freeze_ratio': 0.7,
        'enable_spm_tuning': True,
        'device': 'cuda',
        'verbose': True
    }
    
    if config is not None:
        default_config.update(config)
    
    return OnlineCalibrator(pinn_model, **default_config)
