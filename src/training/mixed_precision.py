"""
Mixed Precision Training Utilities for PINN Battery Prognostics.

Optimized for Intel Core Ultra 9 + RTX 4060:
1. Automatic Mixed Precision (AMP) for 2x speedup on Tensor Cores
2. Gradient scaling to prevent underflow in FP16
3. Memory-efficient loss computation
4. Batch-optimized for maximum VRAM utilization

Key Features:
- Seamless integration with PhysicsConstraint system
- Automatic fallback to FP32 when needed
- Per-iteration gradient scaling updates
- Comprehensive NaN/Inf detection and handling
"""

import logging
from typing import Dict, Any, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)


class MixedPrecisionTrainer:
    """
    Mixed precision training wrapper for PINN models.
    
    Automatically manages:
    1. FP16 forward passes (autocast)
    2. Gradient scaling to prevent underflow
    3. NaN/Inf detection and recovery
    4. Memory optimization for RTX 4060
    
    Usage:
        trainer = MixedPrecisionTrainer(model, optimizer)
        loss = trainer.train_step(data, targets)
    """
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 init_scale: float = 2.**16,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000,
                 enabled: bool = True):
        """
        Initialize mixed precision trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer for the model
            init_scale: Initial gradient scaling factor
            growth_factor: Factor to increase scale when no NaN/Inf
            backoff_factor: Factor to decrease scale when NaN/Inf detected
            growth_interval: Steps between scale increases
            enabled: Whether mixed precision is enabled
        """
        self.model = model
        self.optimizer = optimizer
        self.enabled = enabled and torch.cuda.is_available()
        self.device = next(model.parameters()).device
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=self.enabled
        )
        
        # Training statistics
        self.step_count = 0
        self.nan_count = 0
        self.max_nan_before_fallback = 5
        
        logger.info(f"Mixed Precision Trainer initialized: enabled={self.enabled}, "
                   f"device={self.device}, init_scale={init_scale}")
    
    def train_step(self,
                  data: torch.Tensor,
                  targets: torch.Tensor,
                  loss_fn: Callable,
                  constraint_manager: Optional[Any] = None,
                  constraint_inputs: Optional[Dict[str, torch.Tensor]] = None,
                  cycles: Optional[torch.Tensor] = None,
                  max_cycle: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform one training step with mixed precision.
        
        Args:
            data: Input data tensor
            targets: Target values tensor
            loss_fn: Data loss function (e.g., MSE)
            constraint_manager: PhysicsConstraint manager (optional)
            constraint_inputs: Additional inputs for constraints (optional)
            cycles: Cycle numbers for adaptive weighting (optional)
            max_cycle: Maximum cycle for normalization (optional)
            
        Returns:
            total_loss: Total loss value
            loss_dict: Dictionary with loss breakdown
        """
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=self.enabled):
            # Model forward pass
            predictions = self.model(data)
            
            # Data loss
            data_loss = loss_fn(predictions, targets)
            
            # Constraint losses (if provided)
            constraint_loss = torch.tensor(0.0, device=self.device)
            constraint_breakdown = {}
            
            if constraint_manager is not None and constraint_inputs is not None:
                constraint_loss, constraint_breakdown = constraint_manager.compute_total_loss(
                    predictions, constraint_inputs, cycles, max_cycle
                )
            
            # Total loss
            total_loss = data_loss + constraint_loss
        
        # Backward pass with gradient scaling
        self.scaler.scale(total_loss).backward()
        
        # Unscale gradients and check for NaN/Inf
        self.scaler.unscale_(self.optimizer)
        
        # Check for NaN/Inf gradients
        if self._check_gradient_nan_inf():
            self.nan_count += 1
            logger.warning(f"NaN/Inf detected in gradients (count: {self.nan_count})")
            
            # Skip this update
            self.scaler.update()
            
            # Fallback to FP32 if too many NaN/Inf
            if self.nan_count >= self.max_nan_before_fallback:
                logger.warning("Too many NaN/Inf, disabling mixed precision")
                self.enabled = False
                self.scaler._enabled = False
            
            return total_loss.detach(), {
                "data_loss": data_loss.item(),
                "constraint_loss": constraint_loss.item(),
                "total_loss": total_loss.item(),
                "nan_detected": True,
                **constraint_breakdown
            }
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step with gradient scaling
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.step_count += 1
        self.nan_count = 0  # Reset NaN counter on successful step
        
        return total_loss.detach(), {
            "data_loss": data_loss.item(),
            "constraint_loss": constraint_loss.item(),
            "total_loss": total_loss.item(),
            "nan_detected": False,
            **constraint_breakdown
        }
    
    def _check_gradient_nan_inf(self) -> bool:
        """Check if any parameter has NaN or Inf gradients."""
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                    return True
        return False
    
    def state_dict(self) -> Dict[str, Any]:
        """Get trainer state dictionary."""
        return {
            "scaler": self.scaler.state_dict(),
            "step_count": self.step_count,
            "nan_count": self.nan_count,
            "enabled": self.enabled
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load trainer state dictionary."""
        self.scaler.load_state_dict(state_dict["scaler"])
        self.step_count = state_dict.get("step_count", 0)
        self.nan_count = state_dict.get("nan_count", 0)
        self.enabled = state_dict.get("enabled", self.enabled)
        self.scaler._enabled = self.enabled


class MixedPrecisionLoss:
    """
    Standalone mixed precision loss computation.
    
    Useful for validation or inference where no gradient update is needed.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute(self,
               model: nn.Module,
               data: torch.Tensor,
               targets: torch.Tensor,
               loss_fn: Callable,
               constraint_manager: Optional[Any] = None,
               constraint_inputs: Optional[Dict[str, torch.Tensor]] = None,
               cycles: Optional[torch.Tensor] = None,
               max_cycle: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
        """
        Compute loss with mixed precision (no gradients).
        
        Args:
            model: PyTorch model
            data: Input data tensor
            targets: Target values tensor
            loss_fn: Data loss function
            constraint_manager: PhysicsConstraint manager (optional)
            constraint_inputs: Additional inputs for constraints (optional)
            cycles: Cycle numbers for adaptive weighting (optional)
            max_cycle: Maximum cycle for normalization (optional)
            
        Returns:
            total_loss: Total loss value (float)
            loss_dict: Dictionary with loss breakdown
        """
        model.eval()
        
        with torch.no_grad(), autocast(enabled=self.enabled):
            # Model forward pass
            predictions = model(data)
            
            # Data loss
            data_loss = loss_fn(predictions, targets)
            
            # Constraint losses (if provided)
            constraint_loss = torch.tensor(0.0, device=self.device)
            constraint_breakdown = {}
            
            if constraint_manager is not None and constraint_inputs is not None:
                constraint_loss, constraint_breakdown = constraint_manager.compute_total_loss(
                    predictions, constraint_inputs, cycles, max_cycle
                )
            
            # Total loss
            total_loss = data_loss + constraint_loss
        
        model.train()
        
        return total_loss.item(), {
            "data_loss": data_loss.item(),
            "constraint_loss": constraint_loss.item(),
            "total_loss": total_loss.item(),
            **constraint_breakdown
        }


def create_mixed_precision_loss_fn(loss_fn: Callable, enabled: bool = True) -> Callable:
    """
    Create a mixed precision wrapper for any loss function.
    
    Args:
        loss_fn: Original loss function
        enabled: Whether mixed precision is enabled
        
    Returns:
        Wrapped loss function that uses mixed precision
    """
    def wrapped_loss_fn(*args, **kwargs):
        with autocast(enabled=enabled):
            return loss_fn(*args, **kwargs)
    
    return wrapped_loss_fn


# Example loss functions with mixed precision support
class MixedPrecisionMSELoss:
    """MSE loss with mixed precision support."""
    
    def __init__(self, reduction: str = 'mean', enabled: bool = True):
        self.reduction = reduction
        self.enabled = enabled
    
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with autocast(enabled=self.enabled):
            return F.mse_loss(input, target, reduction=self.reduction)


class MixedPrecisionMAELoss:
    """MAE loss with mixed precision support."""
    
    def __init__(self, reduction: str = 'mean', enabled: bool = True):
        self.reduction = reduction
        self.enabled = enabled
    
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with autocast(enabled=self.enabled):
            return F.l1_loss(input, target, reduction=self.reduction)


def get_optimal_mixed_precision_config(device_name: str = "RTX 4060") -> Dict[str, Any]:
    """
    Get optimal mixed precision configuration for specific hardware.
    
    Args:
        device_name: Name of GPU device
        
    Returns:
        Configuration dictionary with optimal settings
    """
    configs = {
        "RTX 4060": {
            "enabled": True,
            "init_scale": 2.**16,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "max_batch_size_multiplier": 2.0  # Can use 2x larger batches with AMP
        },
        "RTX 4090": {
            "enabled": True,
            "init_scale": 2.**16,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "max_batch_size_multiplier": 2.0
        },
        "A100": {
            "enabled": True,
            "init_scale": 2.**16,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "max_batch_size_multiplier": 2.0
        },
        "default": {
            "enabled": torch.cuda.is_available(),
            "init_scale": 2.**16,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "max_batch_size_multiplier": 1.5
        }
    }
    
    return configs.get(device_name, configs["default"])