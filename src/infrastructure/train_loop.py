"""
Industrial-Grade Training Loop with Anomaly Monitoring for PINN Battery Prognostics.

This module provides:
- TrainingMonitor callback class for real-time loss monitoring
- Graceful handling of NaN/Inf losses with checkpoint recovery
- Comprehensive TrainingLoop with integrated monitoring
- Support for PINNModel as a black box (no core algorithm modifications)
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.infrastructure.config_schema import PINNConfig

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """
    Training monitor callback for anomaly detection and graceful recovery.
    
    This class monitors the training process for numerical anomalies (NaN/Inf losses)
    and implements early stopping logic to prevent model corruption. It maintains
    training history, automatically saves checkpoints, and provides graceful recovery
    mechanisms for safety-critical battery prognostics applications.
    
    Features:
    - Real-time monitoring of loss values
    - Detection of NaN/Inf anomalies
    - Configurable tolerance thresholds
    - Automatic checkpoint saving on anomaly detection
    - Graceful exit without program crash
    
    Attributes:
        nan_tolerance: Maximum allowed consecutive NaN losses before stopping
        inf_tolerance: Maximum allowed consecutive Inf losses before stopping
        save_on_anomaly: Flag indicating if checkpoint should be saved on anomaly
        anomaly_checkpoint_name: Base filename for anomaly recovery checkpoints
        checkpoint_dir: Directory path for saving all checkpoints
        track_gradients: Flag indicating if gradient statistics should be tracked
        track_weights: Flag indicating if weight statistics should be tracked
        consecutive_nan_count: Counter for consecutive NaN loss occurrences
        consecutive_inf_count: Counter for consecutive Inf loss occurrences
        best_loss: Lowest loss value observed during training
        best_epoch: Epoch number where best_loss was achieved
        last_valid_checkpoint_path: Path to the most recent valid checkpoint
        history: Dictionary storing training metrics over epochs
    """
    
    def __init__(
        self,
        nan_tolerance: int = 3,
        inf_tolerance: int = 3,
        save_on_anomaly: bool = True,
        anomaly_checkpoint_name: str = "anomaly_recovery",
        checkpoint_dir: str = "checkpoints",
        track_gradients: bool = False,
        track_weights: bool = False,
    ):
        """
        Initialize TrainingMonitor.
        
        Args:
            nan_tolerance: Number of consecutive NaN losses before triggering exit
            inf_tolerance: Number of consecutive Inf losses before triggering exit
            save_on_anomaly: Save checkpoint when anomaly is detected for post-mortem
            anomaly_checkpoint_name: Base name for anomaly recovery checkpoint files
            checkpoint_dir: Directory path for saving training checkpoints
            track_gradients: Track gradient statistics during training for debugging
            track_weights: Track weight statistics during training for debugging
        """
        self.nan_tolerance: int = nan_tolerance
        self.inf_tolerance: int = inf_tolerance
        self.save_on_anomaly: bool = save_on_anomaly
        self.anomaly_checkpoint_name: str = anomaly_checkpoint_name
        self.checkpoint_dir: Path = Path(checkpoint_dir)
        self.track_gradients: bool = track_gradients
        self.track_weights: bool = track_weights
        
        self.consecutive_nan_count: int = 0
        self.consecutive_inf_count: int = 0
        self.best_loss: float = float("inf")
        self.best_epoch: int = 0
        self.last_valid_checkpoint_path: Optional[Path] = None
        self.history: Dict[str, list[Any]] = {
            "epoch": [],
            "loss": [],
            "data_loss": [],
            "constraint_loss": [],
            "learning_rate": [],
        }
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"TrainingMonitor initialized: nan_tolerance={nan_tolerance}, "
            f"inf_tolerance={inf_tolerance}, save_on_anomaly={save_on_anomaly}"
        )
    
    def on_epoch_end(
        self,
        epoch: int,
        loss_dict: Dict[str, float],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> bool:
        """
        Callback at the end of each epoch.
        
        Args:
            epoch: Current epoch number
            loss_dict: Dictionary containing loss values (total_loss, data_loss, constraint_loss)
            model: PyTorch model being trained
            optimizer: Optional optimizer for checkpoint saving
            scheduler: Optional learning rate scheduler
            
        Returns:
            bool: True if training should continue, False if should stop
        """
        total_loss = loss_dict.get("total_loss", float("nan"))
        
        self.history["epoch"].append(epoch)
        self.history["loss"].append(total_loss)
        self.history["data_loss"].append(loss_dict.get("data_loss", float("nan")))
        self.history["constraint_loss"].append(loss_dict.get("constraint_loss", float("nan")))
        
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else 0.0
        else:
            current_lr = 0.0
        self.history["learning_rate"].append(current_lr)
        
        if self._check_anomaly(total_loss, epoch, model, optimizer):
            return False
        
        if total_loss < self.best_loss and not np.isnan(total_loss) and not np.isinf(total_loss):
            self.best_loss = total_loss
            self.best_epoch = epoch
            self.last_valid_checkpoint_path = self._save_checkpoint(
                model, optimizer, epoch, loss_dict, "best"
            )
        
        return True
    
    def _check_anomaly(
        self,
        loss: float,
        epoch: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> bool:
        """
        Check for NaN/Inf anomalies and handle them.
        Returns True if training should stop.
        """
        if np.isnan(loss):
            self.consecutive_nan_count += 1
            self.consecutive_inf_count = 0
            
            logger.critical(
                f"Epoch {epoch}: NaN loss detected! "
                f"Consecutive NaN count: {self.consecutive_nan_count}/{self.nan_tolerance}"
            )
            
            if self.consecutive_nan_count >= self.nan_tolerance:
                logger.critical(
                    f"NaN tolerance exceeded ({self.nan_tolerance}). "
                    "Initiating graceful exit..."
                )
                self._handle_anomaly(model, optimizer, epoch, "nan")
                return True
        else:
            self.consecutive_nan_count = 0
        
        if np.isinf(loss):
            self.consecutive_inf_count += 1
            self.consecutive_nan_count = 0
            
            logger.critical(
                f"Epoch {epoch}: Inf loss detected! "
                f"Consecutive Inf count: {self.consecutive_inf_count}/{self.inf_tolerance}"
            )
            
            if self.consecutive_inf_count >= self.inf_tolerance:
                logger.critical(
                    f"Inf tolerance exceeded ({self.inf_tolerance}). "
                    "Initiating graceful exit..."
                )
                self._handle_anomaly(model, optimizer, epoch, "inf")
                return True
        else:
            self.consecutive_inf_count = 0
        
        return False
    
    def _handle_anomaly(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        anomaly_type: str,
    ) -> None:
        """
        Handle detected anomaly with graceful recovery.
        """
        if self.save_on_anomaly and self.last_valid_checkpoint_path is not None:
            logger.info(
                f"Anomaly detected. Last valid checkpoint available at: "
                f"{self.last_valid_checkpoint_path}"
            )
            
            anomaly_path = self._save_checkpoint(
                model, optimizer, epoch, {"total_loss": float("nan")}, f"{self.anomaly_checkpoint_name}_{anomaly_type}"
            )
            logger.info(f"Anomaly state saved to: {anomaly_path}")
        
        logger.info("Training stopped gracefully due to anomaly detection.")
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        loss_dict: Dict[str, float],
        checkpoint_name: str,
    ) -> Path:
        """
        Save model checkpoint.
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}_epoch_{epoch}.pt"
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss_dict": loss_dict,
            "history": self.history,
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        return checkpoint_path
    
    def get_history(self) -> Dict[str, list]:
        """Return training history."""
        return self.history
    
    def get_best_epoch_info(self) -> Dict[str, Any]:
        """Return information about the best epoch."""
        return {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "checkpoint_path": str(self.last_valid_checkpoint_path) if self.last_valid_checkpoint_path else None,
        }


class TrainingLoop:
    """
    Industrial-grade training loop with integrated monitoring.
    
    This class wraps the PINNModel training process with:
    - Integrated TrainingMonitor for anomaly detection
    - Configurable callbacks and metrics tracking
    - Support for validation and early stopping
    - Comprehensive logging and checkpointing
    """
    
    def __init__(
        self,
        config: PINNConfig,
        monitor: Optional[TrainingMonitor] = None,
    ):
        """
        Initialize TrainingLoop.
        
        Args:
            config: PINN configuration object
            monitor: Optional TrainingMonitor instance (created from config if None)
        """
        self.config = config
        
        if monitor is None and config.monitor.enable_monitoring:
            monitor = TrainingMonitor(
                nan_tolerance=config.monitor.nan_tolerance,
                inf_tolerance=config.monitor.inf_tolerance,
                save_on_anomaly=config.monitor.save_on_anomaly,
                anomaly_checkpoint_name=config.monitor.anomaly_checkpoint_name,
                checkpoint_dir=config.train.checkpoint_dir,
                track_gradients=config.monitor.track_gradients,
                track_weights=config.monitor.track_weights,
            )
        
        self.monitor = monitor
        self.device = torch.device(config.hardware.device)
        
        logger.info(f"TrainingLoop initialized on device: {self.device}")
    
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[Callable] = None,
        epoch_start: int = 0,
    ) -> Dict[str, Any]:
        """
        Execute training loop with monitoring.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Optional validation data loader
            optimizer: Optional optimizer (created if None)
            scheduler: Optional learning rate scheduler
            loss_fn: Optional loss function
            epoch_start: Starting epoch number
            
        Returns:
            Dict containing training results and history
        """
        model = model.to(self.device)
        model.train()
        
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.train.lr,
                weight_decay=self.config.train.weight_decay,
            )
        
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.train.epochs
            )
        
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        best_loss = float("inf")
        patience_counter = 0
        
        logger.info(f"Starting training from epoch {epoch_start} to {self.config.train.epochs}")
        
        for epoch in range(epoch_start, self.config.train.epochs):
            epoch_loss = self._train_epoch(model, train_loader, optimizer, loss_fn)
            
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(model, val_loader, loss)
            
            loss_dict = {
                "total_loss": epoch_loss,
                "data_loss": epoch_loss,
                "constraint_loss": 0.0,
                "val_loss": val_loss if val_loss is not None else float("nan"),
            }
            
            if scheduler is not None:
                scheduler.step()
            
            if self.monitor is not None:
                should_continue = self.monitor.on_epoch_end(
                    epoch + 1, loss_dict, model, optimizer, scheduler
                )
                if not should_continue:
                    logger.info("Training stopped by monitor")
                    break
            
            if (epoch + 1) % self.config.train.log_interval == 0:
                log_msg = f"Epoch {epoch + 1}/{self.config.train.epochs}: Loss={epoch_loss:.6f}"
                if val_loss is not None:
                    log_msg += f", Val Loss={val_loss:.6f}"
                logger.info(log_msg)
            
            if val_loss is not None:
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.train.patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
        
        results = {
            "final_epoch": epoch + 1,
            "best_loss": best_loss,
            "final_loss": epoch_loss,
        }
        
        if self.monitor is not None:
            results["history"] = self.monitor.get_history()
            results["best_epoch_info"] = self.monitor.get_best_epoch_info()
        
        logger.info(f"Training completed: {results}")
        return results
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            predictions = model(features)
            loss = loss_fn(predictions.squeeze(), targets)
            
            loss.backward()
            
            if self.config.train.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.train.grad_clip_norm
                )
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float("nan")
    
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: Callable,
    ) -> float:
        """Validate the model."""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                predictions = model(features)
                loss = loss_fn(predictions.squeeze(), targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float("nan")


def train_pinn_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[PINNConfig] = None,
) -> Dict[str, Any]:
    """
    Convenience function to train a PINN model with monitoring.
    
    Args:
        model: PINN model to train
        train_loader: Training data loader
        val_loader: Optional validation data loader
        config: PINN configuration (uses defaults if None)
        
    Returns:
        Dict containing training results
    """
    if config is None:
        config = PINNConfig()
    
    training_loop = TrainingLoop(config)
    return training_loop.train(model, train_loader, val_loader)
