"""
Industrial-Grade Configuration Management for PINN Battery Prognostics.

This module provides Pydantic-based strongly-typed configuration management
to eliminate hardcoded values scattered across the codebase.

Key Features:
- Type-safe configuration with automatic validation
- Single source of truth from config.yaml
- Hardware-aware settings (CPU/GPU, mixed precision)
- Physics parameter management
- Training hyperparameter centralization
"""

from pathlib import Path
from typing import Literal, Optional

import torch
import yaml
from pydantic import BaseModel, Field, field_validator


class HardwareConfig(BaseModel):
    """Hardware configuration for device and precision settings."""
    
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="Device to use for training. 'auto' selects CUDA if available."
    )
    
    use_mixed_precision: bool = Field(
        default=True,
        description="Enable automatic mixed precision training for RTX 4060 Tensor Cores."
    )
    
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of worker processes for data loading. 0 for single-process."
    )
    
    pin_memory: bool = Field(
        default=True,
        description="Pin memory for faster GPU data transfer."
    )
    
    deterministic: bool = Field(
        default=False,
        description="Enable deterministic operations for reproducibility (may impact performance)."
    )
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate and resolve device selection."""
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if v == "cuda" and not torch.cuda.is_available():
            import warnings
            warnings.warn("CUDA requested but not available, falling back to CPU")
            return "cpu"
        return v


class PhysicsConfig(BaseModel):
    """Physics configuration for battery degradation model parameters."""
    
    rated_capacity: float = Field(
        default=2.0,
        gt=0,
        description="Rated capacity of the battery in Ah."
    )
    
    eol_fraction: float = Field(
        default=0.7,
        gt=0,
        le=1,
        description="End-of-life fraction of rated capacity (default: 70%)."
    )
    
    lambda_physics: float = Field(
        default=0.1,
        ge=0,
        description="Base weight for physics constraint loss."
    )
    
    lambda_mono: float = Field(
        default=0.05,
        ge=0,
        description="Base weight for monotonicity constraint loss."
    )
    
    adaptive_weighting: bool = Field(
        default=True,
        description="Enable adaptive loss weighting based on battery lifecycle stage."
    )
    
    lambda_physics_min: float = Field(
        default=0.01,
        ge=0,
        description="Minimum physics constraint weight (early lifecycle)."
    )
    
    lambda_physics_max: float = Field(
        default=1.0,
        ge=0,
        description="Maximum physics constraint weight (late lifecycle)."
    )
    
    lambda_mono_min: float = Field(
        default=0.01,
        ge=0,
        description="Minimum monotonicity constraint weight."
    )
    
    lambda_mono_max: float = Field(
        default=0.2,
        ge=0,
        description="Maximum monotonicity constraint weight."
    )
    
    transition_sharpness: float = Field(
        default=10.0,
        gt=0,
        description="Sharpness of sigmoid transition for adaptive weighting."
    )
    
    transition_center: float = Field(
        default=0.6,
        gt=0,
        le=1,
        description="Center point of sigmoid transition (normalized cycle position)."
    )


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration."""
    
    data_dir: str = Field(
        default="data/battery_data",
        description="Root directory for battery datasets."
    )
    
    datasets: list[str] = Field(
        default=["nasa"],
        description="List of datasets to load: ['nasa', 'calce', 'oxford', 'mit_stanford']."
    )
    
    battery_ids: Optional[list[str]] = Field(
        default=None,
        description="Specific battery IDs to load. None loads all available."
    )
    
    val_fraction: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="Fraction of data to use for validation."
    )
    
    test_fraction: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="Fraction of data to use for testing."
    )
    
    batch_size: int = Field(
        default=32,
        gt=0,
        description="Batch size for training."
    )
    
    shuffle: bool = Field(
        default=True,
        description="Shuffle training data."
    )
    
    drop_last: bool = Field(
        default=True,
        description="Drop last incomplete batch."
    )
    
    enable_anomaly_detection: bool = Field(
        default=True,
        description="Enable automatic anomaly detection in data loading."
    )
    
    nan_replacement: str = Field(
        default="interpolate",
        description="Strategy for handling NaN values: 'interpolate', 'zero', 'drop'."
    )
    
    clip_outliers: bool = Field(
        default=True,
        description="Clip extreme outlier values."
    )
    
    outlier_std_threshold: float = Field(
        default=5.0,
        gt=0,
        description="Number of standard deviations for outlier clipping."
    )


class TrainConfig(BaseModel):
    """Training configuration for PINN model."""
    
    input_dim: int = Field(
        default=2,
        gt=0,
        description="Input feature dimension."
    )
    
    hidden_dim: int = Field(
        default=64,
        gt=0,
        description="Hidden layer dimension for neural network."
    )
    
    dropout: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="Dropout rate for neural network."
    )
    
    lr: float = Field(
        default=1e-3,
        gt=0,
        description="Learning rate for optimizer."
    )
    
    epochs: int = Field(
        default=150,
        gt=0,
        description="Maximum number of training epochs."
    )
    
    patience: int = Field(
        default=15,
        gt=0,
        description="Early stopping patience (epochs without improvement)."
    )
    
    mc_samples: int = Field(
        default=100,
        gt=0,
        description="Number of Monte Carlo samples for uncertainty quantification."
    )
    
    weight_decay: float = Field(
        default=1e-4,
        ge=0,
        description="L2 regularization weight decay."
    )
    
    grad_clip_norm: float = Field(
        default=1.0,
        ge=0,
        description="Gradient clipping norm threshold."
    )
    
    checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory for saving model checkpoints."
    )
    
    save_best_only: bool = Field(
        default=True,
        description="Only save the best checkpoint (based on validation loss)."
    )
    
    log_interval: int = Field(
        default=10,
        gt=0,
        description="Log training metrics every N epochs."
    )


class MonitorConfig(BaseModel):
    """Training monitoring and safety configuration."""
    
    enable_monitoring: bool = Field(
        default=True,
        description="Enable training monitoring and anomaly detection."
    )
    
    nan_tolerance: int = Field(
        default=3,
        ge=0,
        description="Number of consecutive NaN/Inf losses before graceful exit."
    )
    
    inf_tolerance: int = Field(
        default=3,
        ge=0,
        description="Number of consecutive Inf losses before graceful exit."
    )
    
    save_on_anomaly: bool = Field(
        default=True,
        description="Save checkpoint when anomaly is detected."
    )
    
    anomaly_checkpoint_name: str = Field(
        default="anomaly_recovery",
        description="Base name for anomaly recovery checkpoint."
    )
    
    track_gradients: bool = Field(
        default=False,
        description="Track gradient statistics during training."
    )
    
    track_weights: bool = Field(
        default=False,
        description="Track weight statistics during training."
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level."
    )
    
    log_file: Optional[str] = Field(
        default=None,
        description="Path to log file. None logs to console only."
    )
    
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format."
    )


class PINNConfig(BaseModel):
    """
    Master configuration class for PINN Battery Prognostics.
    
    This class aggregates all sub-configurations and provides a single
    source of truth for the entire system.
    """
    
    name: str = Field(
        default="pinn_battery_prognostics",
        description="Experiment name."
    )
    
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility."
    )
    
    hardware: HardwareConfig = Field(
        default_factory=HardwareConfig,
        description="Hardware and device configuration."
    )
    
    physics: PhysicsConfig = Field(
        default_factory=PhysicsConfig,
        description="Physics model parameters."
    )
    
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data loading and preprocessing configuration."
    )
    
    train: TrainConfig = Field(
        default_factory=TrainConfig,
        description="Training hyperparameters."
    )
    
    monitor: MonitorConfig = Field(
        default_factory=MonitorConfig,
        description="Training monitoring and safety configuration."
    )
    
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration."
    )
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "PINNConfig":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file.
            
        Returns:
            PINNConfig: Validated configuration instance.
            
        Raises:
            FileNotFoundError: If config file does not exist.
            ValidationError: If configuration is invalid.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
        
        return cls(**raw_config)
    
    def to_yaml(self, output_path: str | Path) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path to save YAML configuration.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(mode="python"), f, default_flow_style=False, sort_keys=False)
    
    def get_pinn_model_kwargs(self) -> dict:
        """
        Get keyword arguments for PINNModel initialization.
        
        Returns:
            dict: Keyword arguments compatible with PINNModel.__init__.
        """
        return {
            "input_dim": self.train.input_dim,
            "hidden_dim": self.train.hidden_dim,
            "dropout": self.train.dropout,
            "lr": self.train.lr,
            "epochs": self.train.epochs,
            "patience": self.train.patience,
            "lambda_physics": self.physics.lambda_physics,
            "lambda_mono": self.physics.lambda_mono,
            "adaptive_weighting": self.physics.adaptive_weighting,
            "lambda_physics_min": self.physics.lambda_physics_min,
            "lambda_physics_max": self.physics.lambda_physics_max,
            "lambda_mono_min": self.physics.lambda_mono_min,
            "lambda_mono_max": self.physics.lambda_mono_max,
            "transition_sharpness": self.physics.transition_sharpness,
            "transition_center": self.physics.transition_center,
            "mc_samples": self.train.mc_samples,
            "device": self.hardware.device,
            "use_mixed_precision": self.hardware.use_mixed_precision,
        }
    
    def get_dataloader_kwargs(self) -> dict:
        """
        Get keyword arguments for DataLoader initialization.
        
        Returns:
            dict: Keyword arguments compatible with torch.utils.data.DataLoader.
        """
        return {
            "batch_size": self.data.batch_size,
            "shuffle": self.data.shuffle,
            "num_workers": self.hardware.num_workers,
            "pin_memory": self.hardware.pin_memory,
            "drop_last": self.data.drop_last,
        }


def load_config(config_path: str | Path = "configs/pinn_config.yaml") -> PINNConfig:
    """
    Convenience function to load PINN configuration from YAML.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        PINNConfig: Validated configuration instance.
    """
    return PINNConfig.from_yaml(config_path)
