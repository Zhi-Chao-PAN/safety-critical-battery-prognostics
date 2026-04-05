"""
Infrastructure module for PINN Battery Prognostics.

This module provides industrial-grade infrastructure components:
- Configuration management (config_schema.py)
- Robust data loading (dataset.py)
- Training loop with monitoring (train_loop.py)
"""

from src.infrastructure.config_schema import (
    DataConfig,
    HardwareConfig,
    LoggingConfig,
    MonitorConfig,
    PhysicsConfig,
    PINNConfig,
    TrainConfig,
    load_config,
)
from src.infrastructure.dataset import (
    BatteryDataset,
    create_battery_dataloaders,
    load_and_create_dataloaders,
)
from src.infrastructure.train_loop import (
    TrainingLoop,
    TrainingMonitor,
    train_pinn_model,
)

__all__ = [
    "PINNConfig",
    "HardwareConfig",
    "PhysicsConfig",
    "DataConfig",
    "TrainConfig",
    "MonitorConfig",
    "LoggingConfig",
    "load_config",
    "BatteryDataset",
    "create_battery_dataloaders",
    "load_and_create_dataloaders",
    "TrainingMonitor",
    "TrainingLoop",
    "train_pinn_model",
]
