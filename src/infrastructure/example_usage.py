"""
Example usage of the industrial-grade infrastructure for PINN Battery Prognostics.

This example demonstrates:
1. Loading configuration from YAML
2. Creating robust DataLoaders with anomaly detection
3. Training PINNModel with monitoring
4. Handling anomalies gracefully
"""

import logging

import numpy as np
import torch

from src.infrastructure import (
    load_and_create_dataloaders,
    load_config,
    train_pinn_model,
)
from src.models.pinn_model import PINNModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== PINN Battery Prognostics - Industrial Infrastructure Demo ===")
    
    config = load_config("configs/pinn_config.yaml")
    logger.info(f"Configuration loaded: {config.name}")
    logger.info(f"Device: {config.hardware.device}")
    logger.info(f"Mixed precision: {config.hardware.use_mixed_precision}")
    
    train_loader, val_loader, test_loader = load_and_create_dataloaders(
        data_dir=config.data.data_dir,
        datasets=config.data.datasets,
        battery_ids=config.data.battery_ids,
        feature_columns=["cycle", "capacity"],
        target_column="rul",
        val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.hardware.num_workers,
        pin_memory=config.hardware.pin_memory,
        drop_last=config.data.drop_last,
        enable_anomaly_detection=config.data.enable_anomaly_detection,
        nan_replacement=config.data.nan_replacement,
        clip_outliers=config.data.clip_outliers,
        outlier_std_threshold=config.data.outlier_std_threshold,
        seed=config.seed,
    )
    
    logger.info("DataLoaders created successfully")
    
    model = PINNModel(**config.get_pinn_model_kwargs())
    logger.info(f"PINNModel initialized: {model.name}")
    
    X_train = []
    y_train = []
    for features, targets in train_loader:
        X_train.append(features.numpy())
        y_train.append(targets.numpy())
    
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    
    logger.info("Starting PINN training with monitoring...")
    model.fit(X_train, y_train)
    
    logger.info("Training completed successfully!")
    
    if test_loader is not None:
        logger.info("Running inference on test set...")
        X_test = []
        for features, _ in test_loader:
            X_test.append(features.numpy())
        X_test = np.concatenate(X_test, axis=0)
        
        mean_pred, lower, upper = model.predict(X_test)
        logger.info(f"Predictions: mean={mean_pred.mean():.4f}, "
                   f"std={(upper - lower).mean() / 3.92:.4f}")
    
    logger.info("=== Demo completed successfully ===")


if __name__ == "__main__":
    main()
