"""
Robust PyTorch Dataset and DataLoader for Battery Prognostics.

This module provides industrial-grade data loading with:
- Native PyTorch Dataset and DataLoader integration
- pin_memory=True for accelerated GPU data transfer
- Automatic anomaly detection and handling (NaN, outliers)
- Configurable data preprocessing and validation
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.unified_loader import UnifiedDataLoader

logger = logging.getLogger(__name__)


class BatteryDataset(Dataset):
    """
    PyTorch Dataset for battery prognostics with robust anomaly handling.
    
    This dataset class manages battery cycle data with built-in preprocessing
    capabilities to handle real-world data quality issues common in industrial
    battery monitoring systems. It supports configurable anomaly detection,
    NaN handling, and outlier clipping to ensure data quality for model training.
    
    Features:
    - Automatic NaN detection and interpolation
    - Outlier clipping based on statistical thresholds
    - Configurable preprocessing pipeline
    - Memory-efficient data access
    
    Attributes:
        data: Preprocessed DataFrame containing battery cycle data
        feature_columns: List of column names used as input features
        target_column: Name of column containing target values (RUL or capacity)
        enable_anomaly_detection: Flag indicating if anomaly detection is enabled
        nan_replacement: Strategy for handling missing values
        clip_outliers: Flag indicating if outlier clipping is enabled
        outlier_std_threshold: Number of standard deviations for outlier detection
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "rul",
        enable_anomaly_detection: bool = True,
        nan_replacement: str = "interpolate",
        clip_outliers: bool = True,
        outlier_std_threshold: float = 5.0,
    ):
        """
        Initialize BatteryDataset.
        
        Args:
            data: DataFrame containing battery cycle data with the following columns:
                cycle: Cycle number (count of charge-discharge cycles completed)
                capacity: Measured discharge capacity (Ah), decreases with degradation
                resistance: Internal resistance (Ohm), increases with degradation
                temperature: Average operating temperature (°C) during cycle
                rul: Remaining Useful Life (cycles until capacity drops below 80% of rated)
            feature_columns: List of column names to use as input features
            target_column: Name of target column (default: 'rul')
            enable_anomaly_detection: Enable automatic anomaly detection and handling
            nan_replacement: Strategy for handling NaN values:
                'interpolate': Linear interpolation between adjacent valid values
                'zero': Replace NaN with 0.0
                'drop': Remove samples containing NaN values
            clip_outliers: Enable outlier clipping to statistical thresholds
            outlier_std_threshold: Number of standard deviations from mean considered an outlier
        """
        self.data: pd.DataFrame = data.copy()
        self.feature_columns: list[str] = feature_columns
        self.target_column: str = target_column
        self.enable_anomaly_detection: bool = enable_anomaly_detection
        self.nan_replacement: str = nan_replacement
        self.clip_outliers: bool = clip_outliers
        self.outlier_std_threshold: float = outlier_std_threshold
        
        self._validate_columns()
        self._preprocess_data()
        
        logger.info(
            f"BatteryDataset initialized: {len(self.data)} samples, "
            f"{len(feature_columns)} features"
        )
    
    def _validate_columns(self) -> None:
        """Validate that required columns exist in data."""
        missing_features = [col for col in self.feature_columns if col not in self.data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        
        if self.target_column not in self.data.columns:
            raise ValueError(f"Missing target column: {self.target_column}")
    
    def _preprocess_data(self) -> None:
        """Preprocess data with anomaly detection and handling."""
        if not self.enable_anomaly_detection:
            return
        
        original_len = len(self.data)
        
        for col in self.feature_columns + [self.target_column]:
            self._handle_nan_values(col)
            if self.clip_outliers:
                self._clip_outliers(col)
        
        cleaned_len = len(self.data)
        if cleaned_len < original_len:
            logger.warning(
                f"Anomaly detection removed {original_len - cleaned_len} samples "
                f"({100 * (original_len - cleaned_len) / original_len:.2f}%)"
            )
    
    def _handle_nan_values(self, column: str) -> None:
        """Handle NaN values in specified column."""
        nan_mask = self.data[column].isna()
        nan_count = nan_mask.sum()
        
        if nan_count == 0:
            return
        
        logger.warning(f"Found {nan_count} NaN values in column '{column}'")
        
        if self.nan_replacement == "drop":
            self.data = self.data[~nan_mask].reset_index(drop=True)
        elif self.nan_replacement == "zero":
            self.data.loc[nan_mask, column] = 0.0
        elif self.nan_replacement == "interpolate":
            self.data[column] = self.data[column].interpolate(method="linear", limit_direction="both")
            remaining_nan = self.data[column].isna().sum()
            if remaining_nan > 0:
                logger.warning(
                    f"Interpolation failed for {remaining_nan} values in '{column}', "
                    "falling back to forward fill"
                )
                self.data[column] = self.data[column].fillna(method="ffill").fillna(method="bfill")
        else:
            raise ValueError(f"Unknown nan_replacement strategy: {self.nan_replacement}")
    
    def _clip_outliers(self, column: str) -> None:
        """Clip outlier values based on statistical threshold."""
        values = self.data[column].values
        mean = np.nanmean(values)
        std = np.nanstd(values)
        
        if std == 0:
            return
        
        lower_bound = mean - self.outlier_std_threshold * std
        upper_bound = mean + self.outlier_std_threshold * std
        
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            logger.warning(
                f"Clipping {outlier_count} outliers in column '{column}' "
                f"({100 * outlier_count / len(values):.2f}%)"
            )
            self.data[column] = np.clip(values, lower_bound, upper_bound)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (features, target) as torch.Tensor
        """
        try:
            row = self.data.iloc[idx]
            
            features = row[self.feature_columns].values.astype(np.float32)
            target = np.float32(row[self.target_column])
            
            features_tensor = torch.from_numpy(features)
            target_tensor = torch.tensor(target)
            
            return features_tensor, target_tensor
            
        except Exception as e:
            logger.error(f"Error accessing sample at index {idx}: {e}")
            raise
    
    def get_statistics(self) -> dict:
        """Get dataset statistics for monitoring."""
        stats = {
            "num_samples": len(self.data),
            "num_features": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
        }
        
        for col in self.feature_columns + [self.target_column]:
            values = self.data[col].values
            stats[f"{col}_mean"] = float(np.nanmean(values))
            stats[f"{col}_std"] = float(np.nanstd(values))
            stats[f"{col}_min"] = float(np.nanmin(values))
            stats[f"{col}_max"] = float(np.nanmax(values))
            stats[f"{col}_nan_count"] = int(np.isnan(values).sum())
        
        return stats


def create_battery_dataloaders(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "rul",
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    enable_anomaly_detection: bool = True,
    nan_replacement: str = "interpolate",
    clip_outliers: bool = True,
    outlier_std_threshold: float = 5.0,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create train, validation, and test DataLoaders from a DataFrame.
    
    Args:
        data: DataFrame containing battery cycle data
        feature_columns: List of column names to use as features
        target_column: Name of target column
        val_fraction: Fraction of data for validation
        test_fraction: Fraction of data for testing
        batch_size: Batch size for DataLoaders
        shuffle: Shuffle training data
        num_workers: Number of worker processes
        pin_memory: Pin memory for GPU transfer
        drop_last: Drop last incomplete batch
        enable_anomaly_detection: Enable anomaly detection
        nan_replacement: Strategy for handling NaN
        clip_outliers: Enable outlier clipping
        outlier_std_threshold: Std threshold for outliers
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    np.random.seed(seed)
    
    total_samples = len(data)
    test_size = int(total_samples * test_fraction)
    val_size = int(total_samples * val_fraction)
    train_size = total_samples - test_size - val_size
    
    indices = np.random.permutation(total_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = data.iloc[train_indices].reset_index(drop=True)
    val_data = data.iloc[val_indices].reset_index(drop=True) if val_size > 0 else None
    test_data = data.iloc[test_indices].reset_index(drop=True) if test_size > 0 else None
    
    train_dataset = BatteryDataset(
        data=train_data,
        feature_columns=feature_columns,
        target_column=target_column,
        enable_anomaly_detection=enable_anomaly_detection,
        nan_replacement=nan_replacement,
        clip_outliers=clip_outliers,
        outlier_std_threshold=outlier_std_threshold,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    
    val_loader = None
    if val_data is not None and len(val_data) > 0:
        val_dataset = BatteryDataset(
            data=val_data,
            feature_columns=feature_columns,
            target_column=target_column,
            enable_anomaly_detection=enable_anomaly_detection,
            nan_replacement=nan_replacement,
            clip_outliers=clip_outliers,
            outlier_std_threshold=outlier_std_threshold,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    
    test_loader = None
    if test_data is not None and len(test_data) > 0:
        test_dataset = BatteryDataset(
            data=test_data,
            feature_columns=feature_columns,
            target_column=target_column,
            enable_anomaly_detection=enable_anomaly_detection,
            nan_replacement=nan_replacement,
            clip_outliers=clip_outliers,
            outlier_std_threshold=outlier_std_threshold,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    
    logger.info(
        f"DataLoaders created: train={len(train_dataset)}, "
        f"val={len(val_dataset) if val_loader else 0}, "
        f"test={len(test_dataset) if test_loader else 0}"
    )
    
    return train_loader, val_loader, test_loader


def load_and_create_dataloaders(
    data_dir: str = "data/battery_data",
    datasets: list[str] = ["nasa"],
    battery_ids: Optional[list[str]] = None,
    feature_columns: Optional[list[str]] = None,
    target_column: str = "rul",
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    enable_anomaly_detection: bool = True,
    nan_replacement: str = "interpolate",
    clip_outliers: bool = True,
    outlier_std_threshold: float = 5.0,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Load battery data and create DataLoaders in one step.
    
    Args:
        data_dir: Root directory for battery datasets
        datasets: List of datasets to load
        battery_ids: Specific battery IDs to load
        feature_columns: List of column names to use as features
        target_column: Name of target column
        val_fraction: Fraction of data for validation
        test_fraction: Fraction of data for testing
        batch_size: Batch size for DataLoaders
        shuffle: Shuffle training data
        num_workers: Number of worker processes
        pin_memory: Pin memory for GPU transfer
        drop_last: Drop last incomplete batch
        enable_anomaly_detection: Enable anomaly detection
        nan_replacement: Strategy for handling NaN
        clip_outliers: Enable outlier clipping
        outlier_std_threshold: Std threshold for outliers
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    loader = UnifiedDataLoader()
    
    if "nasa" in datasets:
        data = loader.load_nasa(data_dir, battery_ids)
    elif "calce" in datasets:
        data = loader.load_calce(data_dir.replace("battery_data", "calce"))
    else:
        data = loader.load_all(
            nasa_dir=data_dir,
            calce_dir=data_dir.replace("battery_data", "calce")
        )
    
    if feature_columns is None:
        feature_columns = ["cycle", "capacity"]
    
    return create_battery_dataloaders(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin=pin_memory,
        drop_last=drop_last,
        enable_anomaly_detection=enable_anomaly_detection,
        nan_replacement=nan_replacement,
        clip_outliers=clip_outliers,
        outlier_std_threshold=outlier_std_threshold,
        seed=seed,
    )
