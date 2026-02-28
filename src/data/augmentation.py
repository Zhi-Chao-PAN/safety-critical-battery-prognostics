"""
Data Augmentation for Battery Time Series.
Techniques: jitter, scaling, window slicing, mixup.
"""

import numpy as np
import pandas as pd


def jitter(X: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to features."""
    return X + np.random.randn(*X.shape) * sigma


def scaling(X: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Random scaling per feature."""
    factors = np.random.normal(1.0, sigma, size=(1, X.shape[1]))
    return X * factors


def window_slice(X: np.ndarray, y: np.ndarray, reduce_ratio: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Randomly slice a window from each sequence."""
    target_len = max(int(len(X) * reduce_ratio), 2)
    start = np.random.randint(0, len(X) - target_len + 1)
    return X[start:start + target_len], y[start:start + target_len]


def mixup(X1: np.ndarray, y1: np.ndarray, X2: np.ndarray, y2: np.ndarray,
          alpha: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Mixup augmentation between two batteries."""
    min_len = min(len(X1), len(X2))
    lam = np.random.beta(alpha, alpha)
    X_mix = lam * X1[:min_len] + (1 - lam) * X2[:min_len]
    y_mix = lam * y1[:min_len] + (1 - lam) * y2[:min_len]
    return X_mix, y_mix


def augment_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "rul",
    group_col: str = "battery_id",
    n_augmented: int = 2,
    jitter_sigma: float = 0.01,
    scale_sigma: float = 0.05,
) -> pd.DataFrame:
    """
    Augment entire dataset by generating synthetic batteries.

    Args:
        df: Original DataFrame
        n_augmented: Number of augmented copies per battery
        jitter_sigma: Noise level for jitter
        scale_sigma: Noise level for scaling

    Returns:
        Augmented DataFrame (original + synthetic)
    """
    augmented = [df]

    for bat_id in df[group_col].unique():
        sub = df[df[group_col] == bat_id].sort_values("cycle")

        for aug_i in range(n_augmented):
            aug = sub.copy()
            aug[group_col] = f"{bat_id}_aug{aug_i}"

            # Apply jitter + scaling to features
            X = aug[feature_cols].values
            X = jitter(X, sigma=jitter_sigma)
            X = scaling(X, sigma=scale_sigma)
            aug[feature_cols] = X

            augmented.append(aug)

    result = pd.concat(augmented, ignore_index=True)
    return result
