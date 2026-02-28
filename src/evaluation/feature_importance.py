"""
Feature Importance Analysis - Permutation importance + SHAP-like analysis.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BatteryModel
from src.uncertainty.scoring import rmse

logger = logging.getLogger(__name__)


def permutation_importance(
    model: BatteryModel,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance for each feature.
    Shuffle one feature at a time, measure RMSE increase.
    """
    rng = np.random.default_rng(seed)

    # Baseline RMSE
    mean, _, _ = model.predict(X)
    y_eval = y[-len(mean):]
    baseline_rmse = rmse(y_eval, mean)

    results = []
    for i, fname in enumerate(feature_names):
        importances = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            mean_perm, _, _ = model.predict(X_perm)
            y_perm = y[-len(mean_perm):]
            perm_rmse = rmse(y_perm, mean_perm)
            importances.append(perm_rmse - baseline_rmse)

        results.append({
            "feature": fname,
            "importance_mean": np.mean(importances),
            "importance_std": np.std(importances),
        })

    df = pd.DataFrame(results).sort_values("importance_mean", ascending=False)
    logger.info(f"Top 5 features:\n{df.head()}")
    return df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: str = "figures/fig10_feature_importance.png",
):
    """Plot horizontal bar chart of feature importance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    df = importance_df.head(top_n).sort_values("importance_mean")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        df["feature"], df["importance_mean"],
        xerr=df["importance_std"], capsize=3,
        color="#0072B2", alpha=0.8,
    )
    ax.set_xlabel("RMSE Increase (Permutation Importance)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
