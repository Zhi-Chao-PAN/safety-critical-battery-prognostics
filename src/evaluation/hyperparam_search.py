"""
Hyperparameter Search - Grid/Random search for optimal model configs.
"""

import copy
import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BatteryModel
from src.data.splitter import DataSplitter
from src.uncertainty.scoring import compute_all_metrics

logger = logging.getLogger(__name__)


def grid_search(
    model_class: type,
    param_grid: dict[str, list],
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "rul",
    metric: str = "RMSE",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Grid search over hyperparameters using LOGO-CV.

    Args:
        model_class: BatteryModel subclass
        param_grid: {param_name: [values]}
        metric: Metric to optimize (lower is better)

    Returns:
        DataFrame with all configs and their scores
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    logger.info(f"Grid search: {len(combos)} configurations")
    results = []

    for combo in combos:
        params = dict(zip(keys, combo))
        params["input_dim"] = len(feature_cols)

        try:
            model = model_class(**params)
        except Exception as e:
            logger.warning(f"Skip {params}: {e}")
            continue

        fold_metrics = []
        np.random.seed(seed)

        for train_df, test_df, test_id in DataSplitter.logo_cv(df):
            m = copy.deepcopy(model)
            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values
            X_test = test_df[feature_cols].values
            y_test = test_df[target_col].values

            try:
                m.fit(X_train, y_train)
                mean, lower, upper = m.predict(X_test)
                if len(mean) > 0:
                    y_eval = y_test[-len(mean):]
                    metrics = compute_all_metrics(y_eval, mean, lower, upper)
                    fold_metrics.append(metrics[metric])
            except Exception:
                continue

        if fold_metrics:
            result = {**params, f"{metric}_mean": np.mean(fold_metrics),
                      f"{metric}_std": np.std(fold_metrics)}
            results.append(result)
            logger.info(f"  {params} → {metric}={np.mean(fold_metrics):.4f}±{np.std(fold_metrics):.4f}")

    results_df = pd.DataFrame(results).sort_values(f"{metric}_mean")
    return results_df


def random_search(
    model_class: type,
    param_distributions: dict[str, Any],
    df: pd.DataFrame,
    feature_cols: list[str],
    n_iter: int = 20,
    target_col: str = "rul",
    metric: str = "RMSE",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Random search over hyperparameters.

    Args:
        param_distributions: {param_name: (low, high, type)}
            type: 'int', 'float', 'log_float', 'choice'
    """
    rng = np.random.default_rng(seed)
    results = []

    for i in range(n_iter):
        params = {"input_dim": len(feature_cols)}
        for name, spec in param_distributions.items():
            if spec[2] == "int":
                params[name] = int(rng.integers(spec[0], spec[1] + 1))
            elif spec[2] == "float":
                params[name] = float(rng.uniform(spec[0], spec[1]))
            elif spec[2] == "log_float":
                params[name] = float(np.exp(rng.uniform(np.log(spec[0]), np.log(spec[1]))))
            elif spec[2] == "choice":
                params[name] = rng.choice(spec[0])

        try:
            model = model_class(**params)
        except Exception:
            continue

        fold_metrics = []
        for train_df, test_df, test_id in DataSplitter.logo_cv(df):
            m = copy.deepcopy(model)
            try:
                m.fit(train_df[feature_cols].values, train_df[target_col].values)
                mean, _, _ = m.predict(test_df[feature_cols].values)
                if len(mean) > 0:
                    y_eval = test_df[target_col].values[-len(mean):]
                    metrics = compute_all_metrics(y_eval, mean, np.zeros_like(mean), np.zeros_like(mean))
                    fold_metrics.append(metrics[metric])
            except Exception:
                continue

        if fold_metrics:
            result = {**params, f"{metric}_mean": np.mean(fold_metrics)}
            results.append(result)
            logger.info(f"  [{i+1}/{n_iter}] {metric}={np.mean(fold_metrics):.4f}")

    return pd.DataFrame(results).sort_values(f"{metric}_mean")
