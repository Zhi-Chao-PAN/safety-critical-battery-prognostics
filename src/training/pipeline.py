"""
Training Pipeline - Unified training with logging, checkpointing, early stopping.
"""

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.splitter import DataSplitter
from src.models.base import BatteryModel
from src.uncertainty.scoring import compute_all_metrics

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    End-to-end training pipeline:
    1. Data split
    2. Feature selection
    3. Model training with validation
    4. Checkpoint best model
    5. Evaluate on test set
    6. Log everything
    """

    def __init__(
        self,
        features: list[str],
        target: str = "capacity",
        group_col: str = "battery_id",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        self.features = features
        self.target = target
        self.group_col = group_col
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        model: BatteryModel,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Full LOGO-CV training + evaluation.
        Returns aggregated metrics.
        """
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass

        if getattr(model, "name", None) == "pinn" and self.target != "capacity":
            raise ValueError(
                "PINNModel requires target='capacity' because its physics prior models "
                "capacity fade, not RUL. Override the pipeline target before training."
            )

        all_metrics = []
        best_model = None
        best_rmse = float("inf")

        for train_df, val_df, test_df, test_id in DataSplitter.nested_cv(df, self.group_col):
            logger.info(f"Training fold: test={test_id}")

            m = copy.deepcopy(model)
            X_train = train_df[self.features].values
            y_train = train_df[self.target].values

            t0 = time.time()
            m.fit(X_train, y_train)
            train_time = time.time() - t0

            # Evaluate on test (for reporting)
            X_test = test_df[self.features].values
            y_test = test_df[self.target].values
            mean, lower, upper = m.predict(X_test)

            if len(mean) == 0:
                logger.warning(f"Empty predictions for fold {test_id}")
                continue

            y_eval = y_test[-len(mean):]
            metrics = compute_all_metrics(y_eval, mean, lower, upper)
            metrics["fold"] = test_id
            metrics["train_time_s"] = round(train_time, 2)
            all_metrics.append(metrics)

            # Track best model using VALIDATION set (not test set)
            # This prevents test-set information leak in checkpoint selection.
            X_val = val_df[self.features].values
            y_val = val_df[self.target].values
            val_mean, _, _ = m.predict(X_val)
            if len(val_mean) > 0:
                y_val_eval = y_val[-len(val_mean):]
                val_rmse = float(np.sqrt(np.mean((val_mean - y_val_eval) ** 2)))
            else:
                val_rmse = float("inf")

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_model = m

            logger.info(f"  Fold {test_id}: RMSE={metrics['RMSE']:.4f}, CRPS={metrics['CRPS']:.4f}, val_RMSE={val_rmse:.4f}")

        # Save best model
        if best_model is not None:
            ckpt_path = self.checkpoint_dir / f"{model.name}_best.pt"
            best_model.save(ckpt_path)
            logger.info(f"Best model saved: {ckpt_path}")

        # Aggregate
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            agg = {
                "model": model.name,
                "seed": seed,
                "n_folds": len(all_metrics),
            }
            for col in ["RMSE", "MAE", "CRPS", "PICP", "MPIW", "R2"]:
                if col in metrics_df.columns:
                    agg[f"{col}_mean"] = round(metrics_df[col].mean(), 4)
                    agg[f"{col}_std"] = round(metrics_df[col].std(), 4)

            # Save log
            log_path = self.log_dir / f"{model.name}_seed{seed}.json"
            with open(log_path, "w") as f:
                json.dump({"summary": agg, "folds": all_metrics}, f, indent=2, default=str)

            return agg

        return {"model": model.name, "error": "No valid folds"}

    def train_all_models(
        self,
        df: pd.DataFrame,
        models: dict[str, BatteryModel],
        seeds: list[int] | None = None,
    ) -> pd.DataFrame:
        """Train and evaluate all models across seeds."""
        seeds = seeds or [42]
        results = []

        for name, model in models.items():
            for seed in seeds:
                logger.info(f"=== {name} | seed={seed} ===")
                result = self.train_and_evaluate(df, model, seed=seed)
                results.append(result)

        results_df = pd.DataFrame(results)
        results_df.to_csv(self.log_dir / "all_results.csv", index=False)
        return results_df
