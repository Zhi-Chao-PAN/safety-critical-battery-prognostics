"""
Benchmark Runner - Unified evaluation pipeline for all models.

Runs: 10 seeds x N models x LOGO CV, computes full metrics suite,
exports results with provenance.
"""

import logging
import time
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BatteryModel
from src.uncertainty.scoring import compute_all_metrics
from src.data.splitter import DataSplitter

logger = logging.getLogger(__name__)


def _config_hash(params: dict) -> str:
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


class BenchmarkRunner:
    """Run comprehensive evaluation across models, folds, and seeds."""

    def __init__(
        self,
        features: list[str],
        target: str = "rul",
        group_col: str = "battery_id",
        n_seeds: int = 10,
        results_dir: str = "results",
    ):
        self.features = features
        self.target = target
        self.group_col = group_col
        self.n_seeds = n_seeds
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        df: pd.DataFrame,
        models: dict[str, BatteryModel],
        seeds: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Run full benchmark: models x folds x seeds.

        Returns:
            DataFrame with columns: model, fold, seed, + all metrics
        """
        seeds = seeds or list(range(42, 42 + self.n_seeds))
        all_results = []

        for model_name, model_template in models.items():
            logger.info(f"=== Benchmarking: {model_name} ===")

            for train_df, test_df, test_id in DataSplitter.logo_cv(df, self.group_col):
                for seed in seeds:
                    np.random.seed(seed)
                    try:
                        import torch
                        torch.manual_seed(seed)
                    except ImportError:
                        pass

                    import copy
                    model = copy.deepcopy(model_template)

                    X_train = train_df[self.features].values
                    y_train = train_df[self.target].values
                    X_test = test_df[self.features].values
                    y_test = test_df[self.target].values

                    t0 = time.time()
                    try:
                        model.fit(X_train, y_train)
                        train_time = time.time() - t0

                        t1 = time.time()
                        mean, lower, upper = model.predict(X_test)
                        infer_time = time.time() - t1

                        if len(mean) == 0:
                            continue

                        # Align y_test to prediction length (sequence models drop first N)
                        y_eval = y_test[-len(mean):]

                        metrics = compute_all_metrics(y_eval, mean, lower, upper)
                        metrics.update({
                            "model": model_name,
                            "fold": test_id,
                            "seed": seed,
                            "train_time_s": round(train_time, 2),
                            "infer_time_ms": round(infer_time * 1000, 2),
                            "config_hash": _config_hash(model.get_params()),
                        })
                        all_results.append(metrics)

                    except Exception as e:
                        logger.error(f"{model_name} fold={test_id} seed={seed}: {e}")
                        continue

        results_df = pd.DataFrame(all_results)

        # Save
        out_path = self.results_dir / "benchmark_results.csv"
        results_df.to_csv(out_path, index=False)
        logger.info(f"Results saved to {out_path}")

        # Summary
        if len(results_df) > 0:
            summary = results_df.groupby("model").agg(
                RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                CRPS_mean=("CRPS", "mean"), CRPS_std=("CRPS", "std"),
                PICP_mean=("PICP", "mean"),
                train_time=("train_time_s", "mean"),
            ).round(4)
            logger.info(f"\n{summary}")

        return results_df

    def run_ablation(
        self,
        df: pd.DataFrame,
        ablation_configs: dict[str, dict[str, Any]],
        base_model_class: type,
        seeds: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Run ablation study with different configurations.

        Args:
            ablation_configs: {name: {param: value}} overrides
            base_model_class: Model class to instantiate
        """
        seeds = seeds or [42, 43, 44]
        models = {}
        for name, overrides in ablation_configs.items():
            params = {"input_dim": len(self.features)}
            params.update(overrides)
            models[name] = base_model_class(**params)

        return self.run(df, models, seeds=seeds)
