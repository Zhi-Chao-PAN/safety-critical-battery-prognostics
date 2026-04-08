"""
Benchmark Runner - Unified evaluation pipeline for all models.

Runs: 10 seeds x N models x LOGO CV, computes full metrics suite,
exports results with provenance.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.splitter import DataSplitter
from src.evaluation.target_adapter import (
    adapt_predictions_to_target,
    build_prediction_data,
    build_training_data,
)
from src.models.base import BatteryModel
from src.uncertainty.scoring import compute_all_metrics

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
        eol_threshold: float = 1.4,
    ):
        self.features = features
        self.target = target
        self.group_col = group_col
        self.n_seeds = n_seeds
        self.eol_threshold = eol_threshold
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

                    train_ordered, X_train, y_train, fit_kwargs = build_training_data(
                        train_df,
                        self.features,
                        model,
                        group_col=self.group_col,
                    )
                    test_ordered, X_test, predict_kwargs = build_prediction_data(
                        test_df,
                        self.features,
                        group_col=self.group_col,
                    )

                    t0 = time.time()
                    try:
                        model.fit(X_train, y_train, **fit_kwargs)
                        train_time = time.time() - t0

                        t1 = time.time()
                        mean, lower, upper = model.predict(X_test, **predict_kwargs)
                        infer_time = time.time() - t1

                        if len(mean) == 0:
                            continue

                        y_eval, mean_eval, lower_eval, upper_eval, _ = adapt_predictions_to_target(
                            model=model,
                            test_df=test_ordered,
                            mean=mean,
                            lower=lower,
                            upper=upper,
                            evaluation_target=self.target,
                            group_col=self.group_col,
                            eol_threshold=self.eol_threshold,
                        )

                        metrics = compute_all_metrics(y_eval, mean_eval, lower_eval, upper_eval)
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
