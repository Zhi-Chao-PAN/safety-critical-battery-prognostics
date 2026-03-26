"""
Cross-Chemistry Transfer Learning Experiment.

Demonstrates few-shot adaptation across battery types.
Pre-train on N-1 batteries, fine-tune on K cycles of target battery.

Usage:
    python scripts/run_transfer.py --n-shots 5 10 20 30
"""

import argparse
import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.splitter import DataSplitter
from src.data.unified_loader import UnifiedDataLoader
from src.data.validator import DataValidator
from src.features.extractor import FeatureExtractor
from src.models import LSTMModel, PINNModel
from src.uncertainty.scoring import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_transfer_experiment(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_battery: str,
    n_shots_list: list[int],
    epochs_pretrain: int = 80,
    epochs_finetune: int = 30,
    seeds: list[int] = None,
) -> pd.DataFrame:
    seeds = seeds or [42, 43, 44]
    results = []
    input_dim = len(feature_cols)

    for n_shots in n_shots_list:
        for seed in seeds:
            np.random.seed(seed)

            pretrain_df, finetune_df, test_df = DataSplitter.few_shot_split(
                df, target_battery=target_battery, n_shots=n_shots
            )

            if len(test_df) < 5:
                logger.warning(f"Not enough test data for {target_battery} with {n_shots} shots")
                continue

            X_pre = pretrain_df[feature_cols].values
            y_pre = pretrain_df["rul"].values
            X_ft = finetune_df[feature_cols].values
            y_ft = finetune_df["rul"].values
            X_test = test_df[feature_cols].values
            y_test = test_df["rul"].values

            for model_name, model_cls, kwargs in [
                ("lstm", LSTMModel, {"hidden_dim": 64, "num_layers": 2, "seq_length": 20}),
                ("pinn", PINNModel, {"hidden_dim": 64}),
            ]:
                import torch
                torch.manual_seed(seed)

                # 1. Pre-train on other batteries
                model = model_cls(input_dim=input_dim, epochs=epochs_pretrain, **kwargs)
                model.fit(X_pre, y_pre)

                # 2. Evaluate zero-shot (no fine-tuning)
                mean, lower, upper = model.predict(X_test)
                if len(mean) > 0:
                    y_eval = y_test[-len(mean):]
                    metrics_zero = compute_all_metrics(y_eval, mean, lower, upper)
                    results.append({
                        "model": model_name, "target": target_battery,
                        "n_shots": 0, "seed": seed, "type": "zero_shot",
                        **metrics_zero,
                    })

                # 3. Fine-tune on K shots
                if len(X_ft) > 5:
                    ft_model = copy.deepcopy(model)
                    ft_model.epochs = epochs_finetune
                    # Combine pretrain + finetune data (with emphasis on finetune)
                    X_combined = np.vstack([X_pre[-50:], X_ft])  # Last 50 pretrain + finetune
                    y_combined = np.concatenate([y_pre[-50:], y_ft])
                    ft_model.fit(X_combined, y_combined)

                    mean, lower, upper = ft_model.predict(X_test)
                    if len(mean) > 0:
                        y_eval = y_test[-len(mean):]
                        metrics_ft = compute_all_metrics(y_eval, mean, lower, upper)
                        results.append({
                            "model": model_name, "target": target_battery,
                            "n_shots": n_shots, "seed": seed, "type": "fine_tuned",
                            **metrics_ft,
                        })

            logger.info(f"Completed: target={target_battery}, n_shots={n_shots}, seed={seed}")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shots", nargs="+", type=int, default=[5, 10, 20, 30])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    # Load data
    loader = UnifiedDataLoader()
    df = loader.load_all(nasa_dir=str(ROOT / "data" / "battery_data"))
    validator = DataValidator()
    df, _ = validator.validate(df)
    extractor = FeatureExtractor()
    df = extractor.extract_all(df)
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ("cycle", "rul")]
    df = df.dropna(subset=feature_cols + ["rul"]).reset_index(drop=True)

    # Run for each battery as target
    all_results = []
    for target in df["battery_id"].unique():
        logger.info(f"=== Transfer target: {target} ===")
        res = run_transfer_experiment(
            df, feature_cols, target_battery=target,
            n_shots_list=args.n_shots, seeds=args.seeds,
        )
        all_results.append(res)

    results_df = pd.concat(all_results, ignore_index=True)
    out_path = ROOT / "results" / "transfer_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    # Summary
    summary = results_df.groupby(["model", "type", "n_shots"]).agg(
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        CRPS_mean=("CRPS", "mean"),
    ).round(4)
    logger.info(f"\n{summary}")
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
