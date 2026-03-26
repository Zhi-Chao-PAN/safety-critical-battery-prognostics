"""
Cross-Chemistry Transfer Experiment.
Pre-train on NASA (LiCoO2), few-shot adapt to CALCE/Oxford.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.splitter import DataSplitter
from src.data.unified_loader import UnifiedDataLoader
from src.features.extractor import FeatureExtractor
from src.models import GRUModel, LSTMModel
from src.uncertainty.scoring import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_transfer_experiment(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "rul",
    n_shots_list: list[int] = None,
    model_configs: dict = None,
):
    """
    Run few-shot transfer: Pre-train on source, fine-tune on N shots of target.

    Args:
        source_df: Source domain data (e.g., NASA LiCoO2)
        target_df: Target domain data (e.g., CALCE CS2)
        feature_cols: Feature column names
        n_shots_list: List of shot counts to evaluate
        model_configs: {name: model_instance}
    """
    if n_shots_list is None:
        n_shots_list = [5, 10, 20, 50]

    if model_configs is None:
        common = dict(input_dim=len(feature_cols), seq_length=20, epochs=50, mc_samples=50)
        model_configs = {
            "lstm": LSTMModel(**common, hidden_dim=64, num_layers=2),
            "gru": GRUModel(**common, hidden_dim=64, num_layers=2),
        }

    results = []
    target_batteries = target_df["battery_id"].unique()

    for target_bat in target_batteries:
        for n_shots in n_shots_list:
            pretrain_df, finetune_df, test_df = DataSplitter.few_shot_split(
                target_df, target_battery=target_bat, n_shots=n_shots
            )
            # Combine source + other target batteries for pre-training
            full_pretrain = pd.concat([source_df, pretrain_df], ignore_index=True)

            for model_name, model_template in model_configs.items():
                import copy
                model = copy.deepcopy(model_template)

                # Pre-train on source + other target batteries
                X_pretrain = full_pretrain[feature_cols].values
                y_pretrain = full_pretrain[target_col].values
                model.fit(X_pretrain, y_pretrain)

                # Fine-tune on N shots (re-fit on combined)
                X_finetune = np.vstack([X_pretrain, finetune_df[feature_cols].values])
                y_finetune = np.concatenate([y_pretrain, finetune_df[target_col].values])
                model.fit(X_finetune, y_finetune)

                # Evaluate on remaining target cycles
                X_test = test_df[feature_cols].values
                y_test = test_df[target_col].values
                mean, lower, upper = model.predict(X_test)

                if len(mean) > 0:
                    y_eval = y_test[-len(mean):]
                    metrics = compute_all_metrics(y_eval, mean, lower, upper)
                    metrics.update({
                        "model": model_name,
                        "target_battery": target_bat,
                        "n_shots": n_shots,
                        "test_size": len(y_eval),
                    })
                    results.append(metrics)
                    logger.info(
                        f"{model_name} | {target_bat} | {n_shots}-shot: "
                        f"RMSE={metrics['RMSE']:.2f}, PICP={metrics['PICP']:.2f}"
                    )

    results_df = pd.DataFrame(results)
    out_path = ROOT / "results" / "transfer_results.csv"
    out_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info(f"Transfer results saved to {out_path}")

    # Summary
    if len(results_df) > 0:
        summary = results_df.groupby(["model", "n_shots"]).agg(
            RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
            PICP_mean=("PICP", "mean"),
        ).round(4)
        logger.info(f"\nTransfer Summary:\n{summary}")

    return results_df


def main():
    """Run transfer experiment with NASA data (cross-battery as proxy for cross-chemistry)."""
    loader = UnifiedDataLoader()
    df = loader.load_all(nasa_dir=str(ROOT / "data" / "battery_data"))

    extractor = FeatureExtractor()
    df = extractor.extract_all(df)

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ("cycle", "rul")]
    df = df.dropna(subset=feature_cols + ["rul"]).reset_index(drop=True)

    # Use B0005/B0006/B0007 as source, B0018 as target (different aging profile)
    source_df = df[df["battery_id"].isin(["B0005", "B0006", "B0007"])]
    target_df = df[df["battery_id"] == "B0018"]

    run_transfer_experiment(
        source_df, target_df, feature_cols,
        n_shots_list=[5, 10, 20],
    )


if __name__ == "__main__":
    main()
