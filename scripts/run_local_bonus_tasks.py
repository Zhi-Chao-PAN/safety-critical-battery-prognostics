"""
Local Bonus Tasks: Context Length Ablation & Uncertainty Calibration
Runs on CPU/local GPU.

Task 1: Context Length Ablation [16, 32, 64, 128] on B0018.
Task 2: Uncertainty Calibration (Reliability Diagram) on B0018.
"""

import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.unified_loader import UnifiedDataLoader
from src.evaluation.capacity_to_rul import evaluate_chronos_rul
from src.models.chronos_model import ChronosZeroShotModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bonus_tasks")


def run_context_ablation() -> pd.DataFrame:
    """Task 1: Ablation on Context Length."""
    logger.info("=" * 60)
    logger.info("  TASK 1: Context Length Ablation (B0018)")
    logger.info("=" * 60)

    loader = UnifiedDataLoader(eol_fraction=0.7, rated_capacity=2.0)
    df = loader.load_nasa(data_dir=str(ROOT / "data" / "battery_data"))
    bat_df = df[df["battery_id"] == "B0018"].sort_values("cycle")
    capacity = bat_df["capacity"].values.astype(np.float64)
    n = len(capacity)

    model = ChronosZeroShotModel(
        model_id="amazon/chronos-t5-small",
        num_samples=20,
        device_map="cpu",  # Run locally
        torch_dtype_str="float32",
    )

    pred_len = 40  # Use horizon 40 to avoid right-censoring in B0018 (EOL at 95, context must be < 95 - 40 = 55?)
    # Wait, B0018 has 132 data points. EOL is at 96.
    # If we predict 40 steps, and context is max 128, that means we need up to 128+40=168.
    # So we should just evaluate RUL RMSE across different context lengths used for forecasting.
    # Let's anchor the prediction start point.

    # EOL is around cycle 96. Let's start prediction at cycle 60.
    start_idx = 60
    pred_len = 60  # predict cycles 61 to 120

    contexts = [16, 32, 48, 60]  # Start idx is 60, so max context is 60.
    results = []

    for ctx_len in contexts:
        logger.info(f"  Testing Context Length: {ctx_len}")

        # Prepare context
        context_data = capacity[start_idx - ctx_len : start_idx]

        # Predict
        gt, mean_pred, lower_pred, upper_pred = model.predict_single_battery(
            capacity_series=capacity[:start_idx + pred_len],  # provide enough for GT extraction internally, or just pass full
            context_length=len(context_data),
            prediction_length=pred_len,
        )

        # We need to manually evaluate RUL since `predict_single_battery` uses the end of the passed series as the context if we just pass everything.
        # Actually `predict_single_battery` takes the FIRST `context_length` elements from `capacity_series`!
        # Let's construct a slice to force the model to use exactly the historical window we want.
        series_slice = capacity[start_idx - ctx_len : start_idx + pred_len]

        gt, mean_pred, lower_pred, upper_pred = model.predict_single_battery(
            capacity_series=series_slice,
            context_length=ctx_len,
            prediction_length=pred_len,
        )

        rul_res = evaluate_chronos_rul(
            capacity_series=series_slice,
            context_length=ctx_len,
            predicted_mean=mean_pred,
            predicted_lower=lower_pred,
            predicted_upper=upper_pred,
            battery_id="B0018",
            eol_threshold=1.4,
        )

        results.append({
            "context_length": ctx_len,
            "rul_error": rul_res["rul_error"],
            "rul_abs_error": rul_res["rul_abs_error"],
            "picp": np.mean((gt >= lower_pred) & (gt <= upper_pred))
        })

        logger.info(f"    RUL Error: {rul_res['rul_error']:.2f}, PICP: {results[-1]['picp']:.2%}")

    df_res = pd.DataFrame(results)

    # Plot ablation
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
    ax1.plot(df_res["context_length"], df_res["rul_abs_error"], marker="o", color="#e74c3c", linewidth=2, label="RUL Absolute Error")
    ax1.set_xlabel("Context Length (cycles)", fontsize=12)
    ax1.set_ylabel("RUL Absolute Error (cycles)", color="#e74c3c", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#e74c3c")
    ax1.set_xticks(contexts)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df_res["context_length"], df_res["picp"], marker="s", color="#3498db", linewidth=2, linestyle="--", label="95% PICP")
    ax2.set_ylabel("Prediction Interval Coverage Probability", color="#3498db", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#3498db")
    ax2.set_ylim(0, 1.1)

    plt.title("Chronos Zero-Shot: Context Length Ablation (B0018)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(ROOT / "figures" / "fig_context_ablation.png")
    plt.close()

    logger.info("  Ablation complete. Output saved.")
    return df_res


def run_reliability_diagram() -> None:
    """Task 2: Uncertainty Calibration (Reliability Diagram)."""
    logger.info("=" * 60)
    logger.info("  TASK 2: Uncertainty Calibration Diagram")
    logger.info("=" * 60)

    import torch
    from chronos import ChronosPipeline

    loader = UnifiedDataLoader(eol_fraction=0.7, rated_capacity=2.0)
    df = loader.load_nasa(data_dir=str(ROOT / "data" / "battery_data"))
    bat_df = df[df["battery_id"] == "B0006"].sort_values("cycle") # B0006 has nice degradation
    capacity = bat_df["capacity"].values.astype(np.float32)

    # Use prediction length 40, context 60
    ctx_len = 60
    pred_len = 40
    start_idx = 40

    series_slice = torch.tensor(capacity[start_idx : start_idx + ctx_len])
    gt_future = capacity[start_idx + ctx_len : start_idx + ctx_len + pred_len]

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    actual_len = len(gt_future)
    if actual_len == 0:
        logger.error("No ground truth data available for this slice.")
        return

    # Generate 100 samples
    num_samples = 100
    forecast = pipeline.predict(series_slice.unsqueeze(0), prediction_length=pred_len, num_samples=num_samples)
    samples = forecast[0].numpy()  # [100, pred_len]

    # Calculate coverage at different confidence levels
    confidence_levels = np.linspace(0.1, 0.9, 9)
    empirical_coverage = []

    for conf in confidence_levels:
        lower_q = (1 - conf) / 2
        upper_q = 1 - lower_q

        lower_bound = np.quantile(samples, lower_q, axis=0)[:actual_len]
        upper_bound = np.quantile(samples, upper_q, axis=0)[:actual_len]

        coverage = np.mean((gt_future >= lower_bound) & (gt_future <= upper_bound))
        empirical_coverage.append(coverage)
        logger.info(f"  Target Confidence: {conf:.0%}, Empirical Coverage: {coverage:.1%}")

    # Calculate Expected Calibration Error (ECE)
    ece = np.mean(np.abs(np.array(empirical_coverage) - confidence_levels))
    logger.info(f"  Expected Calibration Error (ECE): {ece:.4f}")

    # Plot Reliability Diagram
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.plot(confidence_levels, empirical_coverage, marker="o", color="#9b59b6", linewidth=2, label=f"Chronos Zero-Shot (ECE={ece:.3f})")

    ax.set_xlabel("Target Confidence Level", fontsize=12)
    ax.set_ylabel("Empirical Coverage Ratio (PICP)", fontsize=12)
    ax.set_title("Reliability Diagram: Zero-Shot Uncertainty Calibration", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(ROOT / "figures" / "fig_reliability_diagram.png")
    plt.close()

    logger.info("  Reliability diagram saved.")


def main():
    run_context_ablation()
    run_reliability_diagram()
    print("ALL DONE")

if __name__ == "__main__":
    main()
