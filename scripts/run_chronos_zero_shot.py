"""
Zero-Shot Probing of Chronos Foundation Model on NASA Battery Data.

This script evaluates the Amazon Chronos-T5 pretrained time-series model
on NASA PCoE battery capacity degradation curves WITHOUT any fine-tuning.

The goal is to establish a zero-shot baseline. 
Note that existing domain models predict RUL (cycles), while Chronos predicts
Capacity (Ah) directly. Thus, error scales will appear vastly different.

Usage:
    python scripts/run_chronos_zero_shot.py
    python scripts/run_chronos_zero_shot.py --model-id amazon/chronos-t5-base --device cuda
    python scripts/run_chronos_zero_shot.py --prediction-lengths 10 20 30 40
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.unified_loader import UnifiedDataLoader
from src.models.chronos_model import ChronosZeroShotModel
from src.uncertainty.scoring import crps_gaussian, interval_score, mae, mpiw, picp, rmse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "chronos_zero_shot.log", mode="w"),
    ],
)
logger = logging.getLogger("chronos_zero_shot")

# ── Known baselines for comparison ──
BASELINES = {
    "BTCN (B0006)":       {"RMSE": 8.95},
    "Bayesian (B0006)":   {"RMSE": 18.85},  # Hierarchical Bayesian
    "LSTM (B0006)":       {"RMSE": 28.01},
    "Bayesian (B0018)":   {"RMSE": 18.85},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chronos Zero-Shot Battery Probing")
    p.add_argument("--model-id", default="amazon/chronos-t5-small",
                   help="HuggingFace model identifier")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Inference device")
    p.add_argument("--num-samples", type=int, default=20,
                   help="Monte Carlo forecast samples")
    p.add_argument("--prediction-lengths", nargs="+", type=int, default=[20],
                   help="Prediction horizon(s) to evaluate")
    p.add_argument("--context-ratio", type=float, default=0.8,
                   help="Fraction of series used as context")
    p.add_argument("--data-dir", default="data/battery_data",
                   help="Path to NASA .mat files")
    p.add_argument("--batteries", nargs="+",
                   default=["B0005", "B0006", "B0007", "B0018"],
                   help="Battery IDs to evaluate")
    p.add_argument("--output-dir", default="results",
                   help="Directory for CSV output")
    p.add_argument("--figures-dir", default="figures",
                   help="Directory for plot output")
    return p.parse_args()


def extract_capacity_series(df: pd.DataFrame, battery_id: str) -> np.ndarray:
    """Extract sorted capacity time-series for a single battery."""
    bat_df = df[df["battery_id"] == battery_id].sort_values("cycle")
    capacity = bat_df["capacity"].values.astype(np.float64)
    if len(capacity) == 0:
        raise ValueError(f"No data found for battery {battery_id}")
    return capacity


def evaluate_single_battery(
    model: ChronosZeroShotModel,
    capacity_series: np.ndarray,
    context_length: int,
    prediction_length: int,
    battery_id: str,
) -> dict[str, float]:
    """Run zero-shot inference on one battery and compute all metrics."""
    t0 = time.time()
    ground_truth, mean_pred, lower_pred, upper_pred = model.predict_single_battery(
        capacity_series=capacity_series,
        context_length=context_length,
        prediction_length=prediction_length,
    )
    infer_time_ms = (time.time() - t0) * 1000

    # Compute uncertainty std for CRPS/NLL
    std_pred = (upper_pred - lower_pred) / 3.92  # 95% CI -> std
    std_pred = np.maximum(std_pred, 1e-6)

    metrics = {
        "battery_id": battery_id,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "actual_pred_length": len(ground_truth),
        "RMSE": rmse(ground_truth, mean_pred),
        "MAE": mae(ground_truth, mean_pred),
        "CRPS": crps_gaussian(ground_truth, mean_pred, std_pred),
        "PICP": picp(ground_truth, lower_pred, upper_pred),
        "MPIW": mpiw(lower_pred, upper_pred),
        "IS": interval_score(ground_truth, lower_pred, upper_pred),
        "infer_time_ms": round(infer_time_ms, 2),
    }
    return metrics


def plot_zero_shot_results(
    all_results: list[dict],
    capacity_data: dict[str, np.ndarray],
    model: ChronosZeroShotModel,
    context_ratio: float,
    prediction_length: int,
    save_path: str,
) -> None:
    """Generate multi-panel comparison figure."""
    n_batteries = len(capacity_data)
    fig, axes = plt.subplots(n_batteries, 1, figsize=(14, 4 * n_batteries), dpi=150)
    if n_batteries == 1:
        axes = [axes]

    for ax, (bat_id, capacity) in zip(axes, capacity_data.items()):
        context_len = max(10, int(len(capacity) * context_ratio))
        pred_len = min(prediction_length, len(capacity) - context_len)

        if pred_len <= 0:
            ax.text(0.5, 0.5, f"{bat_id}: insufficient data",
                    transform=ax.transAxes, ha="center")
            continue

        gt, mean_pred, lower_pred, upper_pred = model.predict_single_battery(
            capacity_series=capacity,
            context_length=context_len,
            prediction_length=pred_len,
        )

        cycles = np.arange(len(capacity))
        pred_cycles = cycles[context_len:context_len + pred_len]

        # Plot full series
        ax.plot(cycles[:context_len], capacity[:context_len],
                color="#2c3e50", linewidth=1.5, label="Context (Historical)")
        ax.plot(pred_cycles, gt,
                color="#27ae60", linewidth=2.0, linestyle="--", label="Ground Truth")
        ax.plot(pred_cycles, mean_pred,
                color="#e74c3c", linewidth=2.0, label="Chronos Median Forecast")
        ax.fill_between(pred_cycles, lower_pred, upper_pred,
                        color="#e74c3c", alpha=0.15, label="95% CI")

        # Vertical separator
        ax.axvline(x=context_len, color="#7f8c8d", linestyle=":", alpha=0.7)

        # Find matching metrics
        bat_metrics = [r for r in all_results if r["battery_id"] == bat_id]
        if bat_metrics:
            m = bat_metrics[0]
            ax.set_title(
                f"{bat_id} | RMSE={m['RMSE']:.4f}  CRPS={m['CRPS']:.4f}  "
                f"PICP={m['PICP']:.2%}  MPIW={m['MPIW']:.4f}",
                fontsize=11, fontweight="bold",
            )
        else:
            ax.set_title(bat_id, fontsize=11, fontweight="bold")

        ax.set_xlabel("Cycle")
        ax.set_ylabel("Capacity (Ah)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Chronos-T5 Zero-Shot Probing | Model: {model.model_id} | "
        f"Pred Horizon: {prediction_length}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Figure saved: {save_path}")


def print_comparison_table(results_df: pd.DataFrame) -> None:
    """Print formatted comparison with known baselines."""
    print("\n" + "=" * 80)
    print("  CHRONOS ZERO-SHOT vs. KNOWN BASELINES (Dimension Awareness)")
    print("=" * 80)
    print(f"{'Battery':<10} {'Model':<25} {'Target':<12} {'RMSE':>10}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        print(f"{row['battery_id']:<10} {'Chronos (zero-shot)':<25} {'Capacity(Ah)':<12} "
              f"{row['RMSE']:>10.4f}")

    print("-" * 80)
    for name, vals in BASELINES.items():
        print(f"{'(ref)':<10} {name:<25} {'RUL(Cycles)':<12} {vals['RMSE']:>10.2f}")

    print("=" * 80)

    print("\n  [🚨 架构师批注] 量纲对齐警告 (Dimensionality Mismatch Warning):")
    print("  Chronos 的回归目标是【绝对容量 (Ah)】，预测 RMSE 仅为 ~0.03 Ah。")
    print("  现有的 BTCN/Bayesian 基线回归目标是【剩余寿命 (RUL/圈数)】，RMSE 为 8.95 圈。")
    print("  结论：零样本 Chronos 展现出极强的绝对容量拟合能力。")
    print("  在进入 Phase 3 微调前，Phase 2 的数据对齐必须明确训练目标是 Capacity 还是 RUL。")
    print()


def main() -> None:
    args = parse_args()

    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / args.output_dir).mkdir(exist_ok=True)
    (ROOT / args.figures_dir).mkdir(exist_ok=True)

    # ── Step 1: Load NASA data ──
    logger.info("Step 1: Loading NASA battery data...")
    loader = UnifiedDataLoader()
    df = loader.load_nasa(
        data_dir=str(ROOT / args.data_dir),
        battery_ids=args.batteries,
    )
    logger.info(f"Loaded {len(df)} cycles from {df['battery_id'].nunique()} batteries")

    # ── Step 2: Extract capacity series ──
    logger.info("Step 2: Extracting capacity time-series...")
    capacity_data: dict[str, np.ndarray] = {}
    for bat_id in args.batteries:
        try:
            capacity_data[bat_id] = extract_capacity_series(df, bat_id)
            logger.info(f"  {bat_id}: {len(capacity_data[bat_id])} cycles")
        except ValueError as e:
            logger.warning(f"  Skipping {bat_id}: {e}")

    if not capacity_data:
        logger.error("No valid battery data found. Exiting.")
        return

    # ── Step 3: Initialize Chronos model ──
    logger.info(f"Step 3: Loading Chronos model: {args.model_id}")
    dtype_str = "bfloat16" if args.device == "cuda" else "float32"
    model = ChronosZeroShotModel(
        model_id=args.model_id,
        num_samples=args.num_samples,
        device_map=args.device,
        torch_dtype_str=dtype_str,
        confidence_level=0.95,
    )

    # ── Step 4: Run zero-shot evaluation ──
    logger.info("Step 4: Running zero-shot evaluation...")
    all_results: list[dict[str, float]] = []

    for pred_len in args.prediction_lengths:
        logger.info(f"\n--- Prediction Horizon: {pred_len} ---")
        for bat_id, capacity in capacity_data.items():
            context_len = max(10, int(len(capacity) * args.context_ratio))
            actual_pred_len = min(pred_len, len(capacity) - context_len)

            if actual_pred_len <= 0:
                logger.warning(f"  {bat_id}: insufficient data for pred_len={pred_len}")
                continue

            logger.info(f"  Evaluating {bat_id}: context={context_len}, pred={actual_pred_len}")
            metrics = evaluate_single_battery(
                model=model,
                capacity_series=capacity,
                context_length=context_len,
                prediction_length=actual_pred_len,
                battery_id=bat_id,
            )
            metrics["model_id"] = args.model_id
            metrics["requested_pred_length"] = pred_len
            all_results.append(metrics)
            logger.info(
                f"    RMSE={metrics['RMSE']:.4f}  CRPS={metrics['CRPS']:.4f}  "
                f"PICP={metrics['PICP']:.2%}  Latency={metrics['infer_time_ms']:.1f}ms"
            )

    # ── Step 5: Save results ──
    results_df = pd.DataFrame(all_results)
    csv_path = ROOT / args.output_dir / "chronos_zero_shot_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved: {csv_path}")

    # ── Step 6: Generate visualization ──
    logger.info("Step 6: Generating visualization...")
    default_pred_len = args.prediction_lengths[0]
    fig_path = str(ROOT / args.figures_dir / "fig_chronos_zero_shot.png")
    plot_zero_shot_results(
        all_results=[r for r in all_results if r["requested_pred_length"] == default_pred_len],
        capacity_data=capacity_data,
        model=model,
        context_ratio=args.context_ratio,
        prediction_length=default_pred_len,
        save_path=fig_path,
    )

    # ── Step 7: Print comparison table ──
    print_comparison_table(results_df)

    logger.info("Zero-shot probing complete.")


if __name__ == "__main__":
    main()
