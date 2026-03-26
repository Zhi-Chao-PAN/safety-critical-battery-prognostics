"""
Unified Zero-Shot Evaluation: NASA RUL Re-Mapping + CALCE Cross-Domain Probing.

This script:
  Part A: Re-evaluates Chronos zero-shot on NASA data, converting capacity
          predictions into RUL estimates for fair comparison with BTCN (RMSE 8.95).
  Part B: Runs Chronos zero-shot on CALCE CS2 data to test cross-domain
          generalization without any fine-tuning.

Usage:
    python scripts/run_unified_zero_shot.py
    python scripts/run_unified_zero_shot.py --model-id amazon/chronos-t5-base --device cuda
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
from src.evaluation.capacity_to_rul import (
    compute_rul_metrics,
    evaluate_chronos_rul,
)
from src.models.chronos_model import ChronosZeroShotModel
from src.uncertainty.scoring import crps_gaussian, mae, mpiw, picp, rmse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "unified_zero_shot.log", mode="w"),
    ],
)
logger = logging.getLogger("unified_zero_shot")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified Zero-Shot Evaluation")
    p.add_argument("--model-id", default="amazon/chronos-t5-small")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--prediction-length", type=int, default=20)
    p.add_argument("--context-ratio", type=float, default=0.8)
    p.add_argument("--eol-fraction", type=float, default=0.7)
    p.add_argument("--rated-capacity", type=float, default=2.0)
    p.add_argument("--nasa-dir", default="data/battery_data")
    p.add_argument("--calce-dir", default="data/calce")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--figures-dir", default="figures")
    return p.parse_args()


def extract_capacity_series(df: pd.DataFrame, battery_id: str) -> np.ndarray:
    """Extract sorted capacity time-series for a single battery."""
    bat_df = df[df["battery_id"] == battery_id].sort_values("cycle")
    capacity = bat_df["capacity"].values.astype(np.float64)
    if len(capacity) == 0:
        raise ValueError(f"No data found for battery {battery_id}")
    return capacity


def run_part_a_nasa_rul(args, model: ChronosZeroShotModel) -> pd.DataFrame:
    """
    Part A: NASA Zero-Shot with RUL Re-Mapping.
    
    Re-runs Chronos on NASA data and converts capacity predictions
    into RUL estimates for fair comparison with BTCN.
    """
    logger.info("=" * 70)
    logger.info("  PART A: NASA Zero-Shot + Capacity-to-RUL Conversion")
    logger.info("=" * 70)

    eol_threshold = args.eol_fraction * args.rated_capacity
    logger.info(f"  EOL Threshold: {eol_threshold:.2f} Ah "
                f"({args.eol_fraction*100:.0f}% of {args.rated_capacity:.1f} Ah)")

    loader = UnifiedDataLoader(
        eol_fraction=args.eol_fraction,
        rated_capacity=args.rated_capacity,
    )
    df = loader.load_nasa(data_dir=str(ROOT / args.nasa_dir))
    logger.info(f"  Loaded {len(df)} cycles from {df['battery_id'].nunique()} batteries")

    batteries = sorted(df["battery_id"].unique())
    all_capacity_results = []
    all_rul_results = []

    for bat_id in batteries:
        capacity = extract_capacity_series(df, bat_id)
        context_len = max(10, int(len(capacity) * args.context_ratio))
        pred_len = min(args.prediction_length, len(capacity) - context_len)

        if pred_len <= 0:
            logger.warning(f"  {bat_id}: insufficient data")
            continue

        logger.info(f"\n  --- {bat_id}: {len(capacity)} cycles, context={context_len}, pred={pred_len} ---")

        # Run Chronos inference
        t0 = time.time()
        gt, mean_pred, lower_pred, upper_pred = model.predict_single_battery(
            capacity_series=capacity,
            context_length=context_len,
            prediction_length=pred_len,
        )
        infer_ms = (time.time() - t0) * 1000

        # Capacity-level metrics
        std_pred = np.maximum((upper_pred - lower_pred) / 3.92, 1e-6)
        cap_metrics = {
            "battery_id": bat_id,
            "dataset": "NASA",
            "n_cycles": len(capacity),
            "context_length": context_len,
            "prediction_length": pred_len,
            "capacity_RMSE": rmse(gt, mean_pred),
            "capacity_MAE": mae(gt, mean_pred),
            "capacity_CRPS": crps_gaussian(gt, mean_pred, std_pred),
            "capacity_PICP": picp(gt, lower_pred, upper_pred),
            "capacity_MPIW": mpiw(lower_pred, upper_pred),
            "infer_time_ms": round(infer_ms, 2),
        }
        all_capacity_results.append(cap_metrics)

        logger.info(f"    Capacity RMSE={cap_metrics['capacity_RMSE']:.4f} Ah, "
                    f"PICP={cap_metrics['capacity_PICP']:.2%}")

        # RUL Re-Mapping: use FULL remaining ground truth for actual RUL
        rul_result = evaluate_chronos_rul(
            capacity_series=capacity,
            context_length=context_len,
            predicted_mean=mean_pred,
            predicted_lower=lower_pred,
            predicted_upper=upper_pred,
            battery_id=bat_id,
            eol_threshold=eol_threshold,
        )
        rul_result["dataset"] = "NASA"
        all_rul_results.append(rul_result)

    # Aggregate RUL metrics
    rul_agg = compute_rul_metrics(all_rul_results)

    print("\n" + "=" * 80)
    print("  PART A RESULTS: Chronos Zero-Shot RUL (NASA)")
    print("=" * 80)
    print(f"  EOL Threshold: {eol_threshold:.2f} Ah")
    print(f"  {'Battery':<10} {'Pred RUL':>10} {'Actual RUL':>12} {'Error':>10} {'Crossed':>10}")
    print("-" * 80)
    for r in all_rul_results:
        print(f"  {r['battery_id']:<10} {r['predicted_rul']:>10.1f} "
              f"{r['actual_rul_from_context']:>12.1f} {r['rul_error']:>+10.1f} "
              f"{'Yes' if r['pred_trajectory_crossed'] else 'No':>10}")
    print("-" * 80)
    print(f"  Chronos RUL RMSE: {rul_agg['rul_rmse']:.2f} cycles")
    print(f"  Chronos RUL MAE:  {rul_agg['rul_mae']:.2f} cycles")
    print("  BTCN Baseline:    8.95 cycles (RUL RMSE)")
    print("  Bayesian Baseline:18.85 cycles (RUL RMSE)")
    print("=" * 80)

    # Save results
    cap_df = pd.DataFrame(all_capacity_results)
    rul_df = pd.DataFrame(all_rul_results)

    cap_path = ROOT / args.output_dir / "nasa_capacity_metrics.csv"
    rul_path = ROOT / args.output_dir / "nasa_rul_metrics.csv"
    cap_df.to_csv(cap_path, index=False)
    rul_df.to_csv(rul_path, index=False)
    logger.info(f"  Saved: {cap_path}, {rul_path}")

    return rul_df


def run_part_b_calce_zero_shot(args, model: ChronosZeroShotModel) -> pd.DataFrame:
    """
    Part B: CALCE Cross-Domain Zero-Shot Probing.
    
    Tests whether Chronos can generalize to a completely different
    battery chemistry/protocol without any fine-tuning.
    """
    logger.info("\n" + "=" * 70)
    logger.info("  PART B: CALCE CS2 Cross-Domain Zero-Shot Probing")
    logger.info("=" * 70)

    calce_dir = ROOT / args.calce_dir
    if not calce_dir.exists():
        logger.error(f"  CALCE data directory not found: {calce_dir}")
        return pd.DataFrame()

    csv_files = sorted(calce_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith("_")]  # Skip _summary.csv

    if not csv_files:
        logger.error("  No CALCE CSV files found.")
        return pd.DataFrame()

    all_results = []

    for csv_file in csv_files:
        bat_id = csv_file.stem
        try:
            raw = pd.read_csv(csv_file)
            if "capacity" not in raw.columns or "cycle" not in raw.columns:
                logger.warning(f"  Skipping {bat_id}: missing columns")
                continue

            capacity = raw.sort_values("cycle")["capacity"].values.astype(np.float64)
            n = len(capacity)

            context_len = max(10, int(n * args.context_ratio))
            pred_len = min(args.prediction_length, n - context_len)

            if pred_len <= 2:
                logger.warning(f"  {bat_id}: insufficient data ({n} cycles, need >{context_len + 2})")
                continue

            logger.info(f"\n  --- {bat_id}: {n} cycles, context={context_len}, pred={pred_len} ---")

            t0 = time.time()
            gt, mean_pred, lower_pred, upper_pred = model.predict_single_battery(
                capacity_series=capacity,
                context_length=context_len,
                prediction_length=pred_len,
            )
            infer_ms = (time.time() - t0) * 1000

            std_pred = np.maximum((upper_pred - lower_pred) / 3.92, 1e-6)
            metrics = {
                "battery_id": bat_id,
                "dataset": "CALCE",
                "n_cycles": n,
                "context_length": context_len,
                "prediction_length": pred_len,
                "capacity_RMSE": rmse(gt, mean_pred),
                "capacity_MAE": mae(gt, mean_pred),
                "capacity_CRPS": crps_gaussian(gt, mean_pred, std_pred),
                "capacity_PICP": picp(gt, lower_pred, upper_pred),
                "capacity_MPIW": mpiw(lower_pred, upper_pred),
                "infer_time_ms": round(infer_ms, 2),
            }
            all_results.append(metrics)

            logger.info(f"    RMSE={metrics['capacity_RMSE']:.4f} Ah, "
                        f"PICP={metrics['capacity_PICP']:.2%}, "
                        f"Latency={infer_ms:.0f}ms")

        except Exception as e:
            logger.error(f"  Error processing {bat_id}: {e}")

    if not all_results:
        logger.error("  No valid CALCE results.")
        return pd.DataFrame()

    # Print summary
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("  PART B RESULTS: Chronos Zero-Shot on CALCE CS2 (Cross-Domain)")
    print("=" * 80)
    print(f"  {'Battery':<10} {'Cycles':>8} {'RMSE(Ah)':>10} {'MAE(Ah)':>10} {'PICP':>10} {'MPIW':>10}")
    print("-" * 80)
    for _, r in results_df.iterrows():
        print(f"  {r['battery_id']:<10} {r['n_cycles']:>8} "
              f"{r['capacity_RMSE']:>10.4f} {r['capacity_MAE']:>10.4f} "
              f"{r['capacity_PICP']:>10.2%} {r['capacity_MPIW']:>10.4f}")
    print("-" * 80)
    avg_rmse = results_df["capacity_RMSE"].mean()
    avg_picp = results_df["capacity_PICP"].mean()
    print(f"  CALCE Average RMSE: {avg_rmse:.4f} Ah")
    print(f"  CALCE Average PICP: {avg_picp:.2%}")
    print("  NASA  Average RMSE: 0.0294 Ah (from Phase 1)")
    print(f"  Cross-Domain RMSE Ratio (CALCE/NASA): {avg_rmse / 0.0294:.2f}x")
    print("=" * 80)

    # Save
    csv_path = ROOT / args.output_dir / "calce_zero_shot_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path}")

    return results_df


def generate_combined_figure(
    nasa_rul_df: pd.DataFrame,
    calce_df: pd.DataFrame,
    save_path: str,
) -> None:
    """Generate a combined summary figure for both Part A and Part B."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    # Panel 1: NASA RUL comparison
    ax1 = axes[0]
    if not nasa_rul_df.empty:
        batteries = nasa_rul_df["battery_id"].values
        pred_rul = nasa_rul_df["predicted_rul"].values
        actual_rul = nasa_rul_df["actual_rul_from_context"].values

        x = np.arange(len(batteries))
        width = 0.35
        ax1.bar(x - width/2, actual_rul, width, label="Actual RUL", color="#2ecc71", alpha=0.8)
        ax1.bar(x + width/2, pred_rul, width, label="Chronos Predicted RUL", color="#e74c3c", alpha=0.8)

        ax1.set_xlabel("Battery", fontsize=11)
        ax1.set_ylabel("RUL (Cycles)", fontsize=11)
        ax1.set_title("Part A: NASA RUL Estimation\n(Chronos Zero-Shot via Capacity Trajectory)", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(batteries)
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

    # Panel 2: CALCE cross-domain RMSE
    ax2 = axes[1]
    if not calce_df.empty:
        batteries = calce_df["battery_id"].values
        rmse_vals = calce_df["capacity_RMSE"].values

        colors = ["#3498db" if v < 0.05 else "#e67e22" if v < 0.1 else "#e74c3c" for v in rmse_vals]
        ax2.barh(batteries, rmse_vals, color=colors, alpha=0.85)
        ax2.axvline(x=0.0294, color="#2c3e50", linestyle="--", linewidth=1.5, label="NASA Avg RMSE (0.029)")
        ax2.set_xlabel("Capacity RMSE (Ah)", fontsize=11)
        ax2.set_title("Part B: CALCE Cross-Domain Zero-Shot\n(No Fine-Tuning, No CALCE Training Data)", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"  Combined figure saved: {save_path}")


def main() -> None:
    args = parse_args()

    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / args.output_dir).mkdir(exist_ok=True)
    (ROOT / args.figures_dir).mkdir(exist_ok=True)

    # Initialize model once
    logger.info(f"Loading Chronos model: {args.model_id}")
    dtype_str = "bfloat16" if args.device == "cuda" else "float32"
    model = ChronosZeroShotModel(
        model_id=args.model_id,
        num_samples=args.num_samples,
        device_map=args.device,
        torch_dtype_str=dtype_str,
        confidence_level=0.95,
    )

    # Part A
    nasa_rul_df = run_part_a_nasa_rul(args, model)

    # Part B
    calce_df = run_part_b_calce_zero_shot(args, model)

    # Combined figure
    fig_path = str(ROOT / args.figures_dir / "fig_unified_zero_shot.png")
    generate_combined_figure(nasa_rul_df, calce_df, fig_path)

    print("\n" + "=" * 80)
    print("  UNIFIED ZERO-SHOT EVALUATION COMPLETE")
    print("=" * 80)
    logger.info("Done.")


if __name__ == "__main__":
    main()
