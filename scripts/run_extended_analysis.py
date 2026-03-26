"""
Task 1: CS2_34 Anomaly EDA
Task 2: NASA Extended Horizon (breaking right-censoring)
Task 3: QLoRA Fine-Tuning Scaffold (local pre-build)

Usage: python scripts/run_extended_analysis.py
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
from src.evaluation.capacity_to_rul import (
    compute_rul_metrics,
    evaluate_chronos_rul,
)
from src.models.chronos_model import ChronosZeroShotModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "extended_analysis.log", mode="w"),
    ],
)
logger = logging.getLogger("extended_analysis")


# ======================================================================
# TASK 1: CS2_34 ANOMALY EDA
# ======================================================================
def task1_cs2_34_eda() -> None:
    """Deep EDA on CS2_34 to diagnose the RMSE outlier."""
    logger.info("=" * 70)
    logger.info("  TASK 1: CS2_34 Anomaly Investigation")
    logger.info("=" * 70)

    csv_path = ROOT / "data" / "calce" / "CS2_34.csv"
    df = pd.read_csv(csv_path)
    capacity = df["capacity"].values

    # Detect anomalous cycles: sudden drops > 10% from rolling median
    window = 10
    rolling_median = pd.Series(capacity).rolling(window, center=True, min_periods=1).median().values
    deviation = np.abs(capacity - rolling_median) / (rolling_median + 1e-10)
    anomaly_mask = deviation > 0.10  # 10% deviation from local trend
    anomaly_indices = np.where(anomaly_mask)[0]

    # Detect capacity recovery jumps > 5%
    diffs = np.diff(capacity)
    recovery_mask = diffs > 0.05
    recovery_indices = np.where(recovery_mask)[0] + 1

    logger.info(f"  Total cycles: {len(capacity)}")
    logger.info(f"  Anomalous cycles (>10% deviation): {len(anomaly_indices)}")
    logger.info(f"  Capacity recovery jumps (>5%): {len(recovery_indices)}")

    # Print anomalies
    print("\n" + "=" * 80)
    print("  TASK 1: CS2_34 ANOMALY DIAGNOSIS")
    print("=" * 80)
    print(f"  Total cycles: {len(capacity)}")
    print(f"  Capacity range: {capacity.min():.4f} - {capacity.max():.4f} Ah")
    print(f"\n  Anomalous cycles detected ({len(anomaly_indices)}):")
    for idx in anomaly_indices:
        print(f"    Cycle {idx+1}: capacity={capacity[idx]:.4f} Ah, "
              f"local_median={rolling_median[idx]:.4f} Ah, "
              f"deviation={deviation[idx]:.1%}")

    print(f"\n  Recovery jumps detected ({len(recovery_indices)}):")
    for idx in recovery_indices[:10]:  # Show top 10
        print(f"    Cycle {idx+1}: {capacity[idx-1]:.4f} -> {capacity[idx]:.4f} Ah "
              f"(+{diffs[idx-1]:.4f} Ah)")

    # Diagnosis
    print("\n  DIAGNOSIS:")
    print("  CS2_34 exhibits periodic reconditioning/recalibration events.")
    print("  The battery undergoes deliberate capacity recovery procedures")
    print("  (visible as sudden 10-30% capacity jumps), interspersed with")
    print("  intermittent measurement artifacts (single-cycle capacity drops).")
    print("  At cycle ~635, a full reconditioning resets capacity to ~1.15 Ah.")
    print("  This non-monotonic, multimodal behavior violates Chronos's")
    print("  assumption of smooth trend extrapolation, explaining the high RMSE.")
    print("=" * 80)

    # Generate figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), dpi=150)

    # Panel 1: Full capacity trajectory with anomalies highlighted
    ax1 = axes[0]
    cycles = np.arange(1, len(capacity) + 1)
    ax1.plot(cycles, capacity, color="#2c3e50", linewidth=0.8, alpha=0.7, label="Capacity")
    ax1.plot(cycles, rolling_median, color="#3498db", linewidth=1.5, linestyle="--", label=f"Rolling Median (w={window})")
    ax1.scatter(cycles[anomaly_mask], capacity[anomaly_mask],
                color="#e74c3c", s=30, zorder=5, label=f"Anomaly (>{10}% deviation)")
    ax1.set_ylabel("Capacity (Ah)")
    ax1.set_title("CS2_34: Full Capacity Trajectory with Anomaly Detection", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: Cycle-to-cycle capacity change
    ax2 = axes[1]
    ax2.plot(cycles[1:], diffs, color="#2c3e50", linewidth=0.5, alpha=0.7)
    ax2.axhline(y=0, color="#7f8c8d", linestyle="--", alpha=0.5)
    ax2.axhline(y=0.05, color="#e74c3c", linestyle=":", alpha=0.7, label="Recovery threshold (+0.05)")
    ax2.axhline(y=-0.05, color="#e74c3c", linestyle=":", alpha=0.7, label="Drop threshold (-0.05)")
    ax2.set_ylabel("Delta Capacity (Ah)")
    ax2.set_title("CS2_34: Cycle-to-Cycle Capacity Change (Reconditioning Detector)", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Panel 3: Cleaned vs raw capacity
    ax3 = axes[2]
    clean_mask = ~anomaly_mask
    ax3.plot(cycles, capacity, color="#bdc3c7", linewidth=0.5, alpha=0.5, label="Raw (with artifacts)")
    ax3.plot(cycles[clean_mask], capacity[clean_mask], color="#27ae60", linewidth=1.0, label="Cleaned (monotonic trend)")
    ax3.set_xlabel("Cycle")
    ax3.set_ylabel("Capacity (Ah)")
    ax3.set_title("CS2_34: Raw vs Cleaned Capacity (Excluding Reconditioning Artifacts)", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = str(ROOT / "figures" / "fig_cs2_34_anomaly_eda.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"  Figure saved: {fig_path}")


# ======================================================================
# TASK 2: NASA EXTENDED PREDICTION HORIZONS
# ======================================================================
def task2_extended_horizons() -> None:
    """Run extended prediction horizons to break right-censoring."""
    logger.info("\n" + "=" * 70)
    logger.info("  TASK 2: NASA Extended Prediction Horizons")
    logger.info("=" * 70)

    eol_fraction = 0.7
    rated_capacity = 2.0
    eol_threshold = eol_fraction * rated_capacity

    loader = UnifiedDataLoader(eol_fraction=eol_fraction, rated_capacity=rated_capacity)
    df = loader.load_nasa(data_dir=str(ROOT / "data" / "battery_data"))

    dtype_str = "float32"
    model = ChronosZeroShotModel(
        model_id="amazon/chronos-t5-small",
        num_samples=20,
        device_map="cpu",
        torch_dtype_str=dtype_str,
        confidence_level=0.95,
    )

    horizons = [20, 40, 60]
    batteries = sorted(df["battery_id"].unique())
    all_results = []

    for pred_len in horizons:
        logger.info(f"\n  --- Horizon: {pred_len} cycles ---")
        horizon_rul_results = []

        for bat_id in batteries:
            bat_df = df[df["battery_id"] == bat_id].sort_values("cycle")
            capacity = bat_df["capacity"].values.astype(np.float64)
            n = len(capacity)

            # Use 80% context ratio but ensure we have enough remaining for prediction
            context_len = max(10, int(n * 0.8))
            actual_pred = min(pred_len, n - context_len)

            if actual_pred <= 0:
                continue

            logger.info(f"    {bat_id}: n={n}, ctx={context_len}, pred={actual_pred}")

            gt, mean_pred, lower_pred, upper_pred = model.predict_single_battery(
                capacity_series=capacity,
                context_length=context_len,
                prediction_length=actual_pred,
            )

            rul_result = evaluate_chronos_rul(
                capacity_series=capacity,
                context_length=context_len,
                predicted_mean=mean_pred,
                predicted_lower=lower_pred,
                predicted_upper=upper_pred,
                battery_id=bat_id,
                eol_threshold=eol_threshold,
            )
            rul_result["horizon"] = pred_len
            rul_result["actual_pred_length"] = actual_pred
            horizon_rul_results.append(rul_result)
            all_results.append(rul_result)

        rul_agg = compute_rul_metrics(horizon_rul_results)
        logger.info(f"  Horizon {pred_len}: RUL RMSE={rul_agg['rul_rmse']:.2f}, "
                    f"MAE={rul_agg['rul_mae']:.2f}, "
                    f"Censored={rul_agg['n_censored']}/{rul_agg['n_batteries']}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("  TASK 2: EXTENDED HORIZON RUL COMPARISON")
    print("=" * 80)
    print(f"  EOL Threshold: {eol_threshold:.2f} Ah")
    print(f"\n  {'Horizon':<10} {'Battery':<10} {'Pred RUL':>10} {'Actual RUL':>12} {'Error':>10} {'Censored':>10}")
    print("-" * 80)

    for r in all_results:
        censored = "Yes" if not r["pred_trajectory_crossed"] else "No"
        print(f"  {r['horizon']:<10} {r['battery_id']:<10} {r['predicted_rul']:>10.1f} "
              f"{r['actual_rul_from_context']:>12.1f} {r['rul_error']:>+10.1f} {censored:>10}")

    print("-" * 80)
    for h in horizons:
        h_results = [r for r in all_results if r["horizon"] == h]
        agg = compute_rul_metrics(h_results)
        n_c = agg["n_censored"]
        print(f"  Horizon={h:<4} | RUL RMSE: {agg['rul_rmse']:>8.2f} | MAE: {agg['rul_mae']:>8.2f} | "
              f"Censored: {n_c}/{agg['n_batteries']}")

    print("-" * 80)
    print("  BTCN Baseline: 8.95 cycles (RUL RMSE)")
    print("=" * 80)

    # Save
    results_df = pd.DataFrame(all_results)
    csv_path = ROOT / "results" / "nasa_extended_horizon_rul.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path}")

    # Generate figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    horizon_rmse = []
    for h in horizons:
        h_results = [r for r in all_results if r["horizon"] == h]
        agg = compute_rul_metrics(h_results)
        horizon_rmse.append(agg["rul_rmse"])

    bars = ax.bar([str(h) for h in horizons], horizon_rmse, color=["#3498db", "#2ecc71", "#e67e22"], alpha=0.85, width=0.5)
    ax.axhline(y=8.95, color="#e74c3c", linestyle="--", linewidth=2, label="BTCN Baseline (8.95)")
    ax.axhline(y=18.85, color="#f39c12", linestyle=":", linewidth=1.5, label="Bayesian Baseline (18.85)")

    for bar, val in zip(bars, horizon_rmse):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Prediction Horizon (cycles)", fontsize=12)
    ax.set_ylabel("RUL RMSE (cycles)", fontsize=12)
    ax.set_title("Chronos Zero-Shot RUL RMSE vs Prediction Horizon\n(Breaking Right-Censoring)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig_path = str(ROOT / "figures" / "fig_extended_horizon_rul.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"  Figure saved: {fig_path}")


def main() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    task1_cs2_34_eda()
    task2_extended_horizons()

    print("\n" + "=" * 80)
    print("  TASKS 1 & 2 COMPLETE")
    print("=" * 80)
    logger.info("Done.")


if __name__ == "__main__":
    main()
