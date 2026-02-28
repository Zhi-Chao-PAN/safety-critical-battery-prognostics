"""
Generate all publication figures from results.

Usage:
    python scripts/generate_figures.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.unified_loader import UnifiedDataLoader
from src.data.validator import DataValidator
from src.features.extractor import FeatureExtractor
from src.ui.visualization import (
    plot_degradation_curves,
    plot_feature_correlation,
    plot_model_comparison,
    plot_calibration,
    plot_ablation_summary,
)

FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_data():
    loader = UnifiedDataLoader()
    df = loader.load_all(nasa_dir=str(ROOT / "data" / "battery_data"))
    validator = DataValidator()
    df, _ = validator.validate(df)
    extractor = FeatureExtractor()
    df = extractor.extract_all(df)
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ("cycle", "rul")]
    df = df.dropna(subset=feature_cols + ["rul"]).reset_index(drop=True)
    return df, feature_cols


def fig01_degradation(df):
    """Fig 1: Raw capacity degradation curves."""
    plot_degradation_curves(df, str(FIGURES_DIR / "fig01_degradation.png"))
    print("✓ Fig 01: Degradation curves")


def fig02_correlation(df, feature_cols):
    """Fig 2: Feature correlation heatmap."""
    cols = [c for c in feature_cols if df[c].notna().sum() > 10][:15]
    if cols:
        plot_feature_correlation(df, cols, str(FIGURES_DIR / "fig02_correlation.png"))
        print("✓ Fig 02: Feature correlation")


def fig03_model_comparison():
    """Fig 3: Model comparison from benchmark results."""
    results_path = ROOT / "results" / "benchmark_results.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        plot_model_comparison(df, str(FIGURES_DIR / "fig03_comparison.png"))
        print("✓ Fig 03: Model comparison")
    else:
        print("✗ Fig 03: No benchmark results yet")


def fig07_transfer():
    """Fig 7: Transfer learning results."""
    results_path = ROOT / "results" / "transfer_results.csv"
    if not results_path.exists():
        print("✗ Fig 07: No transfer results yet")
        return

    df = pd.read_csv(results_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, model in zip(axes, ["lstm", "pinn"]):
        sub = df[df["model"] == model]
        for typ, style in [("zero_shot", "o--"), ("fine_tuned", "s-")]:
            t = sub[sub["type"] == typ].groupby("n_shots").agg(
                mean=("RMSE", "mean"), std=("RMSE", "std")
            ).reset_index()
            if len(t) > 0:
                ax.errorbar(t["n_shots"], t["mean"], yerr=t["std"],
                           fmt=style, capsize=4, linewidth=2, label=typ)
        ax.set_xlabel("Number of shots")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{model.upper()} Transfer")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "fig07_transfer.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 07: Transfer learning")


def fig08_safety_timeline(df, feature_cols):
    """Fig 8: Safety decision timeline for one battery."""
    from src.models import LSTMModel
    from src.safety.decision_engine import SafetyDecisionEngine, SafetyLevel

    bat = df[df["battery_id"] == "B0005"].sort_values("cycle")
    train = df[df["battery_id"] != "B0005"]

    model = LSTMModel(input_dim=len(feature_cols), hidden_dim=64, epochs=50, seq_length=20)
    model.fit(train[feature_cols].values, train["rul"].values)
    mean, lower, upper = model.predict(bat[feature_cols].values)

    if len(mean) == 0:
        print("✗ Fig 08: Empty predictions")
        return

    engine = SafetyDecisionEngine()
    decisions = engine.decide_batch(mean, lower, upper)
    cycles = bat["cycle"].values[-len(mean):]

    colors = {"GREEN": "#2ecc71", "YELLOW": "#f39c12", "RED": "#e74c3c"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, height_ratios=[3, 1])

    # Top: Prediction with CI
    gt_rul = bat["rul"].values[-len(mean):]
    ax1.plot(cycles, gt_rul, "k-", linewidth=2, label="Ground Truth")
    ax1.plot(cycles, mean, "b--", linewidth=1.5, label="Prediction")
    ax1.fill_between(cycles, lower, upper, alpha=0.2, color="blue", label="95% CI")
    ax1.set_ylabel("RUL (cycles)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Bottom: Safety status
    for i, (c, d) in enumerate(zip(cycles, decisions)):
        ax2.barh(0, 1, left=c, height=0.8, color=colors[d.level.value], edgecolor="none")
    ax2.set_xlabel("Cycle")
    ax2.set_yticks([])
    ax2.set_ylabel("Safety")

    # Legend for safety
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in colors.items()]
    ax2.legend(handles=legend_elements, loc="upper left", ncol=3)

    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "fig08_safety_timeline.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 08: Safety timeline")


def main():
    print("Loading data...")
    df, feature_cols = load_data()
    print(f"Data: {len(df)} rows, {len(feature_cols)} features\n")

    fig01_degradation(df)
    fig02_correlation(df, feature_cols)
    fig03_model_comparison()
    fig07_transfer()
    fig08_safety_timeline(df, feature_cols)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
