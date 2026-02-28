"""
Generate all publication figures from benchmark results.
Run after main.py completes training.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def main():
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Load data for raw plots
    loader = UnifiedDataLoader()
    df = loader.load_all(nasa_dir=str(ROOT / "data" / "battery_data"))
    validator = DataValidator()
    df, _ = validator.validate(df)
    extractor = FeatureExtractor()
    df = extractor.extract_all(df)
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ("cycle", "rul")]
    df = df.dropna(subset=feature_cols + ["rul"]).reset_index(drop=True)

    # Fig 1: Degradation curves
    print("Generating Fig 1: Degradation curves...")
    plot_degradation_curves(df, str(figures_dir / "fig01_degradation.png"))

    # Fig 2: Feature correlation
    print("Generating Fig 2: Feature correlation...")
    plot_feature_correlation(df, feature_cols[:15], str(figures_dir / "fig02_correlation.png"))

    # Fig 3: Model comparison (from benchmark results)
    bench_path = ROOT / "results" / "benchmark_results.csv"
    if bench_path.exists():
        print("Generating Fig 3: Model comparison...")
        bench_df = pd.read_csv(bench_path)
        plot_model_comparison(bench_df, str(figures_dir / "fig03_comparison.png"))

        # Fig 7: Per-fold results
        print("Generating Fig 7: Per-fold breakdown...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for i, metric in enumerate(["RMSE", "CRPS"]):
            pivot = bench_df.pivot_table(index="fold", columns="model", values=metric, aggfunc="mean")
            pivot.plot(kind="bar", ax=axes[i], rot=0)
            axes[i].set_ylabel(metric)
            axes[i].set_title(f"{metric} by Battery (LOGO-CV)")
            axes[i].legend(fontsize=7, ncol=2)
            axes[i].grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(figures_dir / "fig07_per_fold.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Fig 8: Training time comparison
        if "train_time_s" in bench_df.columns:
            print("Generating Fig 8: Training time...")
            time_df = bench_df.groupby("model")["train_time_s"].mean().sort_values()
            fig, ax = plt.subplots(figsize=(8, 4))
            time_df.plot(kind="barh", ax=ax, color="#0072B2", alpha=0.8)
            ax.set_xlabel("Mean Training Time (s)")
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            fig.savefig(str(figures_dir / "fig08_train_time.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

    # Fig 9: Model complexity table
    print("Generating Fig 9: Model complexity...")
    complexity = pd.DataFrame({
        "Model": ["LSTM", "GRU", "TCN", "CNN1D", "Transformer", "PINN", "BayesianNN"],
        "Parameters": [157122, 121026, 60065, 23137, 103745, 7745, 11394],
        "Size (MB)": [0.599, 0.462, 0.229, 0.088, 0.396, 0.030, 0.043],
    })
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(complexity["Model"], complexity["Parameters"], color="#009E73", alpha=0.8)
    ax.set_xlabel("Number of Parameters")
    for bar, val in zip(bars, complexity["Parameters"]):
        ax.text(bar.get_width() + 1000, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(figures_dir / "fig09_complexity.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"All figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
