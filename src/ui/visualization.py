"""
Publication-quality visualization for battery prognostics.
Generates 12+ figure types for paper and dashboard.
"""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Consistent style
STYLE = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Colorblind-safe palette
COLORS = {
    "ground_truth": "#000000",
    "lstm": "#E69F00",
    "gru": "#56B4E9",
    "tcn": "#009E73",
    "transformer": "#F0E442",
    "ensemble": "#0072B2",
    "bayesian": "#D55E00",
    "pinn": "#CC79A7",
    "aleatoric": "#56B4E9",
    "epistemic": "#E69F00",
}


def _setup():
    plt.rcParams.update(STYLE)


def _save(fig, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_degradation_curves(
    df: pd.DataFrame, save_path: str = "figures/fig01_degradation.png"
):
    """Fig 1: Capacity degradation curves, color by battery."""
    _setup()
    fig, ax = plt.subplots()
    for bat_id in df["battery_id"].unique():
        sub = df[df["battery_id"] == bat_id].sort_values("cycle")
        label = f"{bat_id}"
        if "dataset_source" in sub.columns:
            label += f" ({sub['dataset_source'].iloc[0]})"
        ax.plot(sub["cycle"], sub["capacity"], linewidth=1.5, label=label)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity (Ah)")
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, save_path)


def plot_feature_correlation(
    df: pd.DataFrame, features: list[str], save_path: str = "figures/fig02_correlation.png"
):
    """Fig 2: Feature correlation heatmap."""
    _setup()
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(features, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, save_path)


def plot_model_comparison(
    results_df: pd.DataFrame, save_path: str = "figures/fig03_comparison.png"
):
    """Fig 3: Model comparison bar chart (RMSE + CRPS)."""
    _setup()
    summary = results_df.groupby("model").agg(
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        CRPS_mean=("CRPS", "mean"), CRPS_std=("CRPS", "std"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(summary))

    ax1.bar(x, summary["RMSE_mean"], yerr=summary["RMSE_std"], capsize=4, color="#0072B2", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["model"], rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("RMSE (cycles)")
    ax1.set_title("Deterministic Accuracy")

    ax2.bar(x, summary["CRPS_mean"], yerr=summary["CRPS_std"], capsize=4, color="#D55E00", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary["model"], rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("CRPS")
    ax2.set_title("Probabilistic Quality")

    fig.tight_layout()
    _save(fig, save_path)


def plot_safety_buffer(
    cycles: np.ndarray, gt_rul: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    save_path: str = "figures/fig04_safety_buffer.png",
):
    """Fig 4: Safety buffer comparison - multiple models."""
    _setup()
    fig, ax = plt.subplots()
    ax.plot(cycles, gt_rul, color=COLORS["ground_truth"], linewidth=2.5, label="Ground Truth")

    color_list = list(COLORS.values())[1:]
    for i, (name, (mean, lower, upper)) in enumerate(predictions.items()):
        c = color_list[i % len(color_list)]
        ax.plot(cycles[:len(mean)], mean, color=c, linewidth=1.5, linestyle="--", label=f"{name} (mean)")
        ax.fill_between(cycles[:len(lower)], lower, upper, color=c, alpha=0.15, label=f"{name} (95% CI)")

    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, save_path)


def plot_uncertainty_decomposition(
    cycles: np.ndarray, mean: np.ndarray,
    aleatoric_std: np.ndarray, epistemic_std: np.ndarray,
    save_path: str = "figures/fig05_uncertainty_decomp.png",
):
    """Fig 5: Stacked uncertainty bands (aleatoric inner, epistemic outer)."""
    _setup()
    fig, ax = plt.subplots()
    total_std = np.sqrt(aleatoric_std**2 + epistemic_std**2)

    ax.plot(cycles, mean, color="black", linewidth=2, label="Prediction")
    ax.fill_between(cycles, mean - 1.96 * aleatoric_std, mean + 1.96 * aleatoric_std,
                     color=COLORS["aleatoric"], alpha=0.3, label="Aleatoric (data noise)")
    ax.fill_between(cycles, mean - 1.96 * total_std, mean + 1.96 * total_std,
                     color=COLORS["epistemic"], alpha=0.15, label="+ Epistemic (model ignorance)")

    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    _save(fig, save_path)


def plot_calibration(
    expected: np.ndarray, observed_dict: dict[str, np.ndarray],
    save_path: str = "figures/fig06_calibration.png",
):
    """Fig 6: Reliability diagram - multiple models."""
    _setup()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    color_list = list(COLORS.values())[1:]
    for i, (name, observed) in enumerate(observed_dict.items()):
        c = color_list[i % len(color_list)]
        ax.plot(expected, observed, "o-", color=c, linewidth=1.5, markersize=5, label=name)

    ax.set_xlabel("Expected Coverage")
    ax.set_ylabel("Observed Coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    _save(fig, save_path)


def plot_ablation_summary(
    ablation_df: pd.DataFrame, metric: str = "RMSE",
    save_path: str = "figures/fig11_ablation.png",
):
    """Fig 11: Ablation results grouped bar chart."""
    _setup()
    summary = ablation_df.groupby("model").agg(
        mean=(metric, "mean"), std=(metric, "std")
    ).reset_index().sort_values("mean")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(summary["model"], summary["mean"], xerr=summary["std"],
                    capsize=4, color="#0072B2", alpha=0.8)
    ax.set_xlabel(metric)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
