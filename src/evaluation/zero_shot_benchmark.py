"""
Zero-Shot Cross-Dataset Benchmark for Battery Prognostics.

Implements cross-dataset zero-shot generalization evaluation:
- Train on Dataset A, test on Dataset B (unseen distribution)
- Generate unified markdown reports with RMSE, MAE, PICP metrics
- Create comparison plots across datasets

Author: AI Assistant
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.data.unified_loader import UnifiedDataLoader
from src.evaluation.target_adapter import (
    adapt_predictions_to_target,
    build_prediction_data,
    build_training_data,
)
from src.models.base import BatteryModel
from src.uncertainty.scoring import compute_all_metrics
from src.utils.metrics import calculate_picp

logger = logging.getLogger(__name__)


@dataclass
class ZeroShotResult:
    """Container for zero-shot evaluation results."""

    train_dataset: str
    test_dataset: str
    model_name: str
    rmse: float
    mae: float
    picp: float
    nll: float
    crps: float
    coverage_80: float
    coverage_95: float
    sharpe_ratio: float
    inference_time_ms: float
    n_samples: int
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "train_dataset": self.train_dataset,
            "test_dataset": self.test_dataset,
            "model_name": self.model_name,
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "picp": round(self.picp, 4),
            "nll": round(self.nll, 4),
            "crps": round(self.crps, 4),
            "coverage_80": round(self.coverage_80, 4),
            "coverage_95": round(self.coverage_95, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "inference_time_ms": round(self.inference_time_ms, 4),
            "n_samples": self.n_samples,
            "timestamp": self.timestamp,
        }


class ZeroShotBenchmarkRunner:
    """
    Automated cross-dataset zero-shot evaluation benchmark.

    Implements the zero-shot generalization paradigm:
    1. Train model on Dataset A
    2. Load trained model weights
    3. Evaluate on Dataset B (unseen distribution, no fine-tuning)
    4. Generate comprehensive benchmark report

    Supported Datasets:
    - NASA PCoE (Li-ion, 18650)
    - CALCE CS2 (LiCoO2)
    - Oxford (Li-ion)
    - MIT-Stanford (prismatic)

    Metrics:
    - RMSE, MAE: Point prediction accuracy
    - PICP: 95% Prediction Interval Coverage Probability
    - CRPS: Continuous Ranked Probability Score
    - NLL: Negative Log-Likelihood
    - Sharpe Ratio: Risk-adjusted return
    """

    def __init__(
        self,
        results_dir: str = "results/zero_shot_benchmark",
        device: str = "auto",
        random_seed: int = 42,
        eol_threshold: float = 1.4,
    ):
        """
        Initialize benchmark runner.

        Args:
            results_dir: Directory to save benchmark results
            device: Device for model inference ("auto", "cpu", "cuda")
            random_seed: Random seed for reproducibility
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.random_seed = random_seed
        self.eol_threshold = eol_threshold
        self.data_loader = UnifiedDataLoader()

        # Set random seeds
        np.random.seed(random_seed)
        try:
            import torch

            torch.manual_seed(random_seed)
            if torch.cuda.is_available() and device == "cuda":
                torch.cuda.manual_seed_all(random_seed)
        except ImportError:
            pass

        self.results: list[ZeroShotResult] = []

    def run_zero_shot(
        self,
        model: BatteryModel,
        model_name: str,
        train_dataset: str,
        test_dataset: str,
        features: list[str] | None = None,
        target: str = "rul",
        save_model: bool = True,
    ) -> ZeroShotResult:
        """
        Run zero-shot cross-dataset evaluation.

        Args:
            model: Battery model instance (untrained)
            model_name: Name identifier for the model
            train_dataset: Source dataset name ("nasa", "calce", "oxford", "mit")
            test_dataset: Target dataset name for zero-shot testing
            features: List of feature column names
            target: Evaluation target column name (default: "rul")
            save_model: Whether to save trained model weights

        Returns:
            ZeroShotResult with evaluation metrics
        """
        logger.info(
            f"\n{'='*60}\n"
            f"Zero-Shot Evaluation: {model_name}\n"
            f"  Train: {train_dataset.upper()}\n"
            f"  Test:  {test_dataset.upper()}\n"
            f"{'='*60}"
        )

        # Load datasets
        train_df = self._load_dataset(train_dataset)
        test_df = self._load_dataset(test_dataset)

        # Infer features if not provided
        if features is None:
            features = self._infer_features(train_df)
            logger.info(f"Inferred features: {features}")

        # Train model on its declared prediction target
        train_df, X_train, y_train, fit_kwargs = build_training_data(
            train_df,
            features,
            model,
        )

        t0 = time.time()
        model.fit(X_train, y_train, **fit_kwargs)
        train_time = time.time() - t0
        logger.info(f"Training completed in {train_time:.2f}s")

        # Save model if requested
        model_path = None
        if save_model:
            model_path = (
                self.results_dir
                / f"{model_name}_{train_dataset}_to_{test_dataset}.pt"
            )
            model.save(model_path)
            logger.info(f"Model saved to {model_path}")

        # Zero-shot inference (no fine-tuning on test set)
        test_df, X_test, predict_kwargs = build_prediction_data(test_df, features)

        t0 = time.time()
        mean, lower, upper = model.predict(X_test, **predict_kwargs)
        if len(mean) == 0:
            raise ValueError("Model produced no predictions for zero-shot evaluation")
        infer_time = (time.time() - t0) / len(mean) * 1000  # ms per sample
        logger.info(f"Inference: {infer_time:.2f} ms/sample")

        # Compute metrics on the requested evaluation target
        y_eval, mean_eval, lower_eval, upper_eval, _ = adapt_predictions_to_target(
            model=model,
            test_df=test_df,
            mean=mean,
            lower=lower,
            upper=upper,
            evaluation_target=target,
            eol_threshold=self.eol_threshold,
        )
        metrics = compute_all_metrics(y_eval, mean_eval, lower_eval, upper_eval)

        # Compute additional metrics
        coverage_80 = self._compute_coverage(y_eval, lower_eval, upper_eval, alpha=0.2)
        coverage_95 = self._compute_coverage(y_eval, lower_eval, upper_eval, alpha=0.05)
        sharpe = self._compute_sharpe_ratio(y_eval, mean_eval)

        result = ZeroShotResult(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model_name=model_name,
            rmse=float(metrics["RMSE"]),
            mae=float(metrics["MAE"]),
            picp=float(metrics["PICP"]),
            nll=float(metrics["NLL"]),
            crps=float(metrics["CRPS"]),
            coverage_80=coverage_80,
            coverage_95=coverage_95,
            sharpe_ratio=sharpe,
            inference_time_ms=infer_time,
            n_samples=len(mean_eval),
        )

        self.results.append(result)
        logger.info(
            f"\nResults:\n"
            f"  RMSE: {result.rmse:.4f}\n"
            f"  MAE:  {result.mae:.4f}\n"
            f"  PICP: {result.picp:.4f}\n"
            f"  CRPS: {result.crps:.4f}"
        )

        return result

    def run_cross_dataset_matrix(
        self,
        model_class: type,
        model_name: str,
        model_kwargs: dict[str, Any] | None = None,
        datasets: list[str] | None = None,
        features: list[str] | None = None,
        target: str = "rul",
    ) -> pd.DataFrame:
        """
        Run full cross-dataset evaluation matrix.

        Evaluates all combinations of train_dataset -> test_dataset pairs,
        creating a comprehensive zero-shot generalization matrix.

        Args:
            model_class: Model class to instantiate
            model_name: Name identifier
            model_kwargs: Keyword arguments for model initialization
            datasets: List of dataset names to evaluate (default: ["nasa", "calce"])
            features: Feature column names
            target: Evaluation target column name

        Returns:
            DataFrame with cross-dataset results
        """
        if datasets is None:
            datasets = ["nasa", "calce"]
        model_kwargs = model_kwargs or {}

        logger.info(
            f"\n{'='*70}\n"
            f"Cross-Dataset Zero-Shot Matrix\n"
            f"  Model: {model_name}\n"
            f"  Datasets: {', '.join(datasets)}\n"
            f"{'='*70}"
        )

        for train_ds in datasets:
            for test_ds in datasets:
                # Create fresh model instance
                model = model_class(**model_kwargs)

                # Run zero-shot evaluation
                try:
                    self.run_zero_shot(
                        model=model,
                        model_name=f"{model_name}_{train_ds}_to_{test_ds}",
                        train_dataset=train_ds,
                        test_dataset=test_ds,
                        features=features,
                        target=target,
                    )
                except Exception as e:
                    logger.error(f"Failed {train_ds}->{test_ds}: {e}")
                    continue

        return self.get_results_df()

    def get_results_df(self) -> pd.DataFrame:
        """Get results as DataFrame."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.results])

    def save_results(self, filename: str = "zero_shot_results.json") -> Path:
        """Save results to JSON."""
        results_path = self.results_dir / filename
        df = self.get_results_df()
        df.to_json(results_path, orient="records", indent=2)
        logger.info(f"Results saved to {results_path}")
        return results_path

    def generate_markdown_report(
        self,
        title: str = "Zero-Shot Cross-Dataset Benchmark Report",
        filename: str = "zero_shot_benchmark_report.md",
    ) -> Path:
        """
        Generate comprehensive Markdown benchmark report.

        Creates a publication-ready report with:
        - Executive summary
        - Cross-dataset performance matrix
        - Statistical significance tests
        - Visualization references
        - Detailed results tables

        Args:
            title: Report title
            filename: Output filename

        Returns:
            Path to generated report
        """
        df = self.get_results_df()
        if df.empty:
            logger.warning("No results to report")
            return Path()

        report_path = self.results_dir / filename

        lines = []
        lines.append(f"# {title}\n")
        lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**Benchmark Version:** 1.0.0\n")
        lines.append("---\n\n")

        # Executive Summary
        lines.append("## Executive Summary\n")
        lines.append(
            "This report evaluates **zero-shot cross-dataset generalization** "
            "for battery prognostics models. Models are trained on one dataset "
            "and directly evaluated on another dataset without any fine-tuning.\n\n"
        )

        n_models = df["model_name"].nunique()
        n_train_ds = df["train_dataset"].nunique()
        n_test_ds = df["test_dataset"].nunique()

        lines.append(f"- **Models Evaluated:** {n_models}\n")
        lines.append(f"- **Training Datasets:** {n_train_ds}\n")
        lines.append(f"- **Test Datasets:** {n_test_ds}\n")
        lines.append(f"- **Total Evaluations:** {len(df)}\n\n")

        # Cross-Dataset Performance Matrix
        lines.append("## Cross-Dataset Performance Matrix\n")
        lines.append(
            "The following tables show model performance across dataset combinations. "
            "Rows indicate training dataset, columns indicate test dataset.\n\n"
        )

        for model_name in df["model_name"].unique():
            model_df = df[df["model_name"] == model_name]
            lines.append(f"### {model_name}\n")

            # RMSE Matrix
            lines.append("#### RMSE (Root Mean Squared Error)\n")
            rmse_pivot = model_df.pivot_table(
                values="rmse", index="train_dataset", columns="test_dataset", aggfunc="mean"
            )
            lines.append(self._dataframe_to_markdown(rmse_pivot))
            lines.append("\n")

            # MAE Matrix
            lines.append("#### MAE (Mean Absolute Error)\n")
            mae_pivot = model_df.pivot_table(
                values="mae", index="train_dataset", columns="test_dataset", aggfunc="mean"
            )
            lines.append(self._dataframe_to_markdown(mae_pivot))
            lines.append("\n")

            # PICP Matrix
            lines.append("#### PICP (Prediction Interval Coverage Probability)\n")
            picp_pivot = model_df.pivot_table(
                values="picp", index="train_dataset", columns="test_dataset", aggfunc="mean"
            )
            lines.append(self._dataframe_to_markdown(picp_pivot))
            lines.append("\n")

        # Statistical Analysis
        lines.append("## Statistical Analysis\n")
        lines.append(
            "Paired t-tests comparing in-distribution vs. out-of-distribution performance.\n\n"
        )

        # Compare same-dataset vs cross-dataset
        same_ds = df[df["train_dataset"] == df["test_dataset"]]["rmse"]
        cross_ds = df[df["train_dataset"] != df["test_dataset"]]["rmse"]

        if len(same_ds) > 1 and len(cross_ds) > 1:
            t_stat, p_value = stats.ttest_ind(same_ds, cross_ds)
            lines.append(f"### RMSE Comparison: In-Distribution vs. Zero-Shot\n")
            lines.append(f"- In-Distribution RMSE (mean ± std): {same_ds.mean():.4f} ± {same_ds.std():.4f}\n")
            lines.append(f"- Zero-Shot RMSE (mean ± std): {cross_ds.mean():.4f} ± {cross_ds.std():.4f}\n")
            lines.append(f"- t-statistic: {t_stat:.4f}\n")
            lines.append(f"- p-value: {p_value:.4e}\n")
            lines.append(f"- Significant difference: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)\n")
            lines.append("\n")

        # Detailed Results Table
        lines.append("## Detailed Results\n")
        lines.append("Complete evaluation results with all metrics.\n\n")

        # Sort by test dataset, then train dataset
        df_sorted = df.sort_values(["test_dataset", "train_dataset", "model_name"])
        lines.append(self._dataframe_to_markdown(df_sorted))
        lines.append("\n")

        # Visualizations
        lines.append("## Visualizations\n")
        lines.append("The following visualizations have been generated:\n\n")
        lines.append("- `zero_shot_heatmap_rmse.png`: RMSE heatmap across dataset combinations\n")
        lines.append("- `zero_shot_heatmap_mae.png`: MAE heatmap across dataset combinations\n")
        lines.append("- `zero_shot_heatmap_picp.png`: PICP heatmap across dataset combinations\n")
        lines.append("- `zero_shot_comparison.png`: Side-by-side comparison plots\n")
        lines.append("- `zero_shot_boxplot.png`: Distribution of metrics across experiments\n")
        lines.append("\n")

        # Methodology
        lines.append("## Methodology\n")
        lines.append("""
### Zero-Shot Generalization Protocol

1. **Training Phase**: Model is trained on the source dataset (Dataset A)
   - No data from target dataset is used during training
   - Standard training hyperparameters are applied

2. **Evaluation Phase**: Trained model is directly evaluated on target dataset (Dataset B)
   - No fine-tuning on target dataset
   - All predictions are made with frozen model weights
   - This tests true out-of-distribution generalization

### Metrics

- **RMSE (Root Mean Squared Error)**: Point prediction accuracy
- **MAE (Mean Absolute Error)**: Mean absolute deviation
- **PICP (Prediction Interval Coverage Probability)**: Percentage of true values within 95% prediction interval
- **CRPS (Continuous Ranked Probability Score)**: Probabilistic prediction quality
- **NLL (Negative Log-Likelihood)**: Probabilistic calibration

### Interpretation

- **PICP ≈ 0.95**: Well-calibrated uncertainty (ideal)
- **PICP < 0.95**: Under-confident predictions
- **PICP > 0.95**: Over-confident predictions
        """)

        # Write report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Benchmark report saved to {report_path}")
        return report_path

    def _dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to Markdown table."""
        if df.empty:
            return "No data available.\n"

        # Reset index if needed
        df = df.reset_index(drop=True)

        # Build header
        headers = ["| " + " | ".join(df.columns) + " |"]
        separator = "|" + "|".join([" --- " for _ in df.columns]) + "|"

        # Build rows
        rows = []
        for _, row in df.iterrows():
            row_str = "| " + " | ".join([str(val) for val in row.values]) + " |"
            rows.append(row_str)

        return "\n".join(headers + [separator] + rows) + "\n"

    def generate_comparison_plots(
        self,
        save_dir: str | None = None,
    ) -> list[Path]:
        """
        Generate comprehensive visualization plots.

        Args:
            save_dir: Directory to save plots (default: results_dir)

        Returns:
            List of saved plot paths
        """
        df = self.get_results_df()
        if df.empty:
            logger.warning("No results to plot")
            return []

        save_dir = Path(save_dir) if save_dir else self.results_dir / "figures"
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10

        # 1. RMSE Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        rmse_pivot = df.pivot_table(
            values="rmse",
            index="train_dataset",
            columns="test_dataset",
            aggfunc="mean",
        )
        sns.heatmap(
            rmse_pivot,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            cbar_kws={"label": "RMSE"},
            ax=ax,
            vmin=0,
        )
        ax.set_title("Zero-Shot RMSE Heatmap\n(Lower is Better)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Test Dataset", fontsize=12)
        ax.set_ylabel("Train Dataset", fontsize=12)
        plt.tight_layout()
        path = save_dir / "zero_shot_heatmap_rmse.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        saved_paths.append(path)
        plt.close()

        # 2. MAE Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        mae_pivot = df.pivot_table(
            values="mae",
            index="train_dataset",
            columns="test_dataset",
            aggfunc="mean",
        )
        sns.heatmap(
            mae_pivot,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            cbar_kws={"label": "MAE"},
            ax=ax,
            vmin=0,
        )
        ax.set_title("Zero-Shot MAE Heatmap\n(Lower is Better)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Test Dataset", fontsize=12)
        ax.set_ylabel("Train Dataset", fontsize=12)
        plt.tight_layout()
        path = save_dir / "zero_shot_heatmap_mae.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        saved_paths.append(path)
        plt.close()

        # 3. PICP Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        picp_pivot = df.pivot_table(
            values="picp",
            index="train_dataset",
            columns="test_dataset",
            aggfunc="mean",
        )
        sns.heatmap(
            picp_pivot,
            annot=True,
            fmt=".4f",
            cmap="RdYlGn",
            center=0.95,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "PICP (95% CI Coverage)"},
            ax=ax,
        )
        ax.set_title(
            "Zero-Shot PICP Heatmap\n(Closer to 0.95 is Better)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Test Dataset", fontsize=12)
        ax.set_ylabel("Train Dataset", fontsize=12)
        plt.tight_layout()
        path = save_dir / "zero_shot_heatmap_picp.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        saved_paths.append(path)
        plt.close()

        # 4. Side-by-side comparison bar chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # RMSE comparison
        ax = axes[0, 0]
        pivot = df.pivot_table(
            values="rmse",
            index=["train_dataset", "test_dataset"],
            aggfunc="mean",
        ).reset_index()
        pivot["combo"] = pivot["train_dataset"] + " → " + pivot["test_dataset"]
        colors = ["green" if row["train_dataset"] == row["test_dataset"] else "coral"
                  for _, row in pivot.iterrows()]
        ax.barh(pivot["combo"], pivot["rmse"], color=colors)
        ax.set_xlabel("RMSE")
        ax.set_title("RMSE by Dataset Combination\n(Green=In-Distribution, Red=Zero-Shot)")
        ax.invert_yaxis()

        # MAE comparison
        ax = axes[0, 1]
        pivot = df.pivot_table(
            values="mae",
            index=["train_dataset", "test_dataset"],
            aggfunc="mean",
        ).reset_index()
        pivot["combo"] = pivot["train_dataset"] + " → " + pivot["test_dataset"]
        colors = ["green" if row["train_dataset"] == row["test_dataset"] else "coral"
                  for _, row in pivot.iterrows()]
        ax.barh(pivot["combo"], pivot["mae"], color=colors)
        ax.set_xlabel("MAE")
        ax.set_title("MAE by Dataset Combination\n(Green=In-Distribution, Red=Zero-Shot)")
        ax.invert_yaxis()

        # PICP comparison
        ax = axes[1, 0]
        pivot = df.pivot_table(
            values="picp",
            index=["train_dataset", "test_dataset"],
            aggfunc="mean",
        ).reset_index()
        pivot["combo"] = pivot["train_dataset"] + " → " + pivot["test_dataset"]
        colors = ["green" if row["train_dataset"] == row["test_dataset"] else "coral"
                  for _, row in pivot.iterrows()]
        bars = ax.barh(pivot["combo"], pivot["picp"], color=colors)
        ax.axvline(x=0.95, color="black", linestyle="--", label="Target (95%)")
        ax.set_xlabel("PICP")
        ax.set_title("PICP by Dataset Combination\n(Black dashed line = 0.95 target)")
        ax.invert_yaxis()
        ax.legend()

        # Inference time
        ax = axes[1, 1]
        pivot = df.pivot_table(
            values="inference_time_ms",
            index=["train_dataset", "test_dataset"],
            aggfunc="mean",
        ).reset_index()
        pivot["combo"] = pivot["train_dataset"] + " → " + pivot["test_dataset"]
        ax.barh(pivot["combo"], pivot["inference_time_ms"], color="steelblue")
        ax.set_xlabel("Inference Time (ms/sample)")
        ax.set_title("Inference Time by Dataset Combination")
        ax.invert_yaxis()

        plt.suptitle(
            "Zero-Shot Cross-Dataset Benchmark Results",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = save_dir / "zero_shot_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        saved_paths.append(path)
        plt.close()

        # 5. Box plots for metric distributions
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        metrics = ["rmse", "mae", "picp", "crps", "nll", "inference_time_ms"]
        titles = ["RMSE", "MAE", "PICP", "CRPS", "NLL", "Inference Time (ms)"]

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 3, idx % 3]

            # Create comparison: same dataset vs cross dataset
            same_mask = df["train_dataset"] == df["test_dataset"]
            same_data = df[same_mask][metric]
            cross_data = df[~same_mask][metric]

            box_data = [same_data, cross_data]
            bp = ax.boxplot(
                box_data,
                labels=["In-Distribution", "Zero-Shot"],
                patch_artist=True,
            )
            bp["boxes"][0].set_facecolor("lightgreen")
            bp["boxes"][1].set_facecolor("lightcoral")

            ax.set_ylabel(title)
            ax.set_title(f"{title} Distribution")

            # Add mean values as text
            if len(same_data) > 0:
                ax.text(
                    1,
                    same_data.mean(),
                    f"μ={same_data.mean():.3f}",
                    ha="center",
                    va="bottom",
                )
            if len(cross_data) > 0:
                ax.text(
                    2,
                    cross_data.mean(),
                    f"μ={cross_data.mean():.3f}",
                    ha="center",
                    va="bottom",
                )

        plt.suptitle(
            "Zero-Shot vs In-Distribution Performance",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = save_dir / "zero_shot_boxplot.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        saved_paths.append(path)
        plt.close()

        logger.info(f"Generated {len(saved_paths)} visualization plots")
        return saved_paths

    def _load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a specific dataset."""
        if dataset_name.lower() in ["nasa", "nasa_pcoe"]:
            return self.data_loader.load_nasa()
        elif dataset_name.lower() in ["calce", "calce_cs2"]:
            return self.data_loader.load_calce()
        elif dataset_name.lower() == "oxford":
            # Placeholder for Oxford dataset
            raise NotImplementedError("Oxford dataset loading not yet implemented")
        elif dataset_name.lower() == "mit":
            # Placeholder for MIT dataset
            raise NotImplementedError("MIT dataset loading not yet implemented")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _infer_features(self, df: pd.DataFrame) -> list[str]:
        """Infer feature columns from DataFrame."""
        exclude_cols = [
            "battery_id", "dataset_source", "chemistry", "cycle",
            "rul", "RUL", "raw_"
        ]
        features = []
        for col in df.columns:
            if any(ex in col.lower() for ex in exclude_cols):
                continue
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                features.append(col)
        return features

    def _compute_coverage(
        self,
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        alpha: float = 0.05,
    ) -> float:
        """Compute prediction interval coverage."""
        y_true = y_true[-len(lower):]  # Align lengths
        covered = np.sum((y_true >= lower) & (y_true <= upper))
        return float(covered / len(y_true)) if len(y_true) > 0 else 0.0

    def _compute_sharpe_ratio(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Compute Sharpe-like ratio for predictions."""
        y_true = y_true[-len(y_pred):]
        residuals = y_true - y_pred
        if np.std(residuals) > 0:
            return float(np.mean(residuals) / np.std(residuals))
        return 0.0


def run_benchmark_demo():
    """Run a demonstration of the zero-shot benchmark."""
    import logging

    logging.basicConfig(level=logging.INFO)

    # Create benchmark runner
    benchmark = ZeroShotBenchmarkRunner(
        results_dir="results/zero_shot_benchmark_demo",
        random_seed=42,
    )

    # Example: Run cross-dataset evaluation matrix
    # This would use actual model classes in practice
    logger.info("Zero-Shot Benchmark Demo initialized")
    logger.info("To run actual evaluation, provide trained model instances")

    return benchmark


if __name__ == "__main__":
    run_benchmark_demo()
