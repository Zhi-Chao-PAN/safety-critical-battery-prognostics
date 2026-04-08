"""
Zero-Shot Cross-Dataset Benchmark Runner for Battery Prognostics Models.

This script demonstrates the complete zero-shot evaluation pipeline:
1. Load multiple datasets (NASA, CALCE, etc.)
2. Train model on source dataset
3. Evaluate on target dataset without fine-tuning
4. Generate comprehensive benchmark reports

Usage:
    python scripts/run_zero_shot_benchmark.py --model pinn --train nasa --test calce
    python scripts/run_zero_shot_benchmark.py --run-full-matrix
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.zero_shot_benchmark import ZeroShotBenchmarkRunner
from src.data.unified_loader import UnifiedDataLoader

logger = logging.getLogger(__name__)


def create_pinn_model(input_dim: int = 8, **kwargs):
    """Create a capacity-space PINN model for zero-shot evaluation."""
    from src.models.pinn_model import PINNModel

    model = PINNModel(
        input_dim=input_dim,
        hidden_dim=128,
        lambda_physics=0.1,
        dropout=0.1,
        **kwargs
    )
    return model


def create_lstm_model(input_dim: int = 8, **kwargs):
    """Create LSTM baseline model."""
    from src.models.lstm_model import LSTMModel

    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        **kwargs
    )
    return model


def create_gru_model(input_dim: int = 8, **kwargs):
    """Create GRU baseline model."""
    from src.models.gru_model import GRUModel

    model = GRUModel(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        **kwargs
    )
    return model


MODEL_REGISTRY = {
    "pinn": create_pinn_model,
    "lstm": create_lstm_model,
    "gru": create_gru_model,
}


def run_single_evaluation(
    model_name: str,
    train_dataset: str,
    test_dataset: str,
    results_dir: str = "results/zero_shot_benchmark",
    save_model: bool = True,
) -> dict:
    """
    Run single zero-shot evaluation.

    Args:
        model_name: Model type ("pinn", "lstm", "gru")
        train_dataset: Training dataset name
        test_dataset: Test dataset name
        results_dir: Directory to save results
        save_model: Whether to save trained model

    Returns:
        Dictionary with evaluation results
    """
    logger.info(
        f"Running zero-shot evaluation: {model_name} "
        f"({train_dataset} → {test_dataset})"
    )

    # Create benchmark runner
    benchmark = ZeroShotBenchmarkRunner(
        results_dir=results_dir,
        random_seed=42,
    )

    # Load data to infer input dimension
    data_loader = UnifiedDataLoader()
    sample_df = data_loader.load_nasa() if train_dataset == "nasa" else data_loader.load_calce()

    # Infer features and get input dimension
    exclude_cols = ["battery_id", "dataset_source", "chemistry", "cycle", "rul", "RUL"]
    features = [c for c in sample_df.columns
                if not any(ex in c.lower() for ex in exclude_cols)
                and sample_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    input_dim = len(features)
    logger.info(f"Inferred {input_dim} features: {features}")

    # Create model
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[model_name](input_dim=input_dim)
    logger.info(
        "Model prediction target: %s | evaluation target: rul",
        getattr(model, "prediction_target", "rul"),
    )

    # Run zero-shot evaluation
    result = benchmark.run_zero_shot(
        model=model,
        model_name=model_name,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        features=features,
        target="rul",  # Evaluation target only; capacity models still train on capacity.
        save_model=save_model,
    )

    # Generate report and plots
    report_path = benchmark.generate_markdown_report(
        title=f"Zero-Shot Benchmark: {model_name.upper()} ({train_dataset} → {test_dataset})",
    )
    plot_paths = benchmark.generate_comparison_plots()

    logger.info(f"Report saved to: {report_path}")
    logger.info(f"Generated {len(plot_paths)} visualization plots")

    return {
        "result": result.to_dict(),
        "report_path": str(report_path),
        "plot_paths": [str(p) for p in plot_paths],
    }


def run_full_matrix_evaluation(
    model_name: str,
    datasets: list[str] | None = None,
    results_dir: str = "results/zero_shot_benchmark",
) -> dict:
    """
    Run full cross-dataset evaluation matrix.

    Evaluates all combinations of train_dataset × test_dataset pairs.

    Args:
        model_name: Model type to evaluate
        datasets: List of dataset names (default: ["nasa", "calce"])
        results_dir: Directory to save results

    Returns:
        Dictionary with all results
    """
    if datasets is None:
        datasets = ["nasa", "calce"]

    logger.info(
        f"Running full matrix evaluation for {model_name} "
        f"across datasets: {datasets}"
    )

    # Create benchmark runner
    benchmark = ZeroShotBenchmarkRunner(
        results_dir=results_dir,
        random_seed=42,
    )

    # Run all combinations
    all_results = []
    for train_ds in datasets:
        for test_ds in datasets:
            logger.info(f"Evaluating: {train_ds} → {test_ds}")
            try:
                result = run_single_evaluation(
                    model_name=model_name,
                    train_dataset=train_ds,
                    test_dataset=test_ds,
                    results_dir=f"{results_dir}/{train_ds}_to_{test_ds}",
                    save_model=True,
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed {train_ds} → {test_ds}: {e}")
                continue

    # Generate comprehensive report
    report_path = benchmark.generate_markdown_report(
        title=f"Zero-Shot Cross-Dataset Benchmark: {model_name.upper()}",
    )

    # Generate all visualizations
    plot_paths = benchmark.generate_comparison_plots()

    logger.info(f"\n{'='*60}")
    logger.info("Full matrix evaluation completed!")
    logger.info(f"Report: {report_path}")
    logger.info(f"Visualizations: {len(plot_paths)} plots generated")
    logger.info(f"{'='*60}")

    return {
        "results": all_results,
        "report_path": str(report_path),
        "plot_paths": [str(p) for p in plot_paths],
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Zero-Shot Cross-Dataset Benchmark for Battery Prognostics Models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pinn",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model type to evaluate",
    )
    parser.add_argument(
        "--train",
        type=str,
        default="nasa",
        help="Training dataset name",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="calce",
        help="Test dataset name for zero-shot evaluation",
    )
    parser.add_argument(
        "--run-full-matrix",
        action="store_true",
        help="Run full cross-dataset evaluation matrix",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["nasa", "calce"],
        help="List of datasets for full matrix evaluation",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/zero_shot_benchmark",
        help="Directory to save results",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run benchmark
    if args.run_full_matrix:
        results = run_full_matrix_evaluation(
            model_name=args.model,
            datasets=args.datasets,
            results_dir=args.results_dir,
        )
    else:
        results = run_single_evaluation(
            model_name=args.model,
            train_dataset=args.train,
            test_dataset=args.test,
            results_dir=args.results_dir,
        )

    print(f"\n{'='*60}")
    print("Benchmark completed successfully!")
    print(f"Results saved to: {args.results_dir}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    main()
