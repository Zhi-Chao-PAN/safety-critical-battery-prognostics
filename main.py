"""
main.py - Full pipeline: Load → Validate → Extract Features → Train → Evaluate → Visualize

Usage:
    python main.py
    python main.py --models lstm gru tcn transformer pinn
    python main.py --seeds 42 43 44
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.unified_loader import UnifiedDataLoader
from src.data.validator import DataValidator
from src.data.splitter import DataSplitter
from src.features.extractor import FeatureExtractor
from src.models import LSTMModel, GRUModel, TCNModel, TransformerModel, PINNModel, BayesianNNModel, DeepEnsemble
from src.models.cnn1d_model import CNN1DModel, BayesianNNModel, CNN1DModel
from src.uncertainty.scoring import compute_all_metrics
from src.training.pipeline import TrainingPipeline
from src.evaluation.benchmark import BenchmarkRunner
from src.safety.decision_engine import SafetyDecisionEngine
from src.ui.visualization import (
    plot_degradation_curves,
    plot_model_comparison,
    plot_safety_buffer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "run.log", mode="w"),
    ],
)
logger = logging.getLogger("main")


def parse_args():
    p = argparse.ArgumentParser(description="Battery Prognostics Pipeline")
    p.add_argument("--data-dir", default="data/battery_data", help="Path to .mat files")
    p.add_argument("--models", nargs="+", default=["lstm", "gru", "tcn", "cnn1d", "transformer", "pinn", "bayesian_nn"],
                   help="Models to train")
    p.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p.add_argument("--seq-length", type=int, default=30, help="Sequence length")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--skip-viz", action="store_true", help="Skip visualization")
    return p.parse_args()


def build_models(args, input_dim: int) -> dict:
    """Build model zoo from args."""
    common = dict(input_dim=input_dim, seq_length=args.seq_length,
                  epochs=args.epochs, device=args.device)
    registry = {
        "lstm": lambda: LSTMModel(**common, hidden_dim=64, num_layers=2),
        "gru": lambda: GRUModel(**common, hidden_dim=64, num_layers=2),
        "tcn": lambda: TCNModel(**common, num_channels=[32, 32, 64, 64]),
        "cnn1d": lambda: CNN1DModel(**common, channels=[32, 64, 64]),
        "transformer": lambda: TransformerModel(**common, d_model=64, nhead=4, num_layers=2),
        "pinn": lambda: PINNModel(**common, hidden_dim=64),
        "bayesian_nn": lambda: BayesianNNModel(input_dim=input_dim, hidden_dim=64,
                                                epochs=args.epochs, device=args.device),
    }
    models = {}
    for name in args.models:
        if name in registry:
            models[name] = registry[name]()
        elif name == "ensemble":
            base = LSTMModel(**common, hidden_dim=64, num_layers=2)
            models["ensemble"] = DeepEnsemble(base, n_members=5)
        else:
            logger.warning(f"Unknown model: {name}")
    return models


def main():
    args = parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    # ── Step 1: Load Data ──
    logger.info("Step 1: Loading data...")
    loader = UnifiedDataLoader()
    df = loader.load_all(nasa_dir=str(ROOT / args.data_dir))
    logger.info(f"Loaded {len(df)} rows, {df['battery_id'].nunique()} batteries")

    # ── Step 2: Validate ──
    logger.info("Step 2: Validating data...")
    validator = DataValidator()
    df, report = validator.validate(df)
    logger.info(f"Validation: {report.pass_rate:.1%} pass rate, {report.flagged_rows} flagged rows")

    # ── Step 3: Extract Features ──
    logger.info("Step 3: Extracting features...")
    extractor = FeatureExtractor()
    df = extractor.extract_all(df)

    # Select numeric features (exclude IDs, strings)
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ("cycle", "rul")]
    logger.info(f"Features ({len(feature_cols)}): {feature_cols[:10]}...")

    # Drop NaN rows from feature extraction
    df = df.dropna(subset=feature_cols + ["rul"]).reset_index(drop=True)
    logger.info(f"After cleanup: {len(df)} rows")

    if len(df) < 50:
        logger.error("Not enough data after cleanup. Check data files.")
        return

    # ── Step 4: Visualize raw data ──
    if not args.skip_viz:
        logger.info("Step 4: Generating degradation curves...")
        plot_degradation_curves(df, str(ROOT / "figures" / "fig01_degradation.png"))

    # ── Step 5: Build & Train Models ──
    logger.info("Step 5: Training models...")
    models = build_models(args, input_dim=len(feature_cols))

    pipeline = TrainingPipeline(
        features=feature_cols,
        target="rul",
        checkpoint_dir=str(ROOT / "checkpoints"),
        log_dir=str(ROOT / "logs"),
    )
    results_df = pipeline.train_all_models(df, models, seeds=args.seeds)
    logger.info(f"\n{results_df.to_string()}")

    # ── Step 6: Benchmark ──
    logger.info("Step 6: Running benchmark...")
    runner = BenchmarkRunner(
        features=feature_cols,
        target="rul",
        n_seeds=len(args.seeds),
        results_dir=str(ROOT / "results"),
    )
    bench_df = runner.run(df, models, seeds=args.seeds)

    # ── Step 7: Visualize results ──
    if not args.skip_viz and len(bench_df) > 0:
        logger.info("Step 7: Generating comparison plots...")
        plot_model_comparison(bench_df, str(ROOT / "figures" / "fig03_comparison.png"))

    # ── Step 8: Safety demo ──
    logger.info("Step 8: Safety decision demo...")
    engine = SafetyDecisionEngine()
    if len(bench_df) > 0:
        sample_model = list(models.values())[0]
        sample_bat = df["battery_id"].unique()[0]
        sample_df = df[df["battery_id"] == sample_bat].sort_values("cycle")
        X_sample = sample_df[feature_cols].values
        sample_model.fit(df[feature_cols].values, df["rul"].values)
        mean, lower, upper = sample_model.predict(X_sample)
        if len(mean) > 0:
            decisions = engine.decide_batch(mean, lower, upper)
            red_count = sum(1 for d in decisions if d.level.value == "RED")
            yellow_count = sum(1 for d in decisions if d.level.value == "YELLOW")
            logger.info(f"Safety: {red_count} RED, {yellow_count} YELLOW, "
                        f"{len(decisions) - red_count - yellow_count} GREEN")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
