"""
Ablation Study Runner.
Systematically removes components to measure their contribution.
"""

import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.unified_loader import UnifiedDataLoader
from src.data.validator import DataValidator
from src.features.extractor import FeatureExtractor
from src.models import LSTMModel, GRUModel, TCNModel, TransformerModel, PINNModel, BayesianNNModel, CNN1DModel
from src.evaluation.benchmark import BenchmarkRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    # Load data
    loader = UnifiedDataLoader()
    df = loader.load_all(nasa_dir=str(ROOT / "data" / "battery_data"))
    validator = DataValidator()
    df, _ = validator.validate(df)
    extractor = FeatureExtractor()
    df = extractor.extract_all(df)

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ("cycle", "rul")]
    df = df.dropna(subset=feature_cols + ["rul"]).reset_index(drop=True)
    n_feat = len(feature_cols)

    logger.info(f"Data: {len(df)} rows, {n_feat} features")

    # ── Ablation 1: Architecture Comparison ──
    logger.info("=== Ablation 1: Architecture Comparison ===")
    arch_models = {
        "LSTM": LSTMModel(input_dim=n_feat, hidden_dim=64, num_layers=2, seq_length=30, epochs=50, mc_samples=50),
        "GRU": GRUModel(input_dim=n_feat, hidden_dim=64, num_layers=2, seq_length=30, epochs=50, mc_samples=50),
        "TCN": TCNModel(input_dim=n_feat, num_channels=[32, 32, 64, 64], seq_length=30, epochs=50, mc_samples=50),
        "CNN1D": CNN1DModel(input_dim=n_feat, channels=[32, 64, 64], seq_length=30, epochs=50, mc_samples=50),
        "Transformer": TransformerModel(input_dim=n_feat, d_model=64, nhead=4, seq_length=30, epochs=50, mc_samples=50),
        "PINN": PINNModel(input_dim=n_feat, hidden_dim=64, epochs=50, mc_samples=50),
        "BayesianNN": BayesianNNModel(input_dim=n_feat, hidden_dim=64, epochs=50, n_samples=50),
    }

    runner = BenchmarkRunner(features=feature_cols, n_seeds=3, results_dir=str(ROOT / "results" / "ablation"))
    arch_results = runner.run(df, arch_models, seeds=[42, 43, 44])
    arch_results.to_csv(ROOT / "results" / "ablation" / "architecture_comparison.csv", index=False)

    # ── Ablation 2: Sequence Length ──
    logger.info("=== Ablation 2: Sequence Length ===")
    seq_models = {}
    for sl in [5, 10, 20, 30, 50]:
        seq_models[f"LSTM_seq{sl}"] = LSTMModel(
            input_dim=n_feat, hidden_dim=64, num_layers=2,
            seq_length=sl, epochs=50, mc_samples=50,
        )
    seq_results = runner.run(df, seq_models, seeds=[42, 43])
    seq_results.to_csv(ROOT / "results" / "ablation" / "sequence_length.csv", index=False)

    # ── Ablation 3: Hidden Dimension ──
    logger.info("=== Ablation 3: Hidden Dimension ===")
    dim_models = {}
    for hd in [16, 32, 64, 128]:
        dim_models[f"LSTM_h{hd}"] = LSTMModel(
            input_dim=n_feat, hidden_dim=hd, num_layers=2,
            seq_length=30, epochs=50, mc_samples=50,
        )
    dim_results = runner.run(df, dim_models, seeds=[42, 43])
    dim_results.to_csv(ROOT / "results" / "ablation" / "hidden_dimension.csv", index=False)

    # ── Ablation 4: MC Samples ──
    logger.info("=== Ablation 4: MC Dropout Samples ===")
    mc_models = {}
    for mc in [10, 25, 50, 100, 200]:
        mc_models[f"LSTM_mc{mc}"] = LSTMModel(
            input_dim=n_feat, hidden_dim=64, num_layers=2,
            seq_length=30, epochs=50, mc_samples=mc,
        )
    mc_results = runner.run(df, mc_models, seeds=[42])
    mc_results.to_csv(ROOT / "results" / "ablation" / "mc_samples.csv", index=False)

    logger.info("All ablation experiments complete.")


if __name__ == "__main__":
    main()
