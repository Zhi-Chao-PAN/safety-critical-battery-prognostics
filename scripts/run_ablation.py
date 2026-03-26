"""
Run ablation studies from configs/ablation.yaml.

Usage:
    python scripts/run_ablation.py --study architecture
    python scripts/run_ablation.py --study all
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.unified_loader import UnifiedDataLoader
from src.data.validator import DataValidator
from src.evaluation.benchmark import BenchmarkRunner
from src.features.extractor import FeatureExtractor
from src.models import (
    BayesianNNModel,
    GRUModel,
    LSTMModel,
    PINNModel,
    TCNModel,
    TransformerModel,
)
from src.models.cnn1d_model import CNN1DModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "lstm": LSTMModel,
    "gru": GRUModel,
    "tcn": TCNModel,
    "transformer": TransformerModel,
    "pinn": PINNModel,
    "bayesian_nn": BayesianNNModel,
    "cnn1d": CNN1DModel,
}


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


def run_architecture_ablation(cfg: dict, df, feature_cols):
    """Ablation 1: Compare all architectures."""
    logger.info("=== Architecture Ablation ===")
    seeds = cfg.get("seeds", [42, 43, 44])
    epochs = cfg.get("epochs", 100)
    seq_len = cfg.get("seq_length", 30)
    input_dim = len(feature_cols)

    models = {}
    for m_cfg in cfg["models"]:
        name = m_cfg["name"]
        cls = MODEL_REGISTRY.get(name)
        if cls is None:
            continue
        params = {"input_dim": input_dim, "epochs": epochs, "seq_length": seq_len}
        for k, v in m_cfg.items():
            if k != "name":
                params[k] = v
        # Filter params to only those accepted by the class
        import inspect
        valid = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
        params = {k: v for k, v in params.items() if k in valid}
        models[name] = cls(**params)

    runner = BenchmarkRunner(features=feature_cols, results_dir=str(ROOT / "results" / "ablation_arch"))
    results = runner.run(df, models, seeds=seeds)
    logger.info(f"Architecture ablation: {len(results)} results")
    return results


def run_sequence_length_ablation(cfg: dict, df, feature_cols):
    """Ablation 3: Sequence length impact."""
    logger.info("=== Sequence Length Ablation ===")
    seeds = cfg.get("seeds", [42, 43, 44])
    values = cfg.get("values", [5, 10, 20, 30, 50])
    input_dim = len(feature_cols)

    models = {}
    for sl in values:
        models[f"lstm_seq{sl}"] = LSTMModel(
            input_dim=input_dim, hidden_dim=64, seq_length=sl, epochs=100
        )

    runner = BenchmarkRunner(features=feature_cols, results_dir=str(ROOT / "results" / "ablation_seqlen"))
    return runner.run(df, models, seeds=seeds)


def run_hidden_dim_ablation(cfg: dict, df, feature_cols):
    """Ablation 6: Hidden dimension impact."""
    logger.info("=== Hidden Dimension Ablation ===")
    seeds = cfg.get("seeds", [42, 43, 44])
    values = cfg.get("values", [16, 32, 64, 128])
    input_dim = len(feature_cols)

    models = {}
    for hd in values:
        models[f"lstm_h{hd}"] = LSTMModel(
            input_dim=input_dim, hidden_dim=hd, seq_length=30, epochs=100
        )

    runner = BenchmarkRunner(features=feature_cols, results_dir=str(ROOT / "results" / "ablation_hidden"))
    return runner.run(df, models, seeds=seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="architecture",
                        choices=["architecture", "sequence_length", "hidden_dim", "all"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "ablation.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        all_cfg = yaml.safe_load(f)

    df, feature_cols = load_data()
    logger.info(f"Data: {len(df)} rows, {len(feature_cols)} features")

    studies = {
        "architecture": lambda: run_architecture_ablation(all_cfg["architecture"], df, feature_cols),
        "sequence_length": lambda: run_sequence_length_ablation(all_cfg["sequence_length"], df, feature_cols),
        "hidden_dim": lambda: run_hidden_dim_ablation(all_cfg["hidden_dim"], df, feature_cols),
    }

    if args.study == "all":
        for name, fn in studies.items():
            fn()
    else:
        studies[args.study]()

    logger.info("Ablation complete.")


if __name__ == "__main__":
    main()
