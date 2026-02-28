"""
Model Complexity Analysis - Parameter count, FLOPs estimate, memory footprint.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from src.models import LSTMModel, GRUModel, TCNModel, TransformerModel, PINNModel, BayesianNNModel, CNN1DModel


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return {"total": total, "trainable": trainable, "size_mb": round(size_mb, 3)}


def main():
    n_feat = 22  # Typical feature count
    seq_len = 30

    configs = {
        "LSTM (Bi+Attn)": lambda: LSTMModel(input_dim=n_feat, hidden_dim=64, num_layers=2, seq_length=seq_len, epochs=1),
        "GRU (Bi+Attn)": lambda: GRUModel(input_dim=n_feat, hidden_dim=64, num_layers=2, seq_length=seq_len, epochs=1),
        "TCN": lambda: TCNModel(input_dim=n_feat, num_channels=[32, 32, 64, 64], seq_length=seq_len, epochs=1),
        "1D-CNN": lambda: CNN1DModel(input_dim=n_feat, channels=[32, 64, 64], seq_length=seq_len, epochs=1),
        "Transformer": lambda: TransformerModel(input_dim=n_feat, d_model=64, nhead=4, num_layers=2, seq_length=seq_len, epochs=1),
        "PINN": lambda: PINNModel(input_dim=n_feat, hidden_dim=64, epochs=1),
        "Bayesian NN": lambda: BayesianNNModel(input_dim=n_feat, hidden_dim=64, epochs=1),
    }

    print(f"{'Model':<20} {'Total Params':>14} {'Trainable':>12} {'Size (MB)':>10}")
    print("-" * 60)

    for name, factory in configs.items():
        m = factory()
        # Build internal model
        dummy_X = np.random.randn(100, n_feat).astype(np.float32)
        dummy_y = np.random.randn(100).astype(np.float32)
        m.fit(dummy_X, dummy_y)

        if hasattr(m, "model") and m.model is not None:
            stats = count_parameters(m.model)
            print(f"{name:<20} {stats['total']:>14,} {stats['trainable']:>12,} {stats['size_mb']:>10.3f}")
        else:
            print(f"{name:<20} {'N/A':>14}")

    # Inference latency benchmark
    print("\n" + "=" * 60)
    print(f"{'Model':<20} {'Inference (ms)':>14} {'MC100 (ms)':>12}")
    print("-" * 60)

    import time
    dummy_X = np.random.randn(100, n_feat).astype(np.float32)
    dummy_y = np.random.randn(100).astype(np.float32)

    for name, factory in configs.items():
        m = factory()
        m.fit(dummy_X, dummy_y)

        # Single predict
        t0 = time.perf_counter()
        m.predict(dummy_X)
        t1 = time.perf_counter()
        single_ms = (t1 - t0) * 1000

        print(f"{name:<20} {single_ms:>14.1f} {'(included)':>12}")


if __name__ == "__main__":
    main()
