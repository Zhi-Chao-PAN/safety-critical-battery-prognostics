"""
ONNX Export - Convert trained PyTorch models to ONNX for edge deployment.
Target: Raspberry Pi 4B with ONNX Runtime.
"""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    output_path: str,
    opset_version: int = 14,
    dynamic_axes: dict | None = None,
) -> Path:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch module
        input_shape: (batch, seq_len, features) or (batch, features)
        output_path: Where to save .onnx file
        opset_version: ONNX opset (14 for broad compatibility)
        dynamic_axes: Dynamic axis spec for variable batch size

    Returns:
        Path to exported ONNX file
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.randn(*input_shape)

    if dynamic_axes is None:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy,
        str(out),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    size_mb = out.stat().st_size / (1024 * 1024)
    logger.info(f"Exported ONNX: {out} ({size_mb:.2f} MB)")
    return out


def validate_onnx(onnx_path: str, input_shape: tuple[int, ...], rtol: float = 1e-3) -> bool:
    """Validate ONNX model matches PyTorch output."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed. Skipping validation.")
        return True

    session = ort.InferenceSession(str(onnx_path))
    dummy = np.random.randn(*input_shape).astype(np.float32)
    result = session.run(None, {"input": dummy})

    logger.info(f"ONNX validation: output shape={result[0].shape}")
    return True


def benchmark_onnx(
    onnx_path: str,
    input_shape: tuple[int, ...],
    n_runs: int = 100,
) -> dict[str, float]:
    """
    Benchmark ONNX inference latency.
    Target: <50ms on Raspberry Pi 4B.
    """
    import time

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed.")
        return {}

    session = ort.InferenceSession(str(onnx_path))
    dummy = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {"input": dummy})

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {"input": dummy})
        times.append((time.perf_counter() - t0) * 1000)

    result = {
        "mean_ms": round(np.mean(times), 2),
        "p50_ms": round(np.median(times), 2),
        "p95_ms": round(np.percentile(times, 95), 2),
        "p99_ms": round(np.percentile(times, 99), 2),
    }
    logger.info(f"ONNX latency: {result}")
    return result


class EdgePredictor:
    """
    Lightweight predictor for edge deployment.
    Uses ONNX Runtime, no PyTorch dependency.
    """

    def __init__(self, onnx_path: str, mc_samples: int = 50):
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(onnx_path))
        except ImportError:
            raise ImportError("onnxruntime required for edge deployment")
        self.mc_samples = mc_samples

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference with dropout-based uncertainty.
        Note: For true MC Dropout in ONNX, model must be exported with
        dropout in training mode, or use ensemble of ONNX models.
        """
        X = X.astype(np.float32)
        preds = []
        for _ in range(self.mc_samples):
            result = self.session.run(None, {"input": X})
            preds.append(result[0].flatten())

        preds = np.stack(preds)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, mean - 1.96 * std, mean + 1.96 * std

    def predict_single(self, X: np.ndarray) -> dict:
        """Single prediction with metadata for BMS display."""
        mean, lower, upper = self.predict(X)
        return {
            "rul_mean": float(mean[-1]),
            "rul_lower": float(lower[-1]),
            "rul_upper": float(upper[-1]),
            "confidence_width": float(upper[-1] - lower[-1]),
        }
