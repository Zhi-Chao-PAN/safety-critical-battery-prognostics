"""
Chronos Foundation Model Wrapper for Battery RUL Prediction.

Wraps Amazon Chronos-T5 (a pretrained time-series foundation model) into the
project's BatteryModel ABC interface. Designed for zero-shot evaluation:
the model has NEVER seen battery degradation data during its pretraining.

Architecture:
    Chronos tokenizes continuous time-series values into discrete bins via
    mean-scaling + quantization, then feeds them through a T5 encoder-decoder
    to produce probabilistic forecasts as categorical distributions over bins.

Usage:
    model = ChronosZeroShotModel(model_id="amazon/chronos-t5-small")
    model.fit(X_context, y_dummy)          # no-op: zero-shot, no training
    mean, lower, upper = model.predict(X)  # capacity column extracted internally
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import BatteryModel

logger = logging.getLogger(__name__)

# Lazy imports: chronos is an optional dependency
_chronos_available: bool | None = None


def _check_chronos() -> bool:
    """Check if chronos-forecasting is installed. Cache result."""
    global _chronos_available
    if _chronos_available is None:
        try:
            import chronos  # noqa: F401
            _chronos_available = True
        except ImportError:
            _chronos_available = False
    return _chronos_available


class ChronosZeroShotModel(BatteryModel):
    """
    Zero-shot time-series forecasting via Amazon Chronos-T5.

    This model performs NO training. It uses a pretrained foundation model
    to forecast future capacity values given a historical context window.

    The predict() method expects a 2D array where one column is 'capacity'.
    Since Chronos is univariate, all other physical features are stripped.

    Attributes:
        model_id:         HuggingFace model identifier (e.g. "amazon/chronos-t5-small")
        num_samples:      Number of Monte Carlo forecast samples for uncertainty
        prediction_length: Number of future steps to forecast
        context_ratio:    Fraction of input series used as context (rest is ground truth)
        device_map:       "cpu" or "cuda"
        torch_dtype_str:  "float32" or "bfloat16"
        confidence_level: Confidence level for prediction intervals (default 0.95)
    """

    name: str = "chronos_zero_shot"
    prediction_target: str = "capacity"

    def __init__(
        self,
        model_id: str = "amazon/chronos-t5-small",
        num_samples: int = 20,
        prediction_length: int = 20,
        context_ratio: float = 0.8,
        device_map: str = "cpu",
        torch_dtype_str: str = "float32",
        confidence_level: float = 0.95,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        capacity_col_index: int = 0,
        **kwargs: Any,
    ):
        if not _check_chronos():
            raise ImportError(
                "chronos-forecasting is not installed. "
                "Install with: pip install chronos-forecasting transformers accelerate"
            )

        self.model_id = model_id
        self.num_samples = num_samples
        self.prediction_length = prediction_length
        self.context_ratio = context_ratio
        self.device_map = device_map
        self.torch_dtype_str = torch_dtype_str
        self.confidence_level = confidence_level
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.capacity_col_index = capacity_col_index

        self._pipeline = None
        self._context: np.ndarray | None = None

    def _ensure_pipeline(self) -> None:
        """Lazy-load the Chronos pipeline to avoid unnecessary weight loading."""
        if self._pipeline is not None:
            return

        import torch
        from chronos import ChronosPipeline

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.torch_dtype_str, torch.float32)

        # Safety: force float32 on CPU (bfloat16 can be slow or unsupported)
        if self.device_map == "cpu" and torch_dtype != torch.float32:
            logger.warning(
                f"Overriding torch_dtype from {self.torch_dtype_str} to float32 "
                f"for CPU execution (bfloat16 is not well-supported on CPU)."
            )
            torch_dtype = torch.float32

        logger.info(
            f"Loading Chronos pipeline: model_id={self.model_id}, "
            f"device={self.device_map}, dtype={torch_dtype}"
        )
        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            dtype=torch_dtype,
        )
        logger.info("Chronos pipeline loaded successfully.")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "ChronosZeroShotModel":
        """
        No-op for zero-shot model. Stores the capacity context for later use.

        Args:
            X: Feature matrix (N, D). Column at capacity_col_index is capacity.
            y: Target values (ignored in zero-shot mode).

        Returns:
            self (unchanged)
        """
        # Store capacity series as context for predict() calls
        if X.ndim == 2:
            self._context = X[:, self.capacity_col_index].copy()
        else:
            self._context = X.copy()

        logger.info(
            f"ChronosZeroShotModel.fit(): Stored {len(self._context)} context points. "
            f"No training performed (zero-shot)."
        )
        return self

    def predict(
        self, X: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate probabilistic forecasts using the pretrained Chronos model.

        Strategy:
            1. Extract capacity column from X as the target series
            2. Use context_ratio to split into context (input) and evaluation (ground truth)
            3. Feed context to Chronos, get sampled forecast distribution
            4. Compute median, lower (alpha/2), upper (1-alpha/2) quantiles

        Args:
            X: Feature matrix (N, D) or 1D capacity array.

        Returns:
            mean:  Median forecast (prediction_length,)
            lower: Lower CI bound (prediction_length,)
            upper: Upper CI bound (prediction_length,)
        """
        import torch

        self._ensure_pipeline()

        # Extract capacity series
        if X.ndim == 2:
            capacity_series = X[:, self.capacity_col_index].astype(np.float64)
        else:
            capacity_series = X.astype(np.float64)

        total_len = len(capacity_series)

        # If we have stored context from fit(), use the full series
        # Otherwise, split based on context_ratio
        if self._context is not None and len(self._context) > 0:
            context_array = self._context.astype(np.float64)
            # The X passed to predict is the test portion
            pred_len = min(self.prediction_length, total_len)
        else:
            context_len = max(10, int(total_len * self.context_ratio))
            context_array = capacity_series[:context_len]
            pred_len = min(self.prediction_length, total_len - context_len)

        if pred_len <= 0:
            logger.warning("prediction_length <= 0, returning empty arrays.")
            return np.array([]), np.array([]), np.array([])

        # Build context tensor
        context_tensor = torch.tensor(context_array, dtype=torch.float64)

        # Run inference
        logger.info(
            f"Chronos inference: context_len={len(context_array)}, "
            f"prediction_length={pred_len}, num_samples={self.num_samples}"
        )
        forecast = self._pipeline.predict(
            context_tensor,
            prediction_length=pred_len,
            num_samples=self.num_samples,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        # forecast shape: [1, num_samples, pred_len]
        forecast_np = forecast[0].cpu().numpy()  # (num_samples, pred_len)

        # Compute quantiles
        alpha = 1.0 - self.confidence_level
        lower_q = alpha / 2.0
        upper_q = 1.0 - alpha / 2.0

        mean = np.median(forecast_np, axis=0)
        lower = np.quantile(forecast_np, lower_q, axis=0)
        upper = np.quantile(forecast_np, upper_q, axis=0)

        # Defensive: clean up GPU memory
        if self.device_map != "cpu":
            torch.cuda.empty_cache()

        return mean, lower, upper

    def predict_distribution(
        self, X: np.ndarray, n_samples: int = 100, **kwargs: Any
    ) -> np.ndarray:
        """
        Return full sample distribution from Chronos.

        Overrides the base class Gaussian approximation with the actual
        Chronos Monte Carlo samples.

        Returns:
            samples: (n_samples, prediction_length) array
        """
        import torch

        self._ensure_pipeline()

        if X.ndim == 2:
            capacity_series = X[:, self.capacity_col_index].astype(np.float64)
        else:
            capacity_series = X.astype(np.float64)

        if self._context is not None and len(self._context) > 0:
            context_array = self._context.astype(np.float64)
            pred_len = min(self.prediction_length, len(capacity_series))
        else:
            total_len = len(capacity_series)
            context_len = max(10, int(total_len * self.context_ratio))
            context_array = capacity_series[:context_len]
            pred_len = min(self.prediction_length, total_len - context_len)

        if pred_len <= 0:
            return np.array([]).reshape(n_samples, 0)

        context_tensor = torch.tensor(context_array, dtype=torch.float64)

        forecast = self._pipeline.predict(
            context_tensor,
            prediction_length=pred_len,
            num_samples=n_samples,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        samples = forecast[0].cpu().numpy()  # (n_samples, pred_len)

        if self.device_map != "cpu":
            torch.cuda.empty_cache()

        return samples

    def predict_single_battery(
        self,
        capacity_series: np.ndarray,
        context_length: int,
        prediction_length: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convenience method for single-battery zero-shot probing.

        This bypasses the fit/predict BatteryModel interface for direct
        evaluation of a raw capacity time-series.

        Args:
            capacity_series: Full capacity degradation curve (1D)
            context_length:  Number of historical points to feed as context
            prediction_length: Number of future points to forecast

        Returns:
            ground_truth: Actual values in the prediction window
            mean:         Median forecast
            lower:        Lower CI bound
            upper:        Upper CI bound
        """
        import torch

        self._ensure_pipeline()

        if context_length + prediction_length > len(capacity_series):
            prediction_length = len(capacity_series) - context_length

        if prediction_length <= 0:
            raise ValueError(
                f"Insufficient data: series_len={len(capacity_series)}, "
                f"context_len={context_length}, pred_len={prediction_length}"
            )

        context = capacity_series[:context_length]
        ground_truth = capacity_series[context_length:context_length + prediction_length]

        context_tensor = torch.tensor(context, dtype=torch.float64)

        forecast = self._pipeline.predict(
            context_tensor,
            prediction_length=prediction_length,
            num_samples=self.num_samples,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        forecast_np = forecast[0].cpu().numpy()

        alpha = 1.0 - self.confidence_level
        mean = np.median(forecast_np, axis=0)
        lower = np.quantile(forecast_np, alpha / 2.0, axis=0)
        upper = np.quantile(forecast_np, 1.0 - alpha / 2.0, axis=0)

        if self.device_map != "cpu":
            torch.cuda.empty_cache()

        return ground_truth, mean, lower, upper

    def save(self, path: str | Path) -> None:
        """Save model configuration (weights are from HuggingFace, not local)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "model_id": self.model_id,
            "num_samples": self.num_samples,
            "prediction_length": self.prediction_length,
            "context_ratio": self.context_ratio,
            "device_map": self.device_map,
            "torch_dtype_str": self.torch_dtype_str,
            "confidence_level": self.confidence_level,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "capacity_col_index": self.capacity_col_index,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"ChronosZeroShotModel config saved to {path}")

    def load(self, path: str | Path) -> "ChronosZeroShotModel":
        """Load model configuration and reinitialize pipeline."""
        path = Path(path)
        with open(path) as f:
            config = json.load(f)

        self.model_id = config["model_id"]
        self.num_samples = config["num_samples"]
        self.prediction_length = config["prediction_length"]
        self.context_ratio = config["context_ratio"]
        self.device_map = config["device_map"]
        self.torch_dtype_str = config["torch_dtype_str"]
        self.confidence_level = config["confidence_level"]
        self.temperature = config.get("temperature", 1.0)
        self.top_k = config.get("top_k", 50)
        self.top_p = config.get("top_p", 1.0)
        self.capacity_col_index = config.get("capacity_col_index", 0)

        # Force pipeline reload on next use
        self._pipeline = None
        logger.info(f"ChronosZeroShotModel config loaded from {path}")
        return self

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters for logging and provenance."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "num_samples": self.num_samples,
            "prediction_length": self.prediction_length,
            "context_ratio": self.context_ratio,
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype_str,
            "confidence_level": self.confidence_level,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
