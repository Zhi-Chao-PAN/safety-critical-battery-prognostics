"""
Abstract base class for all battery RUL prediction models.

All models implement the same interface for fair comparison:
  fit(X, y, **kwargs) -> self
  predict(X, **kwargs) -> (mean, lower, upper)
  save(path) / load(path)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np


class BatteryModel(ABC):
    """Unified interface for all battery RUL models."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "BatteryModel":
        """Train the model."""
        ...

    @abstractmethod
    def predict(
        self, X: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty.

        Returns:
            mean: Point predictions (N,)
            lower: Lower bound of 95% CI (N,)
            upper: Upper bound of 95% CI (N,)
        """
        ...

    def predict_distribution(
        self, X: np.ndarray, n_samples: int = 100, **kwargs: Any
    ) -> np.ndarray:
        """
        Generate full predictive distribution samples.

        Returns:
            samples: (n_samples, N) array of predictions

        Default: Use mean +/- Gaussian noise from CI width.
        Override in probabilistic models for proper sampling.
        """
        mean, lower, upper = self.predict(X, **kwargs)
        std = (upper - lower) / 3.92  # 95% CI -> std
        std = np.maximum(std, 1e-6)
        rng = np.random.default_rng(42)
        return rng.normal(loc=mean, scale=std, size=(n_samples, len(mean)))

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> "BatteryModel":
        """Load model from disk."""
        ...

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters for logging."""
        return {"name": self.name}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
