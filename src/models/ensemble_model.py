"""
Deep Ensembles - Train N independent models, aggregate for uncertainty.
Wraps any BatteryModel subclass.
"""

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import BatteryModel

logger = logging.getLogger(__name__)


class DeepEnsemble(BatteryModel):
    """Ensemble of N independently trained models."""

    name = "deep_ensemble"

    def __init__(self, base_model: BatteryModel, n_members: int = 5, seeds: list[int] | None = None):
        self.base_model = base_model
        self.n_members = n_members
        self.seeds = seeds or list(range(42, 42 + n_members))
        self.members: list[BatteryModel] = []

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "DeepEnsemble":
        self.members = []
        for i, seed in enumerate(self.seeds):
            logger.info(f"Training ensemble member {i + 1}/{self.n_members} (seed={seed})")
            import torch
            torch.manual_seed(seed)
            np.random.seed(seed)
            member = copy.deepcopy(self.base_model)
            member.fit(X, y, **kwargs)
            self.members.append(member)
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.members:
            raise RuntimeError("Ensemble not fitted.")

        all_means = []
        for member in self.members:
            m, _, _ = member.predict(X, **kwargs)
            if len(m) > 0:
                all_means.append(m)

        if not all_means:
            e = np.array([])
            return e, e, e

        stacked = np.stack(all_means)  # (N_members, N_samples)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        return mean, mean - 1.96 * std, mean + 1.96 * std

    def predict_distribution(self, X: np.ndarray, n_samples: int = 100, **kwargs: Any) -> np.ndarray:
        """Sample from ensemble: Pick random member, use its MC samples."""
        if not self.members:
            raise RuntimeError("Not fitted.")
        rng = np.random.default_rng(42)
        all_samples = []
        for _ in range(n_samples):
            member = self.members[rng.integers(0, len(self.members))]
            m, lo, hi = member.predict(X, **kwargs)
            std = (hi - lo) / 3.92
            std = np.maximum(std, 1e-6)
            sample = rng.normal(loc=m, scale=std)
            all_samples.append(sample)
        return np.stack(all_samples)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for i, member in enumerate(self.members):
            member.save(path / f"member_{i}.pt")

    def load(self, path: str | Path) -> "DeepEnsemble":
        path = Path(path)
        self.members = []
        for i in range(self.n_members):
            member = copy.deepcopy(self.base_model)
            member.load(path / f"member_{i}.pt")
            self.members.append(member)
        return self

    def get_params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_members": self.n_members,
            "base_model": self.base_model.get_params(),
        }
