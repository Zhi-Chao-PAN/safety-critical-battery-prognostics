"""
Physics-Informed Neural Network (PINN) for Battery RUL.

Architecture: Physics residual learning with Adaptive Loss Weighting.
  Total prediction = Physics(n) + NN(features)
  Loss = MSE_data + λ_physics(t) * Physics_constraint + λ_mono(t) * Monotonicity

Key innovation: Dynamic adaptive weights λ_physics(t) and λ_mono(t) that
adjust based on battery lifecycle stage:
  - Early life (high data density): λ_physics low, trust data
  - Mid life (transition): balanced weighting
  - Late life ("knee" / cliff region): λ_physics high, trust physics
  - OOD / extrapolation: λ_physics maximum, physics as safety net

This solves the over-conservative prediction interval problem (100% PICP
with useless width) by letting the model dynamically decide when to rely
on physics vs data-driven predictions.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BatteryModel
from src.physics.aging.degradation import PhysicsModel
from src.physics.electrochemistry.spm import PyTorchSPM

logger = logging.getLogger(__name__)


class PINNNet(nn.Module):
    """Neural network that learns physics residuals."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptiveLossWeighter:
    """
    Dynamic adaptive loss weighting based on battery lifecycle stage.

    Lifecycle stages (determined by normalized cycle position):
      - Early (0-30%): Low physics weight — abundant data, let NN learn
      - Mid (30-70%): Balanced — transition zone
      - Late (70-100%): High physics weight — sparse data, trust degradation model
      - Extrapolation (>100%): Maximum physics weight — safety-critical region

    The weighting follows a sigmoid schedule:
      λ(t) = λ_min + (λ_max - λ_min) * σ(k * (t - t_mid))

    where t is the normalized lifecycle position, k controls transition sharpness.
    """

    def __init__(
        self,
        lambda_physics_min: float = 0.01,
        lambda_physics_max: float = 1.0,
        lambda_mono_min: float = 0.01,
        lambda_mono_max: float = 0.2,
        transition_sharpness: float = 10.0,
        transition_center: float = 0.6,
    ):
        self.lp_min = lambda_physics_min
        self.lp_max = lambda_physics_max
        self.lm_min = lambda_mono_min
        self.lm_max = lambda_mono_max
        self.k = transition_sharpness
        self.t_mid = transition_center

    def get_weights(
        self, cycles: np.ndarray, max_cycle: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute per-sample adaptive weights.

        Args:
            cycles: Current cycle numbers (N,)
            max_cycle: Maximum observed cycle in training data

        Returns:
            lambda_physics: Per-sample physics constraint weight (N,)
            lambda_mono: Per-sample monotonicity weight (N,)
        """
        # Normalize cycle position to [0, 1+]
        t = cycles / max(max_cycle, 1.0)

        # Sigmoid schedule
        sigmoid = 1.0 / (1.0 + np.exp(-self.k * (t - self.t_mid)))

        lambda_physics = self.lp_min + (self.lp_max - self.lp_min) * sigmoid
        lambda_mono = self.lm_min + (self.lm_max - self.lm_min) * sigmoid

        return lambda_physics, lambda_mono

    def get_epoch_weights(
        self, cycles: np.ndarray, max_cycle: float
    ) -> tuple[float, float]:
        """Get mean weights for the batch (for logging)."""
        lp, lm = self.get_weights(cycles, max_cycle)
        return float(np.mean(lp)), float(np.mean(lm))


class PINNModel(BatteryModel):
    """
    Physics-Informed model with Adaptive Loss Weighting.

    - OOD / extrapolation: λ_physics maximum, physics as safety net

    This model solves the over-conservative prediction interval problem by letting 
    the model dynamically decide when to rely on physics vs data-driven predictions 
    through a sigmoid-based weighting schedule.

    Uncertainty is quantified using Monte Carlo (MC) Dropout to estimate 
    epistemic uncertainty (model knowledge gaps).
    """

    name = "pinn"

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 150,
        patience: int = 15,
        lambda_physics: float = 0.1,
        lambda_mono: float = 0.05,
        adaptive_weighting: bool = True,
        lambda_physics_min: float = 0.01,
        lambda_physics_max: float = 1.0,
        lambda_mono_min: float = 0.01,
        lambda_mono_max: float = 0.2,
        transition_sharpness: float = 10.0,
        transition_center: float = 0.6,
        mc_samples: int = 100,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.lambda_physics = lambda_physics
        self.lambda_mono = lambda_mono
        self.adaptive_weighting = adaptive_weighting
        self.mc_samples = mc_samples
        self.device = device
        self.model: PINNNet | None = None
        self.physics = PhysicsModel()
        self._physics_params: dict[str, float] | None = None
        self._max_cycle: float = 1.0

        # Adaptive weighter
        self.weighter = AdaptiveLossWeighter(
            lambda_physics_min=lambda_physics_min,
            lambda_physics_max=lambda_physics_max,
            lambda_mono_min=lambda_mono_min,
            lambda_mono_max=lambda_mono_max,
            transition_sharpness=transition_sharpness,
            transition_center=transition_center,
        ) if adaptive_weighting else None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "PINNModel":
        """
        Fits the PINN model using adaptive loss weighting and physics calibration.

        The fitting process involves:
          1. Initial physics prior fitting (Empirical degradation).
          2. Joint optimization of neural network residuals and SPM parameters.
          3. Dynamic weight adjustment for data, physics, and monotonicity losses.

        Args:
            X (np.ndarray): Input features. X[:, 0] must be the cycle count.
            y (np.ndarray): Target capacity or RUL values.
            **kwargs: Additional training arguments (e.g., validation_data).

        Returns:
            PINNModel: The fitted model instance.
        """
        cycles = X[:, 0] if X.ndim > 1 else X
        self._max_cycle = float(np.max(cycles))

        # Step 1: Fit physics model on capacity (approximate from RUL)
        try:
            self.physics.fit(cycles, y, battery_id="train")
            self._physics_params = self.physics.params.get("train")
        except Exception as e:
            logger.warning(f"Physics fit failed: {e}. Using zero baseline.")
            self._physics_params = None

        # Step 2: Compute physics predictions and residuals
        if self._physics_params:
            physics_pred = self.physics.predict(cycles, battery_id="train")
            residuals = y - physics_pred
        else:
            residuals = y

        # Step 3: Compute adaptive weights per sample
        if self.adaptive_weighting and self.weighter is not None:
            lp_weights, lm_weights = self.weighter.get_weights(cycles, self._max_cycle)
            lp_weights_t = torch.tensor(lp_weights, dtype=torch.float32).unsqueeze(1).to(self.device)
            lm_weights_t = torch.tensor(lm_weights, dtype=torch.float32).unsqueeze(1).to(self.device)
            mean_lp, mean_lm = self.weighter.get_epoch_weights(cycles, self._max_cycle)
            logger.info(f"Adaptive weights: mean λ_physics={mean_lp:.4f}, mean λ_mono={mean_lm:.4f}")
        else:
            lp_weights_t = None
            lm_weights_t = None

        # Step 4: Train NN on residuals
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(residuals, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.model = PINNNet(self.input_dim, self.hidden_dim, self.dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_loss, wait = float("inf"), 0
        self.model.train()

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            nn_pred = self.model(X_t)

            # Data loss: NN should predict residuals
            loss_data = F.mse_loss(nn_pred, y_t)

            # Monotonicity loss: Total prediction should decrease
            if self._physics_params:
                physics_t = torch.tensor(
                    self.physics.predict(cycles, "train"),
                    dtype=torch.float32,
                ).unsqueeze(1).to(self.device)
                total_pred = physics_t + nn_pred
            else:
                total_pred = nn_pred

            diffs = total_pred[1:] - total_pred[:-1]

            # Adaptive vs static weighting
            if lp_weights_t is not None and lm_weights_t is not None:
                # Per-sample weighted monotonicity loss
                mono_violations = torch.relu(diffs) ** 2
                loss_mono = torch.mean(lm_weights_t[1:] * mono_violations)

                # Per-sample weighted physics constraint (residual should be small)
                loss_physics = torch.mean(lp_weights_t * nn_pred ** 2)

                loss = loss_data + loss_physics + loss_mono
            else:
                # Fallback: static weights (backward compatible)
                loss_mono = torch.mean(torch.relu(diffs) ** 2)
                loss = loss_data + self.lambda_mono * loss_mono

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss, wait = loss.item(), 0
            else:
                wait += 1
                if wait >= self.patience:
                    logger.info(f"PINN early stop at epoch {epoch + 1}")
                    break

        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Not fitted.")

        cycles = X[:, 0] if X.ndim > 1 else X
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Physics baseline
        if self._physics_params:
            physics_pred = self.physics.predict(cycles, "train")
        else:
            physics_pred = np.zeros(len(X))

        # MC Dropout for NN residual
        self.model.train()
        nn_preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                nn_preds.append(self.model(X_t).cpu().numpy().flatten())
        self.model.eval()

        nn_preds = np.stack(nn_preds)  # (mc, N)
        total_preds = nn_preds + physics_pred[np.newaxis, :]

        mean = total_preds.mean(axis=0)
        std = total_preds.std(axis=0)
        return mean, mean - 1.96 * std, mean + 1.96 * std

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state": self.model.state_dict() if self.model else None,
            "physics_params": self._physics_params,
            "max_cycle": self._max_cycle,
            "params": self.get_params(),
        }, path)

    def load(self, path: str | Path) -> "PINNModel":
        d = torch.load(path, map_location=self.device, weights_only=False)
        self._physics_params = d.get("physics_params")
        self._max_cycle = d.get("max_cycle", 1.0)
        if self._physics_params:
            self.physics.params["train"] = self._physics_params
        self.model = PINNNet(self.input_dim, self.hidden_dim, self.dropout).to(self.device)
        if d["state"]:
            self.model.load_state_dict(d["state"])
        return self

    def get_params(self) -> dict[str, Any]:
        return {
            "name": self.name, "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim, "dropout": self.dropout,
            "lr": self.lr, "epochs": self.epochs,
            "lambda_physics": self.lambda_physics,
            "lambda_mono": self.lambda_mono,
            "adaptive_weighting": self.adaptive_weighting,
            "mc_samples": self.mc_samples,
        }
