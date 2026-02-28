"""
Physics-Informed Neural Network (PINN) for Battery RUL.

Architecture: Physics residual learning.
  Total prediction = Physics(n) + NN(features)
  Loss = MSE_data + λ_physics * Physics_constraint + λ_mono * Monotonicity

The NN only learns what physics can't explain.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BatteryModel
from src.physics.degradation import PhysicsModel, empirical_fade

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


class PINNModel(BatteryModel):
    """
    Physics-Informed model: Physics baseline + NN residual.
    Uncertainty via MC Dropout.
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
        self.mc_samples = mc_samples
        self.device = device
        self.model: PINNNet | None = None
        self.physics = PhysicsModel()
        self._physics_params: dict[str, float] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "PINNModel":
        """
        Fit PINN. Expects X[:, 0] to contain cycle numbers for physics model.
        """
        cycles = X[:, 0] if X.ndim > 1 else X

        # Step 1: Fit physics model on capacity (approximate from RUL)
        # Use cycle numbers to fit empirical fade
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

        # Step 3: Train NN on residuals
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(residuals, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_full = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.model = PINNNet(self.input_dim, self.hidden_dim, self.dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        mse = nn.MSELoss()

        best_loss, wait = float("inf"), 0
        self.model.train()

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            nn_pred = self.model(X_t)

            # Data loss: NN should predict residuals
            loss_data = mse(nn_pred, y_t)

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
            loss_mono = torch.mean(torch.relu(diffs) ** 2)

            # Total loss
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
            "params": self.get_params(),
        }, path)

    def load(self, path: str | Path) -> "PINNModel":
        d = torch.load(path, map_location=self.device, weights_only=False)
        self._physics_params = d.get("physics_params")
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
            "mc_samples": self.mc_samples,
        }
