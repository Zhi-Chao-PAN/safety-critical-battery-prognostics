"""
Bayesian Neural Network for Battery RUL Prediction.
Uses variational inference (Flipout layers) for principled uncertainty.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BatteryModel

logger = logging.getLogger(__name__)


class BayesLinear(nn.Module):
    """
    Bayesian Linear layer with weight uncertainty (mean-field VI).
    Implements local reparameterization trick.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters (mean + log variance)
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_log_var = nn.Parameter(torch.empty(out_features, in_features))
        self.b_mu = nn.Parameter(torch.empty(out_features))
        self.b_log_var = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()
        self.kl_div = 0.0

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.w_mu)
        nn.init.constant_(self.w_log_var, -5.0)
        nn.init.zeros_(self.b_mu)
        nn.init.constant_(self.b_log_var, -5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w_std = torch.exp(0.5 * self.w_log_var)
            b_std = torch.exp(0.5 * self.b_log_var)
            w_eps = torch.randn_like(w_std)
            b_eps = torch.randn_like(b_std)
            w = self.w_mu + w_std * w_eps
            b = self.b_mu + b_std * b_eps

            # KL divergence vs standard normal prior
            self.kl_div = 0.5 * torch.sum(
                torch.exp(self.w_log_var) + self.w_mu ** 2 - 1 - self.w_log_var
            ) + 0.5 * torch.sum(
                torch.exp(self.b_log_var) + self.b_mu ** 2 - 1 - self.b_log_var
            )
            return F.linear(x, w, b)
        else:
            return F.linear(x, self.w_mu, self.b_mu)

    def get_kl(self) -> torch.Tensor:
        return self.kl_div


class BayesNet(nn.Module):
    """Bayesian Neural Network with variational layers."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = BayesLinear(input_dim, hidden_dim)
        self.fc2 = BayesLinear(hidden_dim, hidden_dim)
        self.fc3 = BayesLinear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def kl_loss(self) -> torch.Tensor:
        return self.fc1.get_kl() + self.fc2.get_kl() + self.fc3.get_kl()


class BayesianNNModel(BatteryModel):
    """
    Bayesian NN with variational inference.
    True epistemic uncertainty from weight distributions.
    """

    name = "bayesian_nn"

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        epochs: int = 200,
        patience: int = 20,
        kl_weight: float = 0.01,
        n_samples: int = 100,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.kl_weight = kl_weight
        self.n_samples = n_samples
        self.device = device
        self.model: BayesNet | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def _normalize_x(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self._x_mean = X.mean(axis=0)
            self._x_std = X.std(axis=0) + 1e-8
        return (X - self._x_mean) / self._x_std

    def _normalize_y(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self._y_mean = float(y.mean())
            self._y_std = float(y.std()) + 1e-8
        return (y - self._y_mean) / self._y_std

    def _denormalize_y(self, y: np.ndarray) -> np.ndarray:
        return y * self._y_std + self._y_mean

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "BayesianNNModel":
        X_norm = self._normalize_x(X, fit=True)
        y_norm = self._normalize_y(y, fit=True)

        X_t = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.model = BayesNet(self.input_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_loss, wait = float("inf"), 0
        n = len(X)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X_t)
            mse_loss = F.mse_loss(pred, y_t)
            kl_loss = self.model.kl_loss() / n  # Normalize by dataset size
            loss = mse_loss + self.kl_weight * kl_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss, wait = loss.item(), 0
            else:
                wait += 1
                if wait >= self.patience:
                    logger.info(f"BNN early stop at epoch {epoch + 1}")
                    break

        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Not fitted.")

        X_norm = self._normalize_x(X)
        X_t = torch.tensor(X_norm, dtype=torch.float32).to(self.device)

        self.model.train()  # Enable stochastic forward passes
        preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                preds.append(self.model(X_t).cpu().numpy().flatten())
        self.model.eval()

        preds = np.stack(preds)
        # Denormalize
        preds = self._denormalize_y(preds)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, mean - 1.96 * std, mean + 1.96 * std

    def predict_distribution(self, X: np.ndarray, n_samples: int = 100, **kwargs: Any) -> np.ndarray:
        """Return raw samples from posterior predictive."""
        if self.model is None:
            raise RuntimeError("Not fitted.")
        X_norm = self._normalize_x(X)
        X_t = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        self.model.train()
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(self.model(X_t).cpu().numpy().flatten())
        self.model.eval()
        return self._denormalize_y(np.stack(samples))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.model.state_dict() if self.model else None,
                     "params": self.get_params(),
                     "x_mean": self._x_mean, "x_std": self._x_std,
                     "y_mean": self._y_mean, "y_std": self._y_std}, path)

    def load(self, path: str | Path) -> "BayesianNNModel":
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.model = BayesNet(self.input_dim, self.hidden_dim).to(self.device)
        if d["state"]:
            self.model.load_state_dict(d["state"])
        self._x_mean = d.get("x_mean")
        self._x_std = d.get("x_std")
        self._y_mean = d.get("y_mean", 0.0)
        self._y_std = d.get("y_std", 1.0)
        return self

    def get_params(self) -> dict[str, Any]:
        return {"name": self.name, "input_dim": self.input_dim, "hidden_dim": self.hidden_dim,
                "lr": self.lr, "epochs": self.epochs, "kl_weight": self.kl_weight,
                "n_samples": self.n_samples}
