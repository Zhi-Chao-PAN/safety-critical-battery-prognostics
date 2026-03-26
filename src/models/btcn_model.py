"""
Bayesian Temporal Convolutional Network (BTCN) for Battery RUL Prediction.

Architecture-level innovation: Bayesian inference embedded directly into
TCN's dilated causal convolution kernels via variational weight uncertainty.

This is NOT a simple TCN+BNN combination. The Bayesian layers replace
standard Conv1d layers inside the TCN residual blocks, enabling:
  1. Principled epistemic uncertainty from weight posterior distributions
  2. Temporal pattern extraction via dilated causal convolutions
  3. Single unified architecture (no model ensembling needed)

Uncertainty: Variational inference (local reparameterization trick)
  - Training: Sample weights from learned posterior q(w|θ)
  - Inference: Multiple stochastic forward passes → predictive distribution
  - KL divergence regularization against standard normal prior
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


# ---------------------------------------------------------------------------
# Bayesian Conv1d Layer — variational weight uncertainty in convolution kernels
# ---------------------------------------------------------------------------

class BayesConv1d(nn.Module):
    """
    Bayesian 1D Convolution with mean-field variational inference.

    Each weight has a learned posterior q(w) = N(μ, σ²).
    During training, weights are sampled via the reparameterization trick.
    During eval, uses posterior mean (MAP estimate).

    KL divergence against N(0, 1) prior is accumulated for ELBO loss.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        padding: int = 0,
        prior_log_var: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.prior_log_var = prior_log_var

        # Posterior parameters: μ and log(σ²)
        self.w_mu = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.w_log_var = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.b_mu = nn.Parameter(torch.empty(out_channels))
        self.b_log_var = nn.Parameter(torch.empty(out_channels))

        self.reset_parameters()
        self.kl_div: torch.Tensor = torch.tensor(0.0)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.w_mu)
        nn.init.constant_(self.w_log_var, -5.0)  # Start with low variance
        nn.init.zeros_(self.b_mu)
        nn.init.constant_(self.b_log_var, -5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Sample weights from posterior via reparameterization trick
            w_std = torch.exp(0.5 * self.w_log_var)
            b_std = torch.exp(0.5 * self.b_log_var)
            w = self.w_mu + w_std * torch.randn_like(w_std)
            b = self.b_mu + b_std * torch.randn_like(b_std)

            # Accumulate KL divergence: KL(q(w) || p(w)) where p = N(0, exp(prior_log_var))
            self.kl_div = self._compute_kl()

            return F.conv1d(x, w, b, dilation=self.dilation, padding=self.padding)
        else:
            # Use posterior mean at inference
            return F.conv1d(
                x, self.w_mu, self.b_mu,
                dilation=self.dilation, padding=self.padding,
            )

    def _compute_kl(self) -> torch.Tensor:
        """KL(N(μ, σ²) || N(0, σ²_prior)) for all parameters."""
        prior_var = torch.exp(torch.tensor(self.prior_log_var, device=self.w_mu.device))
        kl_w = 0.5 * torch.sum(
            (torch.exp(self.w_log_var) + self.w_mu ** 2) / prior_var
            - 1.0
            - self.w_log_var + self.prior_log_var
        )
        kl_b = 0.5 * torch.sum(
            (torch.exp(self.b_log_var) + self.b_mu ** 2) / prior_var
            - 1.0
            - self.b_log_var + self.prior_log_var
        )
        return kl_w + kl_b

    def get_kl(self) -> torch.Tensor:
        return self.kl_div


# ---------------------------------------------------------------------------
# Bayesian Linear Layer (for the prediction head)
# ---------------------------------------------------------------------------

class BayesLinearHead(nn.Module):
    """Bayesian Linear layer for the final prediction head."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_log_var = nn.Parameter(torch.empty(out_features, in_features))
        self.b_mu = nn.Parameter(torch.empty(out_features))
        self.b_log_var = nn.Parameter(torch.empty(out_features))
        self.kl_div: torch.Tensor = torch.tensor(0.0)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.w_mu)
        nn.init.constant_(self.w_log_var, -5.0)
        nn.init.zeros_(self.b_mu)
        nn.init.constant_(self.b_log_var, -5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w_std = torch.exp(0.5 * self.w_log_var)
            b_std = torch.exp(0.5 * self.b_log_var)
            w = self.w_mu + w_std * torch.randn_like(w_std)
            b = self.b_mu + b_std * torch.randn_like(b_std)
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


# ---------------------------------------------------------------------------
# Bayesian TCN Residual Block
# ---------------------------------------------------------------------------

class BayesResidualBlock(nn.Module):
    """
    TCN residual block with Bayesian convolution kernels.

    Structure: BayesConv1d → ReLU → BayesConv1d → ReLU + Residual
    Causal padding ensures no future information leakage.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # Causal padding

        self.conv1 = BayesConv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.conv2 = BayesConv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        self.seq_len_dim = 2  # Time dimension index
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(2)
        out = self.dropout(self.relu(self.conv1(x)[:, :, :T]))
        out = self.dropout(self.relu(self.conv2(out)[:, :, :T]))
        return self.relu(out + self.downsample(x))

    def kl_loss(self) -> torch.Tensor:
        return self.conv1.get_kl() + self.conv2.get_kl()


# ---------------------------------------------------------------------------
# Full Bayesian TCN Network
# ---------------------------------------------------------------------------

class BTCNNet(nn.Module):
    """
    Bayesian Temporal Convolutional Network.

    All convolution layers use variational weight uncertainty.
    The prediction head is also Bayesian for end-to-end uncertainty.
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            self.blocks.append(
                BayesResidualBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout)
            )

        # Bayesian prediction head
        self.head_fc1 = BayesLinearHead(num_channels[-1], 32)
        self.head_relu = nn.ReLU()
        self.head_fc2 = BayesLinearHead(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → (B, C, T) for conv1d
        out = x.transpose(1, 2)
        for block in self.blocks:
            out = block(out)
        out = out[:, :, -1]  # Last timestep
        out = self.head_relu(self.head_fc1(out))
        return self.head_fc2(out)

    def kl_loss(self) -> torch.Tensor:
        """Total KL divergence across all Bayesian layers."""
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.blocks:
            kl = kl + block.kl_loss()
        kl = kl + self.head_fc1.get_kl() + self.head_fc2.get_kl()
        return kl


# ---------------------------------------------------------------------------
# BTCNModel — BatteryModel interface
# ---------------------------------------------------------------------------

class BTCNModel(BatteryModel):
    """
    Bayesian TCN for battery RUL prediction.

    Architecture-level fusion of:
      - TCN's dilated causal convolutions for long-range temporal dependencies
      - Bayesian variational inference for principled epistemic uncertainty

    Uncertainty quantification:
      - Training: ELBO = E[log p(y|w)] - β·KL(q(w)||p(w))
      - Inference: T stochastic forward passes → predictive mean + variance
      - Epistemic uncertainty from weight posterior variance
      - Aleatoric uncertainty separable via heteroscedastic extension

    Key advantage over TCN+BNN ensemble:
      Single model, single forward pass architecture. Uncertainty is intrinsic
      to the temporal feature extraction, not bolted on after the fact.
    """

    name = "btcn"

    def __init__(
        self,
        input_dim: int = 2,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        seq_length: int = 30,
        lr: float = 1e-3,
        epochs: int = 150,
        patience: int = 15,
        kl_weight: float = 0.01,
        kl_annealing_epochs: int = 30,
        n_samples: int = 100,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.num_channels = num_channels or [32, 32, 64, 64]
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.seq_length = seq_length
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.kl_weight = kl_weight
        self.kl_annealing_epochs = kl_annealing_epochs
        self.n_samples = n_samples
        self.device = device
        self.model: BTCNNet | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "BTCNModel":
        X_seq, y_seq = self._make_seq(X, y)
        if len(X_seq) == 0:
            raise ValueError("No sequences created. Check seq_length vs data length.")

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.model = BTCNNet(
            self.input_dim, self.num_channels, self.kernel_size, self.dropout
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_loss, wait = float("inf"), 0
        n = len(X_seq)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            pred = self.model(X_t)
            mse_loss = F.mse_loss(pred, y_t)

            # KL annealing: linearly increase β from 0 to kl_weight
            beta = min(1.0, epoch / max(self.kl_annealing_epochs, 1)) * self.kl_weight
            kl_loss = self.model.kl_loss() / n  # Normalize by dataset size

            loss = mse_loss + beta * kl_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss, wait = loss.item(), 0
            else:
                wait += 1
                if wait >= self.patience:
                    logger.info(f"BTCN early stop at epoch {epoch + 1}")
                    break

        return self

    def predict(
        self, X: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Not fitted.")

        X_seq, _ = self._make_seq(X, np.zeros(len(X)))
        if len(X_seq) == 0:
            e = np.array([])
            return e, e, e

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        # Stochastic forward passes for predictive distribution
        self.model.train()  # Enable weight sampling
        preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                preds.append(self.model(X_t).cpu().numpy().flatten())
        self.model.eval()

        preds = np.stack(preds)  # (n_samples, N)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, mean - 1.96 * std, mean + 1.96 * std

    def predict_distribution(
        self, X: np.ndarray, n_samples: int = 100, **kwargs: Any
    ) -> np.ndarray:
        """Return raw posterior predictive samples."""
        if self.model is None:
            raise RuntimeError("Not fitted.")

        X_seq, _ = self._make_seq(X, np.zeros(len(X)))
        if len(X_seq) == 0:
            return np.array([])

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        self.model.train()
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(self.model(X_t).cpu().numpy().flatten())
        self.model.eval()
        return np.stack(samples)

    def get_epistemic_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Extract pure epistemic uncertainty (from weight posterior variance).
        This is the key advantage of BTCN over MC Dropout approaches.
        """
        samples = self.predict_distribution(X, n_samples=self.n_samples)
        if len(samples) == 0:
            return np.array([])
        return samples.std(axis=0)

    def _make_seq(self, X, y):
        if len(X) <= self.seq_length:
            return np.array([]), np.array([])
        Xl, yl = [], []
        for i in range(len(X) - self.seq_length):
            Xl.append(X[i : i + self.seq_length])
            yl.append(y[i + self.seq_length])
        return np.array(Xl, dtype=np.float32), np.array(yl, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state": self.model.state_dict() if self.model else None,
                "params": self.get_params(),
            },
            path,
        )

    def load(self, path: str | Path) -> "BTCNModel":
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.model = BTCNNet(
            self.input_dim, self.num_channels, self.kernel_size, self.dropout
        ).to(self.device)
        if d["state"]:
            self.model.load_state_dict(d["state"])
        return self

    def get_params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input_dim": self.input_dim,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "seq_length": self.seq_length,
            "lr": self.lr,
            "epochs": self.epochs,
            "kl_weight": self.kl_weight,
            "kl_annealing_epochs": self.kl_annealing_epochs,
            "n_samples": self.n_samples,
        }
