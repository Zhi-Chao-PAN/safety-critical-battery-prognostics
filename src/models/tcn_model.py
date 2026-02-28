"""
Temporal Convolutional Network (TCN) for Battery RUL Prediction.
Dilated causal convolutions with residual blocks.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BatteryModel


class ResidualBlock(nn.Module):
    """TCN residual block: dilated causal conv + weight norm + dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.conv1(x)[:, :, : x.size(2)]))
        out = self.dropout(self.relu(self.conv2(out)[:, :, : x.size(2)]))
        return self.relu(out + self.downsample(x))


class TCNNet(nn.Module):
    """Temporal Convolutional Network."""

    def __init__(self, input_dim: int, num_channels: list[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            layers.append(ResidualBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(num_channels[-1], 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T) for conv1d
        out = self.network(x.transpose(1, 2))  # (B, channels, T)
        out = out[:, :, -1]  # Last timestep
        return self.head(out)


class TCNModel(BatteryModel):
    """TCN with MC Dropout uncertainty."""

    name = "tcn"

    def __init__(self, input_dim: int = 2, num_channels: list[int] | None = None,
                 kernel_size: int = 3, dropout: float = 0.2, seq_length: int = 30,
                 lr: float = 1e-3, epochs: int = 100, patience: int = 10,
                 mc_samples: int = 100, device: str = "cpu"):
        self.input_dim = input_dim
        self.num_channels = num_channels or [32, 32, 64, 64]
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.seq_length = seq_length
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.mc_samples = mc_samples
        self.device = device
        self.model: TCNNet | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "TCNModel":
        X_seq, y_seq = self._make_seq(X, y)
        if len(X_seq) == 0:
            raise ValueError("No sequences created.")
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.model = TCNNet(self.input_dim, self.num_channels, self.kernel_size, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        crit = nn.MSELoss()
        best, wait = float("inf"), 0

        self.model.train()
        for ep in range(self.epochs):
            opt.zero_grad()
            loss = crit(self.model(X_t), y_t)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            sched.step()
            if loss.item() < best:
                best, wait = loss.item(), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Not fitted.")
        X_seq, _ = self._make_seq(X, np.zeros(len(X)))
        if len(X_seq) == 0:
            e = np.array([])
            return e, e, e
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        self.model.train()
        preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                preds.append(self.model(X_t).cpu().numpy().flatten())
        self.model.eval()
        preds = np.stack(preds)
        m, s = preds.mean(0), preds.std(0)
        return m, m - 1.96 * s, m + 1.96 * s

    def _make_seq(self, X, y):
        if len(X) <= self.seq_length:
            return np.array([]), np.array([])
        Xl, yl = [], []
        for i in range(len(X) - self.seq_length):
            Xl.append(X[i:i + self.seq_length])
            yl.append(y[i + self.seq_length])
        return np.array(Xl, dtype=np.float32), np.array(yl, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.model.state_dict() if self.model else None, "params": self.get_params()}, path)

    def load(self, path: str | Path) -> "TCNModel":
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.model = TCNNet(self.input_dim, self.num_channels, self.kernel_size, self.dropout).to(self.device)
        if d["state"]:
            self.model.load_state_dict(d["state"])
        return self

    def get_params(self) -> dict[str, Any]:
        return {"name": self.name, "input_dim": self.input_dim, "num_channels": self.num_channels,
                "kernel_size": self.kernel_size, "dropout": self.dropout, "seq_length": self.seq_length,
                "lr": self.lr, "epochs": self.epochs, "mc_samples": self.mc_samples}
