"""
1D Convolutional Neural Network for Battery RUL Prediction.
Simple but effective baseline with global average pooling.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BatteryModel


class CNN1DNet(nn.Module):
    def __init__(self, input_dim: int, channels: list[int] = None, dropout: float = 0.2):
        super().__init__()
        channels = channels or [32, 64, 64]
        layers = []
        in_ch = input_dim
        for out_ch in channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(channels[-1], 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        out = self.conv(x.transpose(1, 2))  # (B, ch, T)
        out = out.mean(dim=2)  # Global average pooling
        return self.head(out)


class CNN1DModel(BatteryModel):
    """1D-CNN with MC Dropout uncertainty."""

    name = "cnn1d"

    def __init__(self, input_dim: int = 2, channels: list[int] = None,
                 dropout: float = 0.2, seq_length: int = 30, lr: float = 1e-3,
                 epochs: int = 100, patience: int = 10, mc_samples: int = 100, device: str = "cpu"):
        self.input_dim = input_dim
        self.channels = channels or [32, 64, 64]
        self.dropout = dropout
        self.seq_length = seq_length
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.mc_samples = mc_samples
        self.device = device
        self.model: CNN1DNet | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "CNN1DModel":
        X_seq, y_seq = self._make_seq(X, y)
        if len(X_seq) == 0:
            raise ValueError("No sequences.")
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.model = CNN1DNet(self.input_dim, self.channels, self.dropout).to(self.device)
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

    def load(self, path: str | Path) -> "CNN1DModel":
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.model = CNN1DNet(self.input_dim, self.channels, self.dropout).to(self.device)
        if d["state"]:
            self.model.load_state_dict(d["state"])
        return self

    def get_params(self) -> dict[str, Any]:
        return {"name": self.name, "input_dim": self.input_dim, "channels": self.channels,
                "dropout": self.dropout, "seq_length": self.seq_length,
                "lr": self.lr, "epochs": self.epochs, "mc_samples": self.mc_samples}
