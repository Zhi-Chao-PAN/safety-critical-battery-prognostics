"""
GRU Model - Lighter alternative to LSTM with same interface.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BatteryModel
from src.models.lstm_model import TemporalAttention


class GRUNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.attention = TemporalAttention(hidden_dim * 2)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = self.layer_norm(out)
        ctx, self._attn_w = self.attention(out)
        return self.head(ctx)


class GRUModel(BatteryModel):
    """GRU with attention + MC Dropout uncertainty."""

    name = "gru_attention"

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, seq_length: int = 30, lr: float = 1e-3,
                 epochs: int = 100, patience: int = 10, mc_samples: int = 100, device: str = "cpu"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_length = seq_length
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.mc_samples = mc_samples
        self.device = device
        self.model: GRUNet | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "GRUModel":
        X_seq, y_seq = self._make_seq(X, y)
        if len(X_seq) == 0:
            raise ValueError("No sequences created.")
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.model = GRUNet(self.input_dim, self.hidden_dim, self.num_layers, self.dropout).to(self.device)
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

    def load(self, path: str | Path) -> "GRUModel":
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.model = GRUNet(self.input_dim, self.hidden_dim, self.num_layers, self.dropout).to(self.device)
        if d["state"]:
            self.model.load_state_dict(d["state"])
        return self

    def get_params(self) -> dict[str, Any]:
        return {"name": self.name, "input_dim": self.input_dim, "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers, "dropout": self.dropout, "seq_length": self.seq_length,
                "lr": self.lr, "epochs": self.epochs, "mc_samples": self.mc_samples}
