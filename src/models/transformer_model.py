"""
Transformer Encoder for Battery RUL Prediction.
Positional encoding + multi-head self-attention + CLS token.
"""

from pathlib import Path
from typing import Any
import math

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BatteryModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2]) if d_model % 2 else torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerNet(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 32),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_enc(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)
        x = self.encoder(x)
        return self.head(x[:, 0])  # CLS token output


class TransformerModel(BatteryModel):
    """Transformer encoder with MC Dropout uncertainty."""

    name = "transformer"

    def __init__(self, input_dim: int = 2, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.2, seq_length: int = 30,
                 lr: float = 1e-3, epochs: int = 100, patience: int = 10,
                 mc_samples: int = 100, device: str = "cpu"):
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_length = seq_length
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.mc_samples = mc_samples
        self.device = device
        self.model: TransformerNet | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "TransformerModel":
        X_seq, y_seq = self._make_seq(X, y)
        if len(X_seq) == 0:
            raise ValueError("No sequences.")
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.model = TransformerNet(self.input_dim, self.d_model, self.nhead,
                                    self.num_layers, self.dropout).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
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

    def load(self, path: str | Path) -> "TransformerModel":
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.model = TransformerNet(self.input_dim, self.d_model, self.nhead,
                                    self.num_layers, self.dropout).to(self.device)
        if d["state"]:
            self.model.load_state_dict(d["state"])
        return self

    def get_params(self) -> dict[str, Any]:
        return {"name": self.name, "input_dim": self.input_dim, "d_model": self.d_model,
                "nhead": self.nhead, "num_layers": self.num_layers, "dropout": self.dropout,
                "seq_length": self.seq_length, "lr": self.lr, "epochs": self.epochs}
