"""
Upgraded LSTM with Bidirectional + Temporal Attention + Residual connections.
Implements BatteryModel interface.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BatteryModel

logger = logging.getLogger(__name__)


class TemporalAttention(nn.Module):
    """Attention over LSTM hidden states to weight important timesteps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: (B, T, H)
        Returns:
            context: (B, H) weighted sum
            weights: (B, T) attention weights
        """
        scores = self.attn(lstm_out).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=-1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, H)
        return context, weights


class LSTMNet(nn.Module):
    """Bidirectional LSTM + Attention + Residual."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)
        self.attention = TemporalAttention(hidden_dim * self.num_directions)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)  # (B, T, H*dirs)
        lstm_out = self.layer_norm(lstm_out)
        context, self._attn_weights = self.attention(lstm_out)  # (B, H*dirs)
        return self.head(context)  # (B, 1)

    @property
    def attention_weights(self) -> torch.Tensor | None:
        return getattr(self, "_attn_weights", None)


class LSTMModel(BatteryModel):
    """LSTM model with MC Dropout uncertainty."""

    name = "lstm_attention"

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        seq_length: int = 30,
        lr: float = 1e-3,
        epochs: int = 100,
        patience: int = 10,
        mc_samples: int = 100,
        device: str = "cpu",
    ):
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
        self.model: LSTMNet | None = None
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "LSTMModel":
        group_ids = kwargs.get("group_ids")
        X_seq, y_seq = self._make_sequences(X, y, group_ids=group_ids)
        if len(X_seq) == 0:
            raise ValueError("No sequences created. Check data length vs seq_length.")

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.model = LSTMNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience_counter = 0

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        return self

    def predict(
        self, X: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model not fitted.")

        group_ids = kwargs.get("group_ids")
        X_seq, _ = self._make_sequences(X, np.zeros(len(X)), group_ids=group_ids)
        if len(X_seq) == 0:
            empty = np.array([])
            return empty, empty, empty

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        # MC Dropout
        self.model.train()  # Enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                preds.append(self.model(X_t).cpu().numpy().flatten())

        self.model.eval()  # Restore eval mode

        preds = np.stack(preds)  # (mc_samples, N)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std

        return mean, lower, upper

    def _make_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences."""
        if len(X) <= self.seq_length:
            return np.array([]), np.array([])

        X_list, y_list = [], []
        if group_ids is None:
            group_ids = np.zeros(len(X), dtype=np.int64)

        group_ids = np.asarray(group_ids)
        if len(group_ids) != len(X):
            raise ValueError("group_ids must have the same length as X")

        for group_id in np.unique(group_ids):
            mask = group_ids == group_id
            X_group = X[mask]
            y_group = y[mask]

            if len(X_group) <= self.seq_length:
                continue

            for i in range(len(X_group) - self.seq_length):
                X_list.append(X_group[i : i + self.seq_length])
                y_list.append(y_group[i + self.seq_length])

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "params": self.get_params(),
        }, path)

    def load(self, path: str | Path) -> "LSTMModel":
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.model = LSTMNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        if data["model_state"]:
            self.model.load_state_dict(data["model_state"])
        return self

    def get_params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prediction_target": self.prediction_target,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "seq_length": self.seq_length,
            "lr": self.lr,
            "epochs": self.epochs,
            "mc_samples": self.mc_samples,
        }
