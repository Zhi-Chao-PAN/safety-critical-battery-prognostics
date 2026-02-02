import torch
import torch.nn as nn

class BatteryLSTM(nn.Module):
    """
    LSTM Model for Battery RUL Prediction.
    
    Architecture:
        Input -> LSTM(hidden_dim) -> Dropout -> Linear -> ReLU -> Linear(1)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(BatteryLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1) # RUL prediction
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Input_Dim)
            
        Returns:
            Output RUL predictions of shape (Batch, 1)
        """
        # LSTM output: (Batch, Seq_Len, Hidden_Dim)
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step
        last_step_out = lstm_out[:, -1, :]
        
        out = self.fc(last_step_out)
        return out

    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform MC Dropout inference.
        """
        self.train() # Enable dropout
        preds_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds_list.append(self.forward(x))
        
        preds = torch.stack(preds_list)
        return preds.mean(dim=0), preds.std(dim=0)
