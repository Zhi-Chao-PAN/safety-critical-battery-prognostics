# tests/test_model_shapes.py
"""
Unit tests for model input/output shape validation.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestLSTMShapes:
    """Test BatteryLSTM model input/output shapes."""
    
    @pytest.fixture
    def model_params(self) -> dict:
        """Standard model parameters for testing."""
        return {
            "input_dim": 2,      # discharge_time, max_temp
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.2
        }
    
    @pytest.fixture
    def seq_length(self) -> int:
        return 10
        
    @pytest.fixture
    def batch_size(self) -> int:
        """Standard batch size for testing."""
        return 32
    
    def test_lstm_forward_shape(self, model_params: dict, batch_size: int, seq_length: int) -> None:
        """Test LSTM produces correct output shape."""
        from models.lstm_model import BatteryLSTM
        
        model = BatteryLSTM(**model_params)
        # Input shape for LSTM: (batch, seq_len, features)
        x = torch.randn(batch_size, seq_length, model_params["input_dim"])
        
        output = model(x)
        
        assert output.shape == (batch_size, 1), \
            f"Expected shape ({batch_size}, 1), got {output.shape}"
    
    def test_lstm_single_sample(self, model_params: dict, seq_length: int) -> None:
        """Test LSTM works with single sample."""
        from models.lstm_model import BatteryLSTM
        
        model = BatteryLSTM(**model_params)
        x = torch.randn(1, seq_length, model_params["input_dim"])
        
        output = model(x)
        
        assert output.shape == (1, 1)
    
    def test_lstm_gradient_flow(self, model_params: dict, batch_size: int, seq_length: int) -> None:
        """Test gradients flow through LSTM."""
        from models.lstm_model import BatteryLSTM
        
        model = BatteryLSTM(**model_params)
        x = torch.randn(batch_size, seq_length, model_params["input_dim"], requires_grad=True)
        y = torch.randn(batch_size, 1)
        
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Gradient not computed"
                assert not torch.isnan(param.grad).any(), "NaN in gradients"


class TestModelReproducibility:
    """Test model reproducibility with seeds."""
    
    def test_lstm_reproducible_with_seed(self) -> None:
        """Test LSTM produces identical results with same seed."""
        from models.lstm_model import BatteryLSTM
        
        params = {
            "input_dim": 2,
            "hidden_dim": 32,
            "num_layers": 1
        }
        x = torch.randn(5, 10, 2)
        
        # First run
        torch.manual_seed(42)
        model1 = BatteryLSTM(**params)
        out1 = model1(x).detach()
        
        # Second run with same seed
        torch.manual_seed(42)
        model2 = BatteryLSTM(**params)
        out2 = model2(x).detach()
        
        assert torch.allclose(out1, out2), "Results should be identical with same seed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
