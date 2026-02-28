# tests/test_model_shapes.py
"""
Unit tests for model input/output shape validation.
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLSTMShapes:
    """Test LSTMNet model input/output shapes."""

    @pytest.fixture
    def model_params(self) -> dict:
        return {"input_dim": 2, "hidden_dim": 64, "num_layers": 2, "dropout": 0.2}

    @pytest.fixture
    def seq_length(self) -> int:
        return 10

    @pytest.fixture
    def batch_size(self) -> int:
        return 32

    def test_lstm_forward_shape(self, model_params, batch_size, seq_length):
        from src.models.lstm_model import LSTMNet
        model = LSTMNet(**model_params)
        x = torch.randn(batch_size, seq_length, model_params["input_dim"])
        output = model(x)
        assert output.shape == (batch_size, 1)

    def test_lstm_single_sample(self, model_params, seq_length):
        from src.models.lstm_model import LSTMNet
        model = LSTMNet(**model_params)
        x = torch.randn(1, seq_length, model_params["input_dim"])
        output = model(x)
        assert output.shape == (1, 1)

    def test_lstm_gradient_flow(self, model_params, batch_size, seq_length):
        from src.models.lstm_model import LSTMNet
        model = LSTMNet(**model_params)
        x = torch.randn(batch_size, seq_length, model_params["input_dim"], requires_grad=True)
        y = torch.randn(batch_size, 1)
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


class TestModelReproducibility:
    def test_lstm_reproducible_with_seed(self):
        from src.models.lstm_model import LSTMNet
        params = {"input_dim": 2, "hidden_dim": 32, "num_layers": 1}
        x = torch.randn(5, 10, 2)
        torch.manual_seed(42)
        model1 = LSTMNet(**params)
        out1 = model1(x).detach()
        torch.manual_seed(42)
        model2 = LSTMNet(**params)
        out2 = model2(x).detach()
        assert torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
