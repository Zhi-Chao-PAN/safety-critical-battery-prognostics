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


class TestMLPShapes:
    """Test MLP model input/output shapes."""
    
    @pytest.fixture
    def input_dim(self) -> int:
        """Standard input dimension for testing."""
        return 7  # Matching number of numeric features in schema
    
    @pytest.fixture
    def batch_size(self) -> int:
        """Standard batch size for testing."""
        return 32
    
    def test_mlp_forward_shape(self, input_dim: int, batch_size: int) -> None:
        """Test MLP produces correct output shape."""
        from train_nn_baseline import SimpleMLP
        
        model = SimpleMLP(input_dim=input_dim)
        x = torch.randn(batch_size, input_dim)
        
        output = model(x)
        
        assert output.shape == (batch_size, 1), \
            f"Expected shape ({batch_size}, 1), got {output.shape}"
    
    def test_mlp_single_sample(self, input_dim: int) -> None:
        """Test MLP works with single sample."""
        from train_nn_baseline import SimpleMLP
        
        model = SimpleMLP(input_dim=input_dim)
        x = torch.randn(1, input_dim)
        
        output = model(x)
        
        assert output.shape == (1, 1)
    
    def test_mlp_gradient_flow(self, input_dim: int, batch_size: int) -> None:
        """Test gradients flow through MLP."""
        from train_nn_baseline import SimpleMLP
        
        model = SimpleMLP(input_dim=input_dim)
        x = torch.randn(batch_size, input_dim, requires_grad=True)
        y = torch.randn(batch_size, 1)
        
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None, "Gradient not computed"
            assert not torch.isnan(param.grad).any(), "NaN in gradients"


class TestSpatialEmbeddingMLP:
    """Test Spatial Embedding MLP model shapes."""
    
    @pytest.fixture
    def model_config(self) -> dict:
        """Model configuration for testing."""
        return {
            "num_numeric": 7,
            "num_clusters": 10,
            "emb_dim": 8
        }
    
    @pytest.fixture
    def batch_size(self) -> int:
        return 32
    
    def test_spatial_mlp_forward_shape(self, model_config: dict, batch_size: int) -> None:
        """Test Spatial MLP produces correct output shape."""
        from train_nn_spatial import SpatialEmbMLP
        
        model = SpatialEmbMLP(**model_config)
        x_numeric = torch.randn(batch_size, model_config["num_numeric"])
        x_cluster = torch.randint(0, model_config["num_clusters"], (batch_size,))
        
        output = model(x_numeric, x_cluster)
        
        assert output.shape == (batch_size, 1)
    
    def test_embedding_dimension(self, model_config: dict) -> None:
        """Test embedding layer has correct dimensions."""
        from train_nn_spatial import SpatialEmbMLP
        
        model = SpatialEmbMLP(**model_config)
        
        # Check embedding layer
        assert model.spatial_emb.num_embeddings == model_config["num_clusters"]
        assert model.spatial_emb.embedding_dim == model_config["emb_dim"]
    
    def test_cluster_index_bounds(self, model_config: dict, batch_size: int) -> None:
        """Test model handles all valid cluster indices."""
        from train_nn_spatial import SpatialEmbMLP
        
        model = SpatialEmbMLP(**model_config)
        x_numeric = torch.randn(batch_size, model_config["num_numeric"])
        
        # Test each cluster index
        for cluster_id in range(model_config["num_clusters"]):
            x_cluster = torch.full((batch_size,), cluster_id, dtype=torch.long)
            output = model(x_numeric, x_cluster)
            assert output.shape == (batch_size, 1)


class TestModelReproducibility:
    """Test model reproducibility with seeds."""
    
    def test_mlp_reproducible_with_seed(self) -> None:
        """Test MLP produces identical results with same seed."""
        from train_nn_baseline import SimpleMLP
        
        input_dim = 7
        x = torch.randn(10, input_dim)
        
        # First run
        torch.manual_seed(42)
        model1 = SimpleMLP(input_dim)
        out1 = model1(x).detach()
        
        # Second run with same seed
        torch.manual_seed(42)
        model2 = SimpleMLP(input_dim)
        out2 = model2(x).detach()
        
        assert torch.allclose(out1, out2), "Results should be identical with same seed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
