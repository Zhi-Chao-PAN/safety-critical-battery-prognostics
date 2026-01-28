# src/train_nn_spatial.py
"""
Spatial Embedding Neural Network for Housing Price Prediction.

This module implements an MLP with learnable spatial cluster embeddings,
testing whether explicit spatial features improve neural network performance
compared to the baseline MLP.

Key Features:
    - Learnable embeddings for spatial cluster IDs
    - Concatenation of numeric features with spatial embeddings
    - Same MLP architecture as baseline for fair comparison
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.schema import load_schema


class SpatialEmbMLP(nn.Module):
    """
    MLP with Spatial Cluster Embeddings.
    
    Combines learnable spatial embeddings with numeric features
    to test whether explicit spatial structure improves predictions.
    
    Architecture:
        - Embedding layer for cluster IDs
        - Concatenate: [numeric_features, cluster_embedding]
        - MLP: Linear(64) -> ReLU -> Dropout -> Linear(32) -> ReLU -> Dropout -> Linear(1)
    
    Args:
        num_numeric: Number of numeric input features.
        num_clusters: Total number of spatial clusters.
        emb_dim: Dimension of the embedding vector for each cluster.
    
    Example:
        >>> model = SpatialEmbMLP(num_numeric=7, num_clusters=10, emb_dim=8)
        >>> x_num = torch.randn(32, 7)
        >>> x_cluster = torch.randint(0, 10, (32,))
        >>> output = model(x_num, x_cluster)
        >>> output.shape
        torch.Size([32, 1])
    """
    
    def __init__(self, num_numeric: int, num_clusters: int, emb_dim: int = 4) -> None:
        super(SpatialEmbMLP, self).__init__()
        
        # 1. Spatial Embedding Layer (Inductive Bias)
        self.spatial_emb = nn.Embedding(num_embeddings=num_clusters, embedding_dim=emb_dim)
        
        # 2. Concat [Numeric Features + Embedding]
        input_dim = num_numeric + emb_dim
        
        # 3. Main Network
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_numeric: torch.Tensor, x_cluster: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining numeric features with spatial embeddings.
        
        Args:
            x_numeric: Numeric features tensor of shape (batch_size, num_numeric).
            x_cluster: Cluster indices tensor of shape (batch_size,) with dtype=long.
            
        Returns:
            Predictions of shape (batch_size, 1).
        """
        # Look up embedding
        emb = self.spatial_emb(x_cluster)  # (batch, emb_dim)
        
        # Concatenate
        combined = torch.cat([x_numeric, emb], dim=1)
        
        return self.net(combined)


def main() -> None:
    """
    Main entry point for training the Spatial Embedding NN.
    
    Loads data, trains the model with cluster embeddings, and saves results.
    """
    schema = load_schema()
    data_path: str = schema["dataset"]["path"]
    target_str: str = schema["target"]["name"]
    group_var: str = schema["bayesian"]["hierarchical"]["group"]
    
    # Numeric features (including lat/lon for maximum power "Hybrid Model")
    features: list = schema["features"]["numeric"]
    
    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)
    
    # 1. Prepare Features
    X_num = df[features].values
    y = df[target_str].values.reshape(-1, 1)
    
    # 2. Encode Clusters (0 to N-1)
    group_codes, group_uniques = pd.factorize(df[group_var])
    X_cat = group_codes
    n_clusters = len(group_uniques)
    print(f"Found {n_clusters} spatial clusters for embedding.")
    
    # 3. Split (Random for simple comparison)
    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        X_num, X_cat, y, test_size=0.2, random_state=42
    )
    
    # 4. Standardize Numeric
    scaler_x = StandardScaler()
    X_num_train = scaler_x.fit_transform(X_num_train)
    X_num_test = scaler_x.transform(X_num_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # 5. To Tensors
    X_num_train_t = torch.tensor(X_num_train, dtype=torch.float32)
    X_cat_train_t = torch.tensor(X_cat_train, dtype=torch.long)  # Embedding needs Long
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    
    X_num_test_t = torch.tensor(X_num_test, dtype=torch.float32)
    X_cat_test_t = torch.tensor(X_cat_test, dtype=torch.long)
    y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)
    
    # Model Setup
    model = SpatialEmbMLP(num_numeric=len(features), num_clusters=n_clusters, emb_dim=8)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training Spatial Embedding NN...")
    epochs = 500
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        preds = model(X_num_train_t, X_cat_train_t)
        loss = criterion(preds, y_train_t)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_num_test_t, X_cat_test_t)
                val_loss = criterion(val_preds, y_test_t)
            print(f"Epoch {epoch}: Train Loss {loss.item():.4f}, Test Loss {val_loss.item():.4f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_num_test_t, X_cat_test_t).numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    
    print("\n===== Spatial Embedding NN Results =====")
    print(f"Test RMSE: {rmse:.4f}")
    
    # Save results
    results_dir = Path("results/nn_spatial")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), results_dir / "spatial_model.pth")
    
    metrics = pd.DataFrame([{"model": "MLP+Embedding", "rmse": rmse}])
    metrics.to_csv(results_dir / "metrics.csv", index=False)
    print(f"Saved model and metrics to {results_dir}")


if __name__ == "__main__":
    main()
