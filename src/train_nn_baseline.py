# src/train_nn_baseline.py
"""
PyTorch MLP Baseline for Housing Price Prediction.

This module implements a simple Multi-Layer Perceptron (MLP) as a deep learning
baseline for comparison with Bayesian hierarchical models. The architecture
follows standard practices for tabular data with Dropout regularization.

Architecture:
    Input -> Linear(64) -> ReLU -> Dropout(0.2) -> Linear(32) -> ReLU -> Dropout(0.1) -> Linear(1)
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


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for regression.
    
    A 3-layer feedforward network with ReLU activations and Dropout
    regularization. Designed for tabular data regression tasks.
    
    Architecture:
        - Layer 1: Linear(input_dim, 64) + ReLU + Dropout(0.2)
        - Layer 2: Linear(64, 32) + ReLU + Dropout(0.1)
        - Layer 3: Linear(32, 1)
    
    Args:
        input_dim: Number of input features.
        
    Example:
        >>> model = SimpleMLP(input_dim=7)
        >>> x = torch.randn(32, 7)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 1])
    """
    
    def __init__(self, input_dim: int) -> None:
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)  # Regression output
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
            
        Returns:
            Output predictions of shape (batch_size, 1).
        """
        return self.net(x)


def train_evaluate_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    seed: int = 42,
    epochs: int = 500,
    verbose: bool = False
) -> float:
    """
    Train and evaluate an MLP model on given data splits.
    
    Handles scaling internally to prevent data leakage. Uses Adam optimizer
    with MSE loss for regression.
    
    Args:
        X_train: Training features of shape (n_train, n_features).
        y_train: Training targets of shape (n_train, 1).
        X_test: Test features of shape (n_test, n_features).
        y_test: Test targets of shape (n_test, 1).
        input_dim: Number of input features (for model construction).
        seed: Random seed for reproducibility.
        epochs: Number of training epochs.
        verbose: If True, print loss every 50 epochs.
        
    Returns:
        Root Mean Squared Error (RMSE) on the test set.
        
    Note:
        Scaling is performed inside this function on each fold to
        prevent data leakage in cross-validation scenarios.
    """
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Scaling inside the fold to avoid leakage
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)
    
    # Model
    model = SimpleMLP(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()
        
        if verbose and epoch % 50 == 0:
             print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    
    return rmse


def main() -> None:
    """
    Main entry point for training the MLP baseline.
    
    Loads data according to schema, performs train/test split,
    trains the model, and saves metrics.
    """
    schema = load_schema()
    data_path: str = schema["dataset"]["path"]
    target_str: str = schema["target"]["name"]
    features: list = schema["features"]["numeric"]
    
    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)
    X = df[features].values
    y = df[target_str].values.reshape(-1, 1)
    
    # Single run evaluation (Default behavior)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    rmse = train_evaluate_mlp(
        X_train, y_train, X_test, y_test, 
        input_dim=len(features), seed=42, verbose=True
    )
    
    print("\n===== PyTorch MLP Baseline Results =====")
    print(f"Test RMSE: {rmse:.4f}")
    
    results_dir = Path("results/nn_baseline")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([{"model": "MLP", "rmse": rmse}])
    metrics.to_csv(results_dir / "metrics.csv", index=False)

if __name__ == "__main__":
    main()
