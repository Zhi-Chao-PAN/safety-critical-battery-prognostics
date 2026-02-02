import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from src.data_loader import BatteryDataLoader
from src.models.lstm_model import BatteryLSTM
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_config(config_path: str = "experiments/baseline.yaml") -> Dict[str, Any]:
    """Load experiment configuration."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise

def create_sequences(df, seq_length: int, features: List[str], target: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM using vectorized numpy striding.
    
    Args:
        df: Input DataFrame
        seq_length: Length of sequences
        features: List of feature column names
        target: Target column name
        
    Returns:
        X: Sequence data (N, seq_length, num_features)
        y: Target data (N, 1)
    """
    X_list, y_list = [], []
    
    # Process each battery separately to avoid cross-contamination
    for bat_id in df['battery_id'].unique():
        group = df[df['battery_id'] == bat_id].sort_values('cycle')
        data_values = group[features].values.astype(np.float32)
        target_values = group[target].values.astype(np.float32)
        
        num_samples = len(group) - seq_length
        if num_samples <= 0:
            continue
            
        # Vectorized striding
        # X shape: (num_samples, seq_length, num_features)
        # Create a view into the array with the given shape and strides
        stride_0 = data_values.strides[0]
        stride_1 = data_values.strides[1]
        
        X_bat = np.lib.stride_tricks.as_strided(
            data_values,
            shape=(num_samples, seq_length, len(features)),
            strides=(stride_0, stride_0, stride_1)
        )
        
        y_bat = target_values[seq_length:]
        
        X_list.append(X_bat)
        y_list.append(y_bat)
        
    if not X_list:
        return np.array([]), np.array([])
        
    return np.concatenate(X_list), np.concatenate(y_list)

def train_evaluate_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    seq_length: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 32,
    device: str = "cpu",
    verbose: bool = False
) -> Tuple[float, nn.Module]:
    """
    Train LSTM and evaluate on test set.
    Returns RMSE.
    """
    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Model
    model = BatteryLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch+1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t)
        test_loss = criterion(preds, y_test_t)
        rmse = np.sqrt(test_loss.item())
        
    return float(rmse), model

def main():
    """Main training entry point."""
    config = load_config()
    
    logger.info("Loading Data...")
    loader = BatteryDataLoader()
    try:
        df = loader.load_data(config['data']['batteries'])
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Features
    features = config['data']['features']
    target = "rul"
    window_size = config['data']['window_size']
    
    # Train/Test Split (Leave one out)
    test_battery = config['data']['batteries'][-1]
    train_df = df[df['battery_id'] != test_battery].copy()
    test_df = df[df['battery_id'] == test_battery].copy()
    
    # Standardization
    mean = train_df[features].mean()
    std = train_df[features].std()
    
    train_df[features] = (train_df[features] - mean) / std
    test_df[features] = (test_df[features] - mean) / std
    
    # Sequences
    X_train, y_train = create_sequences(train_df, window_size, features, target)
    X_test, y_test = create_sequences(test_df, window_size, features, target)
    
    if len(X_train) == 0:
        logger.error("No training sequences created.")
        return

    logger.info(f"Training on {len(X_train)} sequences, Testing on {len(X_test)} sequences.")

    rmse = train_evaluate_lstm(
        X_train, y_train, X_test, y_test,
        input_dim=len(features),
        seq_length=window_size,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        learning_rate=config['training']['learning_rate'],
        epochs=config['training']['epochs'],
        verbose=True
    )
    
    logger.info(f"Test RMSE: {rmse:.4f}")
    
    # Save Metrics
    save_dir = Path("results/nn_baseline")
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "metrics.csv", "w") as f:
        f.write(f"model,rmse\nLSTM,{rmse}")

if __name__ == "__main__":
    main()
