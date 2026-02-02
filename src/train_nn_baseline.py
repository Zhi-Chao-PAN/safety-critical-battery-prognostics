# src/train_nn_baseline.py
"""
PyTorch LSTM Baseline for Battery RUL Prediction.

This module implements a LSTM-based deep learning baseline for comparison 
with Bayesian hierarchical models. It uses sliding window sequences of 
degradation features to predict Remaining Useful Life (RUL).
"""

from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data_loader import load_battery_data
from src.models.lstm_model import BatteryLSTM

logger = setup_logger(__name__)

def create_sequences(
    data: pd.DataFrame, 
    seq_length: int, 
    features: List[str], 
    target: str,
    group_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for LSTM input.
    
    Args:
        data: DataFrame containing the time series data.
        seq_length: Length of each sequence.
        features: List of feature column names.
        target: Target column name.
        group_col: Column ensuring sequences don't cross battery boundaries.
        
    Returns:
        X: Sequence array of shape (N_samples, seq_length, n_features)
        y: Target array of shape (N_samples, 1) - Target at the last step of sequence
    """
    X_list, y_list = [], []
    
    # Process each battery separately
    for _, group in data.groupby(group_col):
        # Sort by cycle/time if needed, assuming already sorted or cycle_id exists
        if 'cycle_id' in group.columns:
            group = group.sort_values('cycle_id')
            
        values = group[features].values
        target_values = group[target].values
        
        num_samples = len(group)
        if num_samples <= seq_length:
            continue
            
        for i in range(num_samples - seq_length):
            X_list.append(values[i : i + seq_length])
            y_list.append(target_values[i + seq_length]) # Predict RUL at the end of the window
            
    return np.array(X_list), np.array(y_list).reshape(-1, 1)

def train_evaluate_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    seq_length: int,
    seed: int = 42,
    epochs: int = 100, # LSTM takes longer to converge usually
    batch_size: int = 32,
    verbose: bool = False,
    save_path: Path = None
) -> float:
    """
    Train and evaluate LSTM model.
    """
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    # Model
    model = BatteryLSTM(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    logger.info(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if verbose and (epoch + 1) % 10 == 0:
             logger.info(f"Epoch {epoch+1}/{epochs}: Avg Loss {epoch_loss / len(train_loader):.4f}")

    # Save Model
    if save_path:
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).numpy()
    
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    
    return rmse

def main() -> None:
    # Load standardized Pydantic Config
    try:
        config = load_config()
    except Exception as e:
        logger.error("Configuration load failed. Exiting.")
        return

    data_path = config.dataset.path
    target_str = config.target.name
    features = config.features.numeric
    group_col = config.group.name
    
    logger.info(f"Loading data from: {data_path}")
    df = load_battery_data(data_path)
    
    # Standardize Features
    logger.info("Standardizing features...")
    scaler_x = StandardScaler()
    df[features] = scaler_x.fit_transform(df[features])
    
    # Leave-One-Group-Out Split
    # Taking the last battery as test set, others as train
    batteries = df[group_col].unique()
    test_battery = batteries[-1]
    train_batteries = batteries[:-1]
    
    logger.info(f"Train Batteries: {train_batteries}")
    logger.info(f"Test Battery: {test_battery}")
    
    train_df = df[df[group_col].isin(train_batteries)]
    test_df = df[df[group_col] == test_battery]
    
    # Create Sequences
    SEQ_LENGTH = 30
    logger.info(f"Creating sequences (Length={SEQ_LENGTH})...")
    X_train, y_train = create_sequences(train_df, SEQ_LENGTH, features, target_str, group_col)
    X_test, y_test = create_sequences(test_df, SEQ_LENGTH, features, target_str, group_col)
    
    logger.info(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    
    rmse = train_evaluate_lstm(
        X_train, y_train, X_test, y_test, 
        input_dim=len(features), 
        seq_length=SEQ_LENGTH,
        seed=42, 
        verbose=True,
        save_path=Path("results/nn_baseline/lstm_model.pth")
    )
    
    logger.info("\n===== PyTorch LSTM Baseline Results =====")
    logger.info(f"Test RMSE: {rmse:.4f}")
    
    results_dir = Path("results/nn_baseline")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([{"model": "LSTM", "rmse": rmse}])
    metrics.to_csv(results_dir / "metrics.csv", index=False)

if __name__ == "__main__":
    main()
