
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path
from src.data_loader import BatteryDataLoader
from src.models.lstm_model import BatteryLSTM

def load_config(config_path="experiments/baseline.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_sequences(df, seq_length, features, target):
    X = []
    y = []
    
    # Group by battery_id to ensure sequences don't cross batteries
    for bat_id in df['battery_id'].unique():
        group = df[df['battery_id'] == bat_id].sort_values('cycle')
        data_values = group[features].values
        target_values = group[target].values
        
        for i in range(len(group) - seq_length):
            X.append(data_values[i : i + seq_length])
            y.append(target_values[i + seq_length])
            
    return np.array(X), np.array(y)

def train_lstm():
    config = load_config()
    
    # Load Data
    print("Loading Data...")
    loader = BatteryDataLoader()
    df = loader.load_data(config['data']['batteries'])
    
    # Features
    features = config['data']['features']
    target = "rul"
    window_size = config['data']['window_size']
    
    # Train/Test Split (Leave one out)
    test_battery = config['data']['batteries'][-1]
    train_df = df[df['battery_id'] != test_battery]
    test_df = df[df['battery_id'] == test_battery]
    
    # Standardization (Fit on Train, Apply to All for leakage prevention)
    mean = train_df[features].mean()
    std = train_df[features].std()
    
    train_df[features] = (train_df[features] - mean) / std
    test_df[features] = (test_df[features] - mean) / std
    
    # Sequences
    X_train, y_train = create_sequences(train_df, window_size, features, target)
    X_test, y_test = create_sequences(test_df, window_size, features, target)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Model
    model = BatteryLSTM(
        input_dim=len(features),
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Training Loop
    epochs = config['training']['epochs']
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
    # Save Model
    save_dir = Path("results/nn_baseline")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "lstm_model.pth")
    print(f"Model saved to {save_dir}/lstm_model.pth")
    
    # Eval
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t)
        test_loss = criterion(preds, y_test_t)
        
    print(f"Test MSE: {test_loss.item():.4f}")
    
    # Save RMSE metric
    metrics = pd.DataFrame([{"model": "LSTM", "rmse": np.sqrt(test_loss.item())}])
    metrics.to_csv(save_dir / "metrics.csv", index=False)

if __name__ == "__main__":
    train_lstm()
