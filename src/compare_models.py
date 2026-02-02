
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from src.data_loader import BatteryDataLoader
from src.models.lstm_model import BatteryLSTM

def load_config(config_path="experiments/baseline.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_sequences_for_inference(df, seq_length, features):
    data_values = df[features].values
    X = []
    if len(df) <= seq_length:
        return np.array([])
        
    for i in range(len(df) - seq_length):
        X.append(data_values[i : i + seq_length])
        
    return np.array(X)

def compare_models():
    config = load_config()
    battery_ids = config['data']['batteries']
    test_battery = battery_ids[-1] # B0018
    
    print(f"Comparing models for Battery: {test_battery}")
    
    # Load Data
    loader = BatteryDataLoader()
    df = loader.load_data(battery_ids)
    
    # Preprocessing (Need to mimic training stats - ideally load a scaler)
    # For demo, we fit on all OTHER batteries to standardize the test battery
    features = ['discharge_time', 'max_temp']
    target = "rul"
    
    train_df = df[df['battery_id'] != test_battery]
    test_df = df[df['battery_id'] == test_battery]
    
    mean = train_df[features].mean()
    std = train_df[features].std()
    
    test_df_norm = test_df.copy()
    test_df_norm[features] = (test_df_norm[features] - mean) / std
    
    # LSTM Inference
    window_size = config['data']['window_size']
    X_test = create_sequences_for_inference(test_df_norm, window_size, features)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    
    model = BatteryLSTM(
        input_dim=len(features),
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    
    model_path = Path("results/nn_baseline/lstm_model.pth")
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            lstm_preds = model(X_test_t).numpy().flatten()
    else:
        print("LSTM model not found. Run training first.")
        lstm_preds = np.zeros(len(X_test))

    # Ground Truth (aligned)
    gt_rul = test_df['rul'].values[window_size:] 
    
    # Load Bayesian Trace (Placeholder visualization logic from previous phase, simplified)
    # NOTE: In a real standardized script, we would load the 'trace.nc' and predict.
    # For this 'Clean' version, we focus on the Plotting hygiene.
    
    # Basic Plot
    plt.figure(figsize=(10, 6))
    plt.plot(gt_rul, 'k-', label='Ground Truth')
    plt.plot(lstm_preds, 'r--', label='LSTM Prediction')
    
    plt.title(f"RUL Prediction: {test_battery}")
    plt.xlabel("Cycle (window offset)")
    plt.ylabel("RUL")
    plt.legend()
    plt.grid(True)
    
    out_path = Path("results/final_comparison.png")
    plt.savefig(out_path)
    print(f"Saved comparison to {out_path}")

if __name__ == "__main__":
    compare_models()
