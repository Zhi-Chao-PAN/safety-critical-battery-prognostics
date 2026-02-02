
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
    
    # Load Data (Test Battery Only)
    loader = BatteryDataLoader()
    df = loader.load_data(battery_ids)
    test_df = df[df['battery_id'] == test_battery].copy()
    
    # 1. LSTM Inference
    # Need to load training stats to normalize test data effectively
    # (Simplified: fit on others on the fly as approximation or hardcode from train script)
    train_df = df[df['battery_id'] != test_battery]
    features_lstm = config['data']['features'] # ['discharge_time', 'max_temp']
    target = "rul"
    
    mean = train_df[features_lstm].mean()
    std = train_df[features_lstm].std()
    
    test_df_norm = test_df.copy()
    test_df_norm[features_lstm] = (test_df_norm[features_lstm] - mean) / std
    
    window_size = config['data']['window_size']
    X_test_lstm = create_sequences_for_inference(test_df_norm, window_size, features_lstm)
    X_test_t = torch.tensor(X_test_lstm, dtype=torch.float32)
    
    # Load LSTM
    model = BatteryLSTM(
        input_dim=len(features_lstm),
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    
    model_path = Path("results/nn_baseline/lstm_model.pth")
    lstm_preds = np.zeros(len(X_test_lstm))
    
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            lstm_preds = model(X_test_t).numpy().flatten()
    else:
        print("Warning: LSTM model not found.")

    # 2. Bayesian Inference (Zero-Shot for B0018)
    import arviz as az
    trace_path = Path("results/bayes_hierarchical/trace.nc")
    
    bayes_mean = np.zeros(len(test_df))
    bayes_hdi_low = np.zeros(len(test_df))
    bayes_hdi_high = np.zeros(len(test_df))
    
    if trace_path.exists():
        print("Loading Bayesian Trace...")
        trace = az.from_netcdf(trace_path)
        
        # Load scaler params
        scaler = pd.read_csv("results/bayes_hierarchical/scaler_params.csv", index_col=0)
        X_test_bayes = test_df[['discharge_time', 'max_temp']].values
        X_test_bayes_scaled = (X_test_bayes - scaler['mean'].values) / scaler['std'].values
        
        # Extract Posteriors
        # Since B0018 is NEW, we use the Population Priors (mu_alpha, sigma_alpha)
        # to simulate a new alpha_j, combined with the learned Beta.
        
        posterior = trace.posterior
        # Flatten chains
        mu_alpha_samples = posterior['mu_alpha'].values.flatten()
        sigma_alpha_samples = posterior['sigma_alpha'].values.flatten()
        beta_samples = posterior['beta'].values.reshape(-1, 2) # (samples, features)
        sigma_samples = posterior['sigma'].values.flatten()
        
        n_samples = len(mu_alpha_samples)
        n_test = len(test_df)
        
        # Vectorized Prediction: 
        # For each sample s: alpha_new_s ~ N(mu_alpha[s], sigma_alpha[s])
        # mu_s = alpha_new_s + X @ beta[s]
        # y_s ~ N(mu_s, sigma[s])
        
        print(f"Generating Posterior Predictive for {n_test} cycles...")
        
        # Simulate new alpha for B0018 (Hyperpriors)
        alpha_new = np.random.normal(mu_alpha_samples, sigma_alpha_samples)
        
        # Computation: (n_samples, n_test)
        # Term 1: Alpha (n_samples, 1)
        term1 = alpha_new[:, np.newaxis]
        
        # Term 2: X (n_test, 2) @ Beta.T (2, n_samples) -> (n_test, n_samples).T -> (n_samples, n_test)
        term2 = (X_test_bayes_scaled @ beta_samples.T).T
        
        mu_pred = term1 + term2
        
        # Add aleatoric uncertainty
        # y_pred = np.random.normal(mu_pred, sigma_samples[:, np.newaxis])
        # Actually for the "Mean" line we often plot mu_pred mean, 
        # but for Uncertainty Band we generally want Expected Aleatoric range OR Epistemic.
        # Paper req: "Safety Buffer". This usually implies Predictive Interval (Epistemic + Aleatoric).
        
        y_pred_samples = np.random.normal(mu_pred, sigma_samples[:, np.newaxis])
        
        bayes_mean = y_pred_samples.mean(axis=0)
        
        # Calculate HDI (95%)
        hdi = az.hdi(y_pred_samples[:, :, None], hdi_prob=0.95) # az.hdi expects (chain, draw, ...) or (draw, ...)
        # Manually:
        bayes_hdi_low = np.percentile(y_pred_samples, 2.5, axis=0)
        bayes_hdi_high = np.percentile(y_pred_samples, 97.5, axis=0)
        
    else:
        print("Warning: Bayesian trace not found.")

    # 3. Visualization Alignment
    # LSTM predictions are shifted by window_size
    # We align everything to the original cycle index of the test battery
    
    cycles = test_df['cycle'].values
    gt_rul = test_df['rul'].values
    
    # LSTM is shorter by window_size
    lstm_plot = np.full(len(cycles), np.nan)
    lstm_plot[window_size:] = lstm_preds
    
    # Plotting
    plt.style.use('default') 
    # Attempt to set Serif font if available, else standard
    plt.rcParams["font.family"] = "serif"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ground Truth
    ax.plot(cycles, gt_rul, 'k-', linewidth=2, label='True RUL (Ground Truth)')
    
    # Bayesian
    ax.fill_between(cycles, bayes_hdi_low, bayes_hdi_high, color='green', alpha=0.3, label='Bayesian 95% HDI')
    # ax.plot(cycles, bayes_mean, 'g-', linewidth=1, alpha=0.8) # Optional mean line
    
    # LSTM
    ax.plot(cycles, lstm_plot, 'r--', linewidth=2, label='LSTM (Point Estimate)')
    
    # "Safety Buffer Zone" Annotation
    # Heuristic: Find where HDI width > X or where RUL is low
    # We'll just annotate near the EOL
    eol_idx = np.argmin(gt_rul)
    if eol_idx < len(cycles):
        ax.annotate('Safety Buffer Zone\n(Uncertainty Widens)', 
                    xy=(cycles[eol_idx], bayes_hdi_high[eol_idx]), 
                    xytext=(cycles[eol_idx]-50, bayes_hdi_high[eol_idx]+50),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=10, fontweight='bold', color='darkgreen')

    ax.set_title(f"Figure 1: RUL Comparison on Test Battery {test_battery}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Cycle Number", fontsize=12)
    ax.set_ylabel("Remaining Useful Life (RUL)", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    out_path = Path("results/final_comparison_B0018.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved comparison figure to {out_path}")
    
    # Metrics
    # LSTM RMSE
    valid_mask = ~np.isnan(lstm_plot)
    rmse = np.sqrt(np.mean((lstm_plot[valid_mask] - gt_rul[valid_mask])**2))
    
    # Bayesian Coverage
    coverage = np.mean((gt_rul >= bayes_hdi_low) & (gt_rul <= bayes_hdi_high))
    
    print("-" * 30)
    print(f"LSTM RMSE (B0018): {rmse:.2f}")
    print(f"Bayesian HDI Coverage (B0018): {coverage:.2%}")
    print("-" * 30)
    
    # Save text report
    with open("results/verification_report.txt", "w") as f:
        f.write(f"LSTM RMSE: {rmse:.2f}\n")
        f.write(f"Bayesian Coverage: {coverage:.2%}\n")

if __name__ == "__main__":
    compare_models()
