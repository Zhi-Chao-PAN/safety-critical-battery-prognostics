# src/evaluate_rigor.py
"""
Rigorous Cross-Validation Evaluation Script for Battery RUL.

Protocol:
    - Leave-One-Group-Out CV (Grouped by Battery_ID)
    - Compares:
        1. LSTM + MC Dropout (Baseline)
        2. Hierarchical Bayesian Model (Proposed)
    - Generates "Safety Buffer" plots using standardized visualization
"""

from typing import Type, Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data_loader import BatteryDataLoader
from src.train_nn_baseline import train_evaluate_lstm, create_sequences
from src.models.bayes_model import HierarchicalBayesianModel
from src.utils.visualization import plot_safety_comparison

logger = setup_logger(__name__)

def evaluate_fold(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    battery_id: str,
    features: List[str],
    target: str,
    config: Any
) -> List[Dict[str, Any]]:
    """Evaluate one fold (one test battery)."""
    
    results = []
    
    # ==========================================
    # 1. HIERARCHICAL BAYESIAN MODEL (Strategies)
    # ==========================================
    logger.info(f"  > Training Hierarchical Bayesian Model...")
    
    # Prepare Data
    X_train = train_df[features].values
    y_train = train_df[target].values
    groups_train = train_df['battery_id'].values
    
    X_test = test_df[features].values
    y_test = test_df[target].values
    
    # Train
    bayes_model = HierarchicalBayesianModel(samples=500, tune=500, chains=1) # Reduced for speed in demo
    bayes_model.fit(X_train, y_train, groups_train, feature_names=features)
    
    # Predict (Probabilistic)
    b_mean, b_low, b_high = bayes_model.predict(X_test, group_id=battery_id)
    
    # Metrics
    rmse_bayes = np.sqrt(mean_squared_error(y_test, b_mean))
    coverage = np.mean((y_test >= b_low) & (y_test <= b_high)) * 100
    width = np.mean(b_high - b_low)
    
    results.append({
        "Model": "Hierarchical Bayesian",
        "Battery": battery_id,
        "RMSE": rmse_bayes,
        "HDI_Coverage": coverage,
        "Uncertainty_Width": width
    })
    
    # ==========================================
    # 2. LSTM + MC DROPOUT (Baseline)
    # ==========================================
    logger.info(f"  > Training LSTM Baseline...")
    SEQ_LENGTH = config.modeling.lstm.window_size
    
    X_train_seq, y_train_seq = create_sequences(train_df, SEQ_LENGTH, features, target)
    X_test_seq, y_test_seq = create_sequences(test_df, SEQ_LENGTH, features, target)
    
    if len(X_test_seq) > 0:
        rmse_lstm, model = train_evaluate_lstm(
            X_train_seq, y_train_seq, X_test_seq, y_test_seq,
            input_dim=len(features), seq_length=SEQ_LENGTH, verbose=False
        )
        
        # Uncertainty
        device = next(model.parameters()).device
        X_test_t = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
        lstm_mean, lstm_std = model.predict_with_uncertainty(X_test_t, n_samples=50)
        lstm_mean = lstm_mean.cpu().numpy().flatten()
        lstm_std = lstm_std.cpu().numpy().flatten()
        
        # LSTM "HDI" (Normality Assumption)
        l_low = lstm_mean - 1.96 * lstm_std
        l_high = lstm_mean + 1.96 * lstm_std
        
        coverage_lstm = np.mean((y_test_seq.flatten() >= l_low) & (y_test_seq.flatten() <= l_high)) * 100
        
        results.append({
            "Model": "LSTM (MC Dropout)",
            "Battery": battery_id,
            "RMSE": rmse_lstm,
            "HDI_Coverage": coverage_lstm,
            "Uncertainty_Width": np.mean(l_high - l_low)
        })
        
        # ==========================================
        # VISUALIZATION (If Safety Case)
        # ==========================================
        if battery_id == 'B0018':
            logger.info("  > Generating B0018 Safety Plot...")
            
            # Align Bayesian to sequence length for plotting consistency 
            # (LSTM loses first SEQ_LENGTH points)
            # We just take the tail of Bayesian to match LSTM length for the plot
            bayes_metrics_plot = {
                'hdi_low': b_low[-len(lstm_mean):],
                'hdi_high': b_high[-len(lstm_mean):]
            }
            
            # Plot
            plot_path = f"results/final_comparison_{battery_id}.png"
            plot_safety_comparison(
                cycles=np.arange(len(lstm_mean)),
                gt_rul=y_test_seq.flatten(),
                lstm_preds=lstm_mean,
                bayes_metrics=bayes_metrics_plot,
                battery_id=battery_id,
                save_path=plot_path
            )
            
    return results

def main():
    try:
        config = load_config()
        loader = BatteryDataLoader(data_dir=str(config.dataset.path))
        # Restricted set for demo speed, expand for full paper
        battery_ids = ['B0005', 'B0006', 'B0007', 'B0018'] 
        df = loader.load_data(battery_ids)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return

    logo = LeaveOneGroupOut()
    groups = df['battery_id'].values
    X_dummy = np.zeros(len(df)) # Placeholder
    
    all_results = []
    
    logger.info("Starting Rigorous Evaluation Loop...")
    
    fold = 0
    for train_idx, test_idx in logo.split(X_dummy, groups=groups):
        fold += 1
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        test_battery = test_df['battery_id'].iloc[0]
        
        logger.info(f"Fold {fold}: Testing on {test_battery}")
        
        fold_res = evaluate_fold(
            train_df, test_df, test_battery, 
            features=['discharge_time', 'max_temp'], 
            target='rul',
            config=config
        )
        all_results.extend(fold_res)

    # Save Aggregate Results
    res_df = pd.DataFrame(all_results)
    out_dir = Path("results/rigor")
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "metrics_final.csv", index=False)
    
    logger.info("Evaluation Complete. Summary:")
    print(res_df.groupby("Model")[["RMSE", "HDI_Coverage"]].mean())

if __name__ == "__main__":
    main()

