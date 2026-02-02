# src/evaluate_rigor.py
"""
Rigorous Cross-Validation Evaluation Script for Battery RUL.

Protocol:
    - Leave-One-Group-Out CV (Grouped by Battery_ID)
    - Compares Linear Regression (Instantaneous) vs LSTM (Sequence)
    - Generates "Safety Buffer" plots using standardized visualization
"""

from typing import Type, Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.metrics import get_comprehensive_metrics
from src.data_loader import BatteryDataLoader
from src.train_nn_baseline import train_evaluate_lstm, create_sequences
from src.utils.visualization import plot_safety_comparison

logger = setup_logger(__name__)

def evaluate_linear_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_class: Type = LinearRegression,
    **kwargs: Any
) -> Dict[str, float]:
    """Train and evaluate scikit-learn linear model."""
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)
    
    # Scale Y for consistency
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    
    model = model_class(**kwargs)
    model.fit(X_train_s, y_train_s)
    
    # Predict mean
    y_pred_s = model.predict(X_test_s)
    y_pred = scaler_y.inverse_transform(y_pred_s)
    
    # Estimate uncertainty (Analytical approximation for Linear Regression)
    # Variance of prediction = sigma^2 * (1 + x_0^T (X^T X)^-1 x_0)
    # Simplified: Get residual std from training
    y_train_pred_s = model.predict(X_train_s)
    train_residuals = y_train_s - y_train_pred_s
    sigma = np.std(train_residuals) # Std in scaled space
    
    # Scale sigma back to original space? 
    sigma_orig = sigma * scaler_y.scale_[0]
    
    # Deterministic prediction (constant sigma)
    y_std = np.full_like(y_pred, sigma_orig)
    
    # 95% CI
    y_lower = y_pred - 1.96 * y_std
    y_upper = y_pred + 1.96 * y_std
    
    metrics = get_comprehensive_metrics(
        y_test.flatten(), 
        y_pred.flatten(), 
        y_std.flatten(), 
        y_lower.flatten(), 
        y_upper.flatten()
    )
    
    return metrics


def main() -> None:
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Config failed: {e}")
        return

    data_path = config.dataset.path
    target_str = config.target.name
    group_var = config.group.name
    features = config.features.numeric
    
    logger.info(f"Loading data from: {data_path}")
    
    # USE NEW CLASS-BASED LOADER
    loader = BatteryDataLoader(data_dir=str(data_path))
    # Using 'B0005', 'B0006', 'B0007', 'B0018' typically
    battery_ids = ['B0005', 'B0006', 'B0007', 'B0018'] 
    
    try:
        df = loader.load_data(battery_ids)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return
    
    # Groups
    groups = df[group_var].values
    logo = LeaveOneGroupOut()
    
    results: List[Dict[str, Any]] = []
    
    logger.info(f"Starting Evaluation: Leave-One-Battery-Out CV")
    
    fold_idx = 0
    # Create dummy X for splitter
    X_dummy = np.zeros((len(df), 1)) 
    
    for train_idx, test_idx in logo.split(X_dummy, groups=groups):
        fold_idx += 1
        
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        test_battery = test_df[group_var].iloc[0]
        logger.info(f"Fold {fold_idx}: Testing on Battery {test_battery}")
        
        # 1. Linear Regression (Instantaneous)
        X_train_lin = train_df[features].values
        y_train_lin = train_df[target_str].values.reshape(-1, 1)
        X_test_lin = test_df[features].values
        y_test_lin = test_df[target_str].values.reshape(-1, 1)
        
        metrics_lin = evaluate_linear_model(X_train_lin, y_train_lin, X_test_lin, y_test_lin)
        
        res_lin = {"Model": "Linear Regression", "Battery": test_battery}
        res_lin.update(metrics_lin)
        results.append(res_lin)
        
        # 2. LSTM (Sequence)
        SEQ_LENGTH = 30
        # Use new vectorized create_sequences
        X_train_seq, y_train_seq = create_sequences(train_df, SEQ_LENGTH, features, target_str)
        X_test_seq, y_test_seq = create_sequences(test_df, SEQ_LENGTH, features, target_str)
        
        if len(X_test_seq) == 0:
            logger.warning(f"Not enough data for sequence length {SEQ_LENGTH} in battery {test_battery}")
            continue

        # Train & Eval (Get Model)
        rmse_lstm, model = train_evaluate_lstm(
            X_train_seq, y_train_seq, X_test_seq, y_test_seq,
            input_dim=len(features),
            seq_length=SEQ_LENGTH,
            verbose=False
        )
        
        # MC Dropout Uncertainty Quantification
        # Convert Test Data to Tensor
        device = next(model.parameters()).device
        X_test_t = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
        
        mean_pred, std_pred = model.predict_with_uncertainty(X_test_t, n_samples=50)
        mean_pred = mean_pred.cpu().numpy().flatten()
        std_pred = std_pred.cpu().numpy().flatten()
        
        # Calculate NLL (Negative Log Likelihood) or just log the uncertainty
        avg_uncertainty = np.mean(std_pred)
        
        res_lstm = {
            "Model": "LSTM (MC Dropout)", 
            "Battery": test_battery, 
            "RMSE": rmse_lstm,
            "Uncertainty_Sigma": avg_uncertainty
        }
        results.append(res_lstm)
        
        logger.info(f"  Linear RMSE: {metrics_lin['RMSE']:.4f}")
        logger.info(f"  LSTM RMSE:   {rmse_lstm:.4f} (Sigma: {avg_uncertainty:.4f})")

        # Visualization for B0018 (Safety Case)
        if test_battery == 'B0018':
             print("Generating Safety Buffer Plot for B0018...")
             # Re-create full predictions for plot (simplistic, just to show integration)
             # NOTE: In a real run, we'd get these from the train_evaluate_lstm return,
             # but to keep signature clean, we effectively "re-run" or just skip precise alignment here.
             # For the sake of the report, we usually load the PRE-TRAINED model or run standalone.
             # Here we just log that we covered it.
             # Generate full predictions for visualization
             y_test_flat = y_test_seq.flatten()
             
             # Calculate 95% CI (HDI approximation) from MC Dropout
             # y_mean +/- 1.96 * y_std
             hdi_low = mean_pred - 1.96 * std_pred
             hdi_high = mean_pred + 1.96 * std_pred
             
             bayes_metrics_proxy = {
                 'hdi_low': hdi_low,
                 'hdi_high': hdi_high
             }
             
             # Create Cycles array
             cycles = np.arange(len(y_test_flat))
             
             # Save Plot
             out_plot_path = str(Path("results/rigor/comparison_B0018_generated.png")) # ensure string
             plot_safety_comparison(
                 cycles=cycles,
                 gt_rul=y_test_flat,
                 lstm_preds=mean_pred, # Using mean prediction as the "LSTM" line
                 bayes_metrics=bayes_metrics_proxy, # Passing MC Uncertainty as Bayesian Buffer
                 battery_id=test_battery,
                 save_path=out_plot_path
             )
             logger.info(f"Generated comparison plot: {out_plot_path}")

    # Results Summary
    results_df = pd.DataFrame(results)
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("\n" + str(results_df.groupby("Model")[["RMSE"]].mean(numeric_only=True)))
    
    out_dir = Path("results/rigor")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "cv_results_rigorous.csv", index=False)
    logger.info(f"\nSaved results to {out_dir}")

if __name__ == "__main__":
    main()
