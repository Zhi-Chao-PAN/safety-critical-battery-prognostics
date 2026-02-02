# src/evaluate_rigor.py
"""
Rigorous Cross-Validation Evaluation Script for Battery RUL.

Protocol:
    - 4-Fold CV (GroupKFold by Battery_ID) or Leave-One-Group-Out
    - Compares Linear Regression (Instantaneous) vs LSTM (Sequence)
"""

from typing import Type, Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from utils.schema import load_schema
from data_loader import load_battery_data
from train_nn_baseline import train_evaluate_lstm, create_sequences


def evaluate_linear_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_class: Type = LinearRegression,
    **kwargs: Any
) -> float:
    """Train and evaluate scikit-learn linear model."""
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)
    
    # y is already standardized or RUL is large, so scaling y helps convergence usually
    # but LinearRegression solves analytical solution so scaling y affects coefficients but not R2/RMSE rel to scale
    # But let's scale for consistency
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    
    model = model_class(**kwargs)
    model.fit(X_train_s, y_train_s)
    
    y_pred_s = model.predict(X_test_s)
    y_pred = scaler_y.inverse_transform(y_pred_s)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse


def main() -> None:
    schema = load_schema()
    data_path: str = schema["dataset"]["path"]
    target_str: str = schema["target"]["name"]
    group_var: str = schema["group"]["name"]
    features: List[str] = schema["features"]["numeric"]
    
    print(f"Loading data: {data_path}")
    df = load_battery_data(data_path)
    
    # Groups
    groups = df[group_var].values
    logo = LeaveOneGroupOut()
    
    results: List[Dict[str, Any]] = []
    
    print(f"Starting Evaluation: Leave-One-Battery-Out CV")
    
    features_idx = [df.columns.get_loc(c) for c in features]
    
    fold_idx = 0
    # Create X, y for iteration
    # Check if we should iterate on DF or Arrays. DF is easier for create_sequences
    
    # Iterate over splits
    # LeaveOneGroupOut needs numeric groups usually? No, supports strings.
    # We pass groups to split
    
    X_dummy = np.zeros((len(df), 1)) # Dummy X for splitter
    
    for train_idx, test_idx in logo.split(X_dummy, groups=groups):
        fold_idx += 1
        
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        test_battery = test_df[group_var].iloc[0]
        print(f"Fold {fold_idx}: Testing on Battery {test_battery}")
        
        # 1. Linear Regression (Instantaneous)
        X_train_lin = train_df[features].values
        y_train_lin = train_df[target_str].values.reshape(-1, 1)
        X_test_lin = test_df[features].values
        y_test_lin = test_df[target_str].values.reshape(-1, 1)
        
        rmse_linear = evaluate_linear_model(X_train_lin, y_train_lin, X_test_lin, y_test_lin)
        
        results.append({
            "Model": "Linear Regression",
            "Battery": test_battery,
            "RMSE": rmse_linear
        })
        
        # 2. LSTM (Sequence)
        SEQ_LENGTH = 30
        X_train_seq, y_train_seq = create_sequences(train_df, SEQ_LENGTH, features, target_str, group_var)
        X_test_seq, y_test_seq = create_sequences(test_df, SEQ_LENGTH, features, target_str, group_var)
        
        if len(X_test_seq) == 0:
            print(f"Warning: Not enough data for sequence length {SEQ_LENGTH} in battery {test_battery}")
            continue

        rmse_lstm = train_evaluate_lstm(
            X_train_seq, y_train_seq, X_test_seq, y_test_seq,
            input_dim=len(features),
            seq_length=SEQ_LENGTH,
            verbose=False
        )
        
        results.append({
            "Model": "LSTM",
            "Battery": test_battery,
            "RMSE": rmse_lstm
        })
        
        print(f"  Linear RMSE: {rmse_linear:.4f}")
        print(f"  LSTM RMSE:   {rmse_lstm:.4f}")

    # Results Summary
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.groupby("Model")["RMSE"].agg(["mean", "std"]))
    
    out_dir = Path("results/rigor")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "cv_results.csv", index=False)
    print(f"\nSaved results to {out_dir}")

if __name__ == "__main__":
    main()
