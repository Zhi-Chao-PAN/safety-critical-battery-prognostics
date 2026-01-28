# src/evaluate_rigor.py
"""
Rigorous Cross-Validation Evaluation Script.

This module implements a strict evaluation protocol with:
    - 5-Fold Stratified Cross-Validation (stratified by spatial cluster)
    - 3 Random Seeds per fold (15 total runs)
    - Mean ± Std reporting for statistical significance

This approach ensures that reported performance differences are not
due to lucky data splits or random initialization.
"""

from typing import Type, Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from utils.schema import load_schema
from train_nn_baseline import train_evaluate_mlp


def evaluate_linear_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_class: Type = LinearRegression,
    **kwargs: Any
) -> float:
    """
    Train and evaluate a scikit-learn linear model.
    
    Handles standardization internally to prevent data leakage.
    
    Args:
        X_train: Training features of shape (n_train, n_features).
        y_train: Training targets of shape (n_train, 1).
        X_test: Test features of shape (n_test, n_features).
        y_test: Test targets of shape (n_test, 1).
        model_class: Scikit-learn estimator class (default: LinearRegression).
        **kwargs: Additional arguments passed to model constructor.
        
    Returns:
        Root Mean Squared Error (RMSE) on the test set.
    """
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    
    model = model_class(**kwargs)
    model.fit(X_train_s, y_train_s)
    
    y_pred_s = model.predict(X_test_s)
    y_pred = scaler_y.inverse_transform(y_pred_s)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse


def compute_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute confidence interval for a sample.
    
    Args:
        values: Array of sample values.
        confidence: Confidence level (default: 0.95 for 95% CI).
        
    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    n = len(values)
    mean = np.mean(values)
    std = np.std(values)
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    margin = z * (std / np.sqrt(n))
    return (mean - margin, mean + margin)


def main() -> None:
    """
    Run rigorous cross-validation evaluation.
    
    Evaluates Linear Regression and PyTorch MLP across 5 folds × 3 seeds,
    computing mean ± std RMSE with 95% confidence intervals.
    """
    schema = load_schema()
    data_path: str = schema["dataset"]["path"]
    target_str: str = schema["target"]["name"]
    group_var: str = schema["bayesian"]["hierarchical"]["group"]
    features: List[str] = schema["features"]["numeric"]
    
    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)
    X = df[features].values
    y = df[target_str].values.reshape(-1, 1)
    
    # We use StratifiedKFold based on spatial clusters to ensure
    # each fold has a representative mix of neighborhoods
    groups = df[group_var].values
    # StratifiedKFold requires discrete y, but we use it with groups (y=groups) 
    # to stratify by cluster.
    
    SEEDS: List[int] = [42, 101, 2024]
    N_SPLITS: int = 5
    
    results: List[Dict[str, Any]] = []
    
    print(f"Starting Rigorous Evaluation: {N_SPLITS}-Fold CV x {len(SEEDS)} Seeds")
    print("=" * 60)
    
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        
        # Note: StratifiedKFold.split(X, y) - using groups as y for stratification
        fold_idx = 0
        for train_idx, test_idx in skf.split(X, groups):
            fold_idx += 1
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 1. Linear Regression (Baseline)
            rmse_linear = evaluate_linear_model(X_train, y_train, X_test, y_test)
            results.append({
                "Model": "Linear Regression",
                "Seed": seed,
                "Fold": fold_idx,
                "RMSE": rmse_linear
            })
            
            # 2. PyTorch MLP
            # No verbose to reduce clutter
            rmse_mlp = train_evaluate_mlp(X_train, y_train, X_test, y_test, 
                                          input_dim=len(features), 
                                          seed=seed, 
                                          verbose=False)
            results.append({
                "Model": "PyTorch MLP",
                "Seed": seed,
                "Fold": fold_idx,
                "RMSE": rmse_mlp
            })
            
            print(f"Seed {seed} | Fold {fold_idx}: Linear={rmse_linear:.4f}, MLP={rmse_mlp:.4f}")
            
    # Compile Results
    results_df = pd.DataFrame(results)
    
    # Summary Table with Confidence Intervals
    print("\n" + "=" * 60)
    print("FINAL RIGOROUS RESULTS (Mean ± Std)")
    print("=" * 60)
    
    summary_data = []
    for model in results_df["Model"].unique():
        model_rmse = results_df[results_df["Model"] == model]["RMSE"].values
        mean = np.mean(model_rmse)
        std = np.std(model_rmse)
        ci_low, ci_high = compute_confidence_interval(model_rmse)
        
        summary_data.append({
            "Model": model,
            "RMSE_Mean": mean,
            "RMSE_Std": std,
            "95%_CI_Low": ci_low,
            "95%_CI_High": ci_high,
            "n_runs": len(model_rmse)
        })
        
        print(f"{model:20s}: {mean:.4f} ± {std:.4f}  (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
    
    summary = pd.DataFrame(summary_data)
    
    # Save
    out_dir = Path("results/rigor")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "raw_cv_results.csv", index=False)
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    print(f"\nSaved detailed results to {out_dir}")

if __name__ == "__main__":
    main()
