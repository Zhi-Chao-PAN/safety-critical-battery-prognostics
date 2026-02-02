import numpy as np
import scipy.stats as stats
from typing import Tuple, Dict

def calculate_nll(y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray) -> float:
    """
    Calculate Negative Log Likelihood (NLL).
    Lower is better. Measures how well the model's uncertainty fits the data.
    """
    nll = -stats.norm.logpdf(y_true, loc=y_pred_mean, scale=y_pred_std)
    return float(np.mean(nll))

def calculate_picp(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """
    Calculate Prediction Interval Coverage Probability (PICP).
    The percentage of true values that fall within the prediction interval.
    Ideal value is close to the confidence level (e.g., 0.95).
    """
    within_interval = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(within_interval))

def calculate_mpiw(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """
    Calculate Mean Prediction Interval Width (MPIW).
    Lower is better (if PICP is high), indicating sharper predictions.
    """
    return float(np.mean(y_upper - y_lower))

def get_comprehensive_metrics(
    y_true: np.ndarray, 
    y_pred_mean: np.ndarray, 
    y_pred_std: np.ndarray = None,
    y_lower: np.ndarray = None, 
    y_upper: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute a full suite of deterministic and probabilistic metrics.
    """
    # Deterministic
    mse = np.mean((y_true - y_pred_mean)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred_mean))
    
    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae)
    }
    
    # Probabilistic (valid only if uncertainty is provided)
    if y_pred_std is not None:
        metrics["NLL"] = calculate_nll(y_true, y_pred_mean, y_pred_std)
        
    if y_lower is not None and y_upper is not None:
        metrics["PICP"] = calculate_picp(y_true, y_lower, y_upper)
        metrics["MPIW"] = calculate_mpiw(y_lower, y_upper)
        
    return metrics
