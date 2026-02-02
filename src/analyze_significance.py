
import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def analyze_significance():
    """
    Perform statistical tests to compare LSTM vs Linear/Bayesian models.
    """
    results_path = Path("results/rigor/cv_results_rigorous.csv")
    
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return
        
    df = pd.read_csv(results_path)
    
    # 1. Compare LSTM vs Linear Regression (RMSE)
    # We want to show LSTM is significantly better (Lower RMSE)??
    # Or maybe Linear is better? The goal is to show LSTM is "Standard" but Bayesian is "Safe".
    
    lstm_rmse = df[df["Model"] == "LSTM"]["RMSE"].values
    lin_rmse = df[df["Model"] == "Linear Regression"]["RMSE"].values
    
    if len(lstm_rmse) == len(lin_rmse):
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(lstm_rmse, lin_rmse)
        
        logger.info("=== Statistical Significance Test (RMSE) ===")
        logger.info(f"LSTM Mean RMSE: {np.mean(lstm_rmse):.4f}")
        logger.info(f"Linear Mean RMSE: {np.mean(lin_rmse):.4f}")
        logger.info(f"Paired t-test: t={t_stat:.4f}, p={p_val:.4f}")
        
        if p_val < 0.05:
            logger.info("Result: Significant difference detected.")
        else:
            logger.info("Result: No significant difference detected.")
            
    # 2. Compare Safety Metrics (e.g. NLL) if we had a Bayesian Benchmark in the CSV
    # Currently evaluate_rigor calculates NLL for Linear (Analytical) vs LSTM (Proxy/None).
    # If Linear has better NLL, it supports "Probabilistic is better".
    
    lin_nll = df[df["Model"] == "Linear Regression"]["NLL"].values
    # LSTM NLL might be NaN or proxy. If proxy, we compare.
    
    logger.info("\n=== Analysis Complete ===")

if __name__ == "__main__":
    analyze_significance()
