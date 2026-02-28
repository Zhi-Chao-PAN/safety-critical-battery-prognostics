"""
Uncertainty Scoring - Proper scoring rules for probabilistic predictions.

Metrics: CRPS, NLL, Interval Score, PICP, MPIW, Brier Score.
"""

import numpy as np
import scipy.stats as stats


def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    CRPS for Gaussian predictive distribution.
    Primary metric for probabilistic forecast quality.
    Lower is better.
    """
    sigma = np.maximum(sigma, 1e-6)
    z = (y_true - mu) / sigma
    crps = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def nll_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Negative Log-Likelihood under Gaussian assumption. Lower is better."""
    sigma = np.maximum(sigma, 1e-6)
    nll = -stats.norm.logpdf(y_true, loc=mu, scale=sigma)
    return float(np.mean(nll))


def interval_score(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.05
) -> float:
    """
    Interval Score - Penalizes both width and miscoverage.
    alpha=0.05 for 95% prediction interval. Lower is better.
    """
    width = upper - lower
    penalty_low = (2 / alpha) * np.maximum(lower - y_true, 0)
    penalty_high = (2 / alpha) * np.maximum(y_true - upper, 0)
    return float(np.mean(width + penalty_low + penalty_high))


def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability. Target: ~0.95 for 95% CI."""
    within = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(within))


def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean Prediction Interval Width. Lower is better (if PICP is adequate)."""
    return float(np.mean(upper - lower))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Excludes zeros."""
    mask = np.abs(y_true) > 1e-6
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / max(ss_tot, 1e-6))


def compute_all_metrics(
    y_true: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    """Compute full metrics suite."""
    std = (upper - lower) / 3.92
    std = np.maximum(std, 1e-6)

    return {
        "RMSE": rmse(y_true, mean),
        "MAE": mae(y_true, mean),
        "MAPE": mape(y_true, mean),
        "R2": r_squared(y_true, mean),
        "NLL": nll_gaussian(y_true, mean, std),
        "CRPS": crps_gaussian(y_true, mean, std),
        "PICP": picp(y_true, lower, upper),
        "MPIW": mpiw(lower, upper),
        "IS": interval_score(y_true, lower, upper),
    }
