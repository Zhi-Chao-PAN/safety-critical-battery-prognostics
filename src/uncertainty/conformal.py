"""
War Zone 5: Conformal Prediction for Guaranteed Safety Bounds

Provides distribution-free, mathematically guaranteed prediction intervals
for RUL/Capacity forecasting. Unlike Bayesian or MC Dropout methods,
conformal prediction offers finite-sample coverage guarantees:

    P(Y_true ∈ [Y_lower, Y_upper]) >= 1 - alpha

This directly addresses the expert critique:
"未报告置信区间，结果可靠性存疑"
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

class SplitConformalPredictor:
    """
    Split Conformal Prediction for regression.
    
    Algorithm:
    1. Train model on training set.
    2. Compute nonconformity scores on calibration set: |y_true - y_pred|
    3. At inference, construct prediction interval using the (1-alpha) quantile
       of calibration scores.
    
    Mathematical Guarantee:
        For any distribution, with probability >= 1-alpha:
        Y_new ∈ [f(X_new) - q, f(X_new) + q]
        where q is the (1-alpha)(1+1/n) quantile of calibration residuals.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Miscoverage rate. Default 0.05 = 95% coverage guarantee.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.calibration_scores: np.ndarray = np.array([])
        self.q_hat: float = float('inf')
        self.is_calibrated: bool = False

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calibrate the conformal predictor on a held-out calibration set.
        
        Args:
            y_true: Ground truth values. Shape: [n_calibration]
            y_pred: Model predictions. Shape: [n_calibration]
            
        Returns:
            q_hat: The conformal quantile threshold.
        """
        n = len(y_true)
        if n < 2:
            raise ValueError(f"Need at least 2 calibration samples, got {n}")

        # Nonconformity score = absolute residual
        self.calibration_scores = np.abs(y_true - y_pred)

        # Compute the adjusted quantile level for finite-sample coverage
        # ceil((n+1)(1-alpha)) / n gives exact finite-sample guarantee
        quantile_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_hat = float(np.quantile(self.calibration_scores, quantile_level))
        self.is_calibrated = True

        logger.info(f"Conformal calibration complete: n={n}, alpha={self.alpha}, "
                     f"q_hat={self.q_hat:.6f}, coverage_target={1-self.alpha:.0%}")
        return self.q_hat

    def predict(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct prediction intervals.
        
        Args:
            y_pred: Point predictions from the model. Shape: [n_test]
            
        Returns:
            (lower_bounds, upper_bounds): Both shape [n_test]
        """
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() before predict()")

        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat

        # Physical clamp: capacity cannot be negative
        lower = np.maximum(lower, 0.0)

        return lower, upper

    def evaluate_coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Evaluate the empirical coverage on a test set.
        
        Returns:
            Dictionary with coverage metrics.
        """
        lower, upper = self.predict(y_pred)
        covered = (y_true >= lower) & (y_true <= upper)
        empirical_coverage = np.mean(covered)
        avg_width = np.mean(upper - lower)

        result = {
            "target_coverage": 1 - self.alpha,
            "empirical_coverage": float(empirical_coverage),
            "coverage_gap": float(empirical_coverage - (1 - self.alpha)),
            "average_interval_width": float(avg_width),
            "q_hat": self.q_hat,
            "n_test": len(y_true),
            "guarantee_met": bool(empirical_coverage >= (1 - self.alpha) - 0.01)
        }

        logger.info(f"Conformal Coverage: {empirical_coverage:.2%} "
                     f"(target: {1-self.alpha:.2%}) | Width: {avg_width:.4f}")
        return result


class QuantileRegressionLoss(torch.nn.Module):
    """
    Pinball loss for training quantile regression models.
    
    Can be used to train separate lower/upper quantile heads
    for conformalized quantile regression (CQR), which gives
    adaptively-sized intervals.
    """

    def __init__(self, quantile: float):
        super().__init__()
        if not 0 < quantile < 1:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.quantile = quantile

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        residual = y_true - y_pred
        loss = torch.max(
            self.quantile * residual,
            (self.quantile - 1) * residual
        )
        return loss.mean()


class ConformizedQuantileRegression:
    """
    CQR: Combines quantile regression with conformal calibration
    for adaptive-width, guaranteed-coverage intervals.
    
    Unlike vanilla split conformal (fixed-width intervals),
    CQR produces narrower intervals where the model is confident
    and wider intervals where it is uncertain.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat: float = 0.0
        self.is_calibrated: bool = False

    def calibrate(self,
                  y_true: np.ndarray,
                  y_pred_lower: np.ndarray,
                  y_pred_upper: np.ndarray) -> float:
        """
        Calibrate using quantile regression outputs on calibration set.
        
        Args:
            y_true: Ground truth. Shape [n_cal]
            y_pred_lower: Lower quantile predictions. Shape [n_cal]
            y_pred_upper: Upper quantile predictions. Shape [n_cal]
        """
        n = len(y_true)
        # CQR nonconformity score = max(lower - y, y - upper)
        scores = np.maximum(y_pred_lower - y_true, y_true - y_pred_upper)

        quantile_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, quantile_level))
        self.is_calibrated = True

        logger.info(f"CQR calibration: n={n}, q_hat={self.q_hat:.6f}")
        return self.q_hat

    def predict(self,
                y_pred_lower: np.ndarray,
                y_pred_upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Construct conformalized adaptive intervals."""
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() first")

        lower = y_pred_lower - self.q_hat
        upper = y_pred_upper + self.q_hat
        lower = np.maximum(lower, 0.0)
        return lower, upper
