"""
Capacity-to-RUL Mapping Engine.

Converts Chronos capacity trajectory predictions into RUL (Remaining Useful
Life) estimates by finding the intersection with an End-of-Life threshold.

This enables fair, same-dimension comparison between:
  - Chronos (predicts future capacity in Ah)
  - BTCN / Bayesian baselines (predict RUL in cycles)

Algorithm:
    Given a predicted capacity trajectory C_pred[t] and an EOL threshold T:
    1. Find the first index k where C_pred[k] < T
    2. Linearly interpolate between C_pred[k-1] and C_pred[k] for sub-cycle precision
    3. RUL_pred = interpolated_crossing_index (relative to prediction start)
    
    If the trajectory never crosses the threshold within the prediction window,
    RUL_pred = prediction_length (right-censored, lower bound estimate).
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RULPrediction:
    """Container for a single battery's RUL prediction from capacity trajectory."""
    battery_id: str
    current_cycle: int
    predicted_rul: float          # Cycles until EOL (from current_cycle)
    actual_rul: float             # Ground truth RUL
    rul_error: float              # predicted - actual
    eol_threshold: float          # Ah
    trajectory_crossed: bool      # Did the predicted trajectory cross EOL?
    predicted_eol_cycle: float    # Absolute cycle number of predicted EOL
    actual_eol_cycle: float       # Absolute cycle number of actual EOL


def find_eol_crossing(
    capacity_trajectory: np.ndarray,
    eol_threshold: float,
) -> tuple[float, bool]:
    """
    Find the fractional index where a capacity trajectory crosses the EOL threshold.

    Uses linear interpolation between adjacent points for sub-cycle precision.
    
    Args:
        capacity_trajectory: 1D array of predicted capacity values (Ah).
        eol_threshold: End-of-Life capacity threshold (Ah).
    
    Returns:
        (crossing_index, did_cross): 
            - crossing_index: Fractional index at which capacity drops below threshold.
            - did_cross: True if the trajectory actually crossed the threshold.
    """
    if len(capacity_trajectory) == 0:
        return 0.0, False

    # Find first index where capacity drops below threshold
    below_mask = capacity_trajectory < eol_threshold

    if not np.any(below_mask):
        # Never crosses: right-censored, return length as lower bound
        return float(len(capacity_trajectory)), False

    first_below_idx = int(np.argmax(below_mask))

    if first_below_idx == 0:
        # Already below threshold at prediction start
        return 0.0, True

    # Linear interpolation for sub-cycle precision
    c_before = capacity_trajectory[first_below_idx - 1]
    c_after = capacity_trajectory[first_below_idx]

    # Prevent division by zero
    denom = c_before - c_after
    if abs(denom) < 1e-10:
        return float(first_below_idx), True

    # Fractional position between [first_below_idx-1, first_below_idx]
    frac = (c_before - eol_threshold) / denom
    crossing_index = (first_below_idx - 1) + frac

    return float(crossing_index), True


def capacity_trajectory_to_rul(
    predicted_trajectory: np.ndarray,
    current_cycle: int,
    eol_threshold: float,
) -> tuple[float, float, bool]:
    """
    Convert a predicted capacity trajectory into a RUL estimate.
    
    Args:
        predicted_trajectory: 1D array of future capacity predictions (Ah).
        current_cycle: The cycle number at which the prediction starts.
        eol_threshold: End-of-Life capacity threshold (Ah).
    
    Returns:
        (predicted_rul, predicted_eol_cycle, crossed):
            - predicted_rul: Estimated remaining cycles until EOL.
            - predicted_eol_cycle: Absolute cycle number of predicted EOL.
            - crossed: Whether the trajectory actually crossed the threshold.
    """
    crossing_idx, crossed = find_eol_crossing(predicted_trajectory, eol_threshold)

    predicted_rul = crossing_idx  # Cycles from prediction start to EOL
    predicted_eol_cycle = current_cycle + predicted_rul

    return predicted_rul, predicted_eol_cycle, crossed


def evaluate_chronos_rul(
    capacity_series: np.ndarray,
    context_length: int,
    predicted_mean: np.ndarray,
    predicted_lower: np.ndarray,
    predicted_upper: np.ndarray,
    battery_id: str,
    eol_threshold: float = 1.4,  # 0.7 * 2.0 Ah rated
) -> dict[str, float]:
    """
    Full RUL evaluation pipeline for a single battery's Chronos predictions.
    
    Takes the ground truth series + Chronos predictions and computes:
    - Predicted RUL (from capacity trajectory crossing)
    - Actual RUL (from ground truth crossing)
    - RUL RMSE contribution
    
    Args:
        capacity_series: Full ground truth capacity array for this battery.
        context_length: Number of cycles used as context (prediction starts after this).
        predicted_mean: Chronos median prediction (future capacity values).
        predicted_lower: Lower bound of 95% CI.
        predicted_upper: Upper bound of 95% CI.
        battery_id: Battery identifier string.
        eol_threshold: EOL threshold in Ah.
    
    Returns:
        Dictionary with RUL metrics.
    """
    prediction_length = len(predicted_mean)
    current_cycle = context_length  # Prediction starts at this cycle index

    # --- Predicted RUL from Chronos trajectory ---
    pred_rul, pred_eol_cycle, pred_crossed = capacity_trajectory_to_rul(
        predicted_mean, current_cycle, eol_threshold,
    )

    # Also compute RUL from lower/upper bounds for uncertainty
    lower_rul, _, lower_crossed = capacity_trajectory_to_rul(
        predicted_lower, current_cycle, eol_threshold,
    )
    upper_rul, _, upper_crossed = capacity_trajectory_to_rul(
        predicted_upper, current_cycle, eol_threshold,
    )

    # --- Actual RUL from ground truth ---
    # Ground truth: remaining capacity after context_length
    remaining_gt = capacity_series[context_length:]
    actual_rul, actual_eol_cycle, actual_crossed = capacity_trajectory_to_rul(
        remaining_gt, current_cycle, eol_threshold,
    )

    # --- Also compute full-series actual EOL for reference ---
    full_eol_crossing, full_crossed = find_eol_crossing(capacity_series, eol_threshold)
    full_actual_eol = full_eol_crossing if full_crossed else float(len(capacity_series))
    full_actual_rul_from_context = max(0.0, full_actual_eol - current_cycle)

    # RUL error
    rul_error = pred_rul - full_actual_rul_from_context

    result = {
        "battery_id": battery_id,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "eol_threshold_ah": eol_threshold,
        # Predicted
        "predicted_rul": round(pred_rul, 2),
        "predicted_eol_cycle": round(pred_eol_cycle, 2),
        "pred_trajectory_crossed": pred_crossed,
        # Predicted bounds
        "predicted_rul_lower": round(lower_rul, 2),
        "predicted_rul_upper": round(upper_rul, 2),
        # Actual
        "actual_rul_from_context": round(full_actual_rul_from_context, 2),
        "actual_eol_cycle": round(full_actual_eol, 2),
        "actual_trajectory_crossed": full_crossed,
        # Error
        "rul_error": round(rul_error, 2),
        "rul_abs_error": round(abs(rul_error), 2),
    }

    logger.info(
        f"  {battery_id}: Pred RUL={pred_rul:.1f} cycles "
        f"(EOL@cycle {pred_eol_cycle:.1f}, crossed={pred_crossed}), "
        f"Actual RUL={full_actual_rul_from_context:.1f} cycles "
        f"(EOL@cycle {full_actual_eol:.1f}), "
        f"Error={rul_error:+.1f} cycles"
    )

    return result


def compute_rul_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    """
    Aggregate per-battery RUL predictions into summary metrics.
    
    Returns:
        Dictionary with RMSE, MAE, and per-battery breakdown.
    """
    if not results:
        return {"rul_rmse": float("nan"), "rul_mae": float("nan")}

    errors = np.array([r["rul_error"] for r in results])
    abs_errors = np.array([r["rul_abs_error"] for r in results])

    rul_rmse = float(np.sqrt(np.mean(errors ** 2)))
    rul_mae = float(np.mean(abs_errors))

    n_crossed = sum(1 for r in results if r["pred_trajectory_crossed"])

    return {
        "rul_rmse": round(rul_rmse, 4),
        "rul_mae": round(rul_mae, 4),
        "n_batteries": len(results),
        "n_crossed_eol": n_crossed,
        "n_censored": len(results) - n_crossed,
    }
