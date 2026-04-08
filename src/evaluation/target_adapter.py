"""
Helpers for aligning model prediction semantics with evaluation targets.

This module centralizes two decisions:
1. Which target column a model should train on.
2. How to adapt model outputs to the requested evaluation target.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.capacity_to_rul import capacity_trajectory_to_rul_series
from src.models.base import BatteryModel

SUPPORTED_TARGETS = {"capacity", "rul"}


def get_prediction_target(model: BatteryModel) -> str:
    """Return the semantic target produced by the model."""
    target = getattr(model, "prediction_target", "rul")
    if target not in SUPPORTED_TARGETS:
        raise ValueError(
            f"Unsupported prediction_target={target!r}. "
            f"Expected one of {sorted(SUPPORTED_TARGETS)}."
        )
    return target


def sort_battery_frame(
    df: pd.DataFrame,
    group_col: str = "battery_id",
    cycle_col: str = "cycle",
) -> pd.DataFrame:
    """Sort a battery dataframe deterministically for training and evaluation."""
    sort_cols = [c for c in (group_col, cycle_col) if c in df.columns]
    if not sort_cols:
        return df.reset_index(drop=True).copy()
    return df.sort_values(sort_cols).reset_index(drop=True).copy()


def build_training_data(
    df: pd.DataFrame,
    features: list[str],
    model: BatteryModel,
    group_col: str = "battery_id",
    cycle_col: str = "cycle",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Build training arrays using the model's declared prediction target.
    """
    ordered_df = sort_battery_frame(df, group_col=group_col, cycle_col=cycle_col)
    target_col = get_prediction_target(model)

    if target_col not in ordered_df.columns:
        raise ValueError(f"Training target column {target_col!r} is missing from the dataframe")

    fit_kwargs: dict[str, Any] = {}
    if group_col in ordered_df.columns:
        fit_kwargs["group_ids"] = ordered_df[group_col].values

    return (
        ordered_df,
        ordered_df[features].values,
        ordered_df[target_col].values,
        fit_kwargs,
    )


def build_prediction_data(
    df: pd.DataFrame,
    features: list[str],
    group_col: str = "battery_id",
    cycle_col: str = "cycle",
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Build prediction arrays and optional group-aware kwargs."""
    ordered_df = sort_battery_frame(df, group_col=group_col, cycle_col=cycle_col)
    predict_kwargs: dict[str, Any] = {}
    if group_col in ordered_df.columns:
        predict_kwargs["group_ids"] = ordered_df[group_col].values
    return ordered_df, ordered_df[features].values, predict_kwargs


def _align_frame_to_predictions(
    model: BatteryModel,
    ordered_df: pd.DataFrame,
    prediction_length: int,
    group_col: str,
    cycle_col: str,
) -> pd.DataFrame:
    """Align rows in the evaluation dataframe to the model output length."""
    if prediction_length == len(ordered_df):
        return ordered_df.copy()

    seq_length = getattr(model, "seq_length", None)
    if seq_length is not None and group_col in ordered_df.columns:
        aligned_parts = []
        for _, group in ordered_df.groupby(group_col, sort=False):
            if len(group) <= seq_length:
                continue
            aligned_parts.append(group.iloc[seq_length:])

        if aligned_parts:
            aligned_df = pd.concat(aligned_parts, ignore_index=True)
            if len(aligned_df) == prediction_length:
                return aligned_df

    if prediction_length < len(ordered_df):
        return ordered_df.iloc[-prediction_length:].reset_index(drop=True)

    raise ValueError(
        f"Prediction length ({prediction_length}) cannot be aligned to dataframe length ({len(ordered_df)})."
    )


def _capacity_to_grouped_rul(
    df: pd.DataFrame,
    predicted_capacity: np.ndarray,
    group_col: str,
    cycle_col: str,
    eol_threshold: float,
) -> np.ndarray:
    """Convert per-cycle capacity predictions to per-cycle RUL within each battery."""
    working = df[[group_col, cycle_col]].copy() if group_col in df.columns else df[[cycle_col]].copy()
    working["prediction"] = predicted_capacity

    rul_parts: list[np.ndarray] = []
    if group_col in working.columns:
        group_iter = working.groupby(group_col, sort=False)
    else:
        group_iter = [(None, working)]

    for _, group in group_iter:
        rul_parts.append(
            capacity_trajectory_to_rul_series(
                predicted_capacity=group["prediction"].values,
                cycles=group[cycle_col].values,
                eol_threshold=eol_threshold,
            )
        )

    if not rul_parts:
        return np.array([], dtype=np.float64)
    return np.concatenate(rul_parts)


def adapt_predictions_to_target(
    model: BatteryModel,
    test_df: pd.DataFrame,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    evaluation_target: str,
    group_col: str = "battery_id",
    cycle_col: str = "cycle",
    eol_threshold: float = 1.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Adapt raw model predictions to the requested evaluation target.

    Returns:
        y_eval, mean_eval, lower_eval, upper_eval, aligned_df
    """
    if evaluation_target not in SUPPORTED_TARGETS:
        raise ValueError(
            f"Unsupported evaluation_target={evaluation_target!r}. "
            f"Expected one of {sorted(SUPPORTED_TARGETS)}."
        )

    ordered_df = sort_battery_frame(test_df, group_col=group_col, cycle_col=cycle_col)
    aligned_df = _align_frame_to_predictions(
        model,
        ordered_df,
        prediction_length=len(mean),
        group_col=group_col,
        cycle_col=cycle_col,
    )

    prediction_target = get_prediction_target(model)
    if prediction_target == evaluation_target:
        if evaluation_target not in aligned_df.columns:
            raise ValueError(f"Evaluation target column {evaluation_target!r} is missing from the dataframe")
        y_eval = aligned_df[evaluation_target].values
        return y_eval, mean, lower, upper, aligned_df

    if prediction_target == "capacity" and evaluation_target == "rul":
        required_cols = {cycle_col, "rul"}
        missing = [col for col in required_cols if col not in aligned_df.columns]
        if missing:
            raise ValueError(
                "Capacity-to-RUL evaluation requires cycle and rul columns. "
                f"Missing: {missing}"
            )

        y_eval = aligned_df["rul"].values
        mean_eval = _capacity_to_grouped_rul(aligned_df, mean, group_col, cycle_col, eol_threshold)
        lower_eval = _capacity_to_grouped_rul(aligned_df, lower, group_col, cycle_col, eol_threshold)
        upper_eval = _capacity_to_grouped_rul(aligned_df, upper, group_col, cycle_col, eol_threshold)
        return y_eval, mean_eval, lower_eval, upper_eval, aligned_df

    raise ValueError(
        f"Cannot evaluate model target {prediction_target!r} against {evaluation_target!r} without an adapter."
    )
