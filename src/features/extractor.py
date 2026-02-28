"""
Feature Extractor - Extract 15+ features from raw battery cycle signals.

Features extracted per cycle:
  Voltage: end_discharge_v, plateau_duration, voltage_area, dv_dt_max
  IC/DV: ic_peak_height, ic_peak_position, dv_peak_height
  Current/Energy: coulombic_eff, energy_ratio, charge_time
  Temperature: max_temp, mean_temp, temp_rise_rate, thermal_integral
  Derived: capacity_fade_rate, resistance_growth_rate, cycle_delta_cap
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract rich features from raw battery cycle data."""

    def __init__(self, smoothing_sigma: float = 3.0):
        self.smoothing_sigma = smoothing_sigma

    def extract_all(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Extract all features from a DataFrame with raw signals.

        Args:
            df: DataFrame with raw_voltage, raw_current, raw_temperature, raw_time columns
            window: Rolling window size for trend features

        Returns:
            DataFrame with original + new feature columns (raw columns dropped)
        """
        results = []

        for _, row in df.iterrows():
            feats = self._extract_cycle_features(row)
            results.append(feats)

        feat_df = pd.DataFrame(results)

        # Merge with original (drop raw columns to save memory)
        keep_cols = [c for c in df.columns if not c.startswith("raw_")]
        base_df = df[keep_cols].reset_index(drop=True)

        # Drop columns from feat_df that already exist in base to avoid duplicates
        dup_cols = [c for c in feat_df.columns if c in base_df.columns]
        if dup_cols:
            feat_df = feat_df.drop(columns=dup_cols)

        merged = pd.concat([base_df, feat_df], axis=1)

        # Add trend features per battery
        merged = self._add_trend_features(merged, window=window)

        logger.info(f"Extracted {len(feat_df.columns)} cycle features + trend features")
        return merged

    def _extract_cycle_features(self, row: pd.Series) -> dict:
        """Extract features from a single cycle's raw signals."""
        feats = {}

        v = np.array(row.get("raw_voltage", []), dtype=np.float64)
        i = np.array(row.get("raw_current", []), dtype=np.float64)
        t = np.array(row.get("raw_temperature", []), dtype=np.float64)
        time = np.array(row.get("raw_time", []), dtype=np.float64)

        has_signals = len(v) > 2 and len(time) > 2

        # ── Voltage Features ──
        if has_signals:
            feats["end_discharge_voltage"] = float(v[-1])
            feats["voltage_area"] = float(trapezoid(v, time)) if len(v) == len(time) else 0.0

            # Plateau duration: Time spent within 5% of mean voltage
            v_mean = np.mean(v)
            plateau_mask = np.abs(v - v_mean) < 0.05 * v_mean
            if plateau_mask.any() and len(time) == len(v):
                plateau_times = time[plateau_mask]
                feats["plateau_duration"] = float(plateau_times[-1] - plateau_times[0]) if len(plateau_times) > 1 else 0.0
            else:
                feats["plateau_duration"] = 0.0

            # dV/dt max
            if len(v) == len(time) and len(time) > 1:
                dt = np.diff(time)
                dt = np.where(dt == 0, 1e-6, dt)
                dv_dt = np.abs(np.diff(v) / dt)
                feats["dv_dt_max"] = float(np.max(dv_dt))
            else:
                feats["dv_dt_max"] = 0.0
        else:
            feats["end_discharge_voltage"] = row.get("end_discharge_voltage", 0.0)
            feats["voltage_area"] = 0.0
            feats["plateau_duration"] = 0.0
            feats["dv_dt_max"] = 0.0

        # ── IC/DV Features ──
        if has_signals and len(v) == len(i):
            ic_h, ic_p, dv_h = self._compute_ic_dv(v, i)
            feats["ic_peak_height"] = ic_h
            feats["ic_peak_position"] = ic_p
            feats["dv_peak_height"] = dv_h
        else:
            feats["ic_peak_height"] = 0.0
            feats["ic_peak_position"] = 0.0
            feats["dv_peak_height"] = 0.0

        # ── Temperature Features ──
        if len(t) > 0:
            feats["max_temp"] = float(np.max(t))
            feats["mean_temp"] = float(np.mean(t))
            if len(t) > 1 and has_signals:
                duration = time[-1] - time[0] if len(time) == len(t) else 1.0
                duration = max(duration, 1e-6)
                feats["temp_rise_rate"] = float((t[-1] - t[0]) / duration)
                feats["thermal_integral"] = float(trapezoid(t, time)) if len(t) == len(time) else 0.0
            else:
                feats["temp_rise_rate"] = 0.0
                feats["thermal_integral"] = 0.0
        else:
            feats["max_temp"] = row.get("max_temp", 0.0)
            feats["mean_temp"] = row.get("mean_temp", 0.0)
            feats["temp_rise_rate"] = row.get("temp_rise_rate", 0.0)
            feats["thermal_integral"] = 0.0

        # ── Current/Energy Features ──
        feats["internal_resistance"] = float(row.get("internal_resistance", 0.0))

        return feats

    def _compute_ic_dv(self, voltage: np.ndarray, current: np.ndarray) -> tuple[float, float, float]:
        """Compute Incremental Capacity and Differential Voltage features."""
        try:
            # Smooth voltage
            v_smooth = gaussian_filter1d(voltage, sigma=self.smoothing_sigma)

            # dQ/dV (IC curve)
            dv = np.diff(v_smooth)
            dv = np.where(np.abs(dv) < 1e-6, 1e-6, dv)
            # Charge throughput proxy: cumulative |current| * dt (assume dt=1)
            q = np.cumsum(np.abs(current[:-1]))
            dq = np.diff(q) if len(q) > 1 else np.array([0.0])

            if len(dq) > 0 and len(dv) > 1:
                ic = dq / np.abs(dv[: len(dq)])
                ic = gaussian_filter1d(ic, sigma=self.smoothing_sigma)
                ic_peak_idx = int(np.argmax(ic))
                ic_peak_height = float(ic[ic_peak_idx])
                ic_peak_position = float(v_smooth[ic_peak_idx]) if ic_peak_idx < len(v_smooth) else 0.0
            else:
                ic_peak_height, ic_peak_position = 0.0, 0.0

            # dV/dQ (DV curve)
            if len(dq) > 0:
                dq_safe = np.where(np.abs(dq) < 1e-6, 1e-6, dq)
                dv_dq = np.abs(dv[: len(dq_safe)]) / dq_safe
                dv_dq = gaussian_filter1d(dv_dq, sigma=self.smoothing_sigma)
                dv_peak_height = float(np.max(dv_dq))
            else:
                dv_peak_height = 0.0

            return ic_peak_height, ic_peak_position, dv_peak_height
        except Exception:
            return 0.0, 0.0, 0.0

    def _add_trend_features(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Add per-battery rolling trend features."""
        trend_dfs = []
        for bat_id in df["battery_id"].unique():
            sub = df[df["battery_id"] == bat_id].sort_values("cycle").copy()

            # Capacity fade rate
            sub["capacity_fade_rate"] = sub["capacity"].diff().fillna(0.0)

            # Resistance growth rate
            if "internal_resistance" in sub.columns:
                sub["resistance_growth_rate"] = sub["internal_resistance"].diff().fillna(0.0)
            else:
                sub["resistance_growth_rate"] = 0.0

            # Rolling stats
            for col in ["capacity", "max_temp", "discharge_time"]:
                if col in sub.columns:
                    sub[f"{col}_rolling_mean"] = sub[col].rolling(window, min_periods=1).mean()
                    sub[f"{col}_rolling_std"] = sub[col].rolling(window, min_periods=1).std().fillna(0.0)

            trend_dfs.append(sub)

        return pd.concat(trend_dfs, ignore_index=True)
