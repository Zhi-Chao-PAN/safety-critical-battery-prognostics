"""
Data Validator - Physical bounds checking and quality assurance.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Summary of data quality checks."""
    total_rows: int = 0
    flagged_rows: int = 0
    issues: list[dict] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return 1.0 - (self.flagged_rows / max(self.total_rows, 1))

    def summary(self) -> str:
        lines = [f"Validation: {self.total_rows} rows, {self.flagged_rows} flagged ({self.pass_rate:.1%} pass)"]
        for issue in self.issues[:10]:
            lines.append(f"  - {issue['type']}: {issue['count']} rows ({issue['desc']})")
        return "\n".join(lines)


class DataValidator:
    """Validate battery data against physical constraints."""

    # Physical bounds
    CAPACITY_MIN, CAPACITY_MAX = 0.0, 10.0  # Ah
    TEMP_MIN, TEMP_MAX = -40.0, 100.0  # Celsius
    VOLTAGE_MIN, VOLTAGE_MAX = 0.0, 5.0  # Volts
    DISCHARGE_TIME_MIN = 0.0  # seconds

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
        """
        Validate DataFrame, flag anomalies, return cleaned df + report.
        Flags but does NOT remove rows (preserving data integrity).
        """
        report = ValidationReport(total_rows=len(df))
        df = df.copy()
        df["_valid"] = True

        # Capacity bounds
        mask = (df["capacity"] < self.CAPACITY_MIN) | (df["capacity"] > self.CAPACITY_MAX)
        if mask.any():
            df.loc[mask, "_valid"] = False
            report.issues.append({
                "type": "capacity_bounds",
                "count": int(mask.sum()),
                "desc": f"Capacity outside [{self.CAPACITY_MIN}, {self.CAPACITY_MAX}] Ah",
            })

        # Temperature bounds
        if "max_temp" in df.columns:
            mask = (df["max_temp"] < self.TEMP_MIN) | (df["max_temp"] > self.TEMP_MAX)
            if mask.any():
                df.loc[mask, "_valid"] = False
                report.issues.append({
                    "type": "temp_bounds",
                    "count": int(mask.sum()),
                    "desc": f"Temperature outside [{self.TEMP_MIN}, {self.TEMP_MAX}] C",
                })

        # Voltage bounds
        if "end_discharge_voltage" in df.columns:
            mask = (df["end_discharge_voltage"] < self.VOLTAGE_MIN) | (
                df["end_discharge_voltage"] > self.VOLTAGE_MAX
            )
            if mask.any():
                df.loc[mask, "_valid"] = False
                report.issues.append({
                    "type": "voltage_bounds",
                    "count": int(mask.sum()),
                    "desc": f"Voltage outside [{self.VOLTAGE_MIN}, {self.VOLTAGE_MAX}] V",
                })

        # Discharge time
        if "discharge_time" in df.columns:
            mask = df["discharge_time"] < self.DISCHARGE_TIME_MIN
            if mask.any():
                df.loc[mask, "_valid"] = False
                report.issues.append({
                    "type": "discharge_time",
                    "count": int(mask.sum()),
                    "desc": "Negative discharge time",
                })

        # RUL non-negative
        if "rul" in df.columns:
            mask = df["rul"] < 0
            if mask.any():
                df.loc[mask, "_valid"] = False
                report.issues.append({
                    "type": "rul_negative",
                    "count": int(mask.sum()),
                    "desc": "Negative RUL",
                })

        # Missing values
        key_cols = ["capacity", "cycle", "battery_id"]
        for col in key_cols:
            if col in df.columns:
                mask = df[col].isna()
                if mask.any():
                    df.loc[mask, "_valid"] = False
                    report.issues.append({
                        "type": f"missing_{col}",
                        "count": int(mask.sum()),
                        "desc": f"Missing values in {col}",
                    })

        # Capacity monotonicity check (per battery)
        non_mono_count = 0
        for bat_id in df["battery_id"].unique():
            sub = df[df["battery_id"] == bat_id].sort_values("cycle")
            cap = sub["capacity"].values
            # Allow small increases (regeneration) but flag large jumps
            diffs = np.diff(cap)
            large_increases = np.sum(diffs > 0.1)  # > 0.1 Ah jump
            non_mono_count += int(large_increases)

        if non_mono_count > 0:
            report.issues.append({
                "type": "capacity_non_monotonic",
                "count": non_mono_count,
                "desc": "Large capacity increases (>0.1 Ah) between cycles (regeneration or sensor drift)",
            })

        report.flagged_rows = int((~df["_valid"]).sum())
        logger.info(report.summary())
        return df, report
