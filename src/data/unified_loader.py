"""
Unified Data Loader for Multi-Dataset Battery Prognostics.

Supports: NASA PCoE, CALCE CS2, Oxford, MIT-Stanford.
All datasets normalized to a common schema.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

logger = logging.getLogger(__name__)


# ── Common Schema ──────────────────────────────────────────
# battery_id, dataset_source, chemistry, cycle, capacity,
# discharge_time, max_temp, mean_temp, temp_rise_rate,
# end_discharge_voltage, internal_resistance, coulombic_efficiency,
# raw_voltage, raw_current, raw_temperature, raw_time, RUL


class UnifiedDataLoader:
    """Load and unify battery datasets into a common DataFrame schema."""

    def __init__(self, rated_capacity: float = 2.0, eol_fraction: float = 0.7):
        self.rated_capacity = rated_capacity
        self.eol_fraction = eol_fraction

    def load_nasa(
        self,
        data_dir: str = "data/battery_data",
        battery_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load NASA PCoE dataset from .mat files."""
        if battery_ids is None:
            battery_ids = ["B0005", "B0006", "B0007", "B0018"]

        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"NASA data directory not found: {data_dir}")

        all_cycles = []
        for bat_id in battery_ids:
            filepath = data_dir / f"{bat_id}.mat"
            if not filepath.exists():
                logger.warning(f"Skipping {bat_id}: file not found at {filepath}")
                continue

            try:
                mat = scipy.io.loadmat(str(filepath))
                cycles = mat[bat_id][0, 0]["cycle"][0]
                cycle_count = 0

                for cycle in cycles:
                    if cycle["type"][0] == "discharge":
                        cycle_count += 1
                        data = cycle["data"]

                        capacity = float(data[0, 0]["Capacity"][0][0])
                        temp = data[0, 0]["Temperature_measured"][0]
                        voltage = data[0, 0]["Voltage_measured"][0]
                        current = data[0, 0]["Current_measured"][0]
                        time_arr = data[0, 0]["Time"][0]

                        # Basic scalar features
                        discharge_time = float(time_arr[-1] - time_arr[0])
                        max_temp = float(np.max(temp))
                        mean_temp = float(np.mean(temp))

                        # Temperature rise rate
                        if len(temp) > 1 and discharge_time > 0:
                            temp_rise = float((temp[-1] - temp[0]) / discharge_time)
                        else:
                            temp_rise = 0.0

                        # End-of-discharge voltage
                        end_v = float(voltage[-1]) if len(voltage) > 0 else 0.0

                        # Internal resistance proxy (dV/dI at start)
                        if len(voltage) > 1 and len(current) > 1:
                            dv = abs(float(voltage[1] - voltage[0]))
                            di = abs(float(current[1] - current[0]))
                            ir = dv / di if di > 1e-6 else 0.0
                        else:
                            ir = 0.0

                        all_cycles.append({
                            "battery_id": bat_id,
                            "dataset_source": "nasa_pcoe",
                            "chemistry": "LiCoO2",
                            "cycle": cycle_count,
                            "capacity": capacity,
                            "discharge_time": discharge_time,
                            "max_temp": max_temp,
                            "mean_temp": mean_temp,
                            "temp_rise_rate": temp_rise,
                            "end_discharge_voltage": end_v,
                            "internal_resistance": ir,
                            "raw_voltage": voltage.tolist(),
                            "raw_current": current.tolist(),
                            "raw_temperature": temp.tolist(),
                            "raw_time": time_arr.tolist(),
                        })
            except Exception as e:
                logger.error(f"Error processing {bat_id}: {e}")
                raise

        if not all_cycles:
            raise ValueError("No valid cycle data extracted from NASA files.")

        df = pd.DataFrame(all_cycles)
        df = self._compute_rul(df)
        logger.info(f"NASA: Loaded {len(df)} cycles from {df['battery_id'].nunique()} batteries")
        return df

    def _compute_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute RUL based on EOL threshold."""
        eol_thresh = self.eol_fraction * self.rated_capacity
        result = []
        for bat_id in df["battery_id"].unique():
            sub = df[df["battery_id"] == bat_id].copy()
            failed = sub[sub["capacity"] < eol_thresh]
            eol = int(failed["cycle"].min()) if not failed.empty else int(sub["cycle"].max())
            sub["rul"] = eol - sub["cycle"]
            result.append(sub[sub["rul"] >= 0])
        return pd.concat(result, ignore_index=True)

    def load_calce(self, data_dir: str = "data/calce") -> pd.DataFrame:
        """
        Load CALCE CS2 dataset from CSV files.
        Expected format: CSV with columns cycle, capacity, voltage, current, temperature.
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"CALCE data directory not found: {data_dir}")

        all_cycles = []
        for csv_file in sorted(data_dir.glob("*.csv")):
            bat_id = csv_file.stem
            try:
                raw = pd.read_csv(csv_file)
                # Normalize column names
                col_map = {}
                for col in raw.columns:
                    cl = col.lower().strip()
                    if "cycle" in cl:
                        col_map[col] = "cycle"
                    elif "capacity" in cl or "cap" in cl:
                        col_map[col] = "capacity"
                    elif "temp" in cl:
                        col_map[col] = "temperature"
                    elif "volt" in cl:
                        col_map[col] = "voltage"
                    elif "current" in cl or "curr" in cl:
                        col_map[col] = "current"
                raw = raw.rename(columns=col_map)

                if "cycle" not in raw.columns or "capacity" not in raw.columns:
                    logger.warning(f"Skipping {bat_id}: missing cycle/capacity columns")
                    continue

                # Aggregate per cycle
                grouped = raw.groupby("cycle").agg(
                    capacity=("capacity", "last"),
                    max_temp=("temperature", "max") if "temperature" in raw.columns else ("capacity", lambda x: 25.0),
                    mean_temp=("temperature", "mean") if "temperature" in raw.columns else ("capacity", lambda x: 25.0),
                ).reset_index()

                for _, row in grouped.iterrows():
                    all_cycles.append({
                        "battery_id": bat_id,
                        "dataset_source": "calce",
                        "chemistry": "LiCoO2",
                        "cycle": int(row["cycle"]),
                        "capacity": float(row["capacity"]),
                        "discharge_time": 0.0,
                        "max_temp": float(row.get("max_temp", 25.0)),
                        "mean_temp": float(row.get("mean_temp", 25.0)),
                        "temp_rise_rate": 0.0,
                        "end_discharge_voltage": 0.0,
                        "internal_resistance": 0.0,
                    })
            except Exception as e:
                logger.error(f"Error loading CALCE {bat_id}: {e}")

        if not all_cycles:
            raise ValueError("No valid CALCE data found.")

        df = pd.DataFrame(all_cycles)
        df = self._compute_rul(df)
        logger.info(f"CALCE: Loaded {len(df)} cycles from {df['battery_id'].nunique()} batteries")
        return df

    def load_all(self, nasa_dir: str = "data/battery_data", calce_dir: str = "data/calce") -> pd.DataFrame:
        """Load all available datasets and merge."""
        frames = []

        # NASA (always available)
        try:
            nasa_df = self.load_nasa(nasa_dir)
            frames.append(nasa_df)
        except Exception as e:
            logger.error(f"Failed to load NASA: {e}")

        # CALCE (optional)
        try:
            calce_df = self.load_calce(calce_dir)
            frames.append(calce_df)
        except FileNotFoundError:
            logger.info("CALCE data not found, skipping.")
        except Exception as e:
            logger.error(f"Failed to load CALCE: {e}")

        if not frames:
            raise ValueError("No datasets loaded successfully.")

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            f"Combined: {len(combined)} cycles, "
            f"{combined['battery_id'].nunique()} batteries, "
            f"{combined['dataset_source'].nunique()} datasets"
        )
        return combined
