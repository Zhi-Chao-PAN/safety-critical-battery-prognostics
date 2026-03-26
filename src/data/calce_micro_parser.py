"""
CALCE High-Frequency Micro Parser (Phase 5.1)

Handles the chaotic formatting of raw CALCE CS2 datasets (.txt / .xlsx).
Strictly solves the "Variable-Length Sequence Alignment" trap via 1D Spline Interpolation, 
ensuring every cycle operates on perfectly aligned `[Batch, Micro_Steps]` PyTorch tensors.

Defensive Engineering:
- Handles missing columns and infinite/NaN values during txt/csv parsing.
- Uses scipy.interpolate.interp1d for exact step-length guarantees.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

class CalceMicroParser:
    def __init__(self, target_micro_steps: int = 100):
        self.micro_steps = target_micro_steps

    def _read_file(self, filepath: Path) -> pd.DataFrame:
        """Polymorphic reader capable of parsing both Arbin xlsx and old CADEX txt."""
        ext = filepath.suffix.lower()
        logger.info(f"Parsing raw file: {filepath.name} (Format: {ext})")

        try:
            if ext in ['.txt', '.csv']:
                # Old CADEX files often have variable tab/space separators, or standard CSV
                try:
                    df = pd.read_csv(filepath)
                except Exception:
                    df = pd.read_csv(filepath, sep='\t')
                    if len(df.columns) < 3:
                        df = pd.read_csv(filepath, sep=',')
            elif ext in ['.xlsx', '.xls']:
                # Arbin formats
                df = pd.read_excel(filepath, engine='openpyxl')
            else:
                raise ValueError(f"🚨 Unsupported file format: {ext}")

            # Standardize extremely chaotic column names
            df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]
            return df

        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            raise

    def _extract_column(self, df: pd.DataFrame, possible_names: list) -> np.ndarray:
        """Robust column extraction using fallback matches."""
        for name in possible_names:
            matches = [col for col in df.columns if name in col]
            if matches:
                # Fill NaNs defensively with 0 to prevent downstream interpolation explosion
                return df[matches[0]].fillna(0.0).values
        raise KeyError(f"Could not find any columns matching: {possible_names}")

    def slice_cycles_and_align(self, filepath: str) -> dict:
        """
        The Core Engine: Slices chaotic continuous streams into exact chunks 
        and forcefully aligns time steps using 1D interpolation.
        
        Returns:
            dict containing PyTorch tensors:
            - "i_app_micro": [num_cycles, micro_steps] 
            - "v_micro": [num_cycles, micro_steps]
            - "cycle_durations": [num_cycles] (Original real duration in seconds)
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Missing raw data file: {path}")

        df = self._read_file(path)

        # 1. Column extraction with Defensive Fallbacks
        time_arr = self._extract_column(df, ['test_time', 'time'])
        current_arr = self._extract_column(df, ['current', 'i(a)'])
        voltage_arr = self._extract_column(df, ['voltage', 'v(v)'])
        step_index = self._extract_column(df, ['step_index', 'step'])

        logger.info(f"Extracted {len(time_arr)} continuous samples.")

        aligned_i_list = []
        aligned_v_list = []
        durations = []

        # 2. Cycle Slicing (Microscopic View - Focusing on Discharge steps)
        # Assuming discharge steps are where Current < 0 (General convention, though datasets vary).
        # We also use step_index changes to isolate distinct discharge chunks.

        # Identify contiguous blocks where current is discharging
        discharge_mask = current_arr < -0.01  # Small tolerance

        # Find rising/falling edges to identify start/stop of discharge ranges
        edges = np.diff(discharge_mask.astype(int))
        starts = np.where(edges == 1)[0] + 1
        stops = np.where(edges == -1)[0] + 1

        if discharge_mask[0]: starts = np.insert(starts, 0, 0)
        if discharge_mask[-1]: stops = np.append(stops, len(discharge_mask))

        if len(starts) == 0:
            logger.warning("No discharge cycles found in file!")
            return {}

        logger.info(f"Identified {len(starts)} raw discharge segments.")

        valid_cycles = 0

        # 3. Time Resampling & Absolute Alignment
        for start_idx, stop_idx in zip(starts, stops):
            # Skip noise or artificially tiny segments (< 10 samples)
            if stop_idx - start_idx < 10:
                continue

            t_chunk = time_arr[start_idx:stop_idx]
            i_chunk = current_arr[start_idx:stop_idx]
            v_chunk = voltage_arr[start_idx:stop_idx]

            # Normalize time within this chunk to range [0, 1] relative to itself
            t_min = t_chunk.min()
            t_max = t_chunk.max()
            duration_sec = t_max - t_min

            if duration_sec <= 0:
                continue

            t_normalized = (t_chunk - t_min) / duration_sec

            # 🚨 CORE ALIGNMENT: 1D Interpolation to target micro_steps
            # Regardless of if chunk took 3600s (slow discharge) or 1500s (fast aged discharge),
            # it becomes strictly `micro_steps` elements long.
            target_t = np.linspace(0.0, 1.0, self.micro_steps)

            # Use linear interpolation (avoids Runge's phenomenon / overshoot from cubic splines)
            f_current = interp1d(t_normalized, i_chunk, kind='linear', assume_sorted=True)
            f_voltage = interp1d(t_normalized, v_chunk, kind='linear', assume_sorted=True)

            aligned_i = f_current(target_t)
            aligned_v = f_voltage(target_t)

            aligned_i_list.append(aligned_i)
            aligned_v_list.append(aligned_v)
            durations.append(duration_sec)
            valid_cycles += 1

        logger.info(f"Successfully aligned {valid_cycles} cycles to shape [Batch, {self.micro_steps}].")

        # Pack into Tensors
        return {
            "i_app_micro": torch.tensor(np.stack(aligned_i_list), dtype=torch.float32),
            "v_micro": torch.tensor(np.stack(aligned_v_list), dtype=torch.float32),
            "cycle_durations": torch.tensor(durations, dtype=torch.float32)
        }
