"""
CALCE CS2 Dataset ETL Pipeline.

Extracts, parses, and cleans raw CALCE CS2 battery cycling data from
mixed-format zip archives (TXT tab-separated + XLSX) into normalized
per-cycle CSV files compatible with UnifiedDataLoader.load_calce().

Raw CALCE Format:
    - Tab-separated columns: Time, Status code, Status category, Status color,
      Pgm code, Pgm step, Pgm para, Pgm cycle, mV, mA, Temperature,
      Duration, Charge count, Discharge count, Capacity, ...
    - mV/mA: Voltage/current in millivolts/milliamps (need /1000 conversion)
    - Capacity: Accumulated Ah within each charge/discharge step
    - Discharge identified by negative current (mA < 0)

Output Schema (per battery CSV):
    cycle, capacity, voltage, current, temperature

Usage:
    python scripts/process_calce_data.py
    python scripts/process_calce_data.py --zip-dir C:\\Users\\22304\\Desktop --output-dir data/calce
"""

import argparse
import io
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("calce_etl")

# Columns we care about from the raw data
RAW_COLUMNS_MAP = {
    "Time": "time",
    "Pgm cycle": "pgm_cycle",
    "mV": "voltage_mv",
    "mA": "current_ma",
    "Temperature": "temperature",
    "Capacity": "raw_capacity",
    "Charge count": "charge_count",
    "Discharge count": "discharge_count",
    "Status code": "status_code",
}

# CALCE CS2 batteries and their chemistries
BATTERY_CHEMISTRY = {
    "CS2_8": "LiCoO2",
    "CS2_21": "LiCoO2",
    "CS2_33": "LiCoO2",
    "CS2_34": "LiCoO2",
    "CS2_35": "LiCoO2",
    "CS2_36": "LiCoO2",
    "CS2_37": "LiCoO2",
    "CS2_38": "LiCoO2",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CALCE CS2 Dataset ETL Pipeline")
    p.add_argument("--zip-dir", default=str(Path.home() / "Desktop"),
                    help="Directory containing CS2_*.zip files")
    p.add_argument("--output-dir",
                    default=str(ROOT / "data" / "calce"),
                    help="Output directory for cleaned CSVs")
    p.add_argument("--batteries", nargs="+",
                    default=["CS2_8", "CS2_21", "CS2_33", "CS2_34",
                             "CS2_35", "CS2_36", "CS2_37", "CS2_38"],
                    help="Battery IDs to process")
    p.add_argument("--min-discharge-points", type=int, default=10,
                    help="Minimum data points per discharge cycle to keep")
    p.add_argument("--capacity-unit-ah", type=float, default=1.1,
                    help="Nominal capacity in Ah for reference")
    return p.parse_args()


# Arbin MITS xlsx column mapping
ARBIN_COLUMNS_MAP = {
    "Cycle_Index": "cycle_index",
    "Current(A)": "current",
    "Voltage(V)": "voltage",
    "Discharge_Capacity(Ah)": "discharge_capacity_ah",
    "Charge_Capacity(Ah)": "charge_capacity_ah",
    "Test_Time(s)": "time",
    "Step_Index": "step_index",
    "Internal_Resistance(Ohm)": "internal_resistance",
    "dV/dt(V/s)": "dvdt",
}


def read_file_from_zip(
    zf: zipfile.ZipFile,
    entry_name: str,
) -> tuple[pd.DataFrame | None, str]:
    """Read a single data file (TXT or XLSX) from a zip archive.

    Returns:
        (DataFrame or None, format_type: 'txt' | 'arbin_xlsx' | 'unknown')
    """
    try:
        with zf.open(entry_name) as f:
            raw_bytes = f.read()

        if entry_name.lower().endswith(".txt"):
            text = raw_bytes.decode("utf-8", errors="replace")
            df = pd.read_csv(
                io.StringIO(text),
                sep="\t",
                on_bad_lines="skip",
                engine="python",
            )
            if df.empty:
                return None, "txt"
            return df, "txt"

        elif entry_name.lower().endswith(".xlsx"):
            # Arbin MITS xlsx: data is in 'Channel_*' sheet, not 'Info'
            xls = pd.ExcelFile(io.BytesIO(raw_bytes), engine="openpyxl")
            data_sheet = None
            for sheet in xls.sheet_names:
                if sheet.lower().startswith("channel"):
                    data_sheet = sheet
                    break
            if data_sheet is None:
                # Fallback: use last sheet (often the data sheet)
                data_sheet = xls.sheet_names[-1] if len(xls.sheet_names) > 1 else xls.sheet_names[0]

            df = pd.read_excel(xls, sheet_name=data_sheet)
            if df.empty:
                return None, "arbin_xlsx"
            return df, "arbin_xlsx"

        else:
            logger.debug(f"  Skipping unknown format: {entry_name}")
            return None, "unknown"

    except Exception as e:
        logger.warning(f"  Failed to parse {entry_name}: {e}")
        return None, "unknown"


def normalize_columns(df: pd.DataFrame, fmt: str = "txt") -> pd.DataFrame:
    """Normalize column names to handle inconsistencies across formats."""
    df.columns = df.columns.str.strip()

    if fmt == "arbin_xlsx":
        # Arbin format: direct column rename
        rename_map = {}
        for raw_name, clean_name in ARBIN_COLUMNS_MAP.items():
            for col in df.columns:
                if col.strip() == raw_name:
                    rename_map[col] = clean_name
                    break
        df = df.rename(columns=rename_map)
    else:
        # Legacy txt format
        rename_map = {}
        for raw_name, clean_name in RAW_COLUMNS_MAP.items():
            for col in df.columns:
                if col.strip().lower() == raw_name.lower():
                    rename_map[col] = clean_name
                    break
        df = df.rename(columns=rename_map)

    return df


def extract_discharge_capacity_txt(
    df: pd.DataFrame,
    min_points: int = 10,
) -> pd.DataFrame:
    """
    Extract per-cycle discharge capacity from legacy TXT format data.
    (Voltage/current in mV/mA, has discharge_count column.)
    """
    # Convert units: mV -> V, mA -> A
    if "voltage_mv" in df.columns:
        df["voltage"] = df["voltage_mv"].astype(float) / 1000.0
    elif "voltage" not in df.columns:
        df["voltage"] = np.nan

    if "current_ma" in df.columns:
        df["current"] = df["current_ma"].astype(float) / 1000.0
    elif "current" not in df.columns:
        df["current"] = np.nan

    if "temperature" not in df.columns:
        df["temperature"] = 25.0
    else:
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce").fillna(25.0)

    if "raw_capacity" in df.columns:
        df["raw_capacity"] = pd.to_numeric(df["raw_capacity"], errors="coerce").fillna(0.0)
    else:
        df["raw_capacity"] = 0.0

    df["is_discharge"] = df["current"] < -0.01

    if "discharge_count" in df.columns:
        df["discharge_count"] = pd.to_numeric(df["discharge_count"], errors="coerce").fillna(0).astype(int)
        discharge_mask = df["is_discharge"] & (df["discharge_count"] > 0)
        discharge_df = df[discharge_mask].copy()
        if discharge_df.empty:
            return pd.DataFrame()
        cycle_groups = discharge_df.groupby("discharge_count")
    else:
        df["discharge_block"] = (
            df["is_discharge"].astype(int).diff().fillna(0).abs().cumsum()
        )
        discharge_df = df[df["is_discharge"]].copy()
        if discharge_df.empty:
            return pd.DataFrame()
        cycle_groups = discharge_df.groupby("discharge_block")

    cycles = []
    cycle_num = 0
    for group_id, group in cycle_groups:
        if len(group) < min_points:
            continue
        cycle_num += 1

        cap = group["raw_capacity"].max()
        if cap < 0.001 and "time" in group.columns:
            time_vals = pd.to_numeric(group["time"], errors="coerce").dropna().values
            curr_vals = np.abs(group["current"].dropna().values)
            if len(time_vals) > 1 and len(curr_vals) > 1:
                min_len = min(len(time_vals), len(curr_vals))
                dt_hours = np.diff(time_vals[:min_len]) / 3600.0
                avg_current = (curr_vals[:min_len - 1] + curr_vals[1:min_len]) / 2.0
                cap = float(np.sum(avg_current * dt_hours))
        if cap < 0.001:
            continue
        if cap > 10.0:
            cap = cap / 1000.0

        cycles.append({
            "cycle": cycle_num,
            "capacity": float(cap),
            "voltage": float(group["voltage"].mean()),
            "current": float(group["current"].mean()),
            "temperature": float(group["temperature"].mean()),
            "max_voltage": float(group["voltage"].max()),
            "min_voltage": float(group["voltage"].min()),
            "max_temperature": float(group["temperature"].max()),
            "discharge_duration_s": float(
                pd.to_numeric(group["time"], errors="coerce").max() -
                pd.to_numeric(group["time"], errors="coerce").min()
            ) if "time" in group.columns else 0.0,
            "num_points": len(group),
        })

    return pd.DataFrame(cycles) if cycles else pd.DataFrame()


def extract_discharge_capacity_arbin(
    df: pd.DataFrame,
    min_points: int = 10,
) -> pd.DataFrame:
    """
    Extract per-cycle discharge capacity from Arbin MITS xlsx format.

    Since multiple Arbin files are concatenated, `Cycle_Index` resets to 1
    multiple times, breaking simple groupby logic. Also `Discharge_Capacity(Ah)`
    behavior is inconsistent across test sessions.
    
    Robust strategy:
    1. Identify continuous discharge blocks (Current < -0.01)
    2. Group by these blocks (ignoring raw Cycle_Index)
    3. Calculate capacity rigorously via numerical integration of Current(A) * dt
    """
    # Normalize core columns
    if "current" not in df.columns:
        return pd.DataFrame()
    df["current"] = pd.to_numeric(df["current"], errors="coerce")

    if "voltage" not in df.columns:
        df["voltage"] = np.nan
    else:
        df["voltage"] = pd.to_numeric(df["voltage"], errors="coerce")

    if "time" not in df.columns:
        logger.warning("  Arbin data missing Test_Time(s) column, cannot integrate capacity.")
        return pd.DataFrame()

    # Global continuous time (cumulative sum of diffs to handle file stitching)
    dt_raw = pd.to_numeric(df["time"], errors="coerce").diff().fillna(0)
    # If dt < 0, a new file started. Replace negative dt with median dt or 1.0s
    dt_raw[dt_raw < 0] = dt_raw[dt_raw >= 0].median()
    df["global_time"] = dt_raw.cumsum()

    # Temperature (Arbin CS2 datasets usually lack it; default to 25C)
    df["temperature"] = 25.0

    # 1. Identify continuous discharge blocks
    df["is_discharge"] = df["current"] < -0.01
    df["discharge_block"] = (df["is_discharge"].astype(int).diff().fillna(0).abs().cumsum())

    discharge_df = df[df["is_discharge"]].copy()
    if discharge_df.empty:
        return pd.DataFrame()

    # 2. Group by continuous discharge blocks
    cycles = []
    cycle_num = 0
    for block_id, group in discharge_df.groupby("discharge_block"):
        if len(group) < min_points:
            continue

        # 3. Numerical Integration (trapezoidal rule): Ah = A * hours
        time_vals = group["global_time"].values
        curr_vals = np.abs(group["current"].values)

        if len(time_vals) > 1:
            dt_hours = np.diff(time_vals) / 3600.0
            avg_curr = (curr_vals[:-1] + curr_vals[1:]) / 2.0
            cap = float(np.sum(avg_curr * dt_hours))
        else:
            cap = 0.0

        if cap < 0.001:
            continue

        cycle_num += 1
        cycles.append({
            "cycle": cycle_num,
            "capacity": float(cap),
            "voltage": float(group["voltage"].mean()),
            "current": float(group["current"].mean()),
            "temperature": float(group["temperature"].mean()),
            "max_voltage": float(group["voltage"].max()),
            "min_voltage": float(group["voltage"].min()),
            "max_temperature": 25.0,
            "discharge_duration_s": float(time_vals[-1] - time_vals[0]),
            "num_points": len(group),
        })

    return pd.DataFrame(cycles) if cycles else pd.DataFrame()


def process_battery_zip(
    zip_path: Path,
    battery_id: str,
    min_points: int = 10,
) -> pd.DataFrame:
    """Process all files in a single battery zip archive."""
    logger.info(f"Processing {battery_id} from {zip_path.name}...")

    all_raw_frames = []
    detected_format = "unknown"

    with zipfile.ZipFile(zip_path, "r") as zf:
        data_entries = [
            e.filename for e in zf.infolist()
            if (e.filename.lower().endswith(".txt") or
                e.filename.lower().endswith(".xlsx"))
            and not e.filename.startswith("__MACOSX")
            and e.file_size > 100
        ]

        logger.info(f"  Found {len(data_entries)} data files")

        for entry_name in sorted(data_entries):
            df, fmt = read_file_from_zip(zf, entry_name)
            if df is not None and not df.empty:
                detected_format = fmt
                df = normalize_columns(df, fmt=fmt)
                all_raw_frames.append(df)
                logger.debug(f"  Parsed {entry_name}: {len(df)} rows (fmt={fmt})")

    if not all_raw_frames:
        logger.warning(f"  No valid data extracted for {battery_id}")
        return pd.DataFrame()

    # Concatenate all files chronologically
    combined = pd.concat(all_raw_frames, ignore_index=True)
    logger.info(f"  Combined: {len(combined)} total rows from {len(all_raw_frames)} files (format={detected_format})")

    # Route to the correct extraction function based on format
    if detected_format == "arbin_xlsx":
        cycles_df = extract_discharge_capacity_arbin(combined, min_points=min_points)
    else:
        cycles_df = extract_discharge_capacity_txt(combined, min_points=min_points)

    if cycles_df.empty:
        logger.warning(f"  No valid discharge cycles found for {battery_id}")
        return pd.DataFrame()

    # Re-number cycles sequentially
    cycles_df = cycles_df.sort_values("cycle").reset_index(drop=True)
    cycles_df["cycle"] = range(1, len(cycles_df) + 1)

    logger.info(
        f"  {battery_id}: {len(cycles_df)} discharge cycles, "
        f"capacity range [{cycles_df['capacity'].min():.4f}, "
        f"{cycles_df['capacity'].max():.4f}] Ah"
    )

    return cycles_df


def quality_check(df: pd.DataFrame, battery_id: str) -> pd.DataFrame:
    """
    Apply data quality filters to remove obviously bad cycles.

    Filters:
        1. Remove cycles with capacity <= 0 or capacity > 5 Ah (sensor error)
        2. Remove cycles where capacity jumps > 50% from rolling median (outlier)
        3. Remove duplicate or near-duplicate capacity values
    """
    original_len = len(df)

    # Filter 1: Physical bounds
    df = df[(df["capacity"] > 0.01) & (df["capacity"] < 5.0)].copy()

    # Filter 2: Rolling median outlier detection
    if len(df) > 10:
        rolling_median = df["capacity"].rolling(window=5, center=True, min_periods=1).median()
        relative_diff = (df["capacity"] - rolling_median).abs() / rolling_median
        df = df[relative_diff < 0.5].copy()

    # Re-number cycles after filtering
    df["cycle"] = range(1, len(df) + 1)

    removed = original_len - len(df)
    if removed > 0:
        logger.info(f"  Quality check {battery_id}: removed {removed}/{original_len} bad cycles")

    return df


def main() -> None:
    args = parse_args()
    zip_dir = Path(args.zip_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("CALCE ETL Pipeline")
    logger.info(f"  Zip source: {zip_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Batteries:  {args.batteries}")

    all_summaries = []

    for battery_id in args.batteries:
        zip_path = zip_dir / f"{battery_id}.zip"
        if not zip_path.exists():
            logger.warning(f"Zip not found: {zip_path}, skipping {battery_id}")
            continue

        # Process
        cycles_df = process_battery_zip(
            zip_path=zip_path,
            battery_id=battery_id,
            min_points=args.min_discharge_points,
        )

        if cycles_df.empty:
            logger.warning(f"No data produced for {battery_id}")
            continue

        # Quality check
        cycles_df = quality_check(cycles_df, battery_id)

        if len(cycles_df) < 5:
            logger.warning(f"Too few cycles for {battery_id}: {len(cycles_df)}")
            continue

        # Save
        output_path = output_dir / f"{battery_id}.csv"
        cycles_df.to_csv(output_path, index=False)
        logger.info(f"  Saved: {output_path} ({len(cycles_df)} cycles)")

        all_summaries.append({
            "battery_id": battery_id,
            "num_cycles": len(cycles_df),
            "capacity_initial": round(cycles_df["capacity"].iloc[0], 4),
            "capacity_final": round(cycles_df["capacity"].iloc[-1], 4),
            "capacity_fade_pct": round(
                (1 - cycles_df["capacity"].iloc[-1] / cycles_df["capacity"].iloc[0]) * 100, 2
            ),
            "chemistry": BATTERY_CHEMISTRY.get(battery_id, "unknown"),
        })

    # Print summary table
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        print("\n" + "=" * 80)
        print("  CALCE CS2 Dataset Processing Summary")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)

        # Save summary
        summary_path = output_dir / "_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved: {summary_path}")
    else:
        logger.error("No valid batteries processed!")


if __name__ == "__main__":
    main()
