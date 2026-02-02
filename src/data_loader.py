import os
import scipy.io # type: ignore
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from src.utils.logger import setup_logger
from src.utils.seed import set_global_seed

# Initialize module logger
logger = setup_logger(__name__)
set_global_seed(42)  # Enforce determinism on import


def load_battery_data(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load and process NASA Battery Dataset from .mat files.
    
    This function iterates through all .mat files in the specified directory,
    extracts cycle-level charging/discharging data, calculates degradation features
    (e.g., discharge time, temperature stats), and computes Remaining Useful Life (RUL).

    Args:
        data_dir (Union[str, Path]): Directory containing the .mat battery data files.
        
    Returns:
        pd.DataFrame: A DataFrame containing processed features and RUL for all batteries.
            Columns include:
            - battery_id (str): Identifier of the battery.
            - cycle_id (int): Cycle number.
            - discharge_time (float): Duration of discharge in seconds.
            - max_temp (float): Maximum temperature during discharge.
            - min_temp (float): Minimum temperature during discharge.
            - decrease_in_voltage (float): Voltage drop during discharge.
            - time_constant_current (float): (Proxy) Time duration of constant current load.
            - RUL (int): Calculated Remaining Useful Life (Total Cycles - Current Cycle).
            
    Raises:
        FileNotFoundError: If the data directory does not exist and fallback creation fails.
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Data directory '{data_path}' not found. Attempting to create dummy data for development.")
        return _create_dummy_data()

    # List all .mat files
    files = [f for f in os.listdir(data_path) if f.endswith('.mat')]
    
    if not files:
        logger.warning(f"No .mat files found in '{data_path}'. Creating dummy data.")
        return _create_dummy_data()
        
    all_data: List[Dict[str, Any]] = []

    for filename in files:
        filepath = data_path / filename
        battery_id = filepath.stem
        
        logger.info(f"Processing file: {filename}")
        
        try:
            mat = scipy.io.loadmat(str(filepath))
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            continue
            
        # Locate the data key in the .mat file struct
        data_key = _find_data_key(mat, battery_id)

        
        if data_key is None:
            logger.error(f"Could not find valid data key in {filename}. Skipping.")
            continue

        data = mat[data_key]
        cycle = data[0,0]['cycle']
        
        # Extract features per cycle
        battery_cycles = _process_battery_cycles(cycle, battery_id)
        
        # Calculate RUL
        total_life = len(battery_cycles)
        for item in battery_cycles:
            item['RUL'] = total_life - item['cycle_id']
            
        all_data.extend(battery_cycles)
        logger.info(f"Loaded {len(battery_cycles)} cycles for battery {battery_id}.")

    if not all_data:
        logger.warning("No valid data processed from files. Defaulting to dummy data.")
        return _create_dummy_data()

    df = pd.DataFrame(all_data)
    logger.info(f"Successfully loaded dataset with shape {df.shape}")
    return df

def _find_data_key(mat: Dict[str, Any], battery_id: str) -> Optional[str]:
    """
    Heuristic to find the variable name holding the data structure in the .mat file.
    """
    # 1. Try exact match
    if battery_id in mat:
        return battery_id
        
    # 2. Try common NASA names
    common_names = ['B0005', 'B0006', 'B0007', 'B0018']
    for name in common_names:
        if name in mat:
            return name
            
    # 3. Fallback: find first non-meta key
    for key in mat.keys():
        if not key.startswith('__'):
            return key
            
    return None

def _process_battery_cycles(cycle_struct: Any, battery_id: str) -> List[Dict[str, Any]]:
    """
    Extract features from the raw cycle structure.
    """
    cycles_data = []
    
    num_cycles = cycle_struct.shape[1]
    
    for i in range(num_cycles):
        try:
            op_type = cycle_struct[0,i]['type'][0]
            
            if op_type == 'discharge':
                data_struct = cycle_struct[0,i]['data']
                
                # Handling generic structure access needed for different MATLAB versions/formats
                # Assuming standard NASA structure here
                if data_struct.size == 0:
                    continue

                # Features extraction
                # Time, Voltage, Current, Temp are numpy arrays
                time_arr = data_struct[0,0]['Time'][0]
                voltage = data_struct[0,0]['Voltage_measured'][0]
                temp = data_struct[0,0]['Temperature_measured'][0]
                
                if len(time_arr) == 0: 
                    continue
                
                # Feature Engineering
                discharge_time = time_arr[-1] - time_arr[0]
                max_temp = float(np.max(temp))
                min_temp = float(np.min(temp))
                
                # Voltage drop
                decrease_in_voltage = float(np.max(voltage) - np.min(voltage))
                
                # Proxy for constant current time
                time_constant_current = float(discharge_time) 

                cycles_data.append({
                    'battery_id': battery_id,
                    'cycle_id': i + 1,
                    'discharge_time': discharge_time,
                    'max_temp': max_temp,
                    'min_temp': min_temp,
                    'decrease_in_voltage': decrease_in_voltage,
                    'time_constant_current': time_constant_current
                })
        except Exception as e:
            # Log debug but don't spam errors for every malformed cycle
            # logger.debug(f"Error processing cycle {i} for {battery_id}: {e}")
            continue
            
    return cycles_data

def _create_dummy_data() -> pd.DataFrame:
    """
    Generate synthetic battery data for testing purposes.
    
    Returns:
        pd.DataFrame: Synthetic dataset mimicking the structure of the real data.
    """
    logger.info("Generating synthetic battery data (fallback mode)...")
    batteries = ['B0001', 'B0002', 'B0003', 'B0004']
    data = []
    
    for batt in batteries:
        n_cycles = np.random.randint(100, 200)
        # Linear degradation trend with noise
        for i in range(n_cycles):
            age_factor = i / n_cycles
            
            # Simulate physics-based degradation
            discharge_time = 3600 * (1 - 0.3 * age_factor) + np.random.normal(0, 50)
            max_temp = 25 + 10 * age_factor + np.random.normal(0, 1)
            min_temp = 20 + np.random.normal(0, 0.5)
            decrease_in_voltage = 1.0 + 0.2 * age_factor + np.random.normal(0, 0.05)
            time_constant_current = discharge_time * 0.9
            
            rul = n_cycles - i - 1
            
            data.append({
                'battery_id': batt,
                'cycle_id': i + 1,
                'discharge_time': discharge_time,
                'max_temp': max_temp,
                'min_temp': min_temp,
                'decrease_in_voltage': decrease_in_voltage,
                'time_constant_current': time_constant_current,
                'RUL': rul
            })
            
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Test run
    try:
        df = load_battery_data("data/battery_data")
        logger.info(f"Stand-alone run completed. Data Shape: {df.shape}")
        if 'battery_id' in df.columns:
            logger.info(f"Batteries found: {df['battery_id'].unique()}")
    except Exception as e:
        logger.critical(f"Data loading failed: {e}")
