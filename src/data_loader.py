import os
import scipy.io
import pandas as pd
import numpy as np

def load_battery_data(data_dir: str) -> pd.DataFrame:
    """
    Load and process NASA Battery Dataset from .mat files.
    
    Expected structure:
    data_dir/
        B0005.mat
        B0006.mat
        ...
        
    Each .mat file contains a struct with 'cycle' field, which is an array of structs.
    We extract 'discharge' operations to calculate degradation features and RUL.
    
    Args:
        data_dir: Directory containing .mat files.
        
    Returns:
        DataFrame with columns: [battery_id, cycle_id, discharge_time, max_temp, ..., RUL]
    """
    if not os.path.exists(data_dir):
        # Fallback to creating dummy data if directory doesn't exist (for development/testing)
        print(f"Data directory {data_dir} not found. Creating dummy data.")
        return _create_dummy_data()

    files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    all_data = []

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        battery_id = os.path.splitext(filename)[0]
        
        try:
            mat = scipy.io.loadmat(filepath)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
            
        # The structure is specific to NASA dataset. 
        # Usually key is the battery name, e.g., 'B0005'
        # We try to find the data key
        data_key = None
        for key in mat.keys():
            if key == battery_id or key in ['B0005', 'B0006', 'B0007', 'B0018']: # Common names
                data_key = key
                break
        
        if data_key is None:
             # Fallback: look for any key that isn't meta
             for key in mat.keys():
                 if not key.startswith('__'):
                     data_key = key
                     break
        
        if data_key is None:
            print(f"Could not find data key in {filename}")
            continue

        data = mat[data_key]
        cycle = data[0,0]['cycle']
        
        # Extract features per cycle
        battery_cycles = []
        
        for i in range(cycle.shape[1]):
            op_type = cycle[0,i]['type'][0]
            if op_type == 'discharge':
                data_struct = cycle[0,i]['data']
                
                # Features
                # Time is usually (1, N) or (N, 1)
                time_arr = data_struct[0,0]['Time'][0]
                voltage = data_struct[0,0]['Voltage_measured'][0]
                current = data_struct[0,0]['Current_measured'][0]
                temp = data_struct[0,0]['Temperature_measured'][0]
                
                if len(time_arr) == 0: continue
                
                discharge_time = time_arr[-1] - time_arr[0]
                max_temp = np.max(temp)
                min_temp = np.min(temp)
                avg_voltage = np.mean(voltage)
                
                # Heuristic for time constant current: usually part of the curve where I is constant
                # Simplification: specific logic removed for brevity, using proxy
                time_constant_current = discharge_time # Placeholder or needs complex logic
                
                decrease_in_voltage = np.max(voltage) - np.min(voltage)

                battery_cycles.append({
                    'battery_id': battery_id,
                    'cycle_id': i + 1,
                    'discharge_time': discharge_time,
                    'max_temp': max_temp,
                    'min_temp': min_temp,
                    'decrease_in_voltage': decrease_in_voltage,
                    'time_constant_current': time_constant_current
                })
        
        # Calculate RUL
        # RUL = Total_Cycles - Current_Cycle
        # Assuming last cycle is failure (or close to it)
        total_life = len(battery_cycles)
        for item in battery_cycles:
            item['RUL'] = total_life - item['cycle_id']
            # Optionally normalize RUL or keep raw
            
        all_data.extend(battery_cycles)

    if not all_data:
        print("No valid data found in .mat files. Creating dummy data.")
        return _create_dummy_data()

    return pd.DataFrame(all_data)

def _create_dummy_data() -> pd.DataFrame:
    """Generate synthetic battery data for testing."""
    print("Generating synthetic battery data...")
    batteries = ['B0001', 'B0002', 'B0003', 'B0004']
    data = []
    
    for batt in batteries:
        n_cycles = np.random.randint(100, 200)
        # Linear degradation trend with noise
        for i in range(n_cycles):
            age_factor = i / n_cycles
            
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
    df = load_battery_data("data/battery_data")
    print("Data loaded:")
    print(df.head())
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    if 'battery_id' in df.columns:
        print(f"Batteries: {df['battery_id'].unique()}")
