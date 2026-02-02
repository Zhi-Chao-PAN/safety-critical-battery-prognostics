"""
Data Pipeline for NASA Li-ion Battery Dataset.
Standardizes cycle-level degradation features for RUL prediction.
"""
import os
import scipy.io
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

class BatteryDataLoader:
    def __init__(self, data_dir: str = "data/battery_data"):
        self.data_dir = data_dir
        self.rated_capacity = 2.0  # NASA 18650 rated capacity in Ah

    def load_data(self, battery_ids: List[str] = ['B0005', 'B0006', 'B0007', 'B0018']) -> pd.DataFrame:
        """
        Loads and processes .mat files into a standardized DataFrame.
        Includes synthetic data generation fallback for reproduction without raw files.
        """
        all_cycles = []
        
        # Check if raw data exists
        if not os.path.exists(self.data_dir) or not any(fname.endswith('.mat') for fname in os.listdir(self.data_dir)):
            print(f"Warning: {self.data_dir} empty or missing. Generating SYNTHETIC physics-based data for demo.")
            return self._generate_synthetic_data(battery_ids)

        for bat_id in battery_ids:
            filepath = os.path.join(self.data_dir, f"{bat_id}.mat")
            if not os.path.exists(filepath):
                continue
                
            try:
                mat = scipy.io.loadmat(filepath)
                cycles = mat[bat_id][0, 0]['cycle'][0]
                
                cycle_count = 0
                for cycle in cycles:
                    if cycle['type'][0] == 'discharge':
                        cycle_count += 1
                        data = cycle['data']
                        
                        # Feature Extraction
                        capacity = data[0, 0]['Capacity'][0][0]
                        temp = data[0, 0]['Temperature_measured'][0]
                        time_vec = data[0, 0]['Time'][0]
                        volt_vec = data[0, 0]['Voltage_measured'][0]
                        
                        # Physics-informed features
                        discharge_time = time_vec[-1] - time_vec[0]
                        max_temp = np.max(temp)
                        voltage_drop = volt_vec[0] - volt_vec[-1] # Internal resistance proxy
                        
                        all_cycles.append({
                            'battery_id': bat_id,
                            'cycle': cycle_count,
                            'capacity': capacity,
                            'discharge_time': discharge_time,
                            'max_temp': max_temp,
                            'voltage_drop': voltage_drop
                        })
            except Exception as e:
                print(f"Error processing {bat_id}: {e}")

        df = pd.DataFrame(all_cycles)
        return self._calculate_rul(df)

    def _calculate_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Ground Truth RUL based on EOL threshold (70% capacity)."""
        eol_thresh = 0.7 * self.rated_capacity
        df_list = []
        
        for bat_id in df['battery_id'].unique():
            sub_df = df[df['battery_id'] == bat_id].copy()
            # Find first cycle below threshold
            failures = sub_df[sub_df['capacity'] < eol_thresh]
            eol_cycle = failures['cycle'].min() if not failures.empty else sub_df['cycle'].max()
            
            sub_df['rul'] = eol_cycle - sub_df['cycle']
            # Cap RUL at 0 for post-failure analysis
            sub_df = sub_df[sub_df['rul'] >= 0] 
            df_list.append(sub_df)
            
        return pd.concat(df_list, ignore_index=True)

    def _generate_synthetic_data(self, battery_ids: List[str]) -> pd.DataFrame:
        """Generates physics-compliant dummy data for code testing."""
        data = []
        for bat_id in battery_ids:
            n_cycles = 200
            for t in range(n_cycles):
                capacity = 2.0 * np.exp(-0.005 * t) + np.random.normal(0, 0.01)
                data.append({
                    'battery_id': bat_id,
                    'cycle': t + 1,
                    'capacity': capacity,
                    'discharge_time': 3600 * (capacity/2.0),
                    'max_temp': 24 + 0.05 * t,
                    'voltage_drop': 0.5 + 0.001 * t
                })
        return self._calculate_rul(pd.DataFrame(data))

if __name__ == "__main__":
    loader = BatteryDataLoader()
    df = loader.load_data()
    print("Data Loaded Successfully!")
    print(df.head())
    print(f"Shape: {df.shape}")

