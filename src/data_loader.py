import os
import scipy.io
import numpy as np
import pandas as pd
from typing import List, Optional

class BatteryDataLoader:
    """
    NASA PCoE Battery Dataset Loader.
    Standardizes cycle-level degradation features for RUL prediction.
    Strictly forbids synthetic data generation for academic rigor.
    """
    def __init__(self, data_dir: str = "data/battery_data", rated_capacity: float = 2.0):
        self.data_dir = data_dir
        self.rated_capacity = rated_capacity

    def load_data(self, battery_ids: List[str] = ['B0005', 'B0006', 'B0007', 'B0018']) -> pd.DataFrame:
        """
        Load and process battery data.
        
        Args:
            battery_ids: List of battery IDs to load
            
        Returns:
            DataFrame with processed features
            
        Raises:
            FileNotFoundError: If any requested data file is missing.
        """
        # Integrity Check
        if not os.path.exists(self.data_dir):
             raise FileNotFoundError(f"Data directory '{self.data_dir}' not found. Please download NASA PCoE dataset.")

        all_cycles = []
        
        for bat_id in battery_ids:
            filepath = os.path.join(self.data_dir, f"{bat_id}.mat")
            
            if not os.path.exists(filepath):
                 raise FileNotFoundError(f"Battery file not found: {filepath}")
            
            try:
                mat = scipy.io.loadmat(filepath)
                cycles = mat[bat_id][0, 0]['cycle'][0]
                
                cycle_count = 0
                for cycle in cycles:
                    if cycle['type'][0] == 'discharge':
                        cycle_count += 1
                        data = cycle['data']
                        
                        # Defensively extract nested mat structure
                        capacity = data[0, 0]['Capacity'][0][0]
                        temp = data[0, 0]['Temperature_measured'][0]
                        time = data[0, 0]['Time'][0]
                        
                        all_cycles.append({
                            'battery_id': bat_id,
                            'cycle': cycle_count,
                            'capacity': capacity,
                            'discharge_time': time[-1] - time[0],
                            'max_temp': np.max(temp)
                        })
            except Exception as e:
                print(f"Error processing {bat_id}: {e}")
                raise ValueError(f"Corrupt or incompatible data format for {bat_id}") from e
                
        if not all_cycles:
             raise ValueError("No valid cycle data extracted from files.")

        return self._calculate_rul(pd.DataFrame(all_cycles))

    def _calculate_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        eol_thresh = 0.7 * self.rated_capacity
        df_list = []
        for bat_id in df['battery_id'].unique():
            sub = df[df['battery_id'] == bat_id].copy()
            failed = sub[sub['capacity'] < eol_thresh]
            eol = failed['cycle'].min() if not failed.empty else sub['cycle'].max()
            sub['rul'] = eol - sub['cycle']
            df_list.append(sub[sub['rul'] >= 0])
        return pd.concat(df_list, ignore_index=True)