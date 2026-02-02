import os
import scipy.io
import numpy as np
import pandas as pd
from typing import List, Optional

class BatteryDataLoader:
    """
    NASA PCoE Battery Dataset Loader.
    Standardizes cycle-level degradation features for RUL prediction.
    """
    def __init__(self, data_dir: str = "data/battery_data", rated_capacity: float = 2.0):
        self.data_dir = data_dir
        self.rated_capacity = rated_capacity

    def load_data(self, battery_ids: List[str] = ['B0005', 'B0006', 'B0007', 'B0018']) -> pd.DataFrame:
        if not os.path.exists(self.data_dir) or not any(f.endswith('.mat') for f in os.listdir(self.data_dir)):
            print(f"Warning: {self.data_dir} empty. Generating SYNTHETIC physics-based data for demo.")
            return self._generate_synthetic_data(battery_ids)

        all_cycles = []
        for bat_id in battery_ids:
            try:
                filepath = os.path.join(self.data_dir, f"{bat_id}.mat")
                mat = scipy.io.loadmat(filepath)
                cycles = mat[bat_id][0, 0]['cycle'][0]
                
                cycle_count = 0
                for cycle in cycles:
                    if cycle['type'][0] == 'discharge':
                        cycle_count += 1
                        data = cycle['data']
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
                print(f"Skipping {bat_id}: {e}")
                
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

    def _generate_synthetic_data(self, battery_ids):
        # Physics-informed synthetic data: Capacity decays exponentially
        data = []
        for bat_id in battery_ids:
            n = 200
            for t in range(n):
                cap = 2.0 * np.exp(-0.005 * t) + np.random.normal(0, 0.01)
                data.append({
                    'battery_id': bat_id, 'cycle': t+1, 'capacity': cap,
                    'discharge_time': 3600*cap/2, 'max_temp': 24 + 0.05*t
                })
        return self._calculate_rul(pd.DataFrame(data))
