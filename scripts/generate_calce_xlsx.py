import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CalceMock")

def generate_calce_arbin_xlsx():
    """
    Generates a highly realistic mock of an Arbin-formatted CALCE .xlsx 
    file to feed the ETL parser. It contains distinct charge/discharge 
    cycles with high-frequency time series data.
    """
    out_dir = Path("data/raw/calce")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "CS2_33.xlsx"

    logger.info(f"Generating realistic CALCE Arbin format data at {out_file}...")

    cycles = 200
    pts_per_cycle = 150

    test_time = []
    step_time = []
    step_index = []
    cycle_idx = []
    current = []
    voltage = []

    global_time = 0.0
    c_nominal = 1.1

    for c in range(1, cycles + 1):
        # Charge Step (Step Index 1) CC-CV
        for t in range(50):
            test_time.append(global_time)
            step_time.append(t)
            step_index.append(1)
            cycle_idx.append(c)
            current.append(0.55) # 0.5C charge
            voltage.append(3.0 + (t / 50) * 1.2 + np.random.normal(0, 0.01))
            global_time += 1.0

        # Discharge Step (Step Index 2) 1C discharge
        capacity_fade = c_nominal * (1.0 - (c * 0.001))

        for t in range(100):
            test_time.append(global_time)
            step_time.append(t)
            step_index.append(2)
            cycle_idx.append(c)
            # Add some current noise and variability
            current.append(-1.1 + np.random.normal(0, 0.05)) # -1C discharge
            voltage_drop = 4.2 - (t / 100) * 1.5 - (c * 0.002) + np.random.normal(0, 0.02)
            voltage.append(voltage_drop)
            global_time += 1.0

        # Rest Step (Step Index 3)
        for t in range(10):
            test_time.append(global_time)
            step_time.append(t)
            step_index.append(3)
            cycle_idx.append(c)
            current.append(0.0)
            voltage.append(voltage[-1] + 0.1) # Voltage relaxation
            global_time += 1.0

    df = pd.DataFrame({
        'Test_Time(s)': test_time,
        'Step_Time(s)': step_time,
        'Step_Index': step_index,
        'Cycle_Index': cycle_idx,
        'Current(A)': current,
        'Voltage(V)': voltage,
        'Capacity(Ah)': np.linspace(1.1, 0.7, len(test_time)) # Mock degradation
    })

    logger.info("Saving Excel binary...")
    df.to_excel(out_file, index=False)
    logger.info(f"Generated {out_file.name} with {len(df)} rows.")

if __name__ == "__main__":
    generate_calce_arbin_xlsx()
