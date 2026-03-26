"""
Operation Aging Coupling
Links the macroscopic operational profile (current, voltage, temp)
over cycles to the underlying degradation ODEs (SEI growth, LAM).
"""

import numpy as np


class OperationAgingCoupling:
    """
    Aggregates per-cycle stress, temperature, and depth of discharge (DoD)
    to drive the empirical/physical capacity fade models.
    """
    def __init__(self, cycle_limit: int = 2000):
        self.cycle_limit = cycle_limit
        self.total_sei_thickness = 0.0
        self.active_material_loss = 0.0

    def step_cycle(self,
                   avg_temp: float,
                   max_stress: float,
                   dod: float,
                   dt_days: float) -> dict:
        """
        Simulate one macroscopic cycle or time window.
        Returns the delta capacity loss.
        """
        # 1. SEI Growth (Time/Temp dependent) -> related to a*sqrt(t) capacity fade
        # Arrhenius dependency
        k_sei = 1e-6 * np.exp(-30000 / (8.314 * (avg_temp + 273.15)))
        delta_sei = k_sei * dt_days
        self.total_sei_thickness += delta_sei

        # 2. LAM (Stress/DoD dependent) -> related to b*n capacity fade
        # Paris' Law simplified
        k_lam = 1e-12
        delta_lam = k_lam * (max_stress ** 2) * dod
        self.active_material_loss += delta_lam

        cap_retention = 1.0 - (self.total_sei_thickness * 100) - self.active_material_loss
        return {
             "sei_thickness": self.total_sei_thickness,
             "lam": self.active_material_loss,
             "capacity_retention": cap_retention
        }
