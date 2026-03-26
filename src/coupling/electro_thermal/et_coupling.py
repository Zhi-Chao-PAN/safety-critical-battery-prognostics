"""
Electro-Thermal Coupling Module
Links internal resistance (R) changes from ECM with heat generation (Q) 
and Arrhenius temperature dependence for true multi-physics simulation.
"""

import numpy as np

from src.physics.electrochemistry.ecm import ECMModel
from src.physics.thermodynamics.lumped_thermal import LumpedThermalModel


class ElectroThermalCoupling:
    """
    Couples ECM and Thermal models.
    As temperature increases, internal resistance decreases (Arrhenius).
    As resistance increases, Joule heating increases.
    """
    def __init__(self, ecm: ECMModel, thermal: LumpedThermalModel):
        self.ecm = ecm
        self.thermal = thermal

    def step_coupled(self, current: float, duration: float, dt: float,
                     t_amb: float, init_temp_core: float, init_soc: float) -> dict:
        """
        Simulate a coupled time step.
        """
        steps = int(duration / dt)
        t_cores = np.zeros(steps)
        socs = np.zeros(steps)
        voltages = np.zeros(steps)

        t_c = init_temp_core
        soc = init_soc
        v_rc1 = 0.0
        v_rc2 = 0.0

        cap_as = self.ecm.capacity_ah * 3600.0

        for i in range(steps):
             # Temperature scaling for R0 (Arrhenius simplified)
             # Higher temp -> lower resistance
             r0_t = self.ecm.r0 * np.exp(2000 * (1.0/t_c - 1.0/298.15))

             # ECM Step
             soc -= (current * dt) / cap_as
             soc = np.clip(soc, 0.0, 1.0)

             dv1 = -v_rc1 / (self.ecm.r1 * self.ecm.c1) + current / self.ecm.c1
             dv2 = -v_rc2 / (self.ecm.r2 * self.ecm.c2) + current / self.ecm.c2
             v_rc1 += dv1 * dt
             v_rc2 += dv2 * dt

             ocv = self.ecm.ocv_soc_curve(soc)
             v = ocv - current * r0_t - v_rc1 - v_rc2

             # Thermal Step
             q_dot = self.thermal.heat_generation(current, v, ocv, 0.0002, t_c)
             t_c, _ = self.thermal.step_temperature(t_c, t_c, t_amb, q_dot, dt)

             # Store
             t_cores[i] = t_c
             socs[i] = soc
             voltages[i] = v

        return {
             "time": np.arange(steps) * dt,
             "temperature": t_cores,
             "voltage": voltages,
             "soc": socs
        }
