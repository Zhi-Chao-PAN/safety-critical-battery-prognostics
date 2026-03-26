"""
Electrochemical Models
Implements an Equivalent Circuit Model (ECM) as a physical prior for Battery RUL.
This simulates internal resistance growth and capacity displacement.
"""

import numpy as np


class ECMModel:
    """
    Second-Order RC Equivalent Circuit Model.
    
    Equations:
      V = OCV(SOC) - I*R_0 - V_RC1 - V_RC2
      dV_RCi/dt = -V_RCi / (R_i * C_i) + I / C_i
    """
    def __init__(self, r0: float = 0.05, r1: float = 0.02, c1: float = 1000.0,
                 r2: float = 0.01, c2: float = 5000.0, capacity_ah: float = 2.0):
        self.r0 = r0
        self.r1 = r1
        self.c1 = c1
        self.r2 = r2
        self.c2 = c2
        self.capacity_ah = capacity_ah # Initial nominal capacity

    def ocv_soc_curve(self, soc: float | np.ndarray) -> np.ndarray:
        """
        Approximate OCV-SOC curve for Li-ion (e.g. NMC/Graphite).
        Returns voltage in V for a given State of Charge (0 to 1).
        """
        # Empirical polynomial fit
        soc_safe = np.clip(soc, 0.01, 0.99)
        ocv = 3.0 + 1.2 * soc_safe - 0.2 * np.exp(-20 * soc_safe) + 0.1 * np.exp(10 * (soc_safe - 1))
        return ocv

    def simulate_discharge(self, current: float, dt: float, duration: float) -> dict:
        """
        Simulate a constant current discharge.
        
        Args:
           current: Discharge current (A)
           dt: Time step (s)
           duration: Total time (s)
        """
        steps = int(duration / dt)
        soc = np.ones(steps)
        v = np.zeros(steps)
        v_rc1, v_rc2 = 0.0, 0.0

        # Capacity in As
        cap_as = self.capacity_ah * 3600.0

        for i in range(steps):
            if i > 0:
                soc[i] = soc[i-1] - (current * dt) / cap_as

            soc[i] = max(0.0, soc[i])

            # Update RC networks (Euler forward)
            dv1 = -v_rc1 / (self.r1 * self.c1) + current / self.c1
            dv2 = -v_rc2 / (self.r2 * self.c2) + current / self.c2
            v_rc1 += dv1 * dt
            v_rc2 += dv2 * dt

            # Terminal voltage
            v[i] = self.ocv_soc_curve(soc[i]) - current * self.r0 - v_rc1 - v_rc2

        return {
            "time": np.arange(steps) * dt,
            "voltage": v,
            "soc": soc
        }
