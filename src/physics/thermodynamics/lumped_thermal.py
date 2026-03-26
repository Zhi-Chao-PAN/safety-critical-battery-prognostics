"""
Thermodynamic Models
Simulates battery heat generation (reversible and irreversible) 
and core/surface temperature evolution based on lumped thermal mass.
"""


class LumpedThermalModel:
    """
    Two-state lumped thermal model.
    State 1: Core temperature (T_c)
    State 2: Surface temperature (T_s)
    """
    def __init__(self,
                 c_p: float = 800.0, # Specific heat capacity J/(kg*K)
                 mass: float = 0.5,  # Cell mass kg
                 r_cond: float = 1.0, # Internal conduction resistance K/W
                 h_conv: float = 10.0, # Convective heat transfer coeff W/(m^2*K)
                 area: float = 0.05):  # Surface area m^2
        self.c_p = c_p
        self.mass = mass
        self.c_th = mass * c_p # Thermal capacitance
        self.r_cond = r_cond
        self.h_conv = h_conv
        self.area = area
        self.r_conv = 1.0 / (h_conv * area)

    def heat_generation(self, current: float, v_terminal: float, ocv: float, entropic_coeff: float, temp: float) -> float:
        """
        Calculates total heat generation Q_dot.
        Q_dot = I * (OCV - V) + I * T * (dU/dT)
        (Irreversible Ohmic/Polarization) + (Reversible Entropic)
        """
        q_irrev = abs(current * (ocv - v_terminal))
        q_rev = current * temp * entropic_coeff
        return q_irrev + q_rev

    def step_temperature(self, t_core: float, t_surf: float, t_amb: float, q_dot: float, dt: float) -> tuple[float, float]:
        """
        Euler forward step for temperatures.
        dT_c/dt = Q_dot / C_c + (T_s - T_c) / (R_cond * C_c)
        dT_s/dt = (T_c - T_s) / (R_cond * C_s) + (T_amb - T_s) / (R_conv * C_s)
        (Simplified by assuming C_s << C_c, or lumped into one body).
        
        Using single body lumped capacitance for simplicity here:
        dT/dt = (Q_dot - (T - T_amb)/R_total) / C_th
        """
        r_tot = self.r_cond + self.r_conv
        dT = (q_dot - (t_core - t_amb) / r_tot) / self.c_th
        t_core_new = t_core + dT * dt
        return t_core_new, t_core_new # Assume uniform for single lumped mass
