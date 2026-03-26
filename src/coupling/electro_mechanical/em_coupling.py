"""
Electro-Mechanical Coupling
Simulates swelling and internal stress caused by lithium intercalation.
"""


class ElectroMechanicalCoupling:
    """
    Couples SOC/Concentration logic to mechanical stress.
    Stress leads to micro-cracking and loss of active material (LAM).
    """
    def __init__(self, youngs_modulus: float = 10e9, partial_molar_volume: float = 3.497e-6):
        self.E = youngs_modulus
        self.omega = partial_molar_volume

    def calculate_surface_stress(self, c_surf: float, c_avg: float) -> float:
        """
        Simplified radial stress at the particle surface.
        sigma_t = (Omega * E) / (3 * (1 - nu)) * (C_avg - C_surf)
        """
        nu = 0.3 # Poisson's ratio
        stress = (self.omega * self.E) / (3 * (1 - nu)) * (c_avg - c_surf)
        return stress
