"""
Aleatoric (Data) Uncertainty Quantification.
Estimates the inherent noise in sensor measurements (Voltage, Current, Temp)
to provide a lower bound on the predictive uncertainty.
"""

import numpy as np


class DataUncertaintyEstimator:
    """
    Heteroscedastic Aleatoric Uncertainty model.
    Learns to predict the output variance (sigma^2) alongside the mean.
    """
    def __init__(self, base_noise_v: float = 0.005,
                       base_noise_i: float = 0.05,
                       base_noise_t: float = 0.5):
        # Baseline sensor noise margins
        self.sigma_v = base_noise_v
        self.sigma_i = base_noise_i
        self.sigma_t = base_noise_t

    def estimate_capacity_noise(self, v_std: float, i_std: float, dt: float) -> float:
        """
        Propagation of uncertainty from raw sensors sequentially 
        into Capacity (Ah) via Coulomb counting integration.
        Variance(Q) = sum(Variance(I * dt))
        """
        # delta_Q = I * dt / 3600
        # Var(delta_Q) = Var(I) * (dt/3600)^2
        var_q_step = (i_std * (dt / 3600.0)) ** 2
        return var_q_step

    def smooth_measurements(self, data: np.ndarray, window: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts smoothed mean and rolling std to quantify local noise.
        """
        smoothed = np.convolve(data, np.ones(window)/window, mode='same')

        # Simple local variance estimation
        local_var = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window//2)
            end = min(len(data), i + window//2 + 1)
            local_var[i] = np.var(data[start:end])

        local_std = np.sqrt(local_var)
        return smoothed, local_std
