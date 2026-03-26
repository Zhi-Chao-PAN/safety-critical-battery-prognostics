"""
Battery Degradation Physics Models.

Empirical capacity fade:
  Q(n) = Q0 - a*sqrt(n) - b*n

  Term 1: SEI layer growth (diffusion-limited, sqrt)
  Term 2: Lithium plating / active material loss (linear)

Arrhenius temperature dependence:
  k(T) = A * exp(-Ea / (R * T))
"""

import logging

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

# Gas constant (J/(mol*K))
R_GAS = 8.314


def empirical_fade(n: np.ndarray, q0: float, a: float, b: float) -> np.ndarray:
    """Q(n) = Q0 - a*sqrt(|n|) - b*n"""
    return q0 - a * np.sqrt(np.abs(n)) - b * n


def arrhenius_rate(T_celsius: float, A: float = 1.0, Ea: float = 30000.0) -> float:
    """Temperature-dependent degradation rate. T in Celsius."""
    T_kelvin = T_celsius + 273.15
    return A * np.exp(-Ea / (R_GAS * T_kelvin))


class PhysicsModel:
    """
    Fit and predict using empirical capacity fade model.
    Serves as physics baseline and PINN prior.
    """

    def __init__(self):
        self.params: dict[str, dict[str, float]] = {}  # battery_id -> {q0, a, b}
        self.global_params: dict[str, float] | None = None

    def fit(self, cycles: np.ndarray, capacities: np.ndarray, battery_id: str = "global") -> dict[str, float]:
        """Fit empirical fade model to one battery's data."""
        try:
            safe_cycles = np.maximum(np.abs(cycles), 1e-6)
            y_max = max(float(np.max(np.abs(capacities))), 1.0)
            popt, _ = curve_fit(
                empirical_fade, safe_cycles, capacities,
                p0=[capacities[0], 0.01 * y_max, 0.001 * y_max],
                bounds=([0, 0, 0], [y_max * 5, y_max, y_max * 0.1]),
                maxfev=5000,
            )
            params = {"q0": float(popt[0]), "a": float(popt[1]), "b": float(popt[2])}
            self.params[battery_id] = params
            logger.info(f"Physics fit [{battery_id}]: Q0={params['q0']:.3f}, a={params['a']:.4f}, b={params['b']:.5f}")
            return params
        except Exception as e:
            logger.warning(f"Physics fit failed for {battery_id}: {e}. Using defaults.")
            params = {"q0": float(capacities[0]), "a": 0.01, "b": 0.001}
            self.params[battery_id] = params
            return params

    def fit_all(self, df, cycle_col: str = "cycle", cap_col: str = "capacity", group_col: str = "battery_id"):
        """Fit to all batteries in DataFrame."""
        for bat_id in df[group_col].unique():
            sub = df[df[group_col] == bat_id].sort_values(cycle_col)
            self.fit(sub[cycle_col].values, sub[cap_col].values, battery_id=bat_id)

        # Global fit (all data pooled)
        all_cycles = df[cycle_col].values
        all_caps = df[cap_col].values
        self.global_params = self.fit(all_cycles, all_caps, battery_id="global")

    def predict(self, cycles: np.ndarray, battery_id: str = "global") -> np.ndarray:
        """Predict capacity using fitted physics model."""
        params = self.params.get(battery_id, self.global_params or {"q0": 2.0, "a": 0.01, "b": 0.001})
        return empirical_fade(cycles, params["q0"], params["a"], params["b"])

    def residuals(self, cycles: np.ndarray, capacities: np.ndarray, battery_id: str = "global") -> np.ndarray:
        """Compute residuals (data - physics). This is what PINN's NN learns."""
        pred = self.predict(cycles, battery_id)
        return capacities - pred

    def physics_loss(self, pred_capacity: np.ndarray, cycles: np.ndarray, battery_id: str = "global") -> float:
        """
        Physics constraint loss for PINN training.
        Penalizes deviation from physics model prediction.
        """
        physics_pred = self.predict(cycles, battery_id)
        return float(np.mean((pred_capacity - physics_pred) ** 2))

    def monotonicity_loss(self, pred_capacity: np.ndarray) -> float:
        """Soft constraint: Capacity should generally decrease."""
        diffs = np.diff(pred_capacity)
        violations = np.maximum(diffs, 0)  # Penalize increases
        return float(np.mean(violations ** 2))
