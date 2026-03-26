"""Battery Prognostics - Models package."""

# Note: All imports are lazy/optional to avoid breaking the entire package
# when some dependencies or modules are missing.

from src.models.base import BatteryModel

# Chronos is an optional dependency (requires chronos-forecasting)
ChronosZeroShotModel = None
try:
    from src.models.chronos_model import ChronosZeroShotModel
except ImportError:
    pass

# Other models are optional and imported lazily
LSTMModel = None
GRUModel = None
TCNModel = None
TransformerModel = None
DeepEnsemble = None
PINNModel = None
BayesianNNModel = None
CNN1DModel = None
BTCNModel = None

__all__ = [
    "BatteryModel",
    "ChronosZeroShotModel",
]
