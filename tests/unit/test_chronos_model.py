"""
Unit tests for ChronosZeroShotModel.

Tests verify:
  1. Interface compliance with BatteryModel ABC
  2. predict() output shapes on synthetic data
  3. predict_distribution() output shapes
  4. fit() is a no-op (zero-shot semantics)
  5. save/load roundtrip preserves configuration
  6. predict_single_battery() convenience method

All tests use synthetic data to avoid requiring actual NASA .mat files.
Tests are skipped if chronos-forecasting is not installed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if chronos is available
try:
    import chronos  # noqa: F401
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False

skip_no_chronos = pytest.mark.skipif(
    not CHRONOS_AVAILABLE,
    reason="chronos-forecasting not installed"
)


class _FakeChronosPipeline:
    """Deterministic local stand-in for ChronosPipeline used in unit tests."""

    def predict(
        self,
        context_tensor,
        prediction_length: int,
        num_samples: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ):
        last_value = float(context_tensor[-1].item()) if context_tensor.numel() else 1.0
        slope = -0.01 * max(temperature, 0.1)
        base_forecast = last_value + slope * torch.arange(
            prediction_length, dtype=torch.float64
        )
        sample_offsets = torch.linspace(-0.02, 0.02, num_samples, dtype=torch.float64)
        forecast = base_forecast.unsqueeze(0) + sample_offsets.unsqueeze(1)
        return forecast.unsqueeze(0)


@pytest.fixture(autouse=CHRONOS_AVAILABLE)
def fake_chronos_pipeline(monkeypatch):
    """Replace live Chronos loading with a deterministic in-memory stub."""
    if not CHRONOS_AVAILABLE:
        yield
        return

    from src.models.chronos_model import ChronosZeroShotModel

    def _ensure_pipeline(self):
        if self._pipeline is None:
            self._pipeline = _FakeChronosPipeline()

    monkeypatch.setattr(ChronosZeroShotModel, "_ensure_pipeline", _ensure_pipeline)
    yield


@skip_no_chronos
class TestChronosInterface:
    """Verify ChronosZeroShotModel conforms to BatteryModel ABC."""

    @pytest.fixture
    def model(self):
        from src.models.chronos_model import ChronosZeroShotModel
        return ChronosZeroShotModel(
            model_id="amazon/chronos-t5-tiny",
            num_samples=5,
            prediction_length=10,
            device_map="cpu",
            torch_dtype_str="float32",
        )

    def test_is_battery_model_subclass(self, model):
        from src.models.base import BatteryModel
        assert isinstance(model, BatteryModel)

    def test_has_required_methods(self, model):
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "save")
        assert hasattr(model, "load")
        assert hasattr(model, "predict_distribution")
        assert hasattr(model, "get_params")

    def test_name_attribute(self, model):
        assert model.name == "chronos_zero_shot"


@skip_no_chronos
class TestChronosPredict:
    """Test prediction shapes and values on synthetic data."""

    @pytest.fixture
    def model(self):
        from src.models.chronos_model import ChronosZeroShotModel
        return ChronosZeroShotModel(
            model_id="amazon/chronos-t5-tiny",
            num_samples=5,
            prediction_length=10,
            context_ratio=0.8,
            device_map="cpu",
            torch_dtype_str="float32",
        )

    @pytest.fixture
    def synthetic_capacity(self) -> np.ndarray:
        """Synthetic exponential decay curve mimicking battery degradation."""
        np.random.seed(42)
        cycles = np.arange(100)
        capacity = 2.0 * np.exp(-0.005 * cycles) + np.random.normal(0, 0.01, 100)
        return capacity

    def test_predict_single_battery_shapes(self, model, synthetic_capacity):
        context_len = 80
        pred_len = 10
        gt, mean, lower, upper = model.predict_single_battery(
            capacity_series=synthetic_capacity,
            context_length=context_len,
            prediction_length=pred_len,
        )
        assert gt.shape == (pred_len,)
        assert mean.shape == (pred_len,)
        assert lower.shape == (pred_len,)
        assert upper.shape == (pred_len,)

    def test_predict_values_are_finite(self, model, synthetic_capacity):
        gt, mean, lower, upper = model.predict_single_battery(
            capacity_series=synthetic_capacity,
            context_length=80,
            prediction_length=10,
        )
        assert np.all(np.isfinite(mean)), "mean contains NaN or Inf"
        assert np.all(np.isfinite(lower)), "lower contains NaN or Inf"
        assert np.all(np.isfinite(upper)), "upper contains NaN or Inf"

    def test_lower_le_upper(self, model, synthetic_capacity):
        gt, mean, lower, upper = model.predict_single_battery(
            capacity_series=synthetic_capacity,
            context_length=80,
            prediction_length=10,
        )
        assert np.all(lower <= upper), "lower bound exceeds upper bound"

    def test_predict_via_abc_interface(self, model, synthetic_capacity):
        """Test the standard BatteryModel predict() path."""
        X = synthetic_capacity.reshape(-1, 1)  # (100, 1) with capacity as col 0

        # fit stores context
        model.fit(X[:80], np.zeros(80))

        # predict on the remaining data
        model.prediction_length = 10
        mean, lower, upper = model.predict(X[80:90])
        assert mean.shape == (10,)
        assert lower.shape == (10,)
        assert upper.shape == (10,)


@skip_no_chronos
class TestChronosDistribution:
    """Test predict_distribution returns proper sample matrix."""

    @pytest.fixture
    def model(self):
        from src.models.chronos_model import ChronosZeroShotModel
        return ChronosZeroShotModel(
            model_id="amazon/chronos-t5-tiny",
            num_samples=5,
            prediction_length=10,
            device_map="cpu",
            torch_dtype_str="float32",
        )

    def test_distribution_shape(self, model):
        np.random.seed(42)
        capacity = 2.0 * np.exp(-0.005 * np.arange(100))
        X = capacity.reshape(-1, 1)
        model.fit(X[:80], np.zeros(80))

        n_samples = 15
        samples = model.predict_distribution(X[80:90], n_samples=n_samples)
        assert samples.shape == (n_samples, 10)


@skip_no_chronos
class TestChronosFitIsNoop:
    """Verify that fit() does not modify model weights."""

    def test_fit_returns_self(self):
        from src.models.chronos_model import ChronosZeroShotModel
        model = ChronosZeroShotModel(
            model_id="amazon/chronos-t5-tiny",
            num_samples=5,
            device_map="cpu",
            torch_dtype_str="float32",
        )
        X = np.random.randn(50, 1)
        y = np.random.randn(50)
        result = model.fit(X, y)
        assert result is model

    def test_fit_stores_context(self):
        from src.models.chronos_model import ChronosZeroShotModel
        model = ChronosZeroShotModel(
            model_id="amazon/chronos-t5-tiny",
            num_samples=5,
            device_map="cpu",
            torch_dtype_str="float32",
        )
        X = np.random.randn(50, 1)
        y = np.random.randn(50)
        model.fit(X, y)
        assert model._context is not None
        assert len(model._context) == 50


@skip_no_chronos
class TestChronosSaveLoad:
    """Test config serialization roundtrip."""

    def test_save_load_roundtrip(self, workspace_tmp_path):
        from src.models.chronos_model import ChronosZeroShotModel
        original = ChronosZeroShotModel(
            model_id="amazon/chronos-t5-small",
            num_samples=42,
            prediction_length=15,
            context_ratio=0.75,
            device_map="cpu",
            torch_dtype_str="float32",
            confidence_level=0.9,
            temperature=0.8,
            top_k=30,
            top_p=0.95,
        )

        save_path = workspace_tmp_path / "chronos_config.json"
        original.save(save_path)

        # Verify file exists and is valid JSON
        assert save_path.exists()
        with open(save_path) as f:
            config = json.load(f)
        assert config["model_id"] == "amazon/chronos-t5-small"
        assert config["num_samples"] == 42

        # Load into new instance
        loaded = ChronosZeroShotModel(
            model_id="amazon/chronos-t5-tiny",  # different initial value
            num_samples=1,
        )
        loaded.load(save_path)
        assert loaded.model_id == "amazon/chronos-t5-small"
        assert loaded.num_samples == 42
        assert loaded.prediction_length == 15
        assert loaded.context_ratio == 0.75
        assert loaded.confidence_level == 0.9
        assert loaded.temperature == 0.8
        assert loaded.top_k == 30
        assert loaded.top_p == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
