import numpy as np
import pytest

from src.evaluation import real_data_validation
from src.evaluation.real_data_validation import (
    apply_capacity_corruption,
    build_condition_signals,
    evaluate_models_on_signals,
)


@pytest.mark.parametrize("corruption", ["gaussian", "bias_drift", "impulse_spikes", "missing_segments"])
def test_supported_capacity_corruptions_are_deterministic_and_finite(corruption):
    capacity = np.linspace(2.0, 1.2, 40)
    corrupted_a = apply_capacity_corruption(capacity, corruption=corruption, severity=0.5, seed=42)
    corrupted_b = apply_capacity_corruption(capacity, corruption=corruption, severity=0.5, seed=42)

    assert corrupted_a.shape == capacity.shape
    assert np.allclose(corrupted_a, corrupted_b)
    assert np.isfinite(corrupted_a).all()
    assert not np.allclose(corrupted_a, capacity)


def test_build_condition_signals_can_include_clean_and_custom_label():
    capacity = np.linspace(2.0, 1.2, 10)
    signals = build_condition_signals(
        capacity,
        corruption="bias_drift",
        severity=0.5,
        seed=7,
        include_clean=True,
        noisy_label="drifted",
    )

    assert list(signals) == ["clean", "drifted"]
    assert np.allclose(signals["clean"], capacity)
    assert np.isfinite(signals["drifted"]).all()


def test_evaluate_models_on_signals_respects_named_conditions():
    class FakeModel:
        def __init__(self, prediction):
            self.prediction = np.asarray(prediction, dtype=np.float64)

        def predict(self, X, **kwargs):
            return self.prediction.copy(), self.prediction.copy(), self.prediction.copy()

    cycles = np.arange(1.0, 7.0)
    capacity = np.array([2.0, 1.92, 1.85, 1.78, 1.70, 1.62])
    signals = {
        "bias_drift": np.array([2.0, 2.05, 1.95, 2.00, 1.88, 1.90]),
        "missing_segments": np.array([2.0, 1.92, 1.92, 1.92, 1.70, 1.62]),
    }

    result = evaluate_models_on_signals(
        models={"pinn": FakeModel(capacity), "lstm": FakeModel(capacity)},
        cell_id="CELL_X",
        cycles=cycles,
        capacity=capacity,
        signals=signals,
        scatter_condition="bias_drift",
    )

    assert np.allclose(result.noisy_capacity, signals["bias_drift"])
    assert set(result.evaluations["pinn"]) == {"bias_drift", "missing_segments"}
    assert result.evaluations["lstm"]["missing_segments"].violation_rate == 0.0


def test_train_reference_models_locks_training_seed(monkeypatch):
    captured = {}

    class FakePINN:
        def __init__(self, **kwargs):
            captured["pinn_kwargs"] = kwargs

        def fit(self, X, y):
            captured["pinn_fit_shape"] = (X.shape, y.shape)
            return self

    class FakeLSTM:
        def __init__(self, **kwargs):
            captured["lstm_kwargs"] = kwargs

        def fit(self, X, y, **kwargs):
            captured["lstm_fit_shape"] = (X.shape, y.shape, tuple(kwargs["group_ids"]))
            return self

    monkeypatch.setattr(real_data_validation, "PINNModel", FakePINN)
    monkeypatch.setattr(real_data_validation, "LSTMModel", FakeLSTM)
    monkeypatch.setattr(real_data_validation, "set_global_seed", lambda seed: captured.setdefault("seed", seed))

    train_cells = {"CELL_A": (np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.8, 1.6]))}
    models = real_data_validation.train_reference_models(train_cells, training_seed=123)

    assert captured["seed"] == 123
    assert set(models) == {"pinn", "lstm"}
    assert captured["pinn_fit_shape"][0] == (3, 2)
    assert captured["lstm_fit_shape"][0] == (3, 2)
