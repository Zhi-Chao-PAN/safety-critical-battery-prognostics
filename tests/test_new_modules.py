"""
Tests for new model zoo - shape, gradient, interface compliance.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base import BatteryModel
from src.models.lstm_model import LSTMModel
from src.models.gru_model import GRUModel
from src.models.tcn_model import TCNModel
from src.models.transformer_model import TransformerModel
from src.models.ensemble_model import DeepEnsemble


# ── Fixtures ──

@pytest.fixture
def dummy_data():
    """Generate dummy sequential data for testing."""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2).astype(np.float32)
    y = np.linspace(100, 0, n).astype(np.float32)
    return X, y


@pytest.fixture
def seq_models():
    """All sequence models with minimal config for fast testing."""
    return [
        LSTMModel(input_dim=2, hidden_dim=16, num_layers=1, seq_length=10, epochs=5, mc_samples=10),
        GRUModel(input_dim=2, hidden_dim=16, num_layers=1, seq_length=10, epochs=5, mc_samples=10),
        TCNModel(input_dim=2, num_channels=[16, 16], seq_length=10, epochs=5, mc_samples=10),
        TransformerModel(input_dim=2, d_model=16, nhead=2, num_layers=1, seq_length=10, epochs=5, mc_samples=10),
    ]


# ── Interface Tests ──

class TestModelInterface:
    def test_all_inherit_base(self, seq_models):
        for model in seq_models:
            assert isinstance(model, BatteryModel)

    def test_fit_returns_self(self, seq_models, dummy_data):
        X, y = dummy_data
        for model in seq_models:
            result = model.fit(X, y)
            assert result is model, f"{model.name} fit() should return self"

    def test_predict_returns_triple(self, seq_models, dummy_data):
        X, y = dummy_data
        for model in seq_models:
            model.fit(X, y)
            mean, lower, upper = model.predict(X)
            assert len(mean) == len(lower) == len(upper), f"{model.name} predict shapes mismatch"
            assert len(mean) > 0, f"{model.name} returned empty predictions"

    def test_predict_bounds_order(self, seq_models, dummy_data):
        X, y = dummy_data
        for model in seq_models:
            model.fit(X, y)
            mean, lower, upper = model.predict(X)
            assert np.all(lower <= mean + 1e-3), f"{model.name}: lower > mean"
            assert np.all(mean <= upper + 1e-3), f"{model.name}: mean > upper"

    def test_get_params(self, seq_models):
        for model in seq_models:
            params = model.get_params()
            assert "name" in params
            assert isinstance(params, dict)


class TestDeepEnsemble:
    def test_ensemble_fit_predict(self, dummy_data):
        X, y = dummy_data
        base = LSTMModel(input_dim=2, hidden_dim=16, num_layers=1, seq_length=10, epochs=3, mc_samples=5)
        ens = DeepEnsemble(base, n_members=2)
        ens.fit(X, y)
        mean, lower, upper = ens.predict(X)
        assert len(mean) > 0
        assert np.all(lower <= upper + 1e-3)


class TestUncertaintyScoring:
    def test_compute_all_metrics(self):
        from src.uncertainty.scoring import compute_all_metrics
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        m = np.array([11.0, 19.0, 31.0, 38.0, 52.0])
        lo = m - 5
        hi = m + 5
        metrics = compute_all_metrics(y, m, lo, hi)
        assert "RMSE" in metrics
        assert "CRPS" in metrics
        assert "PICP" in metrics
        assert metrics["RMSE"] > 0
        assert 0 <= metrics["PICP"] <= 1


class TestDataModules:
    def test_validator(self):
        from src.data.validator import DataValidator
        import pandas as pd
        df = pd.DataFrame({
            "battery_id": ["B1"] * 5,
            "cycle": [1, 2, 3, 4, 5],
            "capacity": [2.0, 1.9, 1.8, 1.7, -0.1],  # Last one invalid
            "max_temp": [30, 31, 32, 33, 150],  # Last one invalid
            "end_discharge_voltage": [3.5, 3.4, 3.3, 3.2, 3.1],
            "discharge_time": [3600, 3500, 3400, 3300, 3200],
            "rul": [4, 3, 2, 1, 0],
        })
        validator = DataValidator()
        validated, report = validator.validate(df)
        assert report.flagged_rows > 0
        assert report.pass_rate < 1.0

    def test_splitter_logo(self):
        from src.data.splitter import DataSplitter
        import pandas as pd
        df = pd.DataFrame({
            "battery_id": ["A"] * 10 + ["B"] * 10,
            "cycle": list(range(10)) * 2,
            "capacity": np.random.rand(20).tolist(),
        })
        folds = list(DataSplitter.logo_cv(df))
        assert len(folds) == 2


class TestSafetyEngine:
    def test_green_decision(self):
        from src.safety.decision_engine import SafetyDecisionEngine, SafetyLevel
        engine = SafetyDecisionEngine()
        d = engine.decide(rul_mean=50, rul_lower=40, rul_upper=60, epistemic_std=2.0)
        assert d.level == SafetyLevel.GREEN

    def test_red_decision(self):
        from src.safety.decision_engine import SafetyDecisionEngine, SafetyLevel
        engine = SafetyDecisionEngine()
        d = engine.decide(rul_mean=5, rul_lower=2, rul_upper=8, epistemic_std=3.0)
        assert d.level == SafetyLevel.RED

    def test_yellow_decision(self):
        from src.safety.decision_engine import SafetyDecisionEngine, SafetyLevel
        engine = SafetyDecisionEngine()
        d = engine.decide(rul_mean=20, rul_lower=15, rul_upper=25, epistemic_std=8.0)
        assert d.level == SafetyLevel.YELLOW
