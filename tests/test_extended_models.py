"""
Tests for BayesianNN, CNN1D, PINN save/load, and cross-model consistency.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.bayesian_nn import BayesianNNModel
from src.models.cnn1d_model import CNN1DModel
from src.models.pinn_model import PINNModel
from src.models.ensemble_model import DeepEnsemble
from src.uncertainty.decomposition import decompose_ensemble
from src.uncertainty.calibration import calibration_curve, ence


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n = 80
    X = np.random.randn(n, 2).astype(np.float32)
    y = np.linspace(80, 0, n).astype(np.float32)
    return X, y


class TestBayesianNN:
    def test_fit_predict(self, dummy_data):
        X, y = dummy_data
        model = BayesianNNModel(input_dim=2, hidden_dim=16, epochs=10, n_samples=10)
        model.fit(X, y)
        mean, lower, upper = model.predict(X)
        assert len(mean) == len(X)
        assert np.all(lower <= upper + 1e-3)

    def test_predict_distribution(self, dummy_data):
        X, y = dummy_data
        model = BayesianNNModel(input_dim=2, hidden_dim=16, epochs=10, n_samples=10)
        model.fit(X, y)
        samples = model.predict_distribution(X, n_samples=20)
        assert samples.shape == (20, len(X))

    def test_save_load(self, dummy_data, tmp_path):
        X, y = dummy_data
        model = BayesianNNModel(input_dim=2, hidden_dim=16, epochs=5, n_samples=5)
        model.fit(X, y)
        path = tmp_path / "bnn.pt"
        model.save(path)
        model2 = BayesianNNModel(input_dim=2, hidden_dim=16)
        model2.load(path)
        m2, _, _ = model2.predict(X)
        assert len(m2) == len(X)


class TestCNN1D:
    def test_fit_predict(self, dummy_data):
        X, y = dummy_data
        model = CNN1DModel(input_dim=2, channels=[16, 16], seq_length=10, epochs=10, mc_samples=10)
        model.fit(X, y)
        mean, lower, upper = model.predict(X)
        assert len(mean) > 0
        assert np.all(lower <= upper + 1e-3)

    def test_save_load(self, dummy_data, tmp_path):
        X, y = dummy_data
        model = CNN1DModel(input_dim=2, channels=[16, 16], seq_length=10, epochs=5, mc_samples=5)
        model.fit(X, y)
        path = tmp_path / "cnn.pt"
        model.save(path)
        model2 = CNN1DModel(input_dim=2, channels=[16, 16], seq_length=10)
        model2.load(path)
        m2, _, _ = model2.predict(X)
        assert len(m2) > 0


class TestUncertaintyDecomposition:
    def test_decompose_ensemble(self):
        preds = np.random.randn(10, 50)  # 10 members, 50 points
        result = decompose_ensemble(preds)
        assert "mean" in result
        assert "total_std" in result
        assert "aleatoric_std" in result
        assert "epistemic_std" in result
        assert len(result["mean"]) == 50
        assert np.all(result["total_std"] >= 0)
        assert np.all(result["epistemic_std"] >= 0)


class TestCalibration:
    def test_calibration_curve(self):
        np.random.seed(42)
        y = np.random.randn(100)
        mu = y + np.random.randn(100) * 0.1
        sigma = np.ones(100) * 0.5
        expected, observed = calibration_curve(y, mu, sigma)
        assert len(expected) == 9
        assert len(observed) == 9
        assert np.all(observed >= 0) and np.all(observed <= 1)

    def test_ence(self):
        np.random.seed(42)
        y = np.random.randn(100)
        mu = y  # Perfect predictions
        sigma = np.ones(100)
        score = ence(y, mu, sigma)
        assert 0 <= score <= 1


class TestDeepEnsembleWithBNN:
    def test_ensemble_of_bnn(self, dummy_data):
        X, y = dummy_data
        base = BayesianNNModel(input_dim=2, hidden_dim=8, epochs=5, n_samples=5)
        ens = DeepEnsemble(base, n_members=2)
        ens.fit(X, y)
        mean, lower, upper = ens.predict(X)
        assert len(mean) == len(X)
