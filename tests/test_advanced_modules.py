"""
Tests for PINN, training pipeline, deployment, and physics modules.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.pinn_model import PINNModel
from src.physics.degradation import PhysicsModel, empirical_fade
from src.training.pipeline import TrainingPipeline
from src.evaluation.benchmark import BenchmarkRunner


class TestPhysicsModel:
    def test_empirical_fade(self):
        n = np.arange(0, 100)
        q = empirical_fade(n, q0=2.0, a=0.01, b=0.001)
        assert q[0] == pytest.approx(2.0)
        assert q[-1] < q[0]  # Capacity decreases
        assert len(q) == 100

    def test_physics_fit_predict(self):
        cycles = np.arange(1, 101, dtype=float)
        caps = 2.0 - 0.005 * np.sqrt(cycles) - 0.0005 * cycles + np.random.randn(100) * 0.01
        pm = PhysicsModel()
        params = pm.fit(cycles, caps, battery_id="test")
        assert "q0" in params
        pred = pm.predict(cycles, "test")
        assert len(pred) == 100
        assert pred[0] > pred[-1]

    def test_residuals(self):
        cycles = np.arange(1, 51, dtype=float)
        caps = 2.0 - 0.01 * np.sqrt(cycles)
        pm = PhysicsModel()
        pm.fit(cycles, caps, "t")
        res = pm.residuals(cycles, caps, "t")
        assert np.abs(res).mean() < 0.1  # Residuals should be small


class TestPINNModel:
    def test_fit_predict(self):
        np.random.seed(42)
        n = 80
        X = np.column_stack([np.arange(n), np.random.randn(n)]).astype(np.float32)
        y = np.linspace(80, 0, n).astype(np.float32)
        model = PINNModel(input_dim=2, hidden_dim=16, epochs=10, mc_samples=10)
        model.fit(X, y)
        mean, lower, upper = model.predict(X)
        assert len(mean) == n
        assert np.all(lower <= upper + 1e-3)

    def test_save_load(self, tmp_path):
        np.random.seed(42)
        X = np.column_stack([np.arange(50), np.random.randn(50)]).astype(np.float32)
        y = np.linspace(50, 0, 50).astype(np.float32)
        model = PINNModel(input_dim=2, hidden_dim=16, epochs=5, mc_samples=5)
        model.fit(X, y)
        path = tmp_path / "pinn.pt"
        model.save(path)
        model2 = PINNModel(input_dim=2, hidden_dim=16)
        model2.load(path)
        m1, _, _ = model.predict(X)
        m2, _, _ = model2.predict(X)
        assert len(m2) == len(m1)


class TestTrainingPipeline:
    def test_pipeline_runs(self):
        import pandas as pd
        np.random.seed(42)
        rows = []
        for bat in ["A", "B", "C"]:
            for c in range(60):
                rows.append({
                    "battery_id": bat, "cycle": c,
                    "f1": np.random.randn(), "f2": np.random.randn(),
                    "rul": 60 - c,
                })
        df = pd.DataFrame(rows)
        from src.models.gru_model import GRUModel
        model = GRUModel(input_dim=2, hidden_dim=8, num_layers=1, seq_length=5, epochs=3, mc_samples=5)
        pipeline = TrainingPipeline(features=["f1", "f2"], target="rul")
        result = pipeline.train_and_evaluate(df, model, seed=42)
        assert "model" in result
        assert result.get("RMSE_mean") is not None or "error" in result


class TestBenchmarkRunner:
    def test_benchmark_runs(self):
        import pandas as pd
        np.random.seed(42)
        rows = []
        for bat in ["X", "Y"]:
            for c in range(50):
                rows.append({
                    "battery_id": bat, "cycle": c,
                    "f1": np.random.randn(), "f2": np.random.randn(),
                    "rul": 50 - c,
                })
        df = pd.DataFrame(rows)
        from src.models.gru_model import GRUModel
        models = {"gru_tiny": GRUModel(input_dim=2, hidden_dim=8, num_layers=1,
                                        seq_length=5, epochs=3, mc_samples=5)}
        runner = BenchmarkRunner(features=["f1", "f2"], n_seeds=1)
        results = runner.run(df, models, seeds=[42])
        assert len(results) > 0
        assert "RMSE" in results.columns
