from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.capacity_to_rul import capacity_trajectory_to_rul_series
from src.models.pinn_model import PINNModel


def create_capacity_eval_df() -> pd.DataFrame:
    rows = []
    for battery_idx, battery_id in enumerate(["BAT_A", "BAT_B", "BAT_C"]):
        cycles = np.arange(1.0, 9.0)
        capacity = 2.05 - 0.11 * cycles - 0.02 * battery_idx
        rul = capacity_trajectory_to_rul_series(capacity, cycles, eol_threshold=1.4)

        for cycle, cap, rul_value in zip(cycles, capacity, rul):
            rows.append(
                {
                    "battery_id": battery_id,
                    "cycle": cycle,
                    "capacity": cap,
                    "rul": rul_value,
                }
            )
    return pd.DataFrame(rows)


def test_benchmark_runner_trains_pinn_on_capacity_and_evaluates_rul(monkeypatch, workspace_tmp_path):
    df = create_capacity_eval_df()
    trained_targets: list[np.ndarray] = []
    fit_group_lengths: list[int] = []

    def fake_fit(self, X, y, **kwargs):
        trained_targets.append(np.asarray(y).copy())
        fit_group_lengths.append(len(kwargs.get("group_ids", [])))
        return self

    def fake_predict(self, X, **kwargs):
        capacity = np.asarray(X)[:, 1].astype(float)
        return capacity, capacity - 0.01, capacity + 0.01

    monkeypatch.setattr(PINNModel, "fit", fake_fit)
    monkeypatch.setattr(PINNModel, "predict", fake_predict)

    runner = BenchmarkRunner(
        features=["cycle", "capacity"],
        target="rul",
        group_col="battery_id",
        n_seeds=1,
        results_dir=str(workspace_tmp_path),
    )
    model = PINNModel(input_dim=2, epochs=1, patience=1, device="cpu")

    results = runner.run(df, {"pinn": model}, seeds=[42])

    assert trained_targets
    assert all(group_len > 0 for group_len in fit_group_lengths)
    for y_train in trained_targets:
        assert np.max(y_train) < 2.2, "PINN should train on capacity values, not RUL countdowns"
        assert np.min(y_train) > 1.0

    assert not results.empty
    assert results["RMSE"].max() < 1e-8
    assert Path(workspace_tmp_path, "benchmark_results.csv").exists()
