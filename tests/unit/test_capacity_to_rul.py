import numpy as np
import pytest

from src.evaluation.capacity_to_rul import capacity_trajectory_to_rul_series


def test_capacity_trajectory_to_rul_series_crossing():
    cycles = np.array([0.0, 1.0, 2.0, 3.0])
    capacity = np.array([2.0, 1.8, 1.3, 1.0])

    rul = capacity_trajectory_to_rul_series(capacity, cycles, eol_threshold=1.4)

    assert rul == pytest.approx([1.8, 0.8, 0.0, 0.0], abs=1e-6)


def test_capacity_trajectory_to_rul_series_right_censored():
    cycles = np.array([10.0, 11.0, 12.0])
    capacity = np.array([2.0, 1.9, 1.8])

    rul = capacity_trajectory_to_rul_series(capacity, cycles, eol_threshold=1.4)

    assert rul == pytest.approx([2.0, 1.0, 0.0], abs=1e-6)


def test_capacity_trajectory_to_rul_series_already_below_threshold():
    cycles = np.array([5.0, 6.0, 7.0])
    capacity = np.array([1.3, 1.2, 1.1])

    rul = capacity_trajectory_to_rul_series(capacity, cycles, eol_threshold=1.4)

    assert rul == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
