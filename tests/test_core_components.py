"""
Unit tests for core components of PINN battery prognostics system.

Tests:
1. test_dataset_nan_handling: Verify BatteryDataset handles NaN and extreme values correctly
2. test_monitor_early_stop: Verify TrainingMonitor triggers early stop on Inf losses
3. test_physics_constraint_shape: Verify MonotonicityConstraint returns scalar loss
"""

import math
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.infrastructure.dataset import BatteryDataset
from src.infrastructure.train_loop import TrainingMonitor
from src.physics.constraints import MonotonicityConstraint


@pytest.mark.unit
def test_dataset_nan_handling():
    """
    Test BatteryDataset correctly handles NaN values and extreme outliers.
    
    Constructs a dataset with known NaN values and extreme anomalies,
    verifies that the dataset properly cleans or intercepts them without
    crashing or producing invalid samples.
    """
    # Create test data with NaN and extreme values
    data = pd.DataFrame({
        'cycle': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'capacity': [1.0, 0.98, np.nan, 0.94, 1e6, 0.90, np.nan, np.nan, 0.82, -100],
        'rul': [99, 98, 97, 96, 95, np.nan, 93, 92, 91, 90]
    })
    
    feature_columns = ['cycle', 'capacity']
    target_column = 'rul'
    
    # Test with interpolation strategy
    dataset = BatteryDataset(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        enable_anomaly_detection=True,
        nan_replacement='interpolate',
        clip_outliers=True,
        outlier_std_threshold=1.0
    )
    
    # Verify all samples are present (interpolation should not drop samples)
    assert len(dataset) == 10, "Dataset should retain all samples with interpolation"
    
    # Verify no NaN values remain in features
    for i in range(len(dataset)):
        features, target = dataset[i]
        assert not torch.isnan(features).any(), f"NaN found in features at index {i}"
        assert not torch.isnan(target), f"NaN found in target at index {i}"
        assert not torch.isinf(features).any(), f"Inf found in features at index {i}"
        assert not torch.isinf(target), f"Inf found in target at index {i}"
    
    # Verify extreme values are clipped
    stats = dataset.get_statistics()
    assert stats['capacity_max'] < 1e6, "Extreme capacity value was not clipped"
    assert stats['capacity_min'] >= -100, "Negative capacity value was not clipped"
    
    # Test with drop strategy
    dataset_drop = BatteryDataset(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        enable_anomaly_detection=True,
        nan_replacement='drop',
        clip_outliers=False
    )
    
    # Should drop rows with NaN
    assert len(dataset_drop) < 10, "Dataset should drop samples with NaN when using 'drop' strategy"


@pytest.mark.unit
def test_monitor_early_stop():
    """
    Test TrainingMonitor correctly triggers early stop on consecutive Inf losses.
    
    Injects 3 consecutive Inf loss values into the monitor and verifies
    it correctly returns False to indicate training should stop after
    exceeding the tolerance threshold.
    """
    monitor = TrainingMonitor(
        nan_tolerance=3,
        inf_tolerance=3,
        save_on_anomaly=False
    )
    
    model = nn.Linear(2, 1)
    
    # First epoch: valid loss
    loss_dict = {"total_loss": 0.5, "data_loss": 0.4, "constraint_loss": 0.1}
    should_continue = monitor.on_epoch_end(1, loss_dict, model)
    assert should_continue is True, "Should continue training after valid loss"
    assert monitor.consecutive_inf_count == 0, "Inf count should be 0 after valid loss"
    
    # Inject 3 consecutive Inf losses
    for epoch in range(2, 5):
        loss_dict = {"total_loss": math.inf, "data_loss": math.inf, "constraint_loss": 0.0}
        should_continue = monitor.on_epoch_end(epoch, loss_dict, model)
        if epoch < 4:
            assert should_continue is True, f"Should continue before exceeding Inf tolerance at epoch {epoch}"
        else:
            assert should_continue is False, "Should stop training after exceeding Inf tolerance"
    
    assert monitor.consecutive_inf_count == 3, "Inf count should be 3 after 3 consecutive Inf losses"


@pytest.mark.unit
def test_physics_constraint_shape():
    """
    Test MonotonicityConstraint returns scalar loss with intact computation graph.
    
    Constructs a dummy tensor input, passes it through the constraint, and
    verifies:
    1. Output loss is a scalar (0-dimensional tensor)
    2. Loss has requires_grad=True indicating computation graph is preserved
    3. Backward pass can be performed without errors
    """
    constraint = MonotonicityConstraint(weight=0.05, adaptive=False)
    
    # Create dummy input tensors with requires_grad enabled
    batch_size = 32
    predictions = torch.randn(batch_size, 1, requires_grad=True)
    cycles = torch.linspace(1, 100, batch_size).unsqueeze(1)
    
    inputs = {
        "cycles": cycles,
        "features": torch.randn(batch_size, 2)
    }
    
    # Compute constraint loss
    loss = constraint.compute_loss(predictions, inputs)
    
    # Verify loss is a scalar
    assert loss.dim() == 0, "Constraint loss should be a scalar (0-dimensional tensor)"
    assert loss.shape == torch.Size([]), "Loss shape should be empty tuple for scalar"
    
    # Verify computation graph is intact (backward pass works)
    assert loss.requires_grad is True, "Loss should have requires_grad=True"
    
    # Perform backward pass to verify no errors
    loss.backward()
    assert predictions.grad is not None, "Gradients should be computed for predictions"
    assert predictions.grad.shape == predictions.shape, "Gradient shape should match predictions shape"
