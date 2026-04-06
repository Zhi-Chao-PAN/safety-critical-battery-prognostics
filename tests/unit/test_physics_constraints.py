"""
Unit Tests for Physics Constraints Abstraction Layer.

Covers:
1. MonotonicityConstraint — quadratic penalty on capacity rebound
2. SPMResidualConstraint — L2 penalty on NN residuals
3. VoltageConstraint — boundary violation detection
4. TemperatureConstraint — thermal safety ceiling
5. ConstraintManager — aggregate loss + adaptive weighting
6. AdaptiveLossWeighter — sigmoid schedule boundary cases

Hardware: Tested on CPU; CUDA paths validated via conditional.
"""

import pytest
import numpy as np
import torch

from src.physics.constraints import (
    PhysicsConstraint,
    MonotonicityConstraint,
    SPMResidualConstraint,
    VoltageConstraint,
    TemperatureConstraint,
    ConstraintManager,
    create_default_constraint_manager,
    NAN_PENALTY_LOSS,
)
from src.models.pinn_model import AdaptiveLossWeighter


# ──────────────────────── Fixtures ──────────────────────────

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def monotonic_constraint():
    return MonotonicityConstraint(weight=0.05, adaptive=True)


@pytest.fixture
def spm_constraint():
    return SPMResidualConstraint(weight=0.1, adaptive=True)


@pytest.fixture
def voltage_constraint():
    return VoltageConstraint(v_min=0.0, v_max=2.5, weight=0.02, adaptive=True)


@pytest.fixture
def temperature_constraint():
    return TemperatureConstraint(t_max=2.2, weight=0.01, adaptive=True)


@pytest.fixture
def constraint_manager(device):
    return create_default_constraint_manager(str(device))


@pytest.fixture
def adaptive_weighter():
    return AdaptiveLossWeighter(
        lambda_physics_min=0.01,
        lambda_physics_max=1.0,
        lambda_mono_min=0.01,
        lambda_mono_max=0.2,
        transition_sharpness=10.0,
        transition_center=0.6,
    )


# ──────────────────── MonotonicityConstraint ────────────────

class TestMonotonicityConstraint:
    """Monotonicity constraint: penalize capacity rebound (increase)."""

    def test_monotonically_decreasing_zero_loss(self, monotonic_constraint):
        """Perfectly monotonic-decreasing sequence → loss should be exactly 0."""
        # Capacity: 2.0, 1.9, 1.8, ..., strictly decreasing
        predictions = torch.linspace(2.0, 1.0, 20).unsqueeze(1)
        inputs = {"cycles": torch.arange(20, dtype=torch.float32).unsqueeze(1)}

        loss = monotonic_constraint.compute_loss(predictions, inputs)

        assert loss.item() == pytest.approx(0.0, abs=1e-7), \
            f"Monotonically decreasing should yield zero loss, got {loss.item()}"

    def test_capacity_rebound_positive_loss(self, monotonic_constraint):
        """Capacity rebound (increase) → should produce non-zero quadratic penalty."""
        # Insert a rebound: 2.0, 1.9, 1.8, **2.1**, 1.7
        predictions = torch.tensor([2.0, 1.9, 1.8, 2.1, 1.7]).unsqueeze(1)
        inputs = {"cycles": torch.arange(5, dtype=torch.float32).unsqueeze(1)}

        loss = monotonic_constraint.compute_loss(predictions, inputs)

        assert loss.item() > 0, \
            "Capacity rebound should produce positive monotonicity loss"

    def test_constant_capacity_zero_loss(self, monotonic_constraint):
        """Flat capacity (no increase, no decrease) → loss should be 0."""
        predictions = torch.ones(10, 1) * 1.5
        inputs = {"cycles": torch.arange(10, dtype=torch.float32).unsqueeze(1)}

        loss = monotonic_constraint.compute_loss(predictions, inputs)

        assert loss.item() == pytest.approx(0.0, abs=1e-7), \
            "Constant capacity should yield zero monotonicity loss"

    def test_nan_input_returns_penalty_loss(self, monotonic_constraint):
        """NaN in predictions → validate() catches it, returns HIGH PENALTY (not zero).
        
        F4 safety fix: NaN must not silently disable physics constraints.
        The constraint returns 100.0 penalty to force optimizer recovery.
        """
        predictions = torch.tensor([1.0, float('nan'), 0.8]).unsqueeze(1)
        inputs = {"cycles": torch.arange(3, dtype=torch.float32).unsqueeze(1)}

        loss = monotonic_constraint.compute_loss(predictions, inputs)

        assert loss.item() == pytest.approx(NAN_PENALTY_LOSS, abs=1e-5), \
            f"NaN input should return HIGH PENALTY (100.0), got {loss.item()}"

    def test_inf_input_returns_penalty_loss(self, monotonic_constraint):
        """Inf in predictions → should return HIGH PENALTY (not zero).
        
        F4 safety fix: Inf must not silently disable physics constraints.
        """
        predictions = torch.tensor([1.0, float('inf'), 0.8]).unsqueeze(1)
        inputs = {"cycles": torch.arange(3, dtype=torch.float32).unsqueeze(1)}

        loss = monotonic_constraint.compute_loss(predictions, inputs)

        assert loss.item() == pytest.approx(NAN_PENALTY_LOSS, abs=1e-5), \
            f"Inf input should return HIGH PENALTY (100.0), got {loss.item()}"

    def test_single_sample_returns_zero_loss(self, monotonic_constraint):
        """Single sample → no diff to compute, loss should be 0."""
        predictions = torch.tensor([[1.5]])
        inputs = {"cycles": torch.tensor([[0.0]])}

        loss = monotonic_constraint.compute_loss(predictions, inputs)

        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_loss_magnitude_quadratic(self, monotonic_constraint):
        """Verify quadratic penalty: doubling the rebound → 4x the loss."""
        inputs = {"cycles": torch.arange(3, dtype=torch.float32).unsqueeze(1)}

        # Small rebound: 1.0 → 1.1 → 0.9 (rebound = +0.1)
        pred_small = torch.tensor([1.0, 1.1, 0.9]).unsqueeze(1)
        loss_small = monotonic_constraint.compute_loss(pred_small, inputs)

        # Double rebound: 1.0 → 1.2 → 0.9 (rebound = +0.2)
        pred_large = torch.tensor([1.0, 1.2, 0.9]).unsqueeze(1)
        loss_large = monotonic_constraint.compute_loss(pred_large, inputs)

        # Quadratic: 0.2² / 0.1² = 4x (approximately, since mean is over 2 diffs)
        ratio = loss_large.item() / max(loss_small.item(), 1e-10)
        assert ratio > 3.0, \
            f"Quadratic penalty should give ~4x ratio for 2x rebound, got {ratio:.2f}x"


# ──────────────────── SPMResidualConstraint ─────────────────

class TestSPMResidualConstraint:
    """SPM Residual constraint: penalize large NN residuals."""

    def test_zero_residual_zero_loss(self, spm_constraint):
        """Zero residual predictions → loss should be 0."""
        predictions = torch.zeros(10, 1)
        inputs = {}

        loss = spm_constraint.compute_loss(predictions, inputs)

        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_nonzero_residual_positive_loss(self, spm_constraint):
        """Non-zero residuals → should produce positive L2 loss."""
        predictions = torch.randn(10, 1) * 0.5
        inputs = {}

        loss = spm_constraint.compute_loss(predictions, inputs)

        assert loss.item() > 0, "Non-zero residuals should produce positive loss"

    def test_loss_scales_with_magnitude(self, spm_constraint):
        """Higher residual magnitude → higher loss."""
        inputs = {}

        pred_small = torch.ones(10, 1) * 0.1
        loss_small = spm_constraint.compute_loss(pred_small, inputs)

        pred_large = torch.ones(10, 1) * 1.0
        loss_large = spm_constraint.compute_loss(pred_large, inputs)

        assert loss_large.item() > loss_small.item(), \
            "Larger residuals should produce larger loss"


# ──────────────────── VoltageConstraint ─────────────────────

class TestVoltageConstraint:
    """Capacity bound constraint: keep within [v_min, v_max] (capacity range, Ah)."""

    def test_within_range_zero_loss(self, voltage_constraint):
        """Capacity within [0.0, 2.5] → loss should be 0."""
        predictions = torch.tensor([0.5, 1.0, 1.5, 2.0]).unsqueeze(1)
        inputs = {}

        loss = voltage_constraint.compute_loss(predictions, inputs)

        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_overvoltage_positive_loss(self, voltage_constraint):
        """Capacity above 2.5 → should trigger penalty."""
        predictions = torch.tensor([2.8, 3.0]).unsqueeze(1)
        inputs = {}

        loss = voltage_constraint.compute_loss(predictions, inputs)

        assert loss.item() > 0, "Over-bound capacity should produce positive loss"

    def test_undervoltage_positive_loss(self, voltage_constraint):
        """Capacity below 0.0 → should trigger penalty."""
        predictions = torch.tensor([-0.1, -0.5]).unsqueeze(1)
        inputs = {}

        loss = voltage_constraint.compute_loss(predictions, inputs)

        assert loss.item() > 0, "Under-bound capacity should produce positive loss"

    def test_boundary_exact_zero_loss(self, voltage_constraint):
        """Exact boundaries [0.0, 2.5] → should be zero loss."""
        predictions = torch.tensor([0.0, 2.5]).unsqueeze(1)
        inputs = {}

        loss = voltage_constraint.compute_loss(predictions, inputs)

        assert loss.item() == pytest.approx(0.0, abs=1e-7)


# ──────────────────── TemperatureConstraint ─────────────────

class TestTemperatureConstraint:
    """Capacity upper bound constraint: below t_max (capacity ceiling, Ah)."""

    def test_below_max_zero_loss(self, temperature_constraint):
        """Capacity below 2.2 Ah → loss should be 0."""
        predictions = torch.tensor([1.0, 1.5, 2.0]).unsqueeze(1)
        inputs = {}

        loss = temperature_constraint.compute_loss(predictions, inputs)

        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_above_max_positive_loss(self, temperature_constraint):
        """Capacity above 2.2 Ah → should trigger penalty."""
        predictions = torch.tensor([2.5, 3.0]).unsqueeze(1)
        inputs = {}

        loss = temperature_constraint.compute_loss(predictions, inputs)

        assert loss.item() > 0, "Over-bound capacity should produce positive loss"

    def test_exact_boundary_zero_loss(self, temperature_constraint):
        """Exactly 2.2 Ah → should be zero loss."""
        predictions = torch.tensor([2.2]).unsqueeze(1)
        inputs = {}

        loss = temperature_constraint.compute_loss(predictions, inputs)

        assert loss.item() == pytest.approx(0.0, abs=1e-7)


# ──────────────────── ConstraintManager ─────────────────────

class TestConstraintManager:
    """ConstraintManager: aggregate loss computation + adaptive weighting."""

    def test_default_manager_has_four_constraints(self, constraint_manager):
        """Default constraint manager should have exactly 4 constraints."""
        assert len(constraint_manager.constraints) == 4
        expected_names = {"monotonicity", "spm_residual", "voltage_safety", "temperature_safety"}
        assert set(constraint_manager.constraints.keys()) == expected_names

    def test_total_loss_returns_tensor_and_breakdown(self, constraint_manager):
        """compute_total_loss should return (tensor, dict) tuple."""
        predictions = torch.randn(10, 1)
        inputs = {"cycles": torch.arange(10, dtype=torch.float32).unsqueeze(1)}
        cycles = torch.arange(10, dtype=torch.float32)

        total_loss, breakdown = constraint_manager.compute_total_loss(
            predictions, inputs, cycles, max_cycle=100.0
        )

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0, "Total loss should be a scalar tensor"
        assert isinstance(breakdown, dict)
        assert len(breakdown) == 4, "Breakdown should have entry for each constraint"

    def test_total_loss_nonnegative(self, constraint_manager):
        """Total loss should always be non-negative."""
        predictions = torch.randn(20, 1)
        inputs = {"cycles": torch.arange(20, dtype=torch.float32).unsqueeze(1)}
        cycles = torch.arange(20, dtype=torch.float32)

        total_loss, _ = constraint_manager.compute_total_loss(
            predictions, inputs, cycles, max_cycle=200.0
        )

        assert total_loss.item() >= 0, "Total constraint loss must be non-negative"

    def test_add_constraint(self, device):
        """Adding custom constraint should be reflected in constraints dict."""
        manager = ConstraintManager(str(device))
        assert len(manager.constraints) == 0

        manager.add_constraint(MonotonicityConstraint(weight=0.1))
        assert len(manager.constraints) == 1
        assert "monotonicity" in manager.constraints

    def test_validate_all_clean_data(self, constraint_manager):
        """Clean data should pass validation."""
        predictions = torch.randn(10, 1) * 0.1
        inputs = {"cycles": torch.arange(10, dtype=torch.float32).unsqueeze(1)}

        valid = constraint_manager.validate_all(predictions, inputs)

        assert valid is True

    def test_validate_all_nan_data(self, constraint_manager):
        """NaN data should fail validation."""
        predictions = torch.tensor([[1.0], [float('nan')], [0.5]])
        inputs = {"cycles": torch.arange(3, dtype=torch.float32).unsqueeze(1)}

        valid = constraint_manager.validate_all(predictions, inputs)

        assert valid is False

    def test_device_transfer(self, constraint_manager):
        """Constraint manager should transfer all constraints to specified device."""
        manager = constraint_manager.to(torch.device("cpu"))

        for name, constraint in manager.constraints.items():
            assert constraint.device == torch.device("cpu"), \
                f"Constraint {name} device should be CPU after transfer"


# ──────────────────── AdaptiveLossWeighter ──────────────────

class TestAdaptiveLossWeighter:
    """AdaptiveLossWeighter: sigmoid-scheduled per-sample weights."""

    def test_early_cycles_low_weight(self, adaptive_weighter):
        """Very early cycles (t≈0) → weights should be near minimum."""
        cycles = np.array([0.0, 1.0, 5.0])
        max_cycle = 1000.0

        lp, lm = adaptive_weighter.get_weights(cycles, max_cycle)

        # At t≈0 the sigmoid ≈ 0, so weights ≈ min
        assert np.all(lp < 0.05), \
            f"Early physics weight should be near min (0.01), got {lp}"
        assert np.all(lm < 0.02), \
            f"Early mono weight should be near min (0.01), got {lm}"

    def test_late_cycles_high_weight(self, adaptive_weighter):
        """Very late cycles (t≈1) → weights should be near maximum."""
        cycles = np.array([900.0, 950.0, 1000.0])
        max_cycle = 1000.0

        lp, lm = adaptive_weighter.get_weights(cycles, max_cycle)

        assert np.all(lp > 0.8), \
            f"Late physics weight should be near max (1.0), got {lp}"
        assert np.all(lm > 0.15), \
            f"Late mono weight should be near max (0.2), got {lm}"

    def test_extrapolation_maximum_weight(self, adaptive_weighter):
        """Extrapolation (t > 1) → weights should be at/near maximum."""
        cycles = np.array([1200.0, 1500.0, 2000.0])
        max_cycle = 1000.0

        lp, lm = adaptive_weighter.get_weights(cycles, max_cycle)

        assert np.all(lp > 0.95), \
            f"Extrapolation physics weight should be ~max, got {lp}"
        assert np.all(lm > 0.18), \
            f"Extrapolation mono weight should be ~max, got {lm}"

    def test_transition_center_midpoint_weight(self, adaptive_weighter):
        """At transition center (t=0.6) → sigmoid ≈ 0.5, weight ≈ midpoint."""
        cycles = np.array([600.0])
        max_cycle = 1000.0

        lp, lm = adaptive_weighter.get_weights(cycles, max_cycle)

        expected_lp_mid = (0.01 + 1.0) / 2  # ≈ 0.505
        expected_lm_mid = (0.01 + 0.2) / 2  # ≈ 0.105

        assert abs(lp[0] - expected_lp_mid) < 0.1, \
            f"Center physics weight should be ~{expected_lp_mid:.2f}, got {lp[0]:.4f}"
        assert abs(lm[0] - expected_lm_mid) < 0.05, \
            f"Center mono weight should be ~{expected_lm_mid:.2f}, got {lm[0]:.4f}"

    def test_monotonic_increase_over_lifecycle(self, adaptive_weighter):
        """Weights should monotonically increase as cycles progress."""
        cycles = np.linspace(0, 1000, 100)
        max_cycle = 1000.0

        lp, lm = adaptive_weighter.get_weights(cycles, max_cycle)

        # Check monotonicity of physics weight
        diffs_lp = np.diff(lp)
        assert np.all(diffs_lp >= -1e-10), \
            "Physics weight should monotonically increase over lifecycle"

        diffs_lm = np.diff(lm)
        assert np.all(diffs_lm >= -1e-10), \
            "Mono weight should monotonically increase over lifecycle"

    def test_max_cycle_zero_no_crash(self, adaptive_weighter):
        """max_cycle=0 edge case → should not crash (divides by max(0, 1))."""
        cycles = np.array([0.0, 10.0])
        max_cycle = 0.0

        lp, lm = adaptive_weighter.get_weights(cycles, max_cycle)

        assert not np.any(np.isnan(lp)), "Should not produce NaN with max_cycle=0"
        assert not np.any(np.isnan(lm)), "Should not produce NaN with max_cycle=0"

    def test_get_epoch_weights_returns_float(self, adaptive_weighter):
        """get_epoch_weights → should return (float, float) tuple."""
        cycles = np.linspace(0, 500, 50)
        max_cycle = 1000.0

        lp_mean, lm_mean = adaptive_weighter.get_epoch_weights(cycles, max_cycle)

        assert isinstance(lp_mean, float)
        assert isinstance(lm_mean, float)
        assert 0.0 < lp_mean < 1.0
        assert 0.0 < lm_mean < 0.2


# ──────────────────── Constraint Weight System ──────────────

class TestConstraintWeightSystem:
    """Test adaptive weight scheduling within individual constraints."""

    def test_static_weight_when_adaptive_false(self):
        """Non-adaptive constraint should always return base_weight."""
        constraint = MonotonicityConstraint(weight=0.05, adaptive=False)

        weight = constraint.get_weight(cycles=None, max_cycle=None)

        assert weight.item() == pytest.approx(0.05, abs=1e-7)

    def test_adaptive_weight_varies_with_cycle(self):
        """Adaptive constraint weight should depend on cycle position."""
        constraint = MonotonicityConstraint(weight=0.05, adaptive=True)

        cycles_early = torch.tensor([10.0])
        cycles_late = torch.tensor([900.0])

        weight_early = constraint.get_weight(cycles_early, max_cycle=1000.0)
        weight_late = constraint.get_weight(cycles_late, max_cycle=1000.0)

        assert weight_late.mean().item() > weight_early.mean().item(), \
            "Late-cycle weight should be higher than early-cycle weight"

    def test_constraint_to_device(self):
        """Constraint.to() should update device attribute."""
        constraint = MonotonicityConstraint(weight=0.05)
        constraint.to(torch.device("cpu"))

        assert constraint.device == torch.device("cpu")


class TestExpert5BugFixes:
    """Tests specifically targeting bugs found by Expert #5 audit."""

    def test_voltage_constraint_capacity_range_defaults(self):
        """VoltageConstraint defaults should be capacity-appropriate (0.0-2.5 Ah).
        
        Expert #5 Bug: Old defaults v_min=2.5, v_max=4.2 (voltage ranges)
        applied to capacity predictions (~1.4-2.0 Ah) would always trigger
        under_voltage penalty, producing incorrect loss.
        """
        vc = VoltageConstraint()  # Use defaults
        assert vc.v_min == 0.0, f"Default v_min should be 0.0 for capacity, got {vc.v_min}"
        assert vc.v_max == 2.5, f"Default v_max should be 2.5 for capacity, got {vc.v_max}"

        # Capacity predictions in valid range should produce zero loss
        valid_capacity = torch.tensor([[1.5], [1.8], [2.0], [1.3]])
        inputs = {"cycles": torch.arange(4, dtype=torch.float32).unsqueeze(1)}
        loss = vc.compute_loss(valid_capacity, inputs)
        assert loss.item() == 0.0, (
            f"Valid capacity predictions [1.3-2.0] within [0.0, 2.5] "
            f"should produce zero loss, got {loss.item()}"
        )

    def test_voltage_constraint_penalizes_out_of_range_capacity(self):
        """Capacity predictions outside [v_min, v_max] should produce positive loss."""
        vc = VoltageConstraint(v_min=0.0, v_max=2.5)
        # 3.0 Ah exceeds v_max=2.5
        out_of_range = torch.tensor([[1.5], [3.0], [2.0]])
        inputs = {"cycles": torch.arange(3, dtype=torch.float32).unsqueeze(1)}
        loss = vc.compute_loss(out_of_range, inputs)
        assert loss.item() > 0.0, "Out-of-range capacity should produce positive loss"

    def test_temperature_constraint_capacity_defaults(self):
        """TemperatureConstraint defaults should be capacity-appropriate."""
        tc = TemperatureConstraint()  # Use defaults
        assert tc.t_max == 2.2, f"Default t_max should be 2.2 Ah, got {tc.t_max}"

        # Valid capacity under 2.2 should produce zero loss
        valid = torch.tensor([[1.5], [1.8], [2.0]])
        inputs = {"cycles": torch.arange(3, dtype=torch.float32).unsqueeze(1)}
        loss = tc.compute_loss(valid, inputs)
        assert loss.item() == 0.0, f"Valid capacity under 2.2 should be zero loss, got {loss.item()}"

    def test_monotonicity_unsorted_batch_defense(self):
        """MonotonicityConstraint should sort by cycle when batch is unsorted.
        
        Expert #5 Bug: Batch-dimension fallback assumed sorted batches.
        The fix adds argsort-by-cycle to handle random sampling.
        """
        mc = MonotonicityConstraint(weight=0.05)

        # Deliberately unsorted batch: cycles=[100, 0, 50], capacity=[1.0, 2.0, 1.5]
        # After sorting by cycle: [0→2.0, 50→1.5, 100→1.0] → monotonically decreasing → loss=0
        predictions = torch.tensor([[1.0], [2.0], [1.5]])
        inputs = {"cycles": torch.tensor([[100.0], [0.0], [50.0]])}

        loss = mc.compute_loss(predictions, inputs)
        assert loss.item() == 0.0, (
            f"After sorting by cycle, sequence [2.0, 1.5, 1.0] is monotone decreasing. "
            f"Loss should be 0.0, got {loss.item()}"
        )
