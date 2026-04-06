import sys
from pathlib import Path

import pytest

# Add project root to sys.path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.safety.fmea.analyzer import FailureMode, FMEAAnalyzer


def test_fmea_rpn_calculation():
    """Verify RPN = S * O * D."""
    mode = FailureMode(
        component="Sensor",
        failure_mode="Drift",
        effects="Inaccuracy",
        severity=5,
        occurrence=4,
        detection=3,
        mitigation="Redundancy"
    )
    assert mode.rpn == 60

def test_fmea_invalid_scores():
    """Verify that scores outside [1, 10] raise ValueError."""
    with pytest.raises(ValueError):
        FailureMode("C", "F", "E", 11, 5, 5, "M")
    with pytest.raises(ValueError):
        FailureMode("C", "F", "E", 5, 0, 5, "M")

def test_fmea_analyzer_critical_filtering():
    """Verify that the analyzer correctly filters high-risk modes."""
    analyzer = FMEAAnalyzer()

    # Base modes: RUL Overestimate (RPN=108), Sensor Drift (RPN=70)
    critical = analyzer.get_critical_failures(rpn_threshold=100)
    assert len(critical) == 1
    assert critical[0].failure_mode == "Overestimating Remaining Useful Life (RUL)"

def test_fmea_report_generation():
    """Verify that the FMEA report is non-empty and contains key terms."""
    analyzer = FMEAAnalyzer()
    report = analyzer.generate_report()
    assert "### SYSTEM FMEA REPORT ###" in report
    assert "Prognostics" in report or "Battery" in report or "Model" in report

if __name__ == "__main__":
    # For manual debugging
    test_fmea_rpn_calculation()
    test_fmea_invalid_scores()
    test_fmea_analyzer_critical_filtering()
    test_fmea_report_generation()
    print("FMEA unit tests passed.")


# ═══════════════════════════════════════════════════════════════════
# F1/F2 Fail-Safe Tests: Safety Decision Engine NaN/Inf/Unknown Guards
# ═══════════════════════════════════════════════════════════════════

import math
import numpy as np
from src.safety.decision_engine import SafetyDecisionEngine, SafetyLevel


class TestDecisionEngineFailSafe:
    """Tests for F1 (NaN guard) and F2 (unknown uncertainty fail-safe)."""

    def setup_method(self):
        self.engine = SafetyDecisionEngine(
            rul_critical=10.0,
            rul_warning=30.0,
            epistemic_threshold_low=5.0,
            epistemic_threshold_high=15.0,
        )

    # ── F1: NaN/Inf inputs must always produce RED ──────────────

    def test_nan_rul_mean_returns_red(self):
        """F1: NaN in rul_mean must fail-safe to RED."""
        decision = self.engine.decide(
            rul_mean=float('nan'), rul_lower=50.0, rul_upper=100.0, epistemic_std=2.0
        )
        assert decision.level == SafetyLevel.RED
        assert "NaN" in decision.reason or "FAIL-SAFE" in decision.reason

    def test_nan_rul_lower_returns_red(self):
        """F1: NaN in rul_lower must fail-safe to RED."""
        decision = self.engine.decide(
            rul_mean=50.0, rul_lower=float('nan'), rul_upper=100.0, epistemic_std=2.0
        )
        assert decision.level == SafetyLevel.RED

    def test_inf_epistemic_returns_red(self):
        """F1: Inf in epistemic_std must fail-safe to RED."""
        decision = self.engine.decide(
            rul_mean=50.0, rul_lower=40.0, rul_upper=60.0, epistemic_std=float('inf')
        )
        assert decision.level == SafetyLevel.RED

    def test_negative_inf_returns_red(self):
        """F1: -Inf in any input must fail-safe to RED."""
        decision = self.engine.decide(
            rul_mean=float('-inf'), rul_lower=40.0, rul_upper=60.0, epistemic_std=2.0
        )
        assert decision.level == SafetyLevel.RED

    def test_all_nan_returns_red(self):
        """F1: All NaN inputs must fail-safe to RED."""
        decision = self.engine.decide(
            rul_mean=float('nan'), rul_lower=float('nan'),
            rul_upper=float('nan'), epistemic_std=float('nan')
        )
        assert decision.level == SafetyLevel.RED

    # ── F2: Unknown uncertainty must NOT default to GREEN ───────

    def test_batch_none_epistemic_not_green(self):
        """F2: decide_batch with epistemic_stds=None must NOT produce GREEN.

        Before fix: None → zeros → GREEN (dangerous).
        After fix: None → eps_high → at minimum YELLOW.
        """
        means = np.array([50.0, 60.0, 70.0])
        lowers = np.array([40.0, 50.0, 60.0])
        uppers = np.array([60.0, 70.0, 80.0])

        decisions = self.engine.decide_batch(means, lowers, uppers, epistemic_stds=None)

        for d in decisions:
            # None uncertainty should never produce GREEN
            assert d.level != SafetyLevel.GREEN, (
                f"FAIL-SAFE VIOLATION: Got GREEN with unknown uncertainty. "
                f"Expected YELLOW or RED."
            )

    def test_normal_inputs_return_green(self):
        """Sanity check: normal, known-safe inputs should return GREEN."""
        decision = self.engine.decide(
            rul_mean=80.0, rul_lower=70.0, rul_upper=90.0, epistemic_std=1.0
        )
        assert decision.level == SafetyLevel.GREEN
