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
