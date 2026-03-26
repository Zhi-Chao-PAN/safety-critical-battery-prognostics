"""
Failure Mode and Effects Analysis (FMEA) Module
Quantifies risk based on Severity (S), Occurrence (O), and Detection (D).
Outputs a Risk Priority Number (RPN) matrix.
"""

from dataclasses import dataclass


@dataclass
class FailureMode:
    component: str
    failure_mode: str
    effects: str
    severity: int  # 1-10
    occurrence: int  # 1-10
    detection: int  # 1-10
    mitigation: str

    def __post_init__(self):
        """Validate FMEA scores are within ISO 26262 standard range [1, 10]."""
        for field in ["severity", "occurrence", "detection"]:
            val = getattr(self, field)
            if not (1 <= val <= 10):
                raise ValueError(f"FMEA {field} must be between 1 and 10, got {val}.")

    @property
    def rpn(self) -> int:
        """Risk Priority Number: RPN = S * O * D."""
        return self.severity * self.occurrence * self.detection


class FMEAAnalyzer:
    """Manages system FMEA generation according to ISO 26262."""

    def __init__(self):
        self.modes: list[FailureMode] = []

        # Populate basic lithium-ion prognostic failures
        self._initialize_base_modes()

    def _initialize_base_modes(self):
        self.modes.append(FailureMode(
            component="Prediction Model",
            failure_mode="Overestimating Remaining Useful Life (RUL)",
            effects="Unexpected shutdown during critical operation",
            severity=9,
            occurrence=4,
            detection=3,
            mitigation="Utilize OOD Detector and Epistemic Uncertainty bounds."
        ))

        self.modes.append(FailureMode(
            component="Sensor Data",
            failure_mode="Voltage sensor drift by >50mV",
            effects="Inaccurate State of Health (SOH) estimation",
            severity=7,
            occurrence=5,
            detection=2,
            mitigation="Aleatoric uncertainty bounds and dual-redundant checks."
        ))

    def get_critical_failures(self, rpn_threshold: int = 100) -> list[FailureMode]:
        return [m for m in self.modes if m.rpn >= rpn_threshold]

    def generate_report(self) -> str:
        report = "### SYSTEM FMEA REPORT ###\n"
        for m in sorted(self.modes, key=lambda x: x.rpn, reverse=True):
            report += f"[{m.rpn}] {m.component} - {m.failure_mode} (S:{m.severity} O:{m.occurrence} D:{m.detection})\n"
            report += f"      Effects: {m.effects}\n"
            report += f"      Mitigation: {m.mitigation}\n\n"
        return report
