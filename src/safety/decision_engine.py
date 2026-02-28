"""
Safety Decision Engine - Map uncertainty to actionable safety decisions.

Three-tier classification:
  GREEN:  Normal operation
  YELLOW: Reduce load, increase monitoring
  RED:    Stop operation, trigger maintenance
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass
class SafetyDecision:
    level: SafetyLevel
    rul_estimate: float
    confidence_lower: float
    confidence_upper: float
    epistemic_std: float
    action: str
    reason: str


class SafetyDecisionEngine:
    """
    Adaptive safety decision engine.
    Maps RUL predictions + uncertainty to safety actions.
    """

    def __init__(
        self,
        rul_critical: float = 10.0,
        rul_warning: float = 30.0,
        epistemic_threshold_low: float = 5.0,
        epistemic_threshold_high: float = 15.0,
    ):
        self.rul_critical = rul_critical
        self.rul_warning = rul_warning
        self.eps_low = epistemic_threshold_low
        self.eps_high = epistemic_threshold_high

    def decide(
        self,
        rul_mean: float,
        rul_lower: float,
        rul_upper: float,
        epistemic_std: float = 0.0,
    ) -> SafetyDecision:
        """Make a safety decision for a single prediction."""

        # RED: Critical RUL or very high uncertainty
        if rul_lower < self.rul_critical or epistemic_std > self.eps_high:
            if rul_lower < self.rul_critical:
                reason = f"RUL lower bound ({rul_lower:.1f}) below critical threshold ({self.rul_critical})"
            else:
                reason = f"Epistemic uncertainty ({epistemic_std:.1f}) exceeds safe limit ({self.eps_high})"
            return SafetyDecision(
                level=SafetyLevel.RED,
                rul_estimate=rul_mean,
                confidence_lower=rul_lower,
                confidence_upper=rul_upper,
                epistemic_std=epistemic_std,
                action="STOP operation. Trigger immediate maintenance inspection.",
                reason=reason,
            )

        # YELLOW: Warning zone
        if rul_mean < self.rul_warning or epistemic_std > self.eps_low:
            if rul_mean < self.rul_warning:
                reason = f"RUL estimate ({rul_mean:.1f}) approaching warning zone ({self.rul_warning})"
            else:
                reason = f"Elevated epistemic uncertainty ({epistemic_std:.1f})"
            return SafetyDecision(
                level=SafetyLevel.YELLOW,
                rul_estimate=rul_mean,
                confidence_lower=rul_lower,
                confidence_upper=rul_upper,
                epistemic_std=epistemic_std,
                action="Reduce load. Increase monitoring frequency. Schedule maintenance.",
                reason=reason,
            )

        # GREEN: Normal
        return SafetyDecision(
            level=SafetyLevel.GREEN,
            rul_estimate=rul_mean,
            confidence_lower=rul_lower,
            confidence_upper=rul_upper,
            epistemic_std=epistemic_std,
            action="Normal operation. Standard monitoring interval.",
            reason="All parameters within safe bounds.",
        )

    def decide_batch(
        self,
        means: np.ndarray,
        lowers: np.ndarray,
        uppers: np.ndarray,
        epistemic_stds: np.ndarray | None = None,
    ) -> list[SafetyDecision]:
        """Make safety decisions for a batch of predictions."""
        if epistemic_stds is None:
            epistemic_stds = np.zeros_like(means)

        return [
            self.decide(float(m), float(lo), float(hi), float(e))
            for m, lo, hi, e in zip(means, lowers, uppers, epistemic_stds)
        ]

    def calibrate_thresholds(
        self,
        y_true: np.ndarray,
        means: np.ndarray,
        lowers: np.ndarray,
        uppers: np.ndarray,
        target_detection_rate: float = 0.99,
        eol_threshold: float = 0.0,
    ) -> dict[str, float]:
        """
        Calibrate safety thresholds on validation data.
        Finds thresholds that achieve target detection rate.
        """
        # True positives: batteries that actually reach EOL
        actual_critical = y_true <= eol_threshold + self.rul_critical

        if not actual_critical.any():
            logger.warning("No critical samples in validation set. Using defaults.")
            return {"rul_critical": self.rul_critical, "rul_warning": self.rul_warning}

        # Find RUL threshold that catches target_detection_rate of true failures
        critical_rul_preds = lowers[actual_critical]
        sorted_preds = np.sort(critical_rul_preds)
        idx = min(int(len(sorted_preds) * target_detection_rate), len(sorted_preds) - 1)
        calibrated_critical = float(sorted_preds[idx])

        self.rul_critical = max(calibrated_critical, 5.0)  # Floor at 5 cycles
        self.rul_warning = self.rul_critical * 3  # Warning at 3x critical

        result = {"rul_critical": self.rul_critical, "rul_warning": self.rul_warning}
        logger.info(f"Calibrated thresholds: {result}")
        return result
