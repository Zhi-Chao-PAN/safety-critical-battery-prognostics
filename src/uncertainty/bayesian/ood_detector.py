"""
Out-of-Distribution (OOD) Detection for Safety-Critical Battery Prognostics.

Detects when the model encounters unseen battery chemistries, extreme
temperatures, or operating conditions outside the training distribution.

Methods:
  1. Mahalanobis Distance — measures how far a test sample is from the
     training distribution in feature space (accounts for correlations)
  2. Epistemic Uncertainty Surge — monitors if BNN/BTCN weight posterior
     variance spikes beyond a learned threshold
  3. Combined Score — weighted fusion for robust OOD detection

Safety protocol:
  When OOD is detected, the system should:
  - Flag predictions as LOW CONFIDENCE
  - Widen prediction intervals (conservative safety margin)
  - Trigger "unknown operating condition — request human intervention"

This is critical for cross-chemistry generalization:
  e.g., model trained on NASA (LiCoO2) tested on CALCE (LiFePO4)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial.distance import mahalanobis

logger = logging.getLogger(__name__)


class OODLevel(Enum):
    """OOD severity levels."""
    IN_DISTRIBUTION = "IN_DISTRIBUTION"
    BORDERLINE = "BORDERLINE"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"


@dataclass
class OODResult:
    """Result of OOD detection for a single sample or batch."""
    level: OODLevel
    mahalanobis_distance: float
    epistemic_ratio: float  # epistemic_std / training_mean_std
    combined_score: float  # Fused OOD score in [0, 1]
    action: str
    details: str


class MahalanobisDetector:
    """
    Mahalanobis distance-based OOD detector.

    Fits a multivariate Gaussian to the training feature distribution,
    then measures how far test samples deviate from it.

    Advantages over Euclidean distance:
      - Accounts for feature correlations
      - Scale-invariant
      - Principled statistical interpretation (chi-squared distribution)
    """

    def __init__(self, regularization: float = 1e-5):
        self.reg = regularization
        self._mean: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None
        self._threshold_borderline: float = 0.0
        self._threshold_ood: float = 0.0

    def fit(
        self,
        X_train: np.ndarray,
        percentile_borderline: float = 95.0,
        percentile_ood: float = 99.0,
    ) -> "MahalanobisDetector":
        """
        Fit the detector on training data.

        Computes mean, regularized covariance inverse, and thresholds
        from the empirical distribution of training Mahalanobis distances.
        """
        self._mean = X_train.mean(axis=0)
        cov = np.cov(X_train, rowvar=False)

        # Regularize for numerical stability
        cov += self.reg * np.eye(cov.shape[0])

        try:
            self._cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix singular. Using pseudo-inverse.")
            self._cov_inv = np.linalg.pinv(cov)

        # Compute training distances for threshold calibration
        train_distances = self.distances(X_train)
        self._threshold_borderline = float(np.percentile(train_distances, percentile_borderline))
        self._threshold_ood = float(np.percentile(train_distances, percentile_ood))

        logger.info(
            f"MahalanobisDetector fitted: dim={X_train.shape[1]}, "
            f"borderline_thresh={self._threshold_borderline:.2f}, "
            f"ood_thresh={self._threshold_ood:.2f}"
        )
        return self

    def distances(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance for each sample."""
        if self._mean is None or self._cov_inv is None:
            raise RuntimeError("Detector not fitted.")

        dists = np.array([
            mahalanobis(x, self._mean, self._cov_inv)
            for x in X
        ])
        return dists

    def detect(self, X: np.ndarray) -> list[OODLevel]:
        """Classify each sample as ID / BORDERLINE / OOD."""
        dists = self.distances(X)
        levels = []
        for d in dists:
            if d >= self._threshold_ood:
                levels.append(OODLevel.OUT_OF_DISTRIBUTION)
            elif d >= self._threshold_borderline:
                levels.append(OODLevel.BORDERLINE)
            else:
                levels.append(OODLevel.IN_DISTRIBUTION)
        return levels


class EpistemicSurgeDetector:
    """
    Detects OOD via epistemic uncertainty surge.

    Principle: A well-calibrated Bayesian model should show increased
    epistemic uncertainty (weight posterior variance) on OOD inputs.

    Fits a baseline distribution of epistemic uncertainty on training data,
    then flags test samples where uncertainty exceeds learned thresholds.
    """

    def __init__(self):
        self._mean_std: float = 0.0
        self._std_of_std: float = 0.0
        self._threshold_borderline: float = 0.0
        self._threshold_ood: float = 0.0

    def fit(
        self,
        training_epistemic_stds: np.ndarray,
        sigma_borderline: float = 2.0,
        sigma_ood: float = 3.0,
    ) -> "EpistemicSurgeDetector":
        """
        Fit on epistemic uncertainty values from training predictions.

        Args:
            training_epistemic_stds: Std dev from MC inference on training data
            sigma_borderline: Number of std devs for borderline threshold
            sigma_ood: Number of std devs for OOD threshold
        """
        self._mean_std = float(np.mean(training_epistemic_stds))
        self._std_of_std = float(np.std(training_epistemic_stds))

        self._threshold_borderline = self._mean_std + sigma_borderline * self._std_of_std
        self._threshold_ood = self._mean_std + sigma_ood * self._std_of_std

        logger.info(
            f"EpistemicSurgeDetector fitted: mean_std={self._mean_std:.4f}, "
            f"borderline={self._threshold_borderline:.4f}, "
            f"ood={self._threshold_ood:.4f}"
        )
        return self

    def detect(self, epistemic_stds: np.ndarray) -> list[OODLevel]:
        """Classify based on epistemic uncertainty magnitude."""
        levels = []
        for s in epistemic_stds:
            if s >= self._threshold_ood:
                levels.append(OODLevel.OUT_OF_DISTRIBUTION)
            elif s >= self._threshold_borderline:
                levels.append(OODLevel.BORDERLINE)
            else:
                levels.append(OODLevel.IN_DISTRIBUTION)
        return levels

    def ratios(self, epistemic_stds: np.ndarray) -> np.ndarray:
        """Compute epistemic ratio (test_std / training_mean_std)."""
        return epistemic_stds / max(self._mean_std, 1e-8)


class OODDetector:
    """
    Combined OOD detector fusing Mahalanobis distance + epistemic surge.

    Usage:
        detector = OODDetector()
        detector.fit(X_train, train_epistemic_stds)
        results = detector.detect(X_test, test_epistemic_stds)

    For cross-chemistry evaluation:
        # Train on NASA data
        detector.fit(X_nasa, stds_nasa)
        # Test on CALCE data — expect OOD detection
        results = detector.detect(X_calce, stds_calce)
    """

    def __init__(
        self,
        mahal_weight: float = 0.5,
        epistemic_weight: float = 0.5,
        safety_margin: float = 2.0,
    ):
        self.mahal_weight = mahal_weight
        self.epistemic_weight = epistemic_weight
        self.safety_margin = safety_margin
        self.mahal_detector = MahalanobisDetector()
        self.epistemic_detector = EpistemicSurgeDetector()
        self._fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        training_epistemic_stds: np.ndarray,
        **kwargs: Any,
    ) -> "OODDetector":
        """Fit both sub-detectors on training data."""
        self.mahal_detector.fit(X_train, **kwargs)
        self.epistemic_detector.fit(training_epistemic_stds)
        self._fitted = True
        return self

    def detect(
        self,
        X_test: np.ndarray,
        test_epistemic_stds: np.ndarray,
    ) -> list[OODResult]:
        """
        Run combined OOD detection.

        Returns per-sample OODResult with:
          - Combined OOD level
          - Individual scores
          - Recommended action
        """
        if not self._fitted:
            raise RuntimeError("OODDetector not fitted.")

        mahal_dists = self.mahal_detector.distances(X_test)
        mahal_levels = self.mahal_detector.detect(X_test)
        epistemic_ratios = self.epistemic_detector.ratios(test_epistemic_stds)
        epistemic_levels = self.epistemic_detector.detect(test_epistemic_stds)

        # Normalize scores to [0, 1]
        mahal_max = max(self.mahal_detector._threshold_ood * 2, 1e-8)
        mahal_scores = np.clip(mahal_dists / mahal_max, 0, 1)
        epistemic_scores = np.clip(epistemic_ratios / 5.0, 0, 1)  # 5x training mean = max

        combined = (
            self.mahal_weight * mahal_scores
            + self.epistemic_weight * epistemic_scores
        )

        results = []
        for i in range(len(X_test)):
            # Take the more severe of the two detectors
            if mahal_levels[i] == OODLevel.OUT_OF_DISTRIBUTION or epistemic_levels[i] == OODLevel.OUT_OF_DISTRIBUTION:
                level = OODLevel.OUT_OF_DISTRIBUTION
                action = (
                    "⚠️ UNKNOWN OPERATING CONDITION. "
                    "Predictions unreliable. Widen safety margins by "
                    f"{self.safety_margin}x. Request human intervention."
                )
            elif mahal_levels[i] == OODLevel.BORDERLINE or epistemic_levels[i] == OODLevel.BORDERLINE:
                level = OODLevel.BORDERLINE
                action = (
                    "⚡ Borderline operating condition. "
                    "Increase monitoring frequency. Flag for review."
                )
            else:
                level = OODLevel.IN_DISTRIBUTION
                action = "✅ Normal operating condition. Standard monitoring."

            details = (
                f"Mahalanobis={mahal_dists[i]:.2f} ({mahal_levels[i].value}), "
                f"Epistemic ratio={epistemic_ratios[i]:.2f}x ({epistemic_levels[i].value}), "
                f"Combined={combined[i]:.3f}"
            )

            results.append(OODResult(
                level=level,
                mahalanobis_distance=float(mahal_dists[i]),
                epistemic_ratio=float(epistemic_ratios[i]),
                combined_score=float(combined[i]),
                action=action,
                details=details,
            ))

        # Summary logging
        n_ood = sum(1 for r in results if r.level == OODLevel.OUT_OF_DISTRIBUTION)
        n_border = sum(1 for r in results if r.level == OODLevel.BORDERLINE)
        n_id = sum(1 for r in results if r.level == OODLevel.IN_DISTRIBUTION)
        logger.info(f"OOD Detection: {n_id} ID, {n_border} BORDERLINE, {n_ood} OOD out of {len(results)}")

        return results

    def adjust_predictions(
        self,
        mean: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        ood_results: list[OODResult],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Widen prediction intervals for OOD samples.

        This is the safety-critical output: when the model doesn't know,
        it should say so by widening its confidence bounds.
        """
        adjusted_lower = lower.copy()
        adjusted_upper = upper.copy()

        for i, result in enumerate(ood_results):
            if i >= len(mean):
                break
            if result.level == OODLevel.OUT_OF_DISTRIBUTION:
                width = upper[i] - lower[i]
                adjusted_lower[i] = mean[i] - self.safety_margin * width / 2
                adjusted_upper[i] = mean[i] + self.safety_margin * width / 2
            elif result.level == OODLevel.BORDERLINE:
                width = upper[i] - lower[i]
                margin = 1.0 + (self.safety_margin - 1.0) * 0.5  # Half the OOD margin
                adjusted_lower[i] = mean[i] - margin * width / 2
                adjusted_upper[i] = mean[i] + margin * width / 2

        return mean, adjusted_lower, adjusted_upper
