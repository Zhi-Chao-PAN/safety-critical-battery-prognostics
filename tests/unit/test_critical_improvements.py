"""
Tests for the critical improvements:
  1. OOD Detection (Mahalanobis + Epistemic Surge)
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. OOD Detection Tests
# ---------------------------------------------------------------------------

class TestOODDetection:
    def test_mahalanobis_fit_detect(self):
        from src.uncertainty.bayesian.ood_detector import MahalanobisDetector, OODLevel
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        det = MahalanobisDetector()
        det.fit(X_train)

        # In-distribution
        X_id = np.random.randn(10, 5)
        levels_id = det.detect(X_id)
        # Most should be ID
        id_count = sum(1 for l in levels_id if l == OODLevel.IN_DISTRIBUTION)
        assert id_count >= 5

        # Out-of-distribution (shifted)
        X_ood = np.random.randn(10, 5) + 10.0
        levels_ood = det.detect(X_ood)
        ood_count = sum(1 for l in levels_ood if l == OODLevel.OUT_OF_DISTRIBUTION)
        assert ood_count >= 5

    def test_epistemic_surge_detector(self):
        from src.uncertainty.bayesian.ood_detector import EpistemicSurgeDetector, OODLevel
        np.random.seed(42)
        train_stds = np.random.uniform(0.1, 0.5, 100)
        det = EpistemicSurgeDetector()
        det.fit(train_stds)

        # Normal uncertainty
        normal_stds = np.array([0.2, 0.3, 0.25])
        levels = det.detect(normal_stds)
        assert all(l == OODLevel.IN_DISTRIBUTION for l in levels)

        # Spiked uncertainty
        spike_stds = np.array([5.0, 10.0, 20.0])
        levels_spike = det.detect(spike_stds)
        ood_count = sum(1 for l in levels_spike if l == OODLevel.OUT_OF_DISTRIBUTION)
        assert ood_count >= 2

    def test_combined_ood_detector(self):
        from src.uncertainty.bayesian.ood_detector import OODDetector, OODLevel
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        train_stds = np.random.uniform(0.1, 0.5, 100)

        det = OODDetector(safety_margin=2.0)
        det.fit(X_train, train_stds)

        # Test with OOD data
        X_ood = np.random.randn(20, 5) + 8.0
        ood_stds = np.random.uniform(2.0, 5.0, 20)
        results = det.detect(X_ood, ood_stds)
        assert len(results) == 20
        ood_count = sum(1 for r in results if r.level == OODLevel.OUT_OF_DISTRIBUTION)
        assert ood_count >= 10

    def test_adjust_predictions(self):
        from src.uncertainty.bayesian.ood_detector import OODDetector, OODLevel, OODResult
        mean = np.array([50.0, 40.0, 30.0])
        lower = np.array([40.0, 30.0, 20.0])
        upper = np.array([60.0, 50.0, 40.0])

        results = [
            OODResult(OODLevel.IN_DISTRIBUTION, 1.0, 1.0, 0.1, "", ""),
            OODResult(OODLevel.OUT_OF_DISTRIBUTION, 10.0, 5.0, 0.9, "", ""),
            OODResult(OODLevel.BORDERLINE, 5.0, 2.5, 0.5, "", ""),
        ]

        det = OODDetector(safety_margin=2.0)
        det._fitted = True
        _, adj_lower, adj_upper = det.adjust_predictions(mean, lower, upper, results)

        # ID sample: unchanged
        assert adj_lower[0] == lower[0]
        assert adj_upper[0] == upper[0]
        # OOD sample: widened by safety_margin
        assert adj_upper[1] - adj_lower[1] > upper[1] - lower[1]
        # Borderline: widened but less than OOD
        assert adj_upper[2] - adj_lower[2] > upper[2] - lower[2]
        assert adj_upper[2] - adj_lower[2] < adj_upper[1] - adj_lower[1]
