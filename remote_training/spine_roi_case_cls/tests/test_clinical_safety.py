from __future__ import annotations

import unittest

import numpy as np

from clinical_safety import (
    apply_temperature,
    disposition,
    fit_temperature,
    select_triage_thresholds,
    selective_metrics,
    validate_input,
    validate_segmentation,
)


class ClinicalSafetyTests(unittest.TestCase):
    def test_temperature_scaling_is_monotonic_and_improves_nll(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        probabilities = np.asarray([0.05, 0.2, 0.7, 0.3, 0.8, 0.95])
        fitted = fit_temperature(labels, probabilities)
        calibrated = apply_temperature(probabilities, fitted["temperature"])
        order = np.argsort(probabilities)
        self.assertTrue(np.all(np.diff(calibrated[order]) > 0))
        self.assertLessEqual(fitted["validation_nll_after"], fitted["validation_nll_before"] + 1e-12)

    def test_triage_thresholds_create_an_abstention_interval(self) -> None:
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        probabilities = np.asarray([0.05, 0.15, 0.25, 0.8, 0.2, 0.75, 0.85, 0.95])
        thresholds = select_triage_thresholds(labels, probabilities, 1.0, 1.0)
        self.assertLess(thresholds["rule_out_below"], thresholds["rule_in_at_or_above"])
        result = selective_metrics(labels, probabilities, thresholds)
        self.assertEqual(result["n"], len(labels))
        self.assertGreater(result["deferred_to_radiologist"], 0)

    def test_input_domain_mismatch_fails_closed(self) -> None:
        profile = {
            "accepted_orientations": ["RAS"],
            "shape_ranges": [[200, 300], [200, 300], [8, 20]],
            "spacing_ranges": [[0.5, 1.5], [0.5, 1.5], [3.0, 5.0]],
            "nonzero_fraction_range": [0.1, 0.8],
        }
        qc = {
            "expected_dimensions": 3,
            "minimum_in_plane_size": 128,
            "minimum_slices": 3,
            "minimum_finite_fraction": 0.999,
            "minimum_nonzero_fraction": 0.01,
            "maximum_nonzero_fraction": 0.99,
            "minimum_dynamic_range": 1e-6,
        }
        metadata = {
            "shape": [256, 256, 12],
            "zooms": [0.8, 0.8, 4.0],
            "orientation": ["L", "A", "S"],
            "finite_fraction": 1.0,
            "nonzero_fraction": 0.4,
            "dynamic_range_p01_p99": 5.0,
            "affine_determinant": 2.56,
        }
        result = validate_input(metadata, profile, qc)
        self.assertFalse(result["passed"])
        self.assertIn("orientation_outside_training_domain", result["errors"])

    def test_bad_segmentation_is_rejected(self) -> None:
        masks = np.zeros((3, 256, 256), dtype=np.uint8)
        result = validate_segmentation(
            masks,
            {"roi_foreground_fraction_range": [0.1, 0.5]},
            {
                "minimum_roi_foreground_fraction": 0.05,
                "maximum_roi_foreground_fraction": 0.65,
                "minimum_segmentation_labels": 4,
            },
        )
        self.assertFalse(result["passed"])
        self.assertIn("segmentation_foreground_too_small", result["errors"])

    def test_unauthorized_policy_never_returns_a_diagnosis(self) -> None:
        thresholds = {"rule_out_below": 0.2, "rule_in_at_or_above": 0.8}
        self.assertEqual(
            disposition(0.95, thresholds, safety_passed=True, authorized=False),
            "RESEARCH_ONLY_MANDATORY_REVIEW",
        )


if __name__ == "__main__":
    unittest.main()
