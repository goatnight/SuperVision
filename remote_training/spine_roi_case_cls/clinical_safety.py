from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def _clip_probability(value: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=np.float64), 1e-7, 1.0 - 1e-7)


def apply_temperature(probabilities: np.ndarray | float, temperature: float) -> np.ndarray:
    probabilities = _clip_probability(probabilities)
    logits = np.log(probabilities / (1.0 - probabilities))
    calibrated_logits = logits / max(float(temperature), 1e-6)
    return 1.0 / (1.0 + np.exp(-np.clip(calibrated_logits, -40.0, 40.0)))


def binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = _clip_probability(probabilities)
    return float(-np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)))


def fit_temperature(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = _clip_probability(probabilities)
    if np.unique(labels).size != 2:
        raise ValueError("Temperature scaling requires both classes")

    # A deterministic log-space search avoids adding another optimizer dependency.
    lower, upper = math.log(0.05), math.log(20.0)
    best_temperature = 1.0
    best_loss = binary_log_loss(labels, probabilities)
    for _ in range(4):
        candidates = np.exp(np.linspace(lower, upper, 401))
        losses = np.asarray(
            [binary_log_loss(labels, apply_temperature(probabilities, value)) for value in candidates]
        )
        index = int(np.argmin(losses))
        best_temperature = float(candidates[index])
        best_loss = float(losses[index])
        left = max(index - 2, 0)
        right = min(index + 2, len(candidates) - 1)
        lower, upper = math.log(candidates[left]), math.log(candidates[right])
    return {
        "temperature": best_temperature,
        "validation_nll_before": binary_log_loss(labels, probabilities),
        "validation_nll_after": best_loss,
    }


def select_triage_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_sensitivity: float,
    target_specificity: float,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if np.unique(labels).size != 2:
        raise ValueError("Triage threshold selection requires both classes")
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    positives = labels == 1
    negatives = labels == 0
    rule_out_candidates = [
        value for value in candidates if np.mean(probabilities[positives] >= value) >= target_sensitivity
    ]
    rule_in_candidates = [
        value for value in candidates if np.mean(probabilities[negatives] < value) >= target_specificity
    ]
    if not rule_out_candidates or not rule_in_candidates:
        raise ValueError("Requested triage operating points cannot be achieved")
    rule_out = float(max(rule_out_candidates))
    rule_in = float(min(rule_in_candidates))
    if rule_out >= rule_in:
        raise ValueError("Rule-out and rule-in thresholds overlap; safe abstention is unavailable")
    return {
        "rule_out_below": rule_out,
        "rule_in_at_or_above": rule_in,
        "target_rule_out_sensitivity": float(target_sensitivity),
        "target_rule_in_specificity": float(target_specificity),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | int | None]:
    if total <= 0:
        return {"estimate": None, "low": None, "high": None, "successes": successes, "total": total}
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "estimate": float(estimate),
        "low": float(max(0.0, center - radius)),
        "high": float(min(1.0, center + radius)),
        "successes": int(successes),
        "total": int(total),
    }


def selective_metrics(labels: np.ndarray, probabilities: np.ndarray, thresholds: dict[str, float]) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    rule_out = probabilities < float(thresholds["rule_out_below"])
    rule_in = probabilities >= float(thresholds["rule_in_at_or_above"])
    decided = rule_out | rule_in
    predictions = rule_in.astype(np.int64)
    negative_correct = int(np.sum(rule_out & (labels == 0)))
    positive_correct = int(np.sum(rule_in & (labels == 1)))
    decided_correct = int(np.sum(decided & (predictions == labels)))
    return {
        "n": int(labels.size),
        "auto_decided": int(np.sum(decided)),
        "deferred_to_radiologist": int(np.sum(~decided)),
        "coverage": float(np.mean(decided)),
        "selective_accuracy": wilson_interval(decided_correct, int(np.sum(decided))),
        "rule_out_npv": wilson_interval(negative_correct, int(np.sum(rule_out))),
        "rule_in_ppv": wilson_interval(positive_correct, int(np.sum(rule_in))),
        "false_negative_rule_out": int(np.sum(rule_out & (labels == 1))),
        "false_positive_rule_in": int(np.sum(rule_in & (labels == 0))),
    }


def nifti_metadata(path: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    import nibabel as nib

    image = nib.load(str(path), mmap=True)
    shape = tuple(int(value) for value in image.shape)
    zooms = tuple(float(value) for value in image.header.get_zooms()[: len(shape)])
    affine = np.asarray(image.affine, dtype=np.float64)
    steps = tuple(max(value // 64, 1) for value in shape)
    sample = np.asarray(image.dataobj[tuple(slice(None, None, step) for step in steps)], dtype=np.float32)
    finite = np.isfinite(sample)
    finite_values = sample[finite]
    if finite_values.size:
        low, high = np.percentile(finite_values, [1.0, 99.0])
        nonzero_fraction = float(np.mean(finite_values != 0))
    else:
        low, high, nonzero_fraction = 0.0, 0.0, 0.0
    metadata = {
        "shape": list(shape),
        "zooms": list(zooms),
        "orientation": list(nib.aff2axcodes(affine)),
        "finite_fraction": float(np.mean(finite)),
        "nonzero_fraction": nonzero_fraction,
        "dynamic_range_p01_p99": float(high - low),
        "affine_determinant": float(np.linalg.det(affine[:3, :3])),
    }
    return metadata, sample


def _robust_range(values: list[float], lower_q: float = 0.005, upper_q: float = 0.995) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    lower, upper = np.quantile(array, [lower_q, upper_q])
    margin = max((upper - lower) * 0.2, abs(lower) * 0.05, 1e-6)
    return [float(max(0.0, lower - margin)), float(upper + margin)]


def build_input_profile(records: list[dict[str, Any]], roi_fractions: list[float]) -> dict[str, Any]:
    if not records:
        raise ValueError("No training input metadata was collected")
    dimensions = len(records[0]["shape"])
    return {
        "training_cases": len(records),
        "dimensions": dimensions,
        "shape_ranges": [
            _robust_range([record["shape"][axis] for record in records]) for axis in range(dimensions)
        ],
        "spacing_ranges": [
            _robust_range([record["zooms"][axis] for record in records]) for axis in range(dimensions)
        ],
        "nonzero_fraction_range": _robust_range([record["nonzero_fraction"] for record in records]),
        "dynamic_range": _robust_range([record["dynamic_range_p01_p99"] for record in records]),
        "accepted_orientations": sorted({"".join(record["orientation"]) for record in records}),
        "roi_foreground_fraction_range": _robust_range(roi_fractions) if roi_fractions else None,
    }


def validate_input(metadata: dict[str, Any], profile: dict[str, Any], qc: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    shape = metadata["shape"]
    if len(shape) != int(qc["expected_dimensions"]):
        errors.append("unexpected_nifti_dimensions")
    if len(shape) == 3:
        if min(shape[:2]) < int(qc["minimum_in_plane_size"]):
            errors.append("in_plane_matrix_too_small")
        if shape[-1] < int(qc["minimum_slices"]):
            errors.append("insufficient_sagittal_slices")
    if metadata["finite_fraction"] < float(qc["minimum_finite_fraction"]):
        errors.append("non_finite_voxels")
    if not float(qc["minimum_nonzero_fraction"]) <= metadata["nonzero_fraction"] <= float(qc["maximum_nonzero_fraction"]):
        errors.append("invalid_nonzero_fraction")
    if metadata["dynamic_range_p01_p99"] < float(qc["minimum_dynamic_range"]):
        errors.append("insufficient_intensity_dynamic_range")
    if not np.isfinite(metadata["affine_determinant"]) or abs(metadata["affine_determinant"]) < 1e-8:
        errors.append("invalid_affine_geometry")

    orientation = "".join(metadata["orientation"])
    if orientation not in set(profile["accepted_orientations"]):
        errors.append("orientation_outside_training_domain")
    for name, values, ranges in (
        ("shape", shape, profile["shape_ranges"]),
        ("spacing", metadata["zooms"], profile["spacing_ranges"]),
    ):
        for axis, (value, bounds) in enumerate(zip(values, ranges)):
            if not bounds[0] <= value <= bounds[1]:
                errors.append(f"{name}_axis_{axis}_outside_training_domain")
    bounds = profile["nonzero_fraction_range"]
    if not bounds[0] <= metadata["nonzero_fraction"] <= bounds[1]:
        warnings.append("nonzero_fraction_outside_training_reference")
    return {"passed": not errors, "errors": sorted(set(errors)), "warnings": sorted(set(warnings)), "metadata": metadata}


def validate_segmentation(masks: np.ndarray, profile: dict[str, Any], qc: dict[str, Any]) -> dict[str, Any]:
    labels = sorted(int(value) for value in np.unique(masks) if value > 0)
    foreground = float(np.mean(masks > 0))
    errors: list[str] = []
    warnings: list[str] = []
    if foreground < float(qc["minimum_roi_foreground_fraction"]):
        errors.append("segmentation_foreground_too_small")
    if foreground > float(qc["maximum_roi_foreground_fraction"]):
        errors.append("segmentation_foreground_too_large")
    if len(labels) < int(qc["minimum_segmentation_labels"]):
        errors.append("insufficient_anatomical_labels")
    reference = profile.get("roi_foreground_fraction_range")
    if reference and not reference[0] <= foreground <= reference[1]:
        warnings.append("segmentation_fraction_outside_training_reference")
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "foreground_fraction": foreground,
        "labels_present": labels,
    }


def disposition(probability: float, thresholds: dict[str, float], safety_passed: bool, authorized: bool) -> str:
    if not safety_passed:
        return "INPUT_REJECTED_MANDATORY_REVIEW"
    if not authorized:
        return "RESEARCH_ONLY_MANDATORY_REVIEW"
    if probability < float(thresholds["rule_out_below"]):
        return "AI_RULE_OUT_CANDIDATE_RADIOLOGIST_CONFIRMATION_REQUIRED"
    if probability >= float(thresholds["rule_in_at_or_above"]):
        return "AI_RULE_IN_CANDIDATE_RADIOLOGIST_CONFIRMATION_REQUIRED"
    return "INDETERMINATE_MANDATORY_REVIEW"
