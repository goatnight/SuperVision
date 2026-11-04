from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)


def optimal_youden_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    score = tpr[finite] - fpr[finite]
    return float(np.clip(thresholds[finite][int(np.argmax(score))], 0.0, 1.0))


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper
        )
        if selected.any():
            confidence = probabilities[selected].mean()
            frequency = labels[selected].mean()
            value += selected.mean() * abs(confidence - frequency)
    return float(value)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "n": int(labels.size),
        "threshold": float(threshold),
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "ppv": float(precision_score(labels, predictions, zero_division=0)),
        "npv": float(npv),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def bootstrap_confidence_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    rng = np.random.default_rng(seed)
    names = ("auc", "accuracy", "sensitivity", "specificity", "f1", "brier")
    values: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(samples):
        indices = rng.integers(0, labels.size, labels.size)
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        result = binary_metrics(sampled_labels, probabilities[indices], threshold)
        for name in names:
            values[name].append(float(result[name]))
    return {
        name: {
            "low": float(np.percentile(metric_values, 2.5)),
            "high": float(np.percentile(metric_values, 97.5)),
        }
        for name, metric_values in values.items()
        if metric_values
    }
