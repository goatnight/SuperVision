from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import save_json
from metrics import binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit overall and center-stratified case metrics")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    predictions = pd.read_csv(args.predictions, dtype={"patient_id": str, "center": str})
    threshold = float(summary["decision_threshold_selected_on_val"])

    required = {"patient_id", "center", "label", "probability", "prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if predictions["patient_id"].duplicated().any():
        raise ValueError("Each row must represent a unique patient")

    recomputed = binary_metrics(
        predictions["label"].to_numpy(), predictions["probability"].to_numpy(), threshold
    )
    recorded = summary["test"]
    metric_keys = ("auc", "accuracy", "sensitivity", "specificity", "f1", "brier", "ece")
    maximum_delta = max(abs(float(recomputed[key]) - float(recorded[key])) for key in metric_keys)

    by_center = {}
    for center, group in predictions.groupby("center", sort=True):
        by_center[str(center)] = binary_metrics(
            group["label"].to_numpy(), group["probability"].to_numpy(), threshold
        )

    audit = {
        "patient_level_rows": int(len(predictions)),
        "unique_patient_ids": int(predictions["patient_id"].nunique()),
        "fixed_validation_threshold": threshold,
        "recorded_vs_recomputed_max_abs_delta": maximum_delta,
        "overall_recomputed": recomputed,
        "by_center": by_center,
    }
    save_json(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
