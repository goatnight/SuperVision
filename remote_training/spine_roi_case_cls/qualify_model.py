from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from clinical_safety import (
    apply_temperature,
    build_input_profile,
    fit_temperature,
    nifti_metadata,
    select_triage_thresholds,
    selective_metrics,
)
from common import file_sha256, load_config, save_json
from data import read_case_table, resolve_sagittal_path
from metrics import binary_metrics, optimal_youden_threshold


def collect_training_profile(
    table: pd.DataFrame, data_root: str, cache_dir: str
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    roi_fractions: list[float] = []
    errors: list[dict[str, str]] = []
    for patient_id in table.loc[table["lsp_split"] == "train", "id"]:
        try:
            image_path = resolve_sagittal_path(data_root, patient_id)
            metadata, _ = nifti_metadata(image_path)
            records.append(metadata)
            cached = np.load(Path(cache_dir) / f"{patient_id}.npz", allow_pickle=False)
            roi_fractions.append(float(np.mean(cached["masks"] > 0)))
        except Exception as exc:
            errors.append({"patient_id": str(patient_id), "error": repr(exc)})
    if errors:
        raise RuntimeError(f"Training-domain profile failed for {len(errors)} cases: {errors[:3]}")
    return build_input_profile(records, roi_fractions), errors


def gate(name: str, passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "required": required}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fail-closed clinical qualification policy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--val-predictions", required=True)
    parser.add_argument("--test-predictions", required=True)
    parser.add_argument("--clinical-config", default="clinical_config.json")
    parser.add_argument("--output-policy", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()

    clinical = load_config(args.clinical_config)
    qualification = clinical["qualification"]
    evidence = clinical["evidence"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint["config"]
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    val = pd.read_csv(args.val_predictions)
    test = pd.read_csv(args.test_predictions)
    table = read_case_table(model_config["label_csv"])

    calibration = fit_temperature(val["label"].to_numpy(), val["probability"].to_numpy())
    val_calibrated = apply_temperature(val["probability"].to_numpy(), calibration["temperature"])
    test_calibrated = apply_temperature(test["probability"].to_numpy(), calibration["temperature"])
    single_threshold = optimal_youden_threshold(val["label"].to_numpy(), val_calibrated)
    thresholds = select_triage_thresholds(
        val["label"].to_numpy(),
        val_calibrated,
        float(clinical["triage"]["target_rule_out_sensitivity"]),
        float(clinical["triage"]["target_rule_in_specificity"]),
    )
    calibrated_test_metrics = binary_metrics(test["label"].to_numpy(), test_calibrated, single_threshold)
    selective = selective_metrics(test["label"].to_numpy(), test_calibrated, thresholds)
    input_profile, profile_errors = collect_training_profile(
        table, model_config["data_root"], model_config["cache_dir"]
    )

    test_ci = summary["test_bootstrap_95_ci"]
    center_counts = test.groupby("center").size().astype(int).to_dict()
    val_class_counts = val.groupby("label").size().astype(int).to_dict()
    gates = [
        gate(
            "locked_external_validation",
            bool(evidence["locked_external_validation"]),
            evidence["validation_design"],
            True,
        ),
        gate("prospective_validation", evidence["prospective_validation"], evidence["prospective_validation"], True),
        gate("mrmc_reader_study", evidence["mrmc_reader_study"], evidence["mrmc_reader_study"], True),
        gate(
            "documented_reference_standard",
            evidence["documented_reference_standard"],
            evidence["documented_reference_standard"],
            True,
        ),
        gate(
            "representative_prevalence",
            evidence["representative_prevalence"],
            "artificially balanced 1:1 cohort",
            True,
        ),
        gate(
            "demographic_subgroup_audit",
            evidence["demographic_subgroup_audit"],
            evidence["demographic_subgroup_audit"],
            True,
        ),
        gate(
            "local_site_acceptance_test",
            evidence["local_site_acceptance_test"],
            evidence["local_site_acceptance_test"],
            True,
        ),
        gate(
            "regulatory_or_irb_authorization",
            evidence["regulatory_or_irb_authorization"],
            evidence["regulatory_or_irb_authorization"],
            True,
        ),
        gate(
            "test_set_remained_locked",
            not evidence["test_set_reused_after_initial_evaluation"],
            not evidence["test_set_reused_after_initial_evaluation"],
            True,
        ),
        gate(
            "external_test_sample_size",
            len(test) >= int(qualification["minimum_external_test_cases"]),
            len(test),
            int(qualification["minimum_external_test_cases"]),
        ),
        gate(
            "minimum_cases_per_center",
            min(center_counts.values()) >= int(qualification["minimum_cases_per_external_center"]),
            center_counts,
            int(qualification["minimum_cases_per_external_center"]),
        ),
        gate(
            "auc_ci_lower",
            float(test_ci["auc"]["low"]) >= float(qualification["minimum_auc_ci_lower"]),
            float(test_ci["auc"]["low"]),
            float(qualification["minimum_auc_ci_lower"]),
        ),
        gate(
            "sensitivity_ci_lower",
            float(test_ci["sensitivity"]["low"]) >= float(qualification["minimum_sensitivity_ci_lower"]),
            float(test_ci["sensitivity"]["low"]),
            float(qualification["minimum_sensitivity_ci_lower"]),
        ),
        gate(
            "specificity_ci_lower",
            float(test_ci["specificity"]["low"]) >= float(qualification["minimum_specificity_ci_lower"]),
            float(test_ci["specificity"]["low"]),
            float(qualification["minimum_specificity_ci_lower"]),
        ),
        gate(
            "calibration_ece",
            float(calibrated_test_metrics["ece"]) <= float(qualification["maximum_ece"]),
            float(calibrated_test_metrics["ece"]),
            float(qualification["maximum_ece"]),
        ),
        gate(
            "validation_cases_per_class",
            min(val_class_counts.values()) >= int(clinical["triage"]["minimum_validation_cases_per_class"]),
            val_class_counts,
            int(clinical["triage"]["minimum_validation_cases_per_class"]),
        ),
        gate(
            "rule_out_npv_ci_lower",
            float(selective["rule_out_npv"]["low"] or 0.0)
            >= float(qualification["minimum_rule_out_npv_ci_lower"]),
            selective["rule_out_npv"],
            float(qualification["minimum_rule_out_npv_ci_lower"]),
        ),
        gate(
            "rule_in_ppv_ci_lower",
            float(selective["rule_in_ppv"]["low"] or 0.0)
            >= float(qualification["minimum_rule_in_ppv_ci_lower"]),
            selective["rule_in_ppv"],
            float(qualification["minimum_rule_in_ppv_ci_lower"]),
        ),
    ]
    clinically_authorized = all(item["passed"] for item in gates)
    checkpoint_hash = file_sha256(args.checkpoint)
    segmentation_hash = checkpoint.get("segmentation_checkpoint_sha256")
    created_at = datetime.now(timezone.utc).isoformat()
    policy = {
        "schema_version": clinical["schema_version"],
        "created_at_utc": created_at,
        "intended_use": clinical["intended_use"],
        "clinical_use_authorized": clinically_authorized,
        "qualification_status": "CLINICALLY_QUALIFIED" if clinically_authorized else "NOT_CLINICALLY_QUALIFIED",
        "model_sha256": checkpoint_hash,
        "segmentation_model_sha256": segmentation_hash,
        "calibration": calibration,
        "single_threshold": single_threshold,
        "triage_thresholds": thresholds,
        "input_qc": clinical["input_qc"],
        "input_profile": input_profile,
        "mandatory_human_review": True,
        "policy_source_report": str(Path(args.output_report).resolve()),
    }
    report = {
        "schema_version": clinical["schema_version"],
        "created_at_utc": created_at,
        "qualification_status": policy["qualification_status"],
        "passed_gates": sum(item["passed"] for item in gates),
        "total_gates": len(gates),
        "gates": gates,
        "calibration": calibration,
        "calibrated_test_metrics": calibrated_test_metrics,
        "selective_test_metrics": selective,
        "center_counts": center_counts,
        "training_input_profile_errors": profile_errors,
        "blocking_statement": (
            "This checkpoint is a research model and must not autonomously diagnose patients. "
            "Clinical authorization remains fail-closed until every evidence and performance gate passes."
        ),
    }
    save_json(args.output_policy, policy)
    save_json(args.output_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
