from __future__ import annotations

import argparse
import json
from pathlib import Path

from spinevision_agent.config import AgentSettings
from spinevision_agent.schemas import CaseCreateRequest
from spinevision_agent.workflow import SpineVisionWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real segmentation/classification Agent chain")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = AgentSettings.from_env()
    if settings.classifier_backend != "roi_case":
        raise AssertionError(f"Expected roi_case backend, got {settings.classifier_backend}")
    workflow = SpineVisionWorkflow(settings)
    case = workflow.create_case(
        CaseCreateRequest(
            image_path=Path(args.image),
            modality="MRI",
            user_goal="真实分割与病例级滑脱分类集成测试",
            patient_hash="real-model-smoke-case",
        )
    )
    completed = workflow.run_case(case.case_id)
    if completed.input_qc is None or not completed.input_qc.qc_passed:
        raise AssertionError("Input QC did not pass")
    if completed.segmentation is None or not completed.segmentation.qc_passed:
        raise AssertionError("Real segmentation did not produce a valid ROI")
    if not completed.segmentation.model_checkpoint_hash:
        raise AssertionError("Segmentation checkpoint provenance is missing")
    if completed.diagnosis is None:
        raise AssertionError("Real classification output is missing")
    if "mock" in completed.diagnosis.model_name.lower():
        raise AssertionError("Workflow used a mock classifier")
    if completed.diagnosis.raw_probability is None or completed.diagnosis.calibrated_probability is None:
        raise AssertionError("Calibrated probability fields are missing")
    if not completed.diagnosis.model_checkpoint_hash or not completed.diagnosis.clinical_policy_hash:
        raise AssertionError("Classification provenance is missing")
    if not completed.human_review_required:
        raise AssertionError("Unqualified clinical policy must force human review")

    payload = completed.model_dump(mode="json")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "case_id": completed.case_id,
                "status": completed.status,
                "tools": [item.tool_name for item in completed.tool_trace],
                "segmentation_labels": completed.segmentation.labels_present,
                "segmentation_foreground_fraction": completed.segmentation.foreground_fraction,
                "classifier": completed.diagnosis.model_name,
                "raw_probability": completed.diagnosis.raw_probability,
                "calibrated_probability": completed.diagnosis.calibrated_probability,
                "candidate_band": completed.diagnosis.candidate_band,
                "safety_disposition": completed.diagnosis.safety_disposition,
                "human_review_required": completed.human_review_required,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
