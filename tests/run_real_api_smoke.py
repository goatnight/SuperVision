from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from spinevision_agent.api import create_app
from spinevision_agent.config import AgentSettings
from spinevision_agent.workflow import SpineVisionWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise real models through the FastAPI routes")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workflow = SpineVisionWorkflow(AgentSettings.from_env())
    client = TestClient(create_app(workflow))
    health = client.get("/health")
    if health.status_code != 200 or health.json()["classification_backend"] != "roi_case":
        raise AssertionError(f"Unexpected health response: {health.text}")
    created = client.post(
        "/cases",
        json={
            "image_path": str(Path(args.image)),
            "modality": "MRI",
            "user_goal": "FastAPI 真实模型接入验证",
            "patient_hash": "real-api-smoke-case",
        },
    )
    if created.status_code != 200:
        raise AssertionError(created.text)
    case_id = created.json()["case_id"]
    completed = client.post(f"/cases/{case_id}/run")
    if completed.status_code != 200:
        raise AssertionError(completed.text)
    payload = completed.json()
    if payload["diagnosis"]["model_name"] != "roi-assisted-swinumamba-case-classifier":
        raise AssertionError("API did not execute the real classifier")
    trace = client.get(f"/cases/{case_id}/trace")
    if trace.status_code != 200 or len(trace.json()) != 6:
        raise AssertionError("API trace is incomplete")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "health": health.json(),
                "case_id": case_id,
                "status": payload["status"],
                "classifier": payload["diagnosis"]["model_name"],
                "calibrated_probability": payload["diagnosis"]["calibrated_probability"],
                "candidate_band": payload["diagnosis"]["candidate_band"],
                "trace_steps": len(trace.json()),
                "human_review_required": payload["human_review_required"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
