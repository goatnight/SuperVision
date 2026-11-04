from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from clinical_safety import (
    apply_temperature,
    disposition,
    nifti_metadata,
    validate_input,
    validate_segmentation,
)
from common import file_sha256, resolve_device, save_json
from data import resolve_sagittal_path
from model import ROIAssistedCaseClassifier
from prepare_roi_cache import load_segmenter, prepare_case


def append_audit_event(path: str | Path, event: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
    with output.open("a", encoding="utf-8") as stream:
        try:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        stream.write(line)
        stream.flush()


def case_reference(patient_id: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), patient_id.encode("utf-8"), hashlib.sha256).hexdigest()[:20]


def candidate_band(probability: float, thresholds: dict[str, float]) -> str:
    if probability < float(thresholds["rule_out_below"]):
        return "RULE_OUT_BAND"
    if probability >= float(thresholds["rule_in_at_or_above"]):
        return "RULE_IN_BAND"
    return "INDETERMINATE_BAND"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed spondylolisthesis decision-support inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-log", required=True)
    parser.add_argument("--audit-secret-env", default="SPINE_AUDIT_SECRET")
    args = parser.parse_args()

    secret = os.getenv(args.audit_secret_env)
    if not secret or len(secret) < 16:
        raise RuntimeError(f"Set {args.audit_secret_env} to a secret of at least 16 characters")
    device = resolve_device(args.device)
    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    actual_model_hash = file_sha256(args.checkpoint)
    if not hmac.compare_digest(actual_model_hash, policy["model_sha256"]):
        raise RuntimeError("Checkpoint hash does not match the qualified policy")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    data_root = args.data_root or config["data_root"]
    image_path = resolve_sagittal_path(data_root, args.patient_id)
    metadata, _ = nifti_metadata(image_path)
    input_qc = validate_input(metadata, policy["input_profile"], policy["input_qc"])
    reference = case_reference(args.patient_id, secret)
    base_event = {
        "schema_version": policy["schema_version"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "case_reference": reference,
        "input_sha256": file_sha256(image_path),
        "model_sha256": actual_model_hash,
        "policy_sha256": file_sha256(args.policy),
        "clinical_use_authorized": bool(policy["clinical_use_authorized"]),
    }

    if not input_qc["passed"]:
        result = {
            **base_event,
            "disposition": disposition(0.5, policy["triage_thresholds"], False, False),
            "requires_radiologist_review": True,
            "input_qc": input_qc,
            "segmentation_qc": None,
            "model_output": None,
        }
        save_json(args.output, result)
        append_audit_event(args.audit_log, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    segmenter, _ = load_segmenter(config, device)
    amp_dtype = torch.bfloat16 if config.get("amp_dtype") == "bfloat16" else None
    images, masks, indices = prepare_case(
        image_path,
        segmenter,
        device,
        int(config["image_size"]),
        int(config["num_slices"]),
        amp_dtype,
    )
    segmentation_qc = validate_segmentation(masks, policy["input_profile"], policy["input_qc"])
    safety_passed = input_qc["passed"] and segmentation_qc["passed"]
    if not safety_passed:
        result = {
            **base_event,
            "disposition": disposition(0.5, policy["triage_thresholds"], False, False),
            "requires_radiologist_review": True,
            "input_qc": input_qc,
            "segmentation_qc": segmentation_qc,
            "model_output": None,
        }
        save_json(args.output, result)
        append_audit_event(args.audit_log, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    model = ROIAssistedCaseClassifier(
        config["segmentation_project"],
        config["segmentation_checkpoint"],
        num_segmentation_classes=int(config["num_segmentation_classes"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        roi_dilation=int(config["roi_dilation"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    image_tensor = torch.from_numpy(images.astype("float32")).unsqueeze(0).unsqueeze(2).to(device)
    mask_tensor = torch.from_numpy(masks.astype("int64")).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and config.get("amp_dtype") == "bfloat16",
    ):
        outputs = model(image_tensor, mask_tensor)
        raw_probability = float(torch.softmax(outputs["logits"].float(), dim=1)[0, 1].cpu())
    calibrated_probability = float(
        apply_temperature(raw_probability, float(policy["calibration"]["temperature"]))
    )
    result = {
        **base_event,
        "disposition": disposition(
            calibrated_probability,
            policy["triage_thresholds"],
            safety_passed,
            bool(policy["clinical_use_authorized"]),
        ),
        "requires_radiologist_review": True,
        "candidate_band": candidate_band(calibrated_probability, policy["triage_thresholds"]),
        "input_qc": input_qc,
        "segmentation_qc": segmentation_qc,
        "model_output": {
            "raw_probability": raw_probability,
            "calibrated_probability": calibrated_probability,
            "single_threshold": float(policy["single_threshold"]),
            "triage_thresholds": policy["triage_thresholds"],
            "slice_indices": indices,
            "slice_attention": outputs["attention"][0].float().cpu().tolist(),
            "roi_gate": outputs["mean_gate"][0].float().cpu().tolist(),
        },
    }
    save_json(args.output, result)
    append_audit_event(args.audit_log, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
