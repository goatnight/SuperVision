from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import resolve_device, save_json
from data import resolve_sagittal_path
from model import ROIAssistedCaseClassifier
from prepare_roi_cache import load_segmenter, prepare_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one patient-level diagnosis")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    data_root = args.data_root or config["data_root"]
    image_path = resolve_sagittal_path(data_root, args.patient_id)
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
        probability = float(torch.softmax(outputs["logits"].float(), dim=1)[0, 1].cpu())
    threshold = float(checkpoint["decision_threshold"])
    result = {
        "patient_id": args.patient_id,
        "image_path": str(image_path),
        "slice_indices": indices,
        "probability_spondylolisthesis": probability,
        "decision_threshold": threshold,
        "prediction": int(probability >= threshold),
        "slice_attention": outputs["attention"][0].float().cpu().tolist(),
        "roi_gate": outputs["mean_gate"][0].float().cpu().tolist(),
        "segmentation_labels_present": sorted(int(value) for value in set(masks.reshape(-1)) if value > 0),
        "research_only": True,
    }
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
