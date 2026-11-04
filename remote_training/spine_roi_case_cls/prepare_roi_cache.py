from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from common import file_sha256, load_config, resolve_device, save_json, set_seed
from data import (
    center_pad_or_crop,
    central_slice_indices,
    normalize_nonzero,
    read_case_table,
    resolve_sagittal_path,
)


def load_segmenter(config: dict, device: torch.device):
    project = Path(config["segmentation_project"]).resolve()
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from spinevision.checkpoint import load_model_from_checkpoint

    model, model_config, _ = load_model_from_checkpoint(
        config["segmentation_checkpoint"], device
    )
    model.eval()
    return model, model_config


def prepare_case(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    num_slices: int,
    amp_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    image = nib.load(str(image_path), mmap=True)
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D NIfTI, got {image.shape}")
    indices = central_slice_indices(image.shape[-1], num_slices)
    slices = []
    for index in indices:
        plane = np.asarray(image.dataobj[:, :, index], dtype=np.float32)
        plane = center_pad_or_crop(plane, image_size)
        slices.append(normalize_nonzero(plane))
    images = np.stack(slices).astype(np.float32)
    tensor = torch.from_numpy(images).unsqueeze(1).to(device)
    with torch.inference_mode():
        if amp_dtype is None or device.type != "cuda":
            logits = model(tensor)
        else:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        masks = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
    return images.astype(np.float16), masks, indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute central-slice segmentation ROI cache")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = resolve_device(args.device)
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    table = read_case_table(config["label_csv"])
    model, segmentation_config = load_segmenter(config, device)
    image_size = int(config["image_size"])
    num_slices = int(config["num_slices"])
    amp_dtype = torch.bfloat16 if config.get("amp_dtype") == "bfloat16" else None

    started = time.time()
    records = []
    errors = []
    empty_masks = 0
    for row in tqdm(table.itertuples(index=False), total=len(table), desc="ROI cache"):
        output_path = cache_dir / f"{row.id}.npz"
        if output_path.is_file() and not args.overwrite:
            records.append({"patient_id": row.id, "status": "existing"})
            continue
        try:
            image_path = resolve_sagittal_path(config["data_root"], row.id)
            images, masks, indices = prepare_case(
                image_path, model, device, image_size, num_slices, amp_dtype
            )
            temporary = output_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    images=images,
                    masks=masks,
                    slice_indices=np.asarray(indices, dtype=np.int16),
                )
            temporary.replace(output_path)
            is_empty = not np.any(masks > 0)
            empty_masks += int(is_empty)
            records.append(
                {
                    "patient_id": row.id,
                    "status": "created",
                    "source": str(image_path),
                    "slice_indices": indices,
                    "labels_present": sorted(int(value) for value in np.unique(masks) if value > 0),
                    "foreground_fraction": float(np.mean(masks > 0)),
                    "empty_mask": is_empty,
                }
            )
        except Exception as exc:
            errors.append({"patient_id": row.id, "error": repr(exc)})

    manifest = {
        "created_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "device": str(device),
        "cases": len(table),
        "cached": len(records),
        "errors": errors,
        "empty_masks": empty_masks,
        "image_size": image_size,
        "num_slices": num_slices,
        "segmentation_checkpoint": str(Path(config["segmentation_checkpoint"]).resolve()),
        "segmentation_checkpoint_sha256": file_sha256(config["segmentation_checkpoint"]),
        "segmentation_config": segmentation_config,
        "records": records,
    }
    save_json(cache_dir / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("cases", "cached", "empty_masks", "errors", "duration_seconds")}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(f"ROI cache failed for {len(errors)} cases")


if __name__ == "__main__":
    main()
