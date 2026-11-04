from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="CPU inference for one NIfTI image with exported TorchScript.")
    parser.add_argument("--model", required=True, help="TorchScript model from export_cpu.py.")
    parser.add_argument("--image", required=True, help="Input CaseXXX.nii.gz.")
    parser.add_argument("--output", required=True, help="Output prediction .nii.gz.")
    parser.add_argument("--image-size", type=int, required=True, help="The size used during training/export.")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--process-empty", action="store_true", help="Also run slices where all voxels are zero.")
    return parser.parse_args()


def center_pad_or_crop(arr: np.ndarray, target_size: int):
    h, w = arr.shape
    target = np.zeros((target_size, target_size), dtype=np.float32)

    crop_h = min(h, target_size)
    crop_w = min(w, target_size)
    src_h0 = max((h - target_size) // 2, 0)
    src_w0 = max((w - target_size) // 2, 0)
    dst_h0 = max((target_size - h) // 2, 0)
    dst_w0 = max((target_size - w) // 2, 0)

    target[dst_h0 : dst_h0 + crop_h, dst_w0 : dst_w0 + crop_w] = arr[src_h0 : src_h0 + crop_h, src_w0 : src_w0 + crop_w]
    meta = (h, w, crop_h, crop_w, src_h0, src_w0, dst_h0, dst_w0)
    return target, meta


def restore_shape(pred: np.ndarray, meta):
    h, w, crop_h, crop_w, src_h0, src_w0, dst_h0, dst_w0 = meta
    out = np.zeros((h, w), dtype=np.uint8)
    out[src_h0 : src_h0 + crop_h, src_w0 : src_w0 + crop_w] = pred[dst_h0 : dst_h0 + crop_h, dst_w0 : dst_w0 + crop_w]
    return out


def normalize_nonzero(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    nonzero = arr != 0
    if np.any(nonzero):
        mean = arr[nonzero].mean()
        std = arr[nonzero].std()
        return (arr - mean) / (std + 1e-8)
    return np.zeros_like(arr, dtype=np.float32)


def main():
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    model = torch.jit.load(args.model, map_location="cpu").eval()
    image_nii = nib.load(args.image, mmap=True)
    image_proxy = image_nii.dataobj
    output = np.zeros(image_nii.shape, dtype=np.uint8)

    with torch.no_grad():
        for slice_idx in range(image_nii.shape[-1]):
            image_slice = np.asarray(image_proxy[:, :, slice_idx], dtype=np.float32)
            if not args.process_empty and not np.any(image_slice != 0):
                continue

            resized, meta = center_pad_or_crop(image_slice, args.image_size)
            resized = normalize_nonzero(resized)
            tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0)
            logits = model(tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            output[:, :, slice_idx] = restore_shape(pred, meta)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_nii = nib.Nifti1Image(output, affine=image_nii.affine, header=image_nii.header)
    pred_nii.set_data_dtype(np.uint8)
    nib.save(pred_nii, str(out_path))
    print(f"[Saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
