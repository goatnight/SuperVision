from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect label values in paXXX/label/mask_caseXXX.nii.gz files.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--max-files", type=int, default=0, help="0 means scan all label files.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_dir)
    label_paths = sorted(root.glob("pa*/label/*.nii.gz"))
    if args.max_files > 0:
        label_paths = label_paths[: args.max_files]
    if not label_paths:
        raise FileNotFoundError(f"No label NIfTI files found under {root}/pa*/label")

    all_values: set[int] = set()
    global_min = None
    global_max = None
    for label_path in label_paths:
        nii = nib.load(str(label_path), mmap=True)
        proxy = nii.dataobj
        file_values: set[int] = set()
        for slice_idx in range(nii.shape[-1]):
            arr = np.asarray(proxy[:, :, slice_idx])
            values = np.unique(arr.astype(np.int64, copy=False))
            file_values.update(int(v) for v in values.tolist())
        all_values.update(file_values)
        file_min = min(file_values)
        file_max = max(file_values)
        global_min = file_min if global_min is None else min(global_min, file_min)
        global_max = file_max if global_max is None else max(global_max, file_max)
        print(f"{label_path}: min={file_min} max={file_max} values={sorted(file_values)}", flush=True)

    values_sorted = sorted(all_values)
    print("\n[Summary]", flush=True)
    print(f"files={len(label_paths)}", flush=True)
    print(f"global_min={global_min}", flush=True)
    print(f"global_max={global_max}", flush=True)
    print(f"unique_values={values_sorted}", flush=True)
    print(f"recommended_num_classes={int(global_max) + 1}", flush=True)


if __name__ == "__main__":
    main()
