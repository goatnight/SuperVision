from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


REQUIRED_COLUMNS = {"id", "lsp", "lsp_split", "center"}
VALID_SPLITS = {"train", "val", "test"}


def read_case_table(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path, encoding="utf-8-sig", dtype={"id": str})
    missing = REQUIRED_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    table = table.copy()
    table["id"] = table["id"].astype(str).str.strip()
    table["lsp"] = table["lsp"].astype(int)
    table["lsp_split"] = table["lsp_split"].astype(str).str.lower().str.strip()
    if table["id"].duplicated().any():
        duplicated = table.loc[table["id"].duplicated(), "id"].tolist()
        raise ValueError(f"Duplicated patient IDs: {duplicated[:10]}")
    if not set(table["lsp_split"]).issubset(VALID_SPLITS):
        raise ValueError(f"Unexpected splits: {sorted(set(table['lsp_split']) - VALID_SPLITS)}")
    if not set(table["lsp"]).issubset({0, 1}):
        raise ValueError("Only binary lsp labels 0/1 are supported")
    return table


def image_folder(case_dir: Path) -> Path:
    for name in ("img", "IMG"):
        candidate = case_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No img/IMG folder under {case_dir}")


def resolve_sagittal_path(data_root: str | Path, patient_id: str) -> Path:
    folder = image_folder(Path(data_root) / patient_id)
    digits = "".join(re.findall(r"\d+", patient_id))
    expected = f"case{digits}.nii.gz".lower()
    candidates = sorted(folder.glob("*.nii.gz"))
    for path in candidates:
        if path.name.lower() == expected:
            return path.resolve()

    # A small number of exported cases contain an extra underscore. Exclude T1/TSE/SPIR
    # auxiliary series and choose the shortest remaining sagittal Case file.
    fallback = [
        path
        for path in candidates
        if path.name.lower().startswith(f"case{digits}".lower())
        and not any(token in path.name.lower() for token in ("_t1", "_tse", "_spir"))
    ]
    if fallback:
        return sorted(fallback, key=lambda item: (len(item.name), item.name.lower()))[0].resolve()

    # Some source exports have a mistyped leading digit in every file name (for
    # example pa42xxx stored as Case32xxx). Fall back only to a plain Case series;
    # auxiliary sagittal and axial acquisitions remain explicitly excluded.
    plain_case = [
        path
        for path in candidates
        if path.name.lower().startswith("case")
        and not any(token in path.name.lower() for token in ("_t1", "_tse", "_spir", "axial"))
    ]
    if plain_case:
        return sorted(plain_case, key=lambda item: (len(item.name), item.name.lower()))[0].resolve()
    raise FileNotFoundError(f"No primary sagittal NIfTI found for {patient_id} in {folder}")


def central_slice_indices(depth: int, count: int) -> list[int]:
    if depth < 1 or count < 1:
        raise ValueError(f"Invalid depth/count: {depth}/{count}")
    center = depth // 2
    start = center - count // 2
    return [min(max(start + offset, 0), depth - 1) for offset in range(count)]


def center_pad_or_crop(array: np.ndarray, target_size: int) -> np.ndarray:
    height, width = array.shape
    output = np.zeros((target_size, target_size), dtype=np.float32)
    crop_h, crop_w = min(height, target_size), min(width, target_size)
    src_h = max((height - target_size) // 2, 0)
    src_w = max((width - target_size) // 2, 0)
    dst_h = max((target_size - height) // 2, 0)
    dst_w = max((target_size - width) // 2, 0)
    output[dst_h : dst_h + crop_h, dst_w : dst_w + crop_w] = array[
        src_h : src_h + crop_h, src_w : src_w + crop_w
    ]
    return output


def normalize_nonzero(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    foreground = np.isfinite(array) & (array != 0)
    output = np.zeros_like(array, dtype=np.float32)
    if foreground.any():
        values = array[foreground]
        output[foreground] = (values - values.mean()) / (values.std() + 1e-8)
        output = np.clip(output, -6.0, 6.0)
    return output


class CaseROIDataset(Dataset):
    """One item equals one patient; no slice can cross a train/val/test boundary."""

    def __init__(
        self,
        table: pd.DataFrame,
        split: str,
        cache_dir: str | Path,
        augmentation: dict[str, Any] | None = None,
    ):
        self.rows = table.loc[table["lsp_split"] == split].reset_index(drop=True)
        self.split = split
        self.cache_dir = Path(cache_dir)
        self.augmentation = augmentation if split == "train" else None
        missing = [pid for pid in self.rows["id"] if not (self.cache_dir / f"{pid}.npz").is_file()]
        if missing:
            raise FileNotFoundError(f"Missing ROI cache for {len(missing)} cases: {missing[:10]}")

    def __len__(self) -> int:
        return len(self.rows)

    def _augment(self, images: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.augmentation or {}
        angle = float(torch.empty(1).uniform_(-config.get("rotation_degrees", 0), config.get("rotation_degrees", 0)))
        fraction = float(config.get("translation_fraction", 0))
        max_shift = int(round(images.shape[-1] * fraction))
        translate = [
            int(torch.randint(-max_shift, max_shift + 1, (1,))) if max_shift else 0,
            int(torch.randint(-max_shift, max_shift + 1, (1,))) if max_shift else 0,
        ]
        scale = float(torch.empty(1).uniform_(config.get("scale_min", 1), config.get("scale_max", 1)))
        images = TF.affine(
            images,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        masks = TF.affine(
            masks.float().unsqueeze(1),
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        ).squeeze(1).round().long()

        foreground = images.ne(0)
        intensity_scale = 1.0 + float(torch.empty(1).uniform_(-config.get("intensity_scale", 0), config.get("intensity_scale", 0)))
        intensity_shift = float(torch.empty(1).uniform_(-config.get("intensity_shift", 0), config.get("intensity_shift", 0)))
        images = torch.where(foreground, images * intensity_scale + intensity_shift, images)
        noise_std = float(config.get("noise_std", 0))
        if noise_std > 0:
            images = torch.where(foreground, images + torch.randn_like(images) * noise_std, images)
        return images, masks

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        cached = np.load(self.cache_dir / f"{row['id']}.npz", allow_pickle=False)
        images = torch.from_numpy(cached["images"].astype(np.float32)).unsqueeze(1)
        masks = torch.from_numpy(cached["masks"].astype(np.int64))
        if self.augmentation:
            images, masks = self._augment(images, masks)
        return {
            "images": images,
            "masks": masks,
            "label": torch.tensor(int(row["lsp"]), dtype=torch.long),
            "patient_id": str(row["id"]),
            "center": str(row["center"]),
        }
