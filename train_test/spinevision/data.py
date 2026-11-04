from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SliceRecord:
    image_path: Path
    label_path: Path
    pa_id: str
    case_id: str
    slice_index: int
    spacing_xy: tuple[float, float]

    @property
    def name(self) -> str:
        return f"{self.pa_id}_{self.case_id}_slice{self.slice_index:03d}"


def _randint_inclusive(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def shift_image(image: np.ndarray, label: np.ndarray, transition_x: int, transition_y: int):
    from scipy import ndimage

    image = ndimage.shift(image, (transition_y, transition_x), order=1, mode="nearest")
    label = ndimage.shift(label, (transition_y, transition_x), order=0, mode="nearest")
    return image, label


def random_shift(image: np.ndarray, label: np.ndarray, shift: Sequence[int], rng: np.random.Generator):
    if shift[0] == 0 and shift[1] == 0:
        return image, label
    transition_x = _randint_inclusive(rng, -int(shift[0]), int(shift[0])) if shift[0] else 0
    transition_y = _randint_inclusive(rng, -int(shift[1]), int(shift[1])) if shift[1] else 0
    return shift_image(image, label, transition_x, transition_y)


def random_rotate(image: np.ndarray, label: np.ndarray, rotate: int, rng: np.random.Generator):
    if rotate == 0:
        return image, label
    from scipy import ndimage

    angle = _randint_inclusive(rng, -int(rotate), int(rotate))
    image = ndimage.rotate(image, angle, order=1, reshape=False, mode="nearest")
    label = ndimage.rotate(label, angle, order=0, reshape=False, mode="nearest")
    return image, label


def elastic_deform_2d_rand(
    image: np.ndarray,
    label: np.ndarray,
    alpha: tuple[float, float],
    sigma: float,
    rng: np.random.Generator,
):
    """Optional elastic deformation kept for compatibility with the original preprocessing code.

    It is disabled by default because it is relatively expensive on CPU.
    """
    from scipy import ndimage

    shape = image.shape
    dx = ndimage.gaussian_filter((rng.random(shape) * 2 - 1), sigma, mode="reflect") * alpha[0]
    dy = ndimage.gaussian_filter((rng.random(shape) * 2 - 1), sigma, mode="reflect") * alpha[1]
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    indices = (np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)))
    image = ndimage.map_coordinates(image, indices, order=1, mode="reflect").reshape(shape)
    label = ndimage.map_coordinates(label, indices, order=0, mode="reflect").reshape(shape)
    return image, label


def rand_elasticdeform(
    image: np.ndarray,
    label: np.ndarray,
    elastic: float | Sequence[float],
    sigma: float,
    rng: np.random.Generator,
):
    if isinstance(elastic, (float, int)):
        elastic_values: Iterable[float] = (float(elastic),)
    else:
        elastic_values = [float(v) for v in elastic]

    for value in elastic_values:
        if value > 0:
            image, label = elastic_deform_2d_rand(image, label, (value, value), sigma, rng)
    return image, label


def random_generator(
    image: np.ndarray,
    label: np.ndarray,
    shift: Sequence[int],
    rotate: int,
    elastic: float | Sequence[float],
    sigma: float,
    rng: np.random.Generator,
):
    image, label = rand_elasticdeform(image, label, elastic, sigma, rng)
    image, label = random_rotate(image, label, rotate, rng)
    image, label = random_shift(image, label, shift, rng)
    return image, label


def scan_sagittal_slices(data_dir: str | Path, keep_empty: bool = False) -> list[SliceRecord]:
    """Scan paXXX/img/label folders and store only slice metadata.

    The original preprocessing loaded every valid slice into RAM. This scanner only
    reads label slices to find useful indices, keeping training usable on small CPUs.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    records: list[SliceRecord] = []
    for pa_folder in sorted(root.iterdir()):
        if not pa_folder.is_dir() or not pa_folder.name.startswith("pa"):
            continue

        img_folder = pa_folder / "img"
        label_folder = pa_folder / "label"
        if not img_folder.exists() or not label_folder.exists():
            print(f"[Warning] skip {pa_folder.name}: missing img/label folder", flush=True)
            continue

        img_files = sorted(p for p in img_folder.iterdir() if p.name.startswith("Case") and p.name.endswith(".nii.gz"))
        if not img_files:
            print(f"[Warning] skip {pa_folder.name}: no Case*.nii.gz image", flush=True)
            continue

        img_path = img_files[0]
        match = re.search(r"Case(\d+)", img_path.name)
        if match is None:
            print(f"[Warning] skip {pa_folder.name}: cannot parse case id from {img_path.name}", flush=True)
            continue

        case_num = match.group(1)
        case_id = img_path.name.replace(".nii.gz", "")
        label_path = label_folder / f"mask_case{case_num}.nii.gz"
        if not label_path.exists():
            print(f"[Warning] skip {pa_folder.name}: missing {label_path.name}", flush=True)
            continue

        try:
            label_nii = nib.load(str(label_path), mmap=True)
            spacing_xy = tuple(float(x) for x in label_nii.header.get_zooms()[:2])
            label_proxy = label_nii.dataobj
            for slice_idx in range(label_nii.shape[-1]):
                label_slice = np.asarray(label_proxy[:, :, slice_idx])
                if keep_empty or np.any(label_slice > 0):
                    records.append(
                        SliceRecord(
                            image_path=img_path,
                            label_path=label_path,
                            pa_id=pa_folder.name,
                            case_id=case_id,
                            slice_index=slice_idx,
                            spacing_xy=spacing_xy,
                        )
                    )
        except Exception as exc:
            print(f"[Warning] skip {pa_folder.name}: failed to scan label, error: {exc}", flush=True)

    print(f"[Data] scanned {len(records)} slices from {root}", flush=True)
    return records


class _NiftiCache:
    def __init__(self, max_cases: int = 2):
        self.max_cases = max(1, int(max_cases))
        self._cache: OrderedDict[tuple[Path, Path], tuple[nib.spatialimages.SpatialImage, nib.spatialimages.SpatialImage]] = OrderedDict()

    def get(self, image_path: Path, label_path: Path):
        key = (image_path, label_path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        value = (nib.load(str(image_path), mmap=True), nib.load(str(label_path), mmap=True))
        self._cache[key] = value
        while len(self._cache) > self.max_cases:
            self._cache.popitem(last=False)
        return value


class SagittalDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path | None = None,
        num_classes: int = 3,
        target_size: int | tuple[int, int] = 320,
        records: Optional[Sequence[SliceRecord]] = None,
        augment: bool = False,
        shift: tuple[int, int] = (0, 0),
        rotate: int = 0,
        elastic: float | Sequence[float] = 0.0,
        sigma: float = 8.0,
        return_one_hot: bool = False,
        label_mode: str = "auto",
        keep_empty: bool = False,
        cache_cases: int = 2,
        seed: int = 2026,
    ):
        if records is None:
            if data_dir is None:
                raise ValueError("Either data_dir or records must be provided.")
            records = scan_sagittal_slices(data_dir, keep_empty=keep_empty)

        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        self.records = list(records)
        self.num_classes = int(num_classes)
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.augment = augment
        self.shift = shift
        self.rotate = int(rotate)
        self.elastic = elastic
        self.sigma = float(sigma)
        self.return_one_hot = return_one_hot
        self.label_mode = label_mode
        self.seed = int(seed)
        self._cache = _NiftiCache(max_cases=cache_cases)

    def __len__(self):
        return len(self.records)

    def _pad_or_crop_to_size(self, tensor: torch.Tensor) -> torch.Tensor:
        target_h, target_w = self.target_size
        h, w = tensor.shape
        if h < target_h or w < target_w:
            pad_h = max(target_h - h, 0)
            pad_w = max(target_w - w, 0)
            pad_h1, pad_h2 = pad_h // 2, pad_h - pad_h // 2
            pad_w1, pad_w2 = pad_w // 2, pad_w - pad_w // 2
            tensor = F.pad(tensor[None, None], (pad_w1, pad_w2, pad_h1, pad_h2), value=0).squeeze(0).squeeze(0)

        h, w = tensor.shape
        if h > target_h or w > target_w:
            crop_h1 = max((h - target_h) // 2, 0)
            crop_w1 = max((w - target_w) // 2, 0)
            tensor = tensor[crop_h1 : crop_h1 + target_h, crop_w1 : crop_w1 + target_w]
        return tensor

    def _recode_label(self, label: torch.Tensor) -> torch.Tensor:
        label = label.round().long()
        label_mode = self.label_mode
        if label_mode == "auto":
            label_mode = "binary" if self.num_classes == 2 else "three" if self.num_classes == 3 else "preserve"

        if label_mode == "binary":
            return (label > 0).long()
        if label_mode == "three":
            recoded = torch.zeros_like(label, dtype=torch.long)
            recoded[(label >= 1) & (label <= 6)] = 1
            recoded[label >= 7] = 2
            return recoded
        if label_mode != "preserve":
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

        max_label = int(label.max().item()) if label.numel() else 0
        min_label = int(label.min().item()) if label.numel() else 0
        if min_label < 0 or max_label >= self.num_classes:
            raise ValueError(
                f"Label value out of range for num_classes={self.num_classes}: "
                f"min={min_label}, max={max_label}. "
                "For PyTorch segmentation, num_classes must include background label 0."
            )
        return label

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image_nii, label_nii = self._cache.get(record.image_path, record.label_path)

        image = np.asarray(image_nii.dataobj[:, :, record.slice_index], dtype=np.float32)
        label = np.asarray(label_nii.dataobj[:, :, record.slice_index], dtype=np.float32)
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)

        if self.augment:
            rng = np.random.default_rng(self.seed + idx)
            image, label = random_generator(image, label, self.shift, self.rotate, self.elastic, self.sigma, rng)

        img_tensor = torch.as_tensor(image, dtype=torch.float32)
        label_tensor = torch.as_tensor(label, dtype=torch.float32)

        img_tensor = self._pad_or_crop_to_size(img_tensor)
        label_tensor = self._pad_or_crop_to_size(label_tensor)

        nonzero = img_tensor != 0
        if torch.any(nonzero):
            mean = img_tensor[nonzero].mean()
            std = img_tensor[nonzero].std()
            img_tensor = (img_tensor - mean) / (std + 1e-8)
        else:
            img_tensor = torch.zeros_like(img_tensor)

        label_tensor = self._recode_label(label_tensor)
        sample = {
            "image": img_tensor.unsqueeze(0),
            "label": label_tensor,
            "filename": record.name,
            "spacing": torch.tensor(record.spacing_xy, dtype=torch.float32),
        }

        if self.return_one_hot:
            sample["one_hot"] = F.one_hot(label_tensor, num_classes=self.num_classes).permute(2, 0, 1).float()
        return sample
