"""MRI 输入检查工具，只验证文件和几何信息，不进行诊断推理。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pydantic import BaseModel

from ..errors import InputValidationError, ModelUnavailableError
from .base import AgentTool


class InspectImageInput(BaseModel):
    image_path: Path
    modality: str


def file_sha256(path: Path) -> str:
    """计算输入/权重指纹，用于结果复现和 provenance，不能替代去标识化。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class InspectImageTool(AgentTool):
    """在昂贵模型推理前执行格式、模态和 NIfTI 几何检查。"""

    name = "inspect_mri_input"
    description = "Validate a lumbar MRI input and return non-identifying file metadata."
    input_model = InspectImageInput

    def __init__(self, clinical_policy: Path | None = None):
        self.clinical_policy = clinical_policy.resolve() if clinical_policy else None
        self._policy: dict[str, Any] | None = None

    def _load_policy(self) -> dict[str, Any] | None:
        if self.clinical_policy is None:
            return None
        if self._policy is None:
            if not self.clinical_policy.is_file():
                raise ModelUnavailableError(f"Clinical policy not found: {self.clinical_policy}")
            self._policy = json.loads(self.clinical_policy.read_text(encoding="utf-8"))
        return self._policy

    @staticmethod
    def _in_range(value: float, bounds: list[float]) -> bool:
        return float(bounds[0]) <= float(value) <= float(bounds[1])

    def run(self, payload: InspectImageInput):
        path = payload.image_path.expanduser().resolve()
        if payload.modality.upper() != "MRI":
            raise InputValidationError("Only MRI is enabled for the validated workflow.")
        if not path.is_file():
            raise InputValidationError(f"Input image does not exist: {path}")
        name = path.name.lower()
        if not (name.endswith(".nii") or name.endswith(".nii.gz")):
            raise InputValidationError("The current model adapter accepts NIfTI (.nii/.nii.gz) only.")
        digest = file_sha256(path)
        data: dict[str, Any] = {
            "size_bytes": path.stat().st_size,
            "input_hash": digest,
            "format": "nifti",
            "modality": "MRI",
        }
        # NIfTI 文件名无法可靠判断 T1/T2 等序列，正式系统应读取经过脱敏的协议元数据。
        warnings = ["MRI sequence/protocol cannot be verified from the file extension alone."]
        try:
            import nibabel as nib

            image = nib.load(str(path), mmap=True)
            shape = list(image.shape)
            spacing = [float(v) for v in image.header.get_zooms()[: len(image.shape)]]
            affine = np.asarray(image.affine, dtype=np.float64)
            orientation = list(nib.aff2axcodes(affine))
            data.update(shape=shape, spacing=spacing, orientation=orientation, affine_valid=False)
            if len(image.shape) != 3:
                raise InputValidationError(f"Expected a 3D NIfTI volume, got shape={image.shape}")
            affine_valid = bool(
                affine.shape == (4, 4)
                and np.all(np.isfinite(affine))
                and abs(float(np.linalg.det(affine[:3, :3]))) >= 1e-8
            )
            data["affine_valid"] = affine_valid
            steps = tuple(max(int(value) // 64, 1) for value in image.shape)
            sample = np.asarray(
                image.dataobj[tuple(slice(None, None, step) for step in steps)], dtype=np.float32
            )
            finite = np.isfinite(sample)
            values = sample[finite]
            finite_fraction = float(np.mean(finite))
            nonzero_fraction = float(np.mean(values != 0)) if values.size else 0.0
            if values.size:
                low, high = np.percentile(values, [1.0, 99.0])
                dynamic_range = float(high - low)
            else:
                dynamic_range = 0.0
            data.update(
                finite_fraction=finite_fraction,
                nonzero_fraction=nonzero_fraction,
                dynamic_range_p01_p99=dynamic_range,
            )

            policy = self._load_policy()
            qc_errors: list[str] = []
            qc_warnings: list[str] = []
            if not affine_valid:
                qc_errors.append("invalid_affine_geometry")
            if policy:
                qc = policy["input_qc"]
                profile = policy["input_profile"]
                if min(shape[:2]) < int(qc["minimum_in_plane_size"]):
                    qc_errors.append("in_plane_matrix_too_small")
                if shape[-1] < int(qc["minimum_slices"]):
                    qc_errors.append("insufficient_sagittal_slices")
                if finite_fraction < float(qc["minimum_finite_fraction"]):
                    qc_errors.append("non_finite_voxels")
                if not (
                    float(qc["minimum_nonzero_fraction"])
                    <= nonzero_fraction
                    <= float(qc["maximum_nonzero_fraction"])
                ):
                    qc_errors.append("invalid_nonzero_fraction")
                if dynamic_range < float(qc["minimum_dynamic_range"]):
                    qc_errors.append("insufficient_intensity_dynamic_range")
                if "".join(orientation) not in set(profile["accepted_orientations"]):
                    qc_errors.append("orientation_outside_training_domain")
                for name, values, ranges in (
                    ("shape", shape, profile["shape_ranges"]),
                    ("spacing", spacing, profile["spacing_ranges"]),
                ):
                    for axis, (value, bounds) in enumerate(zip(values, ranges)):
                        if not self._in_range(value, bounds):
                            qc_errors.append(f"{name}_axis_{axis}_outside_training_domain")
                data["clinical_policy_hash"] = file_sha256(self.clinical_policy)
                if not policy.get("clinical_use_authorized", False):
                    qc_warnings.append("CLINICAL_POLICY_NOT_AUTHORIZED")
            else:
                qc_warnings.append("CLINICAL_POLICY_NOT_CONFIGURED")
            data.update(
                qc_passed=not qc_errors,
                qc_errors=sorted(set(qc_errors)),
                qc_warnings=sorted(set(qc_warnings)),
            )
            warnings.extend(data["qc_errors"])
            warnings.extend(data["qc_warnings"])
        except ImportError:
            raise ModelUnavailableError("nibabel is required for validated input QC.")
        return data, warnings, {}, {"input_hash": digest}
