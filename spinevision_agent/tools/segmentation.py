"""腰椎 MRI 分割模型适配层。

这是当前主链路中真正执行神经网络推理的模块。它负责加载本地 checkpoint、
保持 NIfTI 几何、逐切片推理并输出结构化 QC，而不是语言 Agent。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from ..errors import InputValidationError, ModelUnavailableError, ToolExecutionError
from .base import AgentTool
from .imaging import file_sha256


class SegmentationInput(BaseModel):
    image_path: Path
    case_id: str


def center_pad_or_crop(arr: np.ndarray, target_size: int):
    h, w = arr.shape
    target = np.zeros((target_size, target_size), dtype=np.float32)
    crop_h, crop_w = min(h, target_size), min(w, target_size)
    src_h0, src_w0 = max((h - target_size) // 2, 0), max((w - target_size) // 2, 0)
    dst_h0, dst_w0 = max((target_size - h) // 2, 0), max((target_size - w) // 2, 0)
    target[dst_h0 : dst_h0 + crop_h, dst_w0 : dst_w0 + crop_w] = arr[
        src_h0 : src_h0 + crop_h, src_w0 : src_w0 + crop_w
    ]
    return target, (h, w, crop_h, crop_w, src_h0, src_w0, dst_h0, dst_w0)


def restore_shape(pred: np.ndarray, meta) -> np.ndarray:
    h, w, crop_h, crop_w, src_h0, src_w0, dst_h0, dst_w0 = meta
    out = np.zeros((h, w), dtype=np.uint8)
    out[src_h0 : src_h0 + crop_h, src_w0 : src_w0 + crop_w] = pred[
        dst_h0 : dst_h0 + crop_h, dst_w0 : dst_w0 + crop_w
    ]
    return out


def normalize_nonzero(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    foreground = arr != 0
    if not np.any(foreground):
        return np.zeros_like(arr)
    output = np.zeros_like(arr, dtype=np.float32)
    values = arr[foreground]
    output[foreground] = (values - values.mean()) / (values.std() + 1e-8)
    return np.clip(output, -6.0, 6.0)


class SegmentationService:
    """延迟加载本地 SwinUMambaLite，避免 API 启动时立即占用 GPU。"""

    def __init__(
        self,
        checkpoint: Path,
        artifact_root: Path,
        device: str = "auto",
        project_root: Path | None = None,
    ):
        self.checkpoint = checkpoint.resolve()
        self.artifact_root = artifact_root.resolve()
        self.device_name = device
        self.project_root = (
            project_root.resolve()
            if project_root
            else Path(__file__).resolve().parents[2] / "train_test"
        )
        self._model = None
        self._config: dict[str, Any] = {}
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        if not self.checkpoint.is_file():
            raise ModelUnavailableError(f"Segmentation checkpoint not found: {self.checkpoint}")
        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailableError("PyTorch is required for segmentation.") from exc
        # 复用训练工程里的 checkpoint loader，保证训练和部署的网络构建逻辑一致。
        train_root = self.project_root
        if str(train_root) not in sys.path:
            sys.path.insert(0, str(train_root))
        from spinevision.checkpoint import load_model_from_checkpoint

        if self.device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = self.device_name
        self._device = torch.device(device_name)
        self._model, self._config, _ = load_model_from_checkpoint(self.checkpoint, self._device)
        self._model.eval()

    def predict(self, image_path: Path, case_id: str) -> tuple[dict[str, Any], dict[str, float]]:
        try:
            import nibabel as nib
            import torch
        except ImportError as exc:
            raise ModelUnavailableError("nibabel and PyTorch are required for segmentation.") from exc
        self._load()
        image = nib.load(str(image_path), mmap=True)
        if len(image.shape) != 3:
            raise InputValidationError(f"Expected 3D NIfTI, got {image.shape}")
        target_size = int(self._config.get("image_size", 640))
        output = np.zeros(image.shape, dtype=np.uint8)
        processed = 0
        # 当前网络是 2D 模型，因此沿 NIfTI 最后一维逐切片推理，再恢复为 3D mask。
        with torch.inference_mode():
            for index in range(image.shape[-1]):
                plane = np.asarray(image.dataobj[:, :, index], dtype=np.float32)
                if not np.any(plane != 0):
                    continue
                resized, transform = center_pad_or_crop(plane, target_size)
                tensor = torch.from_numpy(normalize_nonzero(resized)).unsqueeze(0).unsqueeze(0).to(self._device)
                logits = self._model(tensor)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                output[:, :, index] = restore_shape(pred, transform)
                processed += 1
        out_dir = self.artifact_root / case_id
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_path = out_dir / "segmentation.nii.gz"
        mask_image = nib.Nifti1Image(output, affine=image.affine, header=image.header)
        mask_image.set_data_dtype(np.uint8)
        nib.save(mask_image, str(mask_path))
        labels = sorted(int(v) for v in np.unique(output) if int(v) != 0)
        warnings: list[str] = []
        if not labels:
            warnings.append("EMPTY_MASK")
        if processed == 0:
            warnings.append("EMPTY_VOLUME")
        spacing = [float(v) for v in image.header.get_zooms()[:3]]
        # Tool 只输出可验证事实；具体节段语义必须在标签字典通过验证后才能启用。
        return {
            "mask_path": str(mask_path),
            "input_shape": list(image.shape),
            "output_shape": list(output.shape),
            "labels_present": labels,
            "foreground_voxels": int(np.count_nonzero(output)),
            "spacing": spacing,
            "qc_passed": not warnings,
            "qc_warnings": warnings,
            "label_semantics_validated": False,
            "foreground_fraction": float(np.mean(output > 0)),
            "model_checkpoint_hash": file_sha256(self.checkpoint),
            "input_hash": file_sha256(image_path),
        }, {"processed_slices": float(processed)}


class RunSegmentationTool(AgentTool):
    name = "run_spine_segmentation"
    description = "Run the validated SwinUMambaLite checkpoint and preserve NIfTI geometry."
    input_model = SegmentationInput

    def __init__(self, service: SegmentationService):
        self.service = service

    def run(self, payload: SegmentationInput):
        data, metrics = self.service.predict(payload.image_path.resolve(), payload.case_id)
        warnings = list(data.get("qc_warnings", []))
        return data, warnings, metrics, {
            "model_version": "SwinUMambaLite",
            "checkpoint_hash": file_sha256(self.service.checkpoint),
            "input_hash": file_sha256(payload.image_path),
        }


class MockSegmentationTool(AgentTool):
    """仅供测试/Benchmark 使用的假分割工具，不能用于真实病例结论。"""

    name = "run_spine_segmentation"
    description = "Deterministic segmentation test double."
    input_model = SegmentationInput

    def run(self, payload: SegmentationInput):
        data = {
            "mask_path": str(payload.image_path),
            "input_shape": [64, 64, 8],
            "output_shape": [64, 64, 8],
            "labels_present": [1, 2],
            "foreground_voxels": 512,
            "spacing": [1.0, 1.0, 3.0],
            "qc_passed": True,
            "qc_warnings": [],
            "label_semantics_validated": False,
        }
        return data, [], {"processed_slices": 8.0}, {"model_version": "mock-segmentation"}
