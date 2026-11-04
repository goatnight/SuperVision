"""病例级滑脱分类 Tool 及其可替换测试后端。"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel

from ..errors import InputValidationError, ModelUnavailableError, ToolExecutionError
from .base import AgentTool
from .imaging import file_sha256
from .segmentation import center_pad_or_crop, normalize_nonzero


class ClassificationInput(BaseModel):
    image_path: Path
    roi_path: Path
    case_id: str


class ClassifierBackend(Protocol):
    model_name: str

    def predict(self, image_path: Path, roi_path: Path) -> dict[str, Any]: ...


class MockClassifierBackend:
    """返回固定概率的测试替身；当前默认配置使用它。"""

    model_name = "mock-deepmr-dual-branch"

    def __init__(self, confidence: float = 0.82):
        self.confidence = confidence

    def predict(self, image_path: Path, roi_path: Path) -> dict[str, Any]:
        return {
            "predicted_class": "spondylolisthesis_candidate",
            "probabilities": {
                "no_spondylolisthesis": 1.0 - self.confidence,
                "spondylolisthesis_candidate": self.confidence,
            },
            "confidence": self.confidence,
            "uncertainty": 1.0 - self.confidence,
            "model_name": self.model_name,
        }


class CallableClassifierBackend:
    """适配用户自有分类模型；推理入口约定为 ``module:function``。"""

    def __init__(self, callable_path: str, project_root: Path | None = None):
        if ":" not in callable_path:
            raise ValueError("classifier_callable must use module:function syntax")
        if project_root and str(project_root.resolve()) not in sys.path:
            sys.path.insert(0, str(project_root.resolve()))
        module_name, function_name = callable_path.split(":", 1)
        self._function = getattr(importlib.import_module(module_name), function_name)
        self.model_name = callable_path

    def predict(self, image_path: Path, roi_path: Path) -> dict[str, Any]:
        # 自定义函数必须返回 ClassificationOutput 所需字段；Tool 层随后补充复核标记。
        result = self._function(image_path=str(image_path), roi_path=str(roi_path))
        if not isinstance(result, dict):
            raise ToolExecutionError("Classifier callable must return a dictionary.")
        return result


class ROICaseClassifierBackend:
    """加载真实 ROI 辅助病例级 checkpoint，并应用冻结的临床安全策略。"""

    model_name = "roi-assisted-swinumamba-case-classifier"

    def __init__(
        self,
        checkpoint: Path,
        project_root: Path,
        segmentation_project: Path,
        segmentation_checkpoint: Path,
        clinical_policy: Path,
        device: str = "auto",
    ):
        self.checkpoint = checkpoint.resolve()
        self.project_root = project_root.resolve()
        self.segmentation_project = segmentation_project.resolve()
        self.segmentation_checkpoint = segmentation_checkpoint.resolve()
        self.clinical_policy = clinical_policy.resolve()
        self.device_name = device
        self._model = None
        self._device = None
        self._config: dict[str, Any] = {}
        self._policy: dict[str, Any] = {}
        self.checkpoint_hash: str | None = None
        self.segmentation_checkpoint_hash: str | None = None
        self.policy_hash: str | None = None

    @staticmethod
    def _load_model_class(project_root: Path):
        model_file = project_root / "model.py"
        if not model_file.is_file():
            raise ModelUnavailableError(f"Classifier model.py not found: {model_file}")
        spec = importlib.util.spec_from_file_location("spinevision_roi_case_model", model_file)
        if spec is None or spec.loader is None:
            raise ModelUnavailableError(f"Unable to import classifier model: {model_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.ROIAssistedCaseClassifier

    def _load(self) -> None:
        if self._model is not None:
            return
        for name, path in (
            ("classification checkpoint", self.checkpoint),
            ("segmentation checkpoint", self.segmentation_checkpoint),
            ("clinical policy", self.clinical_policy),
        ):
            if not path.is_file():
                raise ModelUnavailableError(f"Missing {name}: {path}")
        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailableError("PyTorch is required for classification.") from exc

        self.checkpoint_hash = file_sha256(self.checkpoint)
        self.segmentation_checkpoint_hash = file_sha256(self.segmentation_checkpoint)
        self.policy_hash = file_sha256(self.clinical_policy)
        self._policy = json.loads(self.clinical_policy.read_text(encoding="utf-8"))
        if self._policy.get("model_sha256") != self.checkpoint_hash:
            raise ModelUnavailableError("Classification checkpoint hash does not match clinical policy")
        expected_segmentation_hash = self._policy.get("segmentation_model_sha256")
        if expected_segmentation_hash and expected_segmentation_hash != self.segmentation_checkpoint_hash:
            raise ModelUnavailableError("Segmentation checkpoint hash does not match clinical policy")

        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self._config = dict(checkpoint["config"])
        # 训练 checkpoint 保存的是服务器绝对路径，部署时必须显式替换为当前受控路径。
        self._config["segmentation_project"] = str(self.segmentation_project)
        self._config["segmentation_checkpoint"] = str(self.segmentation_checkpoint)
        if self.device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = self.device_name
        self._device = torch.device(device_name)
        model_class = self._load_model_class(self.project_root)
        self._model = model_class(
            self._config["segmentation_project"],
            self._config["segmentation_checkpoint"],
            num_segmentation_classes=int(self._config["num_segmentation_classes"]),
            hidden_dim=int(self._config["hidden_dim"]),
            dropout=float(self._config["dropout"]),
            roi_dilation=int(self._config["roi_dilation"]),
        )
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.to(self._device).eval()

    @staticmethod
    def _central_indices(depth: int, count: int) -> list[int]:
        center = depth // 2
        start = center - count // 2
        return [min(max(start + offset, 0), depth - 1) for offset in range(count)]

    @staticmethod
    def _calibrate(probability: float, temperature: float) -> float:
        value = float(np.clip(probability, 1e-7, 1.0 - 1e-7))
        logit = np.log(value / (1.0 - value)) / max(float(temperature), 1e-6)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))

    def predict(self, image_path: Path, roi_path: Path) -> dict[str, Any]:
        try:
            import nibabel as nib
            import torch
        except ImportError as exc:
            raise ModelUnavailableError("nibabel and PyTorch are required for classification.") from exc
        self._load()
        image = nib.load(str(image_path), mmap=True)
        roi = nib.load(str(roi_path), mmap=True)
        if len(image.shape) != 3 or image.shape != roi.shape:
            raise InputValidationError(
                f"MRI/ROI geometry mismatch: image={image.shape}, roi={roi.shape}"
            )
        if not np.allclose(image.affine, roi.affine, atol=1e-4):
            raise InputValidationError("MRI/ROI affine mismatch")

        image_size = int(self._config["image_size"])
        indices = self._central_indices(image.shape[-1], int(self._config["num_slices"]))
        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for index in indices:
            plane = np.asarray(image.dataobj[:, :, index], dtype=np.float32)
            mask = np.asarray(roi.dataobj[:, :, index], dtype=np.int64)
            image_plane, _ = center_pad_or_crop(plane, image_size)
            mask_plane, _ = center_pad_or_crop(mask, image_size)
            images.append(normalize_nonzero(image_plane))
            masks.append(np.rint(mask_plane).astype(np.int64))
        image_array = np.stack(images).astype(np.float32)
        mask_array = np.stack(masks).astype(np.int64)

        labels = sorted(int(value) for value in np.unique(mask_array) if value > 0)
        foreground_fraction = float(np.mean(mask_array > 0))
        qc = self._policy["input_qc"]
        roi_errors: list[str] = []
        if foreground_fraction < float(qc["minimum_roi_foreground_fraction"]):
            roi_errors.append("segmentation_foreground_too_small")
        if foreground_fraction > float(qc["maximum_roi_foreground_fraction"]):
            roi_errors.append("segmentation_foreground_too_large")
        if len(labels) < int(qc["minimum_segmentation_labels"]):
            roi_errors.append("insufficient_anatomical_labels")
        if mask_array.max(initial=0) >= int(self._config["num_segmentation_classes"]):
            roi_errors.append("segmentation_label_out_of_range")
        if roi_errors:
            raise InputValidationError(f"ROI QC failed: {', '.join(roi_errors)}")

        image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(2).to(self._device)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).to(self._device)
        amp_enabled = self._device.type == "cuda" and self._config.get("amp_dtype") == "bfloat16"
        with torch.inference_mode(), torch.autocast(
            device_type=self._device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            outputs = self._model(image_tensor, mask_tensor)
            raw_probability = float(torch.softmax(outputs["logits"].float(), dim=1)[0, 1].cpu())

        calibrated = self._calibrate(
            raw_probability, float(self._policy["calibration"]["temperature"])
        )
        threshold = float(self._policy["single_threshold"])
        triage = self._policy["triage_thresholds"]
        if calibrated < float(triage["rule_out_below"]):
            band = "RULE_OUT_BAND"
        elif calibrated >= float(triage["rule_in_at_or_above"]):
            band = "RULE_IN_BAND"
        else:
            band = "INDETERMINATE_BAND"
        authorized = bool(self._policy.get("clinical_use_authorized", False))
        if not authorized:
            safety_disposition = "RESEARCH_ONLY_MANDATORY_REVIEW"
        elif band == "INDETERMINATE_BAND":
            safety_disposition = "INDETERMINATE_MANDATORY_REVIEW"
        else:
            safety_disposition = f"AI_{band}_RADIOLOGIST_CONFIRMATION_REQUIRED"
        predicted_class = (
            "spondylolisthesis_candidate" if calibrated >= threshold else "no_spondylolisthesis_candidate"
        )
        confidence = max(calibrated, 1.0 - calibrated)
        warnings = []
        if not authorized:
            warnings.append("CLINICAL_POLICY_NOT_AUTHORIZED")
        if band == "INDETERMINATE_BAND":
            warnings.append("INDETERMINATE_CLASSIFICATION")
        return {
            "predicted_class": predicted_class,
            "probabilities": {
                "no_spondylolisthesis_candidate": 1.0 - calibrated,
                "spondylolisthesis_candidate": calibrated,
            },
            "confidence": confidence,
            "uncertainty": 1.0 - confidence,
            "model_name": self.model_name,
            "requires_review": (not authorized) or band == "INDETERMINATE_BAND",
            "raw_probability": raw_probability,
            "calibrated_probability": calibrated,
            "decision_threshold": threshold,
            "candidate_band": band,
            "safety_disposition": safety_disposition,
            "clinical_use_authorized": authorized,
            "roi_qc_passed": True,
            "roi_foreground_fraction": foreground_fraction,
            "slice_indices": indices,
            "slice_attention": outputs["attention"][0].float().cpu().tolist(),
            "roi_gate": outputs["mean_gate"][0].float().cpu().tolist(),
            "model_checkpoint_hash": self.checkpoint_hash,
            "segmentation_checkpoint_hash": self.segmentation_checkpoint_hash,
            "clinical_policy_hash": self.policy_hash,
            "_warnings": warnings,
        }


class RunDiagnosisTool(AgentTool):
    """校验 MRI/ROI 后调用配置的分类后端，并应用低置信度复核阈值。"""

    name = "run_spine_diagnosis"
    description = "Run the configured dual-branch classification backend on MRI and segmentation ROI."
    input_model = ClassificationInput
    deterministic = False

    def __init__(self, backend: ClassifierBackend, confidence_threshold: float):
        self.backend = backend
        self.confidence_threshold = confidence_threshold

    def run(self, payload: ClassificationInput):
        # TODO(VLM): 可在专业分类模型之后增加 VLM 第二读片结果，但不要覆盖原始概率。
        if not payload.image_path.is_file():
            raise ModelUnavailableError(f"Image not found: {payload.image_path}")
        if not payload.roi_path.is_file():
            raise ModelUnavailableError(f"ROI not found: {payload.roi_path}")
        data = self.backend.predict(payload.image_path.resolve(), payload.roi_path.resolve())
        confidence = float(data.get("confidence", 0.0))
        # 阈值判断放在模型外，便于独立校准、审计和调整临床工作点。
        backend_warnings = list(data.pop("_warnings", []))
        data["requires_review"] = bool(data.get("requires_review", False)) or confidence < self.confidence_threshold
        warnings = backend_warnings
        if confidence < self.confidence_threshold:
            warnings.append("LOW_CLASSIFICATION_CONFIDENCE")
        provenance = {"model_version": self.backend.model_name}
        checkpoint_hash = getattr(self.backend, "checkpoint_hash", None)
        if checkpoint_hash:
            provenance["checkpoint_hash"] = checkpoint_hash
        return data, sorted(set(warnings)), {
            "confidence": confidence,
            "calibrated_probability": float(data.get("calibrated_probability", 0.0)),
        }, provenance
