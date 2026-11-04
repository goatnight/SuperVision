"""Agent 服务运行配置；这里只保存路径和开关，不负责加载模型。"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]


class AgentSettings(BaseModel):
    """集中管理可部署配置，避免模型路径和阈值散落在各 Agent 中。"""

    project_root: Path = ROOT
    artifact_root: Path = ROOT / "agent_artifacts"
    segmentation_project: Path = ROOT / "train_test"
    segmentation_checkpoint: Path = ROOT / "runslite_256" / "best.pt"
    segmentation_device: str = "auto"
    classifier_backend: str = "mock"
    classifier_project_root: Path = ROOT / "remote_training" / "spine_roi_case_cls"
    classifier_checkpoint: Path = (
        ROOT / "remote_training" / "spine_roi_case_cls" / "runs" / "best.pt"
    )
    classifier_device: str = "auto"
    clinical_policy: Path = (
        ROOT / "remote_training" / "spine_roi_case_cls" / "runs" / "clinical_policy.json"
    )
    # TODO(LLM): 后续可增加 llm_base_url、llm_model、embedding_model 等字段。
    classifier_callable: str | None = None
    classifier_root: Path | None = None
    knowledge_db: Path = ROOT / "agent_artifacts" / "knowledge.sqlite3"
    confidence_threshold: float = Field(default=0.70, ge=0, le=1)
    max_retries: int = Field(default=1, ge=0, le=3)
    max_steps: int = Field(default=12, ge=6, le=30)
    rag_top_k: int = Field(default=4, ge=1, le=20)
    allow_unvalidated_label_semantics: bool = False

    @classmethod
    def from_env(cls) -> "AgentSettings":
        values: dict[str, object] = {}
        mapping = {
            "SPINEVISION_ARTIFACT_ROOT": ("artifact_root", Path),
            "SPINEVISION_SEGMENTATION_PROJECT": ("segmentation_project", Path),
            "SPINEVISION_SEGMENTATION_CHECKPOINT": ("segmentation_checkpoint", Path),
            "SPINEVISION_SEGMENTATION_DEVICE": ("segmentation_device", str),
            "SPINEVISION_CLASSIFIER_BACKEND": ("classifier_backend", str),
            "SPINEVISION_CLASSIFIER_PROJECT_ROOT": ("classifier_project_root", Path),
            "SPINEVISION_CLASSIFIER_CHECKPOINT": ("classifier_checkpoint", Path),
            "SPINEVISION_CLASSIFIER_DEVICE": ("classifier_device", str),
            "SPINEVISION_CLINICAL_POLICY": ("clinical_policy", Path),
            "SPINEVISION_CLASSIFIER_CALLABLE": ("classifier_callable", str),
            "SPINEVISION_CLASSIFIER_ROOT": ("classifier_root", Path),
            "SPINEVISION_KNOWLEDGE_DB": ("knowledge_db", Path),
        }
        for env_name, (field_name, cast) in mapping.items():
            value = os.getenv(env_name)
            if value:
                values[field_name] = cast(value)
        if os.getenv("SPINEVISION_CONFIDENCE_THRESHOLD"):
            values["confidence_threshold"] = float(os.environ["SPINEVISION_CONFIDENCE_THRESHOLD"])
        return cls(**values)
