"""工作流共享数据契约。

这些 Pydantic Schema 不是大模型，只负责把 Agent 与 Tool 之间的输入输出变成
可校验、可序列化、可追踪的数据。接入 LLM/VLM 时应继续复用这些契约。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    partial = "partial"
    review = "review"
    completed = "completed"
    failed = "failed"


class ToolStatus(str, Enum):
    success = "success"
    partial = "partial"
    failed = "failed"
    rejected = "rejected"


class Provenance(BaseModel):
    tool_name: str
    tool_version: str = "0.1.0"
    model_version: str | None = None
    checkpoint_hash: str | None = None
    input_hash: str | None = None
    started_at: datetime
    finished_at: datetime


class ToolError(BaseModel):
    code: str
    message: str
    recoverable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """任意 Tool 的统一结果封装，区分业务数据、告警、错误和运行指标。"""

    status: ToolStatus
    data: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[ToolError] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    provenance: Provenance


class ToolTrace(BaseModel):
    step_id: str
    agent: str
    tool_name: str
    arguments: dict[str, Any]
    result_status: ToolStatus
    started_at: datetime
    finished_at: datetime
    error_code: str | None = None


class Evidence(BaseModel):
    """可追溯知识片段；报告 Finding 通过 evidence_id 引用它。"""

    evidence_id: str
    title: str
    section: str
    content: str
    source: str
    score: float = Field(ge=-1, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Finding(BaseModel):
    finding_id: str
    anatomical_level: str = "unknown"
    observation: str
    status: Literal["observed", "derived", "inferred", "unknown"]
    confidence: float = Field(ge=0, le=1)
    source_tool: str
    evidence_ids: list[str] = Field(default_factory=list)


class InputQCOutput(BaseModel):
    input_hash: str
    format: str
    modality: str
    shape: list[int]
    spacing: list[float]
    orientation: list[str]
    affine_valid: bool
    finite_fraction: float = Field(ge=0, le=1)
    nonzero_fraction: float = Field(ge=0, le=1)
    dynamic_range_p01_p99: float = Field(ge=0)
    qc_passed: bool
    qc_errors: list[str] = Field(default_factory=list)
    qc_warnings: list[str] = Field(default_factory=list)
    clinical_policy_hash: str | None = None


class SegmentationOutput(BaseModel):
    mask_path: Path
    input_shape: list[int]
    output_shape: list[int]
    labels_present: list[int]
    foreground_voxels: int
    spacing: list[float] | None = None
    qc_passed: bool
    qc_warnings: list[str] = Field(default_factory=list)
    label_semantics_validated: bool = False
    foreground_fraction: float = Field(default=0.0, ge=0, le=1)
    model_checkpoint_hash: str | None = None
    input_hash: str | None = None


class DiagnosisOutput(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    model_name: str
    requires_review: bool = False
    raw_probability: float | None = Field(default=None, ge=0, le=1)
    calibrated_probability: float | None = Field(default=None, ge=0, le=1)
    decision_threshold: float | None = Field(default=None, ge=0, le=1)
    candidate_band: str | None = None
    safety_disposition: str | None = None
    clinical_use_authorized: bool = False
    roi_qc_passed: bool = False
    roi_foreground_fraction: float | None = Field(default=None, ge=0, le=1)
    slice_indices: list[int] = Field(default_factory=list)
    slice_attention: list[float] = Field(default_factory=list)
    roi_gate: list[float] = Field(default_factory=list)
    model_checkpoint_hash: str | None = None
    segmentation_checkpoint_hash: str | None = None
    clinical_policy_hash: str | None = None


class VerificationIssue(BaseModel):
    code: str
    severity: Literal["low", "medium", "high", "critical"]
    message: str
    related_ids: list[str] = Field(default_factory=list)


class VerificationOutput(BaseModel):
    passed: bool
    issues: list[VerificationIssue] = Field(default_factory=list)
    human_review_required: bool = False


class ReportOutput(BaseModel):
    examination: str
    findings: list[Finding]
    impression: list[str]
    evidence: list[Evidence]
    limitations: list[str]
    status: Literal["draft", "reviewed"] = "draft"


class ReActDecision(BaseModel):
    """Planner 的单步结构化决策；未来可直接作为 Qwen structured output。"""

    rationale: str
    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_observation: str


class CaseCreateRequest(BaseModel):
    image_path: Path
    modality: Literal["MRI"] = "MRI"
    user_goal: str = "完成腰椎 MRI 分割、病变候选识别及结构化报告草稿"
    patient_hash: str | None = None


class ReviewRequest(BaseModel):
    decision: Literal["accept", "modify", "reject"]
    reviewer: str
    reason: str
    corrected_report: ReportOutput | None = None


class CaseRecord(BaseModel):
    """单病例完整状态，LangGraph 节点围绕该对象逐步补充结果。"""

    case_id: str
    patient_hash: str
    image_path: Path
    modality: str
    user_goal: str
    status: RunStatus = RunStatus.pending
    current_agent: str = "intake"
    plan: list[ReActDecision] = Field(default_factory=list)
    input_qc: InputQCOutput | None = None
    segmentation: SegmentationOutput | None = None
    diagnosis: DiagnosisOutput | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    report: ReportOutput | None = None
    verification: VerificationOutput | None = None
    tool_trace: list[ToolTrace] = Field(default_factory=list)
    errors: list[ToolError] = Field(default_factory=list)
    retry_count: dict[str, int] = Field(default_factory=dict)
    human_review_required: bool = False
    human_feedback: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class GraphState(TypedDict, total=False):
    """LangGraph 在节点之间传递的最小状态。"""

    case: CaseRecord
    next_agent: str
    step_count: int
