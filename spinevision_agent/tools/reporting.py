"""结构化报告生成 Tool。

当前实现是确定性 Python 模板，没有调用 LLM。模板先保证字段、证据 ID 和安全
限制可测试；接入 Qwen 后应保留 ReportInput/ReportOutput，并使用结构化输出，
禁止模型返回未经 Schema 校验的自由文本。
"""

from __future__ import annotations

from pydantic import BaseModel

from ..schemas import DiagnosisOutput, Evidence, Finding, ReportOutput, SegmentationOutput
from .base import AgentTool


class ReportInput(BaseModel):
    modality: str
    segmentation: SegmentationOutput
    diagnosis: DiagnosisOutput
    evidence: list[Evidence]


class GenerateReportTool(AgentTool):
    """根据模型输出和检索证据生成研究用报告草稿。"""

    name = "generate_grounded_report"
    description = "Generate a structured draft using only model outputs and retrieved evidence."
    input_model = ReportInput
    deterministic = True

    def run(self, payload: ReportInput):
        # TODO(LLM): 在此调用 Qwen Report Agent；Prompt 只允许引用 payload 中的事实与证据。
        segmentation = payload.segmentation
        diagnosis = payload.diagnosis
        evidence_ids = [item.evidence_id for item in payload.evidence]
        findings = [
            Finding(
                finding_id="F-SEG-001",
                observation=(
                    f"分割工具检出 {len(segmentation.labels_present)} 个非背景标签，"
                    f"前景体素数为 {segmentation.foreground_voxels}。"
                ),
                status="observed",
                confidence=1.0 if segmentation.qc_passed else 0.5,
                source_tool="run_spine_segmentation",
                evidence_ids=evidence_ids[:1],
            ),
            Finding(
                finding_id="F-CLS-001",
                observation=f"分类模型输出候选类别：{diagnosis.predicted_class}。",
                status="inferred",
                confidence=diagnosis.confidence,
                source_tool="run_spine_diagnosis",
                evidence_ids=evidence_ids,
            ),
        ]
        # 安全限制由代码强制追加，不能完全依赖 LLM 自觉生成。
        limitations = ["本输出为研究用途的结构化报告草稿，须由医生复核。"]
        if not segmentation.label_semantics_validated:
            limitations.append("分割标签的正式解剖语义尚未验证，未生成具体节段结论。")
        if diagnosis.requires_review:
            limitations.append("分类置信度低于自动完成阈值。")
        if not diagnosis.clinical_use_authorized:
            limitations.append("当前分类模型策略未通过临床准入，任何结果均须医生复核。")
        if diagnosis.candidate_band == "INDETERMINATE_BAND":
            limitations.append("分类概率位于双阈值拒识区间，未形成自动分层结论。")
        if not payload.evidence:
            limitations.append("未检索到可用证据。")
        report = ReportOutput(
            examination=f"腰椎 {payload.modality}",
            findings=findings,
            impression=[f"模型提示 {diagnosis.predicted_class}，仅作为待复核候选。"],
            evidence=payload.evidence,
            limitations=limitations,
        )
        return report.model_dump(mode="json"), [], {}, {}
