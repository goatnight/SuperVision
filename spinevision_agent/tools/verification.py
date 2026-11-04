"""报告一致性验证 Tool。

当前 Verifier 是可审计的硬规则/正则引擎，不是大模型。后续可增加独立 Qwen
语义审查，但硬规则应始终保留并拥有更高优先级，避免 LLM 覆盖数值或引用错误。
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from ..schemas import DiagnosisOutput, ReportOutput, SegmentationOutput, VerificationIssue, VerificationOutput
from .base import AgentTool


class VerificationInput(BaseModel):
    segmentation: SegmentationOutput
    diagnosis: DiagnosisOutput
    report: ReportOutput
    confidence_threshold: float


class VerifyCaseTool(AgentTool):
    """检查分割 QC、诊断置信度、报告数值、节段和证据引用。"""

    name = "verify_case_consistency"
    description = "Check segmentation QC, diagnosis confidence, report numbers, levels and evidence IDs."
    input_model = VerificationInput
    deterministic = True
    requires_human_review = True

    def run(self, payload: VerificationInput):
        # issues 是后续 Reflexion 的机器可读反馈，可据 code 路由回 RAG/Report/Diagnosis。
        issues: list[VerificationIssue] = []
        if not payload.segmentation.qc_passed:
            issues.append(
                VerificationIssue(
                    code="SEGMENTATION_QC_FAILED",
                    severity="critical",
                    message="分割质量控制未通过。",
                )
            )
        if not payload.segmentation.label_semantics_validated:
            issues.append(
                VerificationIssue(
                    code="LABEL_SEMANTICS_UNVALIDATED",
                    severity="high",
                    message="标签语义未验证，禁止生成具体腰椎节段结论。",
                )
            )
        if payload.diagnosis.confidence < payload.confidence_threshold:
            issues.append(
                VerificationIssue(
                    code="LOW_DIAGNOSIS_CONFIDENCE",
                    severity="high",
                    message=(
                        f"诊断置信度 {payload.diagnosis.confidence:.3f} 低于阈值 "
                        f"{payload.confidence_threshold:.3f}。"
                    ),
                )
            )
        if not payload.diagnosis.roi_qc_passed:
            issues.append(
                VerificationIssue(
                    code="CLASSIFICATION_ROI_QC_FAILED",
                    severity="critical",
                    message="分类模型使用的 ROI 质量控制未通过。",
                )
            )
        if not payload.diagnosis.clinical_use_authorized:
            issues.append(
                VerificationIssue(
                    code="CLINICAL_POLICY_NOT_AUTHORIZED",
                    severity="critical",
                    message="当前模型策略未获临床使用授权，输出仅限研究复核。",
                )
            )
        if payload.diagnosis.candidate_band == "INDETERMINATE_BAND":
            issues.append(
                VerificationIssue(
                    code="INDETERMINATE_CLASSIFICATION",
                    severity="high",
                    message="分类概率位于双阈值拒识区间，必须人工判读。",
                )
            )
        evidence_ids = {item.evidence_id for item in payload.report.evidence}
        for finding in payload.report.findings:
            missing = [item for item in finding.evidence_ids if item not in evidence_ids]
            if missing:
                issues.append(
                    VerificationIssue(
                        code="MISSING_EVIDENCE_REFERENCE",
                        severity="high",
                        message=f"Finding {finding.finding_id} 引用了不存在的证据。",
                        related_ids=missing,
                    )
                )
        report_text = " ".join(
            [item.observation for item in payload.report.findings] + payload.report.impression
        )
        if payload.diagnosis.predicted_class not in report_text:
            issues.append(
                VerificationIssue(
                    code="DIAGNOSIS_CLASS_MISMATCH",
                    severity="critical",
                    message="报告中的病变候选与分类工具输出不一致。",
                )
            )
        voxel_matches = re.findall(r"前景体素数为\s*(\d+)", report_text)
        if voxel_matches and any(int(value) != payload.segmentation.foreground_voxels for value in voxel_matches):
            issues.append(
                VerificationIssue(
                    code="NUMERIC_MISMATCH",
                    severity="critical",
                    message="报告中的前景体素数与分割工具输出不一致。",
                )
            )
        inferred_findings = [item for item in payload.report.findings if item.status == "inferred"]
        if inferred_findings and not payload.report.evidence:
            issues.append(
                VerificationIssue(
                    code="INFERRED_FINDING_WITHOUT_EVIDENCE",
                    severity="high",
                    message="报告包含模型推断，但没有可追溯的检索证据。",
                )
            )
        if not payload.segmentation.label_semantics_validated and re.search(r"\bL[1-5](?:[-/ ]S?\d)?\b", report_text):
            issues.append(
                VerificationIssue(
                    code="UNSUPPORTED_ANATOMICAL_LEVEL",
                    severity="critical",
                    message="标签语义未验证，但报告包含具体节段。",
                )
            )
        # TODO(LLM): 在硬规则之后追加 semantic_issues，例如结论是否被证据真正支持。
        high_risk = any(item.severity in {"high", "critical"} for item in issues)
        output = VerificationOutput(passed=not issues, issues=issues, human_review_required=high_risk)
        warnings = [item.code for item in issues]
        return output.model_dump(mode="json"), warnings, {"issue_count": float(len(issues))}, {}
