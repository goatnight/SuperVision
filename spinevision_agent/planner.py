"""病例执行计划。

当前实现是安全可控的固定计划，并没有调用 LLM，因此还不是真正的 ReAct。
后续接入 Qwen 时，建议保留 ``ReActDecision`` 作为结构化输出契约，仅替换
``build_plan`` 的决策来源，并把可选 action 严格限制在 ToolRegistry 中。
"""

from __future__ import annotations

from .schemas import CaseRecord, ReActDecision


class ReActPlanner:
    """受限 Planner；当前返回固定步骤，用于先验证整条工作流。"""

    def build_plan(self, case: CaseRecord, available_tools: set[str]) -> list[ReActDecision]:
        # TODO(LLM): 调用 Qwen 的 structured output / tool calling 动态生成此列表。
        # 即使接入 LLM，也必须继续执行末尾的工具白名单校验，禁止模型调用任意代码。
        steps = [
            ReActDecision(
                rationale="先确认输入格式和模态满足已验证的模型边界。",
                action="inspect_mri_input",
                arguments={"image_path": str(case.image_path), "modality": case.modality},
                expected_observation="获得 NIfTI 几何和非身份化文件元数据。",
            ),
            ReActDecision(
                rationale="分割 ROI 是后续病变分类和证据组织的前置条件。",
                action="run_spine_segmentation",
                arguments={"image_path": str(case.image_path), "case_id": case.case_id},
                expected_observation="获得保持原始几何的 mask 和分割 QC。",
            ),
            ReActDecision(
                rationale="利用原始 MRI 与分割 ROI 调用自主训练的分类模型。",
                action="run_spine_diagnosis",
                arguments={"image_path": str(case.image_path), "roi_path": "<from_segmentation>", "case_id": case.case_id},
                expected_observation="获得类别概率、置信度和不确定性。",
            ),
            ReActDecision(
                rationale="根据模型候选和安全限制检索版本化医学证据。",
                action="retrieve_medical_knowledge",
                arguments={"query": "<from_diagnosis>", "modality": case.modality},
                expected_observation="获得带 evidence_id 的证据片段。",
            ),
            ReActDecision(
                rationale="只使用结构化工具结果和检索证据生成报告草稿。",
                action="generate_grounded_report",
                arguments={},
                expected_observation="获得可追溯的结构化报告草稿。",
            ),
            ReActDecision(
                rationale="在输出前校验节段、数值、置信度和证据引用。",
                action="verify_case_consistency",
                arguments={},
                expected_observation="获得一致性问题和人工复核判定。",
            ),
        ]
        # Planner 只能选择注册过的医疗工具；这是比 Prompt 更可靠的执行边界。
        unknown = [item.action for item in steps if item.action not in available_tools]
        if unknown:
            raise ValueError(f"Planner requested unregistered tools: {unknown}")
        return steps
