"""SpineVision 的 LangGraph 主工作流。

需要特别注意：当前 Planner、RAG、Report、Verifier 等名称表示图中的职责节点，
并不意味着每个节点背后都有一个 LLM。Segmentation 与 Diagnosis 默认加载真实
神经网络，其余节点目前主要由确定性 Python 逻辑实现。
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

from .config import AgentSettings
from .planner import ReActPlanner
from .rag import SQLiteVectorStore
from .schemas import (
    CaseCreateRequest,
    CaseRecord,
    DiagnosisOutput,
    Evidence,
    GraphState,
    InputQCOutput,
    ReportOutput,
    RunStatus,
    SegmentationOutput,
    ToolStatus,
    ToolTrace,
    VerificationOutput,
    utc_now,
)
from .store import InMemoryCaseStore
from .tools.classification import (
    CallableClassifierBackend,
    MockClassifierBackend,
    ROICaseClassifierBackend,
    RunDiagnosisTool,
)
from .tools.imaging import InspectImageTool
from .tools.registry import ToolRegistry
from .tools.reporting import GenerateReportTool
from .tools.retrieval import RetrieveKnowledgeTool
from .tools.segmentation import RunSegmentationTool, SegmentationService
from .tools.verification import VerifyCaseTool


class SpineVisionWorkflow:
    """组装 Tool、维护病例状态，并通过 LangGraph 驱动完整病例流程。"""

    def __init__(
        self,
        settings: AgentSettings | None = None,
        *,
        store: InMemoryCaseStore | None = None,
        registry: ToolRegistry | None = None,
    ):
        self.settings = settings or AgentSettings.from_env()
        self.store = store or InMemoryCaseStore()
        # registry 参数允许测试注入 Mock Tool，也方便以后替换成远程模型服务。
        self.registry = registry or self._build_registry()
        self.planner = ReActPlanner()
        self._compiled = None

    def _build_registry(self) -> ToolRegistry:
        """构建生产默认工具集；这里是替换模型后端最集中的入口。"""

        registry = ToolRegistry()
        # 1) 输入检查：使用冻结策略中的训练域统计执行 fail-closed QC。
        registry.register(InspectImageTool(self.settings.clinical_policy))
        # 2) 分割：当前真实加载本地 SwinUMambaLite checkpoint。
        registry.register(
            RunSegmentationTool(
                SegmentationService(
                    checkpoint=self.settings.segmentation_checkpoint,
                    artifact_root=self.settings.artifact_root,
                    device=self.settings.segmentation_device,
                    project_root=self.settings.segmentation_project,
                )
            )
        )
        # 3) 诊断：生产默认加载 ROI 辅助病例级真实分类 checkpoint。
        if self.settings.classifier_backend == "roi_case":
            classifier = ROICaseClassifierBackend(
                checkpoint=self.settings.classifier_checkpoint,
                project_root=self.settings.classifier_project_root,
                segmentation_project=self.settings.segmentation_project,
                segmentation_checkpoint=self.settings.segmentation_checkpoint,
                clinical_policy=self.settings.clinical_policy,
                device=self.settings.classifier_device,
            )
        elif self.settings.classifier_backend == "callable":
            if not self.settings.classifier_callable:
                raise ValueError("SPINEVISION_CLASSIFIER_CALLABLE is required for callable backend")
            classifier = CallableClassifierBackend(
                self.settings.classifier_callable,
                self.settings.classifier_root,
            )
        elif self.settings.classifier_backend == "mock":
            classifier = MockClassifierBackend()
        else:
            raise ValueError(f"Unsupported classifier backend: {self.settings.classifier_backend}")
        registry.register(RunDiagnosisTool(classifier, self.settings.confidence_threshold))
        # 4) RAG：当前为 SQLite + HashingEmbedder，不是神经网络向量模型。
        vector_store = SQLiteVectorStore(self.settings.knowledge_db)
        seed = Path(__file__).resolve().parent / "knowledge" / "seed_documents.json"
        registry.register(RetrieveKnowledgeTool(vector_store, seed))
        # 5/6) 当前分别是报告模板和硬规则验证器；未来可在 Tool 内接 Qwen。
        registry.register(GenerateReportTool())
        registry.register(VerifyCaseTool())
        return registry

    @staticmethod
    def _patient_hash(value: str | None, image_path: Path) -> str:
        source = value or str(image_path.resolve())
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    def create_case(self, request: CaseCreateRequest) -> CaseRecord:
        # patient_hash 用于演示状态关联，不代表完整的医学数据去标识化方案。
        case = CaseRecord(
            case_id=f"case-{uuid.uuid4().hex[:12]}",
            patient_hash=self._patient_hash(request.patient_hash, request.image_path),
            image_path=request.image_path.resolve(),
            modality=request.modality,
            user_goal=request.user_goal,
        )
        return self.store.create(case)

    def _invoke(self, case: CaseRecord, agent: str, tool_name: str, arguments: dict[str, Any]):
        """统一调用 Tool，并把每次动作写入可观测 trace。"""

        result = self.registry.invoke(tool_name, arguments)
        case.tool_trace.append(
            ToolTrace(
                step_id=f"step-{len(case.tool_trace) + 1:03d}",
                agent=agent,
                tool_name=tool_name,
                arguments=arguments,
                result_status=result.status,
                started_at=result.provenance.started_at,
                finished_at=result.provenance.finished_at,
                error_code=result.errors[0].code if result.errors else None,
            )
        )
        case.errors.extend(result.errors)
        return result

    def _fail_or_retry(self, case: CaseRecord, agent: str, result) -> str:
        """只重试声明为 recoverable 的错误，超过次数后交给人工复核。"""

        recoverable = any(error.recoverable for error in result.errors)
        attempts = case.retry_count.get(agent, 0)
        if recoverable and attempts < self.settings.max_retries:
            case.retry_count[agent] = attempts + 1
            return agent
        case.status = RunStatus.failed
        case.human_review_required = True
        return "finalize"

    def intake_node(self, state: GraphState):
        # 当前 build_plan 返回固定计划；TODO(LLM) 的动态 Planner 替换点在 planner.py。
        case = state["case"]
        case.status = RunStatus.running
        case.current_agent = "Planner"
        case.plan = self.planner.build_plan(case, set(item["name"] for item in self.registry.describe()))
        result = self._invoke(
            case,
            "Planner",
            "inspect_mri_input",
            {"image_path": str(case.image_path), "modality": case.modality},
        )
        if result.status in {ToolStatus.success, ToolStatus.partial}:
            case.input_qc = InputQCOutput.model_validate(result.data)
            if case.input_qc.qc_passed:
                next_agent = "segmentation"
            else:
                case.status = RunStatus.failed
                case.human_review_required = True
                next_agent = "finalize"
        else:
            next_agent = self._fail_or_retry(case, "intake", result)
        self.store.save(case)
        return {"case": case, "next_agent": next_agent, "step_count": state.get("step_count", 0) + 1}

    def segmentation_node(self, state: GraphState):
        # 分割 QC 失败时跳过诊断，避免错误 ROI 继续向后传播。
        case = state["case"]
        case.current_agent = "Segmentation"
        result = self._invoke(
            case,
            "Segmentation",
            "run_spine_segmentation",
            {"image_path": str(case.image_path), "case_id": case.case_id},
        )
        if result.status in {ToolStatus.success, ToolStatus.partial}:
            case.segmentation = SegmentationOutput.model_validate(result.data)
            next_agent = "diagnosis" if case.segmentation.qc_passed else "verifier"
        else:
            next_agent = self._fail_or_retry(case, "segmentation", result)
        self.store.save(case)
        return {"case": case, "next_agent": next_agent, "step_count": state["step_count"] + 1}

    def diagnosis_node(self, state: GraphState):
        # Diagnosis Tool 同时接收原图和 mask；分类模型适配器位于 classification.py。
        case = state["case"]
        case.current_agent = "Diagnosis"
        result = self._invoke(
            case,
            "Diagnosis",
            "run_spine_diagnosis",
            {
                "image_path": str(case.image_path),
                "roi_path": str(case.segmentation.mask_path),
                "case_id": case.case_id,
            },
        )
        if result.status in {ToolStatus.success, ToolStatus.partial}:
            case.diagnosis = DiagnosisOutput.model_validate(result.data)
            next_agent = "rag"
        else:
            next_agent = self._fail_or_retry(case, "diagnosis", result)
        self.store.save(case)
        return {"case": case, "next_agent": next_agent, "step_count": state["step_count"] + 1}

    def rag_node(self, state: GraphState):
        case = state["case"]
        case.current_agent = "RAG"
        # TODO(LLM): 当前查询由模板拼接；可替换成 Qwen Query Rewrite，但应保存原查询。
        query = (
            f"腰椎 MRI {case.diagnosis.predicted_class} 结构化报告、证据追溯、"
            "低置信度与标签语义未验证的人工复核规范"
        )
        result = self._invoke(
            case,
            "RAG",
            "retrieve_medical_knowledge",
            {"query": query, "top_k": self.settings.rag_top_k, "modality": case.modality},
        )
        if result.status in {ToolStatus.success, ToolStatus.partial}:
            case.evidence = [Evidence.model_validate(item) for item in result.data.get("evidence", [])]
            next_agent = "report"
        else:
            next_agent = self._fail_or_retry(case, "rag", result)
        self.store.save(case)
        return {"case": case, "next_agent": next_agent, "step_count": state["step_count"] + 1}

    def report_node(self, state: GraphState):
        # Report Agent 只接收结构化模型结果与 Evidence，不直接读取未约束的中间文本。
        case = state["case"]
        case.current_agent = "Report"
        result = self._invoke(
            case,
            "Report",
            "generate_grounded_report",
            {
                "modality": case.modality,
                "segmentation": case.segmentation.model_dump(mode="json"),
                "diagnosis": case.diagnosis.model_dump(mode="json"),
                "evidence": [item.model_dump(mode="json") for item in case.evidence],
            },
        )
        if result.status in {ToolStatus.success, ToolStatus.partial}:
            case.report = ReportOutput.model_validate(result.data)
            case.findings = case.report.findings
            next_agent = "verifier"
        else:
            next_agent = self._fail_or_retry(case, "report", result)
        self.store.save(case)
        return {"case": case, "next_agent": next_agent, "step_count": state["step_count"] + 1}

    def verifier_node(self, state: GraphState):
        # Verifier 是输出闸门：上游必要结果缺失时直接 fail closed 并要求人工复核。
        case = state["case"]
        case.current_agent = "Verifier"
        if not (case.segmentation and case.diagnosis and case.report):
            case.status = RunStatus.failed
            case.human_review_required = True
            self.store.save(case)
            return {"case": case, "next_agent": "finalize", "step_count": state["step_count"] + 1}
        result = self._invoke(
            case,
            "Verifier",
            "verify_case_consistency",
            {
                "segmentation": case.segmentation.model_dump(mode="json"),
                "diagnosis": case.diagnosis.model_dump(mode="json"),
                "report": case.report.model_dump(mode="json"),
                "confidence_threshold": self.settings.confidence_threshold,
            },
        )
        if result.status in {ToolStatus.success, ToolStatus.partial}:
            case.verification = VerificationOutput.model_validate(result.data)
            case.human_review_required = case.verification.human_review_required
        else:
            case.human_review_required = True
        self.store.save(case)
        return {"case": case, "next_agent": "finalize", "step_count": state["step_count"] + 1}

    def finalize_node(self, state: GraphState):
        # 自动流程只生成 completed/review 状态；医生最终决策由 API review 接口写入。
        case = state["case"]
        case.current_agent = "HumanReview" if case.human_review_required else "Complete"
        if case.status != RunStatus.failed:
            case.status = RunStatus.review if case.human_review_required else RunStatus.completed
        self.store.save(case)
        return {"case": case, "next_agent": "end", "step_count": state.get("step_count", 0) + 1}

    def build_graph(self):
        """声明节点和条件边；业务判断在节点中通过 next_agent 返回。"""

        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError("LangGraph is not installed. Install requirements-agent.txt.") from exc
        graph = StateGraph(GraphState)
        graph.add_node("intake", self.intake_node)
        graph.add_node("segmentation", self.segmentation_node)
        graph.add_node("diagnosis", self.diagnosis_node)
        graph.add_node("rag", self.rag_node)
        graph.add_node("report", self.report_node)
        graph.add_node("verifier", self.verifier_node)
        graph.add_node("finalize", self.finalize_node)
        graph.add_edge(START, "intake")
        # 每个节点都允许在 recoverable 错误时回到自身重试，也可路由到后续节点。
        for node in ["intake", "segmentation", "diagnosis", "rag", "report", "verifier"]:
            graph.add_conditional_edges(
                node,
                lambda state: state["next_agent"],
                {
                    "intake": "intake",
                    "segmentation": "segmentation",
                    "diagnosis": "diagnosis",
                    "rag": "rag",
                    "report": "report",
                    "verifier": "verifier",
                    "finalize": "finalize",
                },
            )
        graph.add_edge("finalize", END)
        self._compiled = graph.compile()
        return self._compiled

    def run_case(self, case_id: str) -> CaseRecord:
        """从病例仓库读取状态，同步执行图，并保存最终结果。"""

        case = self.store.get(case_id)
        graph = self._compiled or self.build_graph()
        result = graph.invoke({"case": case, "next_agent": "intake", "step_count": 0})
        return self.store.save(result["case"])
