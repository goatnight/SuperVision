"""Agent 工程 Benchmark。

除显式标记 ``real_segmentation`` 的耗时测试外，多数样本由 Mock/合成数据生成，
用于验证调用次数、错误恢复和数据契约，不能当作临床性能指标。替换 LLM/VLM 后，
应新增固定病例集、人工金标准和模型版本信息，避免不同实验结果不可比较。
"""

from __future__ import annotations

import json
import math
import statistics
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from ..config import AgentSettings
from ..errors import ToolExecutionError
from ..rag import SQLiteVectorStore
from ..schemas import (
    CaseCreateRequest,
    DiagnosisOutput,
    Evidence,
    ReportOutput,
    SegmentationOutput,
    ToolStatus,
)
from ..tools.base import AgentTool
from ..tools.classification import MockClassifierBackend, RunDiagnosisTool
from ..tools.imaging import InspectImageTool
from ..tools.registry import ToolRegistry
from ..tools.reporting import GenerateReportTool
from ..tools.retrieval import RetrieveKnowledgeTool
from ..tools.segmentation import MockSegmentationTool, RunSegmentationTool, SegmentationService
from ..tools.verification import VerifyCaseTool
from ..workflow import SpineVisionWorkflow


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float]:
    if total == 0:
        return {"estimate": 0.0, "low": 0.0, "high": 0.0}
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return {"estimate": p, "low": max(0.0, center - half), "high": min(1.0, center + half)}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def seed_path() -> Path:
    return Path(__file__).resolve().parents[1] / "knowledge" / "seed_documents.json"


def build_registry(settings: AgentSettings, *, real_segmentation: bool = False) -> ToolRegistry:
    """构建隔离的 Benchmark 工具集；分类器始终使用固定 Mock 输出。"""

    registry = ToolRegistry()
    registry.register(InspectImageTool())
    if real_segmentation:
        registry.register(
            RunSegmentationTool(
                SegmentationService(
                    settings.segmentation_checkpoint,
                    settings.artifact_root,
                    settings.segmentation_device,
                )
            )
        )
    else:
        registry.register(MockSegmentationTool())
    registry.register(RunDiagnosisTool(MockClassifierBackend(0.82), settings.confidence_threshold))
    registry.register(RetrieveKnowledgeTool(SQLiteVectorStore(settings.knowledge_db), seed_path()))
    registry.register(GenerateReportTool())
    registry.register(VerifyCaseTool())
    return registry


def make_nifti(path: Path, slices: int = 3) -> None:
    volume = np.zeros((32, 32, slices), dtype=np.float32)
    volume[8:24, 8:24, :] = 1.0
    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)


def run_workflow_latency(root: Path, cases: int, slices: int, real_segmentation: bool) -> dict[str, Any]:
    """测量工程耗时和 Tool 次数，不评估医学准确性。"""

    settings = AgentSettings(
        artifact_root=root / ("real_artifacts" if real_segmentation else "mock_artifacts"),
        knowledge_db=root / ("real_knowledge.sqlite3" if real_segmentation else "mock_knowledge.sqlite3"),
        segmentation_device="cpu",
        confidence_threshold=0.7,
    )
    workflow = SpineVisionWorkflow(settings, registry=build_registry(settings, real_segmentation=real_segmentation))
    durations: list[float] = []
    tool_calls: list[int] = []
    for index in range(cases):
        image_path = root / f"{'real' if real_segmentation else 'mock'}_{index}.nii.gz"
        make_nifti(image_path, slices=slices)
        case = workflow.create_case(CaseCreateRequest(image_path=image_path))
        start = time.perf_counter()
        result = workflow.run_case(case.case_id)
        durations.append(time.perf_counter() - start)
        tool_calls.append(len(result.tool_trace))
    return {
        "cases": cases,
        "synthetic_slices_per_case": slices,
        "segmentation_backend": "real_best_pt_cpu" if real_segmentation else "mock",
        "classification_backend": "mock",
        "mean_seconds": statistics.mean(durations),
        "median_seconds": statistics.median(durations),
        "p95_seconds": percentile(durations, 0.95),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "mean_tool_calls": statistics.mean(tool_calls),
        "tool_calls": tool_calls,
        "durations_seconds": durations,
    }


def run_rag_benchmark(root: Path) -> dict[str, Any]:
    """在带已知相关文档的查询集上计算 RAG 召回指标。"""

    store = SQLiteVectorStore(root / "rag_benchmark.sqlite3")
    RetrieveKnowledgeTool(store, seed_path())
    queries = [
        ("报告中的测量和节段必须来自工具，不能由语言模型补全", {"SOP-GROUNDING-001"}),
        ("低置信度结果应该如何转人工复核", {"SOP-REVIEW-001"}),
        ("mask 标签字典未验证时能否生成节段诊断", {"SOP-MASK-001"}),
        ("结构化报告草稿需要显示哪些限制", {"SOP-REPORT-001"}),
        ("工具结果与报告证据如何保持可追溯", {"SOP-GROUNDING-001", "SOP-REPORT-001"}),
        ("分割质量控制失败后的处理流程", {"SOP-REVIEW-001", "SOP-MASK-001"}),
        ("自动报告能否作为最终诊断", {"SOP-REPORT-001", "SOP-REVIEW-001"}),
        ("影像模型推断缺少证据时怎么处理", {"SOP-GROUNDING-001", "SOP-REVIEW-001"}),
    ]
    output: dict[str, Any] = {"queries": len(queries), "corpus_documents": store.count(), "by_k": {}}
    reciprocal_ranks: list[float] = []
    for query, relevant in queries:
        ranked = store.search(query, top_k=store.count(), metadata_filter={"modality": "MRI"})
        ids = [item.evidence_id for item in ranked]
        ranks = [ids.index(item) + 1 for item in relevant if item in ids]
        reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
    for k in range(1, store.count() + 1):
        recalls = []
        hits = 0
        for query, relevant in queries:
            retrieved = {item.evidence_id for item in store.search(query, top_k=k, metadata_filter={"modality": "MRI"})}
            recalls.append(len(retrieved & relevant) / len(relevant))
            hits += int(bool(retrieved & relevant))
        output["by_k"][str(k)] = {
            "mean_recall": statistics.mean(recalls),
            "hit_rate": hits / len(queries),
        }
    output["mrr"] = statistics.mean(reciprocal_ranks)
    return output


def base_outputs(root: Path):
    segmentation = SegmentationOutput(
        mask_path=root / "mask.nii.gz",
        input_shape=[64, 64, 8],
        output_shape=[64, 64, 8],
        labels_present=[1, 2],
        foreground_voxels=512,
        spacing=[1.0, 1.0, 3.0],
        qc_passed=True,
        label_semantics_validated=True,
    )
    diagnosis = DiagnosisOutput(
        predicted_class="spondylolisthesis_candidate",
        probabilities={"negative": 0.18, "spondylolisthesis_candidate": 0.82},
        confidence=0.82,
        uncertainty=0.18,
        model_name="benchmark-classifier",
    )
    evidence = [
        Evidence(
            evidence_id="E-1",
            title="Benchmark evidence",
            section="reporting",
            content="模型推断须保留证据并由医生复核。",
            source="benchmark_fixture",
            score=0.9,
        )
    ]
    report_result = GenerateReportTool().invoke(
        {
            "modality": "MRI",
            "segmentation": segmentation.model_dump(mode="json"),
            "diagnosis": diagnosis.model_dump(mode="json"),
            "evidence": [item.model_dump(mode="json") for item in evidence],
        }
    )
    return segmentation, diagnosis, ReportOutput.model_validate(report_result.data)


def report_is_contract_correct(report: ReportOutput, diagnosis: DiagnosisOutput) -> bool:
    evidence_ids = {item.evidence_id for item in report.evidence}
    references_valid = all(set(item.evidence_ids) <= evidence_ids for item in report.findings)
    statuses_valid = all(item.status in {"observed", "derived", "inferred", "unknown"} for item in report.findings)
    diagnosis_grounded = any(
        item.source_tool == "run_spine_diagnosis"
        and item.status == "inferred"
        and diagnosis.predicted_class in item.observation
        for item in report.findings
    )
    return bool(
        report.examination
        and report.findings
        and report.impression
        and report.limitations
        and references_valid
        and statuses_valid
        and diagnosis_grounded
        and report.status == "draft"
    )


def run_report_benchmark(root: Path, samples: int = 100) -> dict[str, Any]:
    """检查报告是否满足 Schema/引用契约，不代表报告临床正确。"""

    tool = GenerateReportTool()
    correct = 0
    for index in range(samples):
        segmentation, diagnosis, report = base_outputs(root)
        diagnosis.confidence = 0.5 + (index % 50) / 100
        diagnosis.uncertainty = 1 - diagnosis.confidence
        diagnosis.requires_review = diagnosis.confidence < 0.7
        result = tool.invoke(
            {
                "modality": "MRI",
                "segmentation": segmentation.model_dump(mode="json"),
                "diagnosis": diagnosis.model_dump(mode="json"),
                "evidence": [item.model_dump(mode="json") for item in report.evidence],
            }
        )
        if result.status in {ToolStatus.success, ToolStatus.partial}:
            generated = ReportOutput.model_validate(result.data)
            correct += int(report_is_contract_correct(generated, diagnosis))
    return {"samples": samples, "correct": correct, "proportion": wilson(correct, samples)}


def verify(tool: VerifyCaseTool, segmentation, diagnosis, report) -> set[str]:
    result = tool.invoke(
        {
            "segmentation": segmentation.model_dump(mode="json"),
            "diagnosis": diagnosis.model_dump(mode="json"),
            "report": report.model_dump(mode="json"),
            "confidence_threshold": 0.7,
        }
    )
    return {item["code"] for item in result.data["issues"]}


def run_verifier_benchmark(root: Path, repetitions: int = 20) -> dict[str, Any]:
    """向合成报告注入已知错误，测量规则 Verifier 的检出能力。"""

    tool = VerifyCaseTool()
    perturbations = {
        "low_confidence": "LOW_DIAGNOSIS_CONFIDENCE",
        "missing_evidence_reference": "MISSING_EVIDENCE_REFERENCE",
        "unsupported_level": "UNSUPPORTED_ANATOMICAL_LEVEL",
        "numeric_mismatch": "NUMERIC_MISMATCH",
        "diagnosis_mismatch": "DIAGNOSIS_CLASS_MISMATCH",
        "inferred_without_evidence": "INFERRED_FINDING_WITHOUT_EVIDENCE",
    }
    detected = 0
    total = 0
    by_type: dict[str, dict[str, int]] = {}
    for name, expected_code in perturbations.items():
        type_hits = 0
        for _ in range(repetitions):
            segmentation, diagnosis, report = base_outputs(root)
            if name == "low_confidence":
                diagnosis.confidence = 0.4
            elif name == "missing_evidence_reference":
                report.findings[0].evidence_ids = ["DOES-NOT-EXIST"]
            elif name == "unsupported_level":
                segmentation.label_semantics_validated = False
                report.findings[0].observation += " 具体节段为 L4-L5。"
            elif name == "numeric_mismatch":
                report.findings[0].observation = report.findings[0].observation.replace("512", "999")
            elif name == "diagnosis_mismatch":
                for finding in report.findings:
                    finding.observation = finding.observation.replace(diagnosis.predicted_class, "other_disease")
                report.impression = [item.replace(diagnosis.predicted_class, "other_disease") for item in report.impression]
            elif name == "inferred_without_evidence":
                report.evidence = []
                for finding in report.findings:
                    finding.evidence_ids = []
            hit = expected_code in verify(tool, segmentation, diagnosis, report)
            type_hits += int(hit)
            detected += int(hit)
            total += 1
        by_type[name] = {"detected": type_hits, "total": repetitions}

    clean_passed = 0
    clean_total = repetitions
    for _ in range(clean_total):
        segmentation, diagnosis, report = base_outputs(root)
        clean_passed += int(not verify(tool, segmentation, diagnosis, report))
    return {
        "inconsistency_types": len(perturbations),
        "injected_cases": total,
        "detected": detected,
        "sensitivity": wilson(detected, total),
        "clean_cases": clean_total,
        "clean_passed": clean_passed,
        "specificity": wilson(clean_passed, clean_total),
        "by_type": by_type,
    }


class FlakyTool(AgentTool):
    """第一次调用故意失败的包装器，用于验证 recoverable error 重试。"""

    def __init__(self, wrapped: AgentTool, failures: int = 1):
        self.wrapped = wrapped
        self.name = wrapped.name
        self.description = wrapped.description
        self.input_model = wrapped.input_model
        self.version = wrapped.version
        self.deterministic = wrapped.deterministic
        self.requires_human_review = wrapped.requires_human_review
        self.failures = failures
        self.calls = 0

    def run(self, payload):
        self.calls += 1
        if self.calls <= self.failures:
            raise ToolExecutionError("Injected transient failure")
        result = self.wrapped.invoke(payload.model_dump(mode="python"))
        if result.status not in {ToolStatus.success, ToolStatus.partial}:
            raise ToolExecutionError("Wrapped tool failed after recovery")
        return result.data, result.warnings, result.metrics, {
            "model_version": result.provenance.model_version,
            "checkpoint_hash": result.provenance.checkpoint_hash,
            "input_hash": result.provenance.input_hash,
        }


def run_recovery_benchmark(root: Path, repetitions: int = 10) -> dict[str, Any]:
    stages = [
        "inspect_mri_input",
        "run_spine_segmentation",
        "run_spine_diagnosis",
        "retrieve_medical_knowledge",
        "generate_grounded_report",
    ]
    recovered = 0
    total = 0
    by_stage: dict[str, dict[str, int]] = {}
    image_path = root / "recovery_case.nii.gz"
    make_nifti(image_path, slices=1)
    for stage in stages:
        stage_recovered = 0
        for index in range(repetitions):
            run_root = root / f"recovery_{stage}_{index}"
            settings = AgentSettings(
                artifact_root=run_root / "artifacts",
                knowledge_db=run_root / "knowledge.sqlite3",
                max_retries=1,
            )
            base = build_registry(settings, real_segmentation=False)
            registry = ToolRegistry()
            for item in base.describe():
                tool = base.get(item["name"])
                registry.register(FlakyTool(tool) if tool.name == stage else tool)
            workflow = SpineVisionWorkflow(settings, registry=registry)
            case = workflow.create_case(CaseCreateRequest(image_path=image_path))
            result = workflow.run_case(case.case_id)
            success = result.report is not None and len(result.tool_trace) == 7
            stage_recovered += int(success)
            recovered += int(success)
            total += 1
        by_stage[stage] = {"recovered": stage_recovered, "total": repetitions}
    return {"injected_cases": total, "recovered": recovered, "proportion": wilson(recovered, total), "by_stage": by_stage}


def manual_step_analysis() -> dict[str, Any]:
    """基于流程定义统计人工步骤变化，不是实际临床观察数据。"""

    baseline = [
        "导入 MRI 至分割程序",
        "启动分割",
        "检查并导出 ROI",
        "将 ROI/MRI 导入分类程序",
        "启动分类",
        "人工检索知识",
        "撰写报告",
        "核对并确认报告",
    ]
    agent = ["提交 MRI 与任务", "查看自动聚合结果及告警", "接受、修改或驳回报告"]
    return {
        "manual_baseline_steps": len(baseline),
        "agent_assisted_steps": len(agent),
        "step_reduction": (len(baseline) - len(agent)) / len(baseline),
        "baseline_definition": baseline,
        "agent_definition": agent,
        "evidence_level": "workflow_count_not_time_motion_study",
    }


def run_benchmark(output_dir: Path, *, real_cases: int = 5, real_slices: int = 3) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
        root = Path(temp_dir)
        results = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scope": "engineering_benchmark_not_clinical_validation",
            "agents": {
                "automated_agents": 6,
                "names": ["Planner", "Segmentation", "Diagnosis", "RAG", "Report", "Verifier"],
                "human_review_nodes": 1,
            },
            "tools": {
                "registered_tools": 6,
                "names": [
                    "inspect_mri_input",
                    "run_spine_segmentation",
                    "run_spine_diagnosis",
                    "retrieve_medical_knowledge",
                    "generate_grounded_report",
                    "verify_case_consistency",
                ],
            },
            "mock_workflow": run_workflow_latency(root, cases=30, slices=3, real_segmentation=False),
            "real_segmentation_workflow": run_workflow_latency(
                root, cases=real_cases, slices=real_slices, real_segmentation=True
            ),
            "rag": run_rag_benchmark(root),
            "report_contract": run_report_benchmark(root),
            "verifier": run_verifier_benchmark(root),
            "recovery": run_recovery_benchmark(root),
            "human_steps": manual_step_analysis(),
        }
    json_path = output_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
