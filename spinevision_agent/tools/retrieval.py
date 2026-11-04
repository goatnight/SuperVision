"""医学知识检索 Tool，将向量库结果转换为可追溯 Evidence。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ..rag import SQLiteVectorStore
from .base import AgentTool


class RetrievalInput(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=4, ge=1, le=20)
    modality: str = "MRI"


class RetrieveKnowledgeTool(AgentTool):
    """执行带模态过滤的 Top-k 检索；当前不包含 LLM Query Rewrite。"""

    name = "retrieve_medical_knowledge"
    description = "Retrieve versioned evidence from the local medical knowledge vector store."
    input_model = RetrievalInput
    deterministic = True

    def __init__(self, store: SQLiteVectorStore, seed_path: Path):
        self.store = store
        # 首次启动时写入少量演示知识；正式系统应使用有版本和许可的医学语料。
        if self.store.count() == 0:
            with seed_path.open("r", encoding="utf-8") as stream:
                self.store.upsert(json.load(stream))

    def run(self, payload: RetrievalInput):
        # TODO(LLM): 检索前可让 Qwen 根据诊断候选生成多个医学查询，但必须记录原始查询。
        evidence = self.store.search(payload.query, payload.top_k, {"modality": payload.modality})
        warnings = [] if evidence else ["INSUFFICIENT_EVIDENCE"]
        return {
            "query": payload.query,
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "insufficient_evidence": not evidence,
        }, warnings, {"retrieved_count": float(len(evidence))}, {}
