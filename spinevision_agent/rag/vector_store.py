"""轻量本地 RAG 存储。

当前 HashingEmbedder 是确定性哈希基线，不是神经网络 Embedding 模型。它的作用
是让项目离线可运行、便于测试数据流；正式版本应替换为 Qwen3-Embedding 等
真实语义向量模型，并增加独立 Reranker。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

from ..schemas import Evidence


class HashingEmbedder:
    """离线哈希向量基线，不具备医学语义理解能力。"""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())
        return words + ["".join(words[i : i + 2]) for i in range(max(0, len(words) - 1))]

    def embed(self, text: str) -> list[float]:
        # TODO(RAG): 保持 embed(text)->list[float] 接口，替换为本地 Embedding 服务。
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "little") % self.dimensions
            sign = 1.0 if digest[0] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


class SQLiteVectorStore:
    """小规模文档的 SQLite 向量库；当前用 Python 全量计算余弦相似度。"""

    def __init__(self, path: Path, embedder: HashingEmbedder | None = None):
        self.path = path.resolve()
        self.embedder = embedder or HashingEmbedder()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    evidence_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    section TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
                """
            )

    def upsert(self, documents: list[dict[str, Any]]) -> None:
        rows = []
        for doc in documents:
            content = str(doc["content"])
            rows.append(
                (
                    str(doc["evidence_id"]),
                    str(doc["title"]),
                    str(doc.get("section", "general")),
                    content,
                    str(doc.get("source", "internal_sop")),
                    json.dumps(doc.get("metadata", {}), ensure_ascii=False, sort_keys=True),
                    json.dumps(self.embedder.embed(content)),
                )
            )
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evidence_id) DO UPDATE SET
                    title=excluded.title, section=excluded.section,
                    content=excluded.content, source=excluded.source,
                    metadata=excluded.metadata, embedding=excluded.embedding
                """,
                rows,
            )

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def search(self, query: str, top_k: int = 4, metadata_filter: dict[str, Any] | None = None) -> list[Evidence]:
        # 数据量增大后可替换为 pgvector/Qdrant/Milvus，但保持 Evidence 输出契约。
        query_vector = self.embedder.embed(query)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT evidence_id, title, section, content, source, metadata, embedding FROM documents"
            ).fetchall()
        ranked: list[Evidence] = []
        for evidence_id, title, section, content, source, metadata_json, embedding_json in rows:
            metadata = json.loads(metadata_json)
            if metadata_filter and any(metadata.get(key) != value for key, value in metadata_filter.items()):
                continue
            ranked.append(
                Evidence(
                    evidence_id=evidence_id,
                    title=title,
                    section=section,
                    content=content,
                    source=source,
                    score=cosine(query_vector, json.loads(embedding_json)),
                    metadata=metadata,
                )
            )
        # TODO(RAG): 对初召回 Top-N 使用 Qwen3-Reranker，再截取最终 Top-k。
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]
