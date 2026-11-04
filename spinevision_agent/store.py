"""病例状态存储抽象的 MVP 实现。"""

from __future__ import annotations

import threading
from copy import deepcopy

from .schemas import CaseRecord, utc_now


class InMemoryCaseStore:
    """线程安全的内存状态库；服务重启会丢失数据，部署时应换成 Redis/PostgreSQL。"""

    def __init__(self):
        self._cases: dict[str, CaseRecord] = {}
        self._lock = threading.RLock()

    def create(self, case: CaseRecord) -> CaseRecord:
        with self._lock:
            if case.case_id in self._cases:
                raise ValueError(f"Case already exists: {case.case_id}")
            self._cases[case.case_id] = deepcopy(case)
            return deepcopy(case)

    def get(self, case_id: str) -> CaseRecord:
        # 返回 deepcopy，防止调用方绕过 save 在锁外静默修改共享状态。
        with self._lock:
            if case_id not in self._cases:
                raise KeyError(case_id)
            return deepcopy(self._cases[case_id])

    def save(self, case: CaseRecord) -> CaseRecord:
        with self._lock:
            case.updated_at = utc_now()
            self._cases[case.case_id] = deepcopy(case)
            return deepcopy(case)

    def list(self) -> list[CaseRecord]:
        with self._lock:
            return [deepcopy(item) for item in self._cases.values()]
