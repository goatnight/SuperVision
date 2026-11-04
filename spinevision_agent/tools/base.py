"""Agent Tool 的统一执行边界。

工具负责真正的业务能力，Agent/LLM 只能选择工具并提供参数。这里集中完成
Pydantic 校验、异常归一化、耗时记录和 provenance 生成，避免模型输出直接
进入底层推理代码。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from ..errors import AgentError
from ..schemas import Provenance, ToolError, ToolResult, ToolStatus


class EmptyInput(BaseModel):
    pass


class AgentTool(ABC):
    """所有可调用工具的基类；新增模型能力时优先实现新的 Tool 子类。"""

    name: str
    description: str
    version = "0.1.0"
    input_model: type[BaseModel] = EmptyInput
    deterministic = True
    requires_human_review = False

    def invoke(self, arguments: dict[str, Any]) -> ToolResult:
        # invoke 是稳定外壳，子类只实现 run。将来接 LLM 时也应调用 invoke，
        # 不要让模型绕过输入 Schema 或直接调用 run。
        started = datetime.now(timezone.utc)
        try:
            validated = self.input_model.model_validate(arguments)
            data, warnings, metrics, provenance_data = self.run(validated)
            status = ToolStatus.partial if warnings else ToolStatus.success
            errors: list[ToolError] = []
        except AgentError as exc:
            data, warnings, metrics, provenance_data = {}, [], {}, {}
            status = ToolStatus.failed
            errors = [ToolError(code=exc.code, message=str(exc), recoverable=exc.recoverable, details=exc.details)]
        except Exception as exc:  # 工具边界把未知实现错误转换为稳定错误契约。
            data, warnings, metrics, provenance_data = {}, [], {}, {}
            status = ToolStatus.failed
            errors = [ToolError(code="UNEXPECTED_TOOL_ERROR", message=str(exc), recoverable=False)]
        finished = datetime.now(timezone.utc)
        provenance = Provenance(
            tool_name=self.name,
            tool_version=self.version,
            started_at=started,
            finished_at=finished,
            **provenance_data,
        )
        return ToolResult(
            status=status,
            data=data,
            warnings=warnings,
            errors=errors,
            metrics=metrics,
            provenance=provenance,
        )

    @abstractmethod
    def run(self, payload: BaseModel) -> tuple[dict[str, Any], list[str], dict[str, float], dict[str, Any]]:
        """执行领域逻辑，依次返回 data、warnings、metrics、provenance 补充字段。"""
        raise NotImplementedError
