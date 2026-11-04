from __future__ import annotations


class AgentError(RuntimeError):
    """可安全跨 Tool 边界传播的领域错误基类。"""
    """Base error carrying a stable machine-readable code."""

    code = "AGENT_ERROR"
    recoverable = False

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class InputValidationError(AgentError):
    code = "INPUT_INVALID"


class ToolExecutionError(AgentError):
    code = "TOOL_EXECUTION_FAILED"
    recoverable = True


class ModelUnavailableError(AgentError):
    code = "MODEL_UNAVAILABLE"


class HumanReviewRequired(AgentError):
    code = "HUMAN_REVIEW_REQUIRED"
