"""工具注册表，也是未来 LLM Tool Calling 的能力白名单。"""

from __future__ import annotations

from .base import AgentTool


class ToolRegistry:
    """按稳定名称注册和调用工具，屏蔽具体模型或服务的实现差异。"""

    def __init__(self):
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> AgentTool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def invoke(self, name: str, arguments: dict):
        return self.get(name).invoke(arguments)

    def describe(self) -> list[dict]:
        # input_schema 可以直接转换成 OpenAI-compatible function/tool schema，
        # 供 Qwen Planner 判断可用动作；真正执行仍须回到 registry.invoke。
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "version": tool.version,
                "deterministic": tool.deterministic,
                "requires_human_review": tool.requires_human_review,
                "input_schema": tool.input_model.model_json_schema(),
            }
            for tool in self._tools.values()
        ]
