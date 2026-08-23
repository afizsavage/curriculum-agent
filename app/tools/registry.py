"""Tool registry for the Curriculum Q&A agent."""

from __future__ import annotations

from typing import Any

from app.exceptions import ToolFailureError
from app.tools.base import Tool, ToolResult


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolFailureError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolFailureError(f"Unknown tool '{name}'")
        return tool

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return sorted(self._tools)

    def execute(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self.get(name)
        try:
            return tool.execute(**kwargs)
        except ToolFailureError:
            raise
        except Exception as exc:
            raise ToolFailureError(f"Tool '{name}' failed: {exc}") from exc

    def llm_tool_specs(self) -> list[dict[str, Any]]:
        return [tool.as_llm_tool() for tool in self.list()]


def build_default_registry(*, include_echo: bool = True) -> ToolRegistry:
    """Sprint 1 registry is empty of curriculum tools; optional echo for tests."""
    registry = ToolRegistry()
    if include_echo:
        from app.tools.base import EchoTool

        registry.register(EchoTool())
    return registry
