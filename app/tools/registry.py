"""Tool registry for the Curriculum Q&A agent."""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.curriculum.client import CurriculumAPIClient
from app.exceptions import ToolFailureError
from app.tools.base import Tool, ToolResult
from app.tools.curriculum import build_curriculum_tools
from app.tools.document import build_document_tools


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


def build_default_registry(
    *,
    settings: Settings | None = None,
    include_echo: bool = False,
    client: CurriculumAPIClient | None = None,
) -> ToolRegistry:
    """Register Phase 2 curriculum tools (read-only Curriculum API)."""
    registry = ToolRegistry()
    settings = settings or get_settings()
    api_client = client or CurriculumAPIClient(settings=settings)
    for tool in build_curriculum_tools(api_client):
        registry.register(tool)
    for tool in build_document_tools(settings=settings, client=api_client):
        registry.register(tool)
    if include_echo:
        from app.tools.base import EchoTool

        registry.register(EchoTool())
    return registry
