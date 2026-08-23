"""Tool abstraction for future curriculum retrieval tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    success: bool
    data: Any = None
    error: str | None = None


class Tool(ABC):
    """Generic tool interface. Curriculum tools arrive in Sprint 2."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema describing tool inputs."""

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        ...

    def as_llm_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }


class EchoTool(Tool):
    """Sprint 1 mock tool for registry/tests. Not a curriculum tool."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes the provided message. Used for foundation tests only."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Text to echo"},
            },
            "required": ["message"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        message = kwargs.get("message")
        if message is None:
            return ToolResult(success=False, error="message is required")
        return ToolResult(success=True, data={"echo": str(message)})
