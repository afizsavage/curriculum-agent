"""LLM provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: str
    content: str


class ToolCallRequest(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: Optional[str] = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
    model: Optional[str] = None


class LLMProvider(ABC):
    """Provider-agnostic LLM interface for the Curriculum Q&A agent."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. stub, openai)."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Configured model name."""

    @abstractmethod
    def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a free-form completion."""

    @abstractmethod
    def generate_structured(
        self,
        messages: list[LLMMessage],
        *,
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Generate a response conforming to a JSON schema."""

    @abstractmethod
    def generate_with_tools(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate a completion that may include tool calls."""
