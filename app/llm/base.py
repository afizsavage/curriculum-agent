"""LLM provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: str
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    # DeepSeek Responses API thinking items that must be replayed on follow-up turns.
    reasoning: list[dict[str, Any]] | None = None

    def to_api_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            payload["tool_calls"] = self.tool_calls
        return payload


class ToolCallRequest(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: Optional[str] = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    reasoning: list[dict[str, Any]] = Field(default_factory=list)
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
