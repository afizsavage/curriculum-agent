"""Concrete LLM providers and factory."""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.exceptions import AgentError, ConfigurationError, LLMProviderError
from app.llm.base import LLMMessage, LLMProvider, LLMResponse


class StubLLMProvider(LLMProvider):
    """Deterministic provider for local runs and tests. No network calls."""

    def __init__(self, *, model: str = "stub-model") -> None:
        self._model = model

    @property
    def name(self) -> str:
        return "stub"

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        last = messages[-1].content if messages else ""
        return LLMResponse(
            content=f"[stub] Acknowledged: {last[:200]}",
            model=self._model,
            raw={"provider": "stub"},
        )

    def generate_structured(
        self,
        messages: list[LLMMessage],
        *,
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        return {"provider": "stub", "schema_title": schema.get("title"), "ok": True}

    def generate_with_tools(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> LLMResponse:
        return LLMResponse(
            content=None,
            tool_calls=[],
            model=self._model,
            raw={"provider": "stub", "available_tools": [t.get("name") for t in tools]},
        )


class ConfigurableLLMProvider(LLMProvider):
    """Selects a concrete provider from settings without coupling the agent to an SDK.

    Sprint 1 ships a stub. Additional providers plug in here later.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        provider = self._settings.llm_provider.strip().lower()
        if provider in {"", "stub", "none", "mock"}:
            self._inner: LLMProvider = StubLLMProvider(model=self._settings.llm_model)
        else:
            raise ConfigurationError(
                f"LLM provider '{self._settings.llm_provider}' is not configured in "
                "Sprint 1. Use LLM_PROVIDER=stub or implement a concrete provider."
            )

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def model(self) -> str:
        return self._inner.model

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        try:
            return self._inner.generate(
                messages, temperature=temperature, max_tokens=max_tokens
            )
        except AgentError:
            raise
        except Exception as exc:  # pragma: no cover
            raise LLMProviderError(f"LLM generate failed: {exc}") from exc

    def generate_structured(
        self,
        messages: list[LLMMessage],
        *,
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        try:
            return self._inner.generate_structured(
                messages, schema=schema, temperature=temperature
            )
        except AgentError:
            raise
        except Exception as exc:  # pragma: no cover
            raise LLMProviderError(f"LLM structured generate failed: {exc}") from exc

    def generate_with_tools(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> LLMResponse:
        try:
            return self._inner.generate_with_tools(
                messages, tools=tools, temperature=temperature
            )
        except AgentError:
            raise
        except Exception as exc:  # pragma: no cover
            raise LLMProviderError(f"LLM tool generate failed: {exc}") from exc


def build_llm_provider(settings: Settings | None = None) -> LLMProvider:
    return ConfigurableLLMProvider(settings)
