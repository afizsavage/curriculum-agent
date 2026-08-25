"""Concrete LLM providers and factory."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import httpx

from app.config import Settings, get_settings
from app.exceptions import (
    AgentError,
    ConfigurationError,
    LLMProviderError,
    LLMTimeoutError,
)
from app.llm.base import LLMMessage, LLMProvider, LLMResponse, ToolCallRequest
from app.llm.tool_selection import select_tool_calls


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
        calls = select_tool_calls(messages, tools)
        return LLMResponse(
            content=None if calls else "Retrieval complete.",
            tool_calls=calls,
            model=self._model,
            raw={"provider": "stub", "available_tools": [t.get("name") for t in tools]},
        )


_OPENAI_COMPATIBLE_DEFAULTS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
}


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI Chat Completions API (native tool/function calling)."""

    def __init__(self, settings: Settings, *, provider_name: str = "openai") -> None:
        if not settings.llm_api_key:
            raise ConfigurationError(
                f"LLM_API_KEY is required for {provider_name} provider"
            )
        self._settings = settings
        self._provider_name = provider_name
        self._model = settings.llm_model
        base_url = settings.llm_base_url.rstrip("/")
        default_url = _OPENAI_COMPATIBLE_DEFAULTS.get(provider_name)
        if default_url and base_url == _OPENAI_COMPATIBLE_DEFAULTS["openai"]:
            base_url = default_url
        self._client = httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(settings.llm_timeout_seconds),
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def name(self) -> str:
        return self._provider_name

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
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_api_dict() for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        data = self._post("/chat/completions", body)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return LLMResponse(
            content=message.get("content"),
            model=data.get("model") or self._model,
            raw=data,
        )

    def generate_structured(
        self,
        messages: list[LLMMessage],
        *,
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        body = {
            "model": self._model,
            "messages": [m.to_api_dict() for m in messages],
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("title") or "response",
                    "schema": schema,
                },
            },
        }
        data = self._post("/chat/completions", body)
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
        if not content:
            raise LLMProviderError("Empty structured response from LLM")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMProviderError("LLM returned invalid JSON") from exc

    def generate_with_tools(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> LLMResponse:
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            }
            for tool in tools
        ]
        body = {
            "model": self._model,
            "messages": [m.to_api_dict() for m in messages],
            "tools": openai_tools,
            "tool_choice": "auto",
            "temperature": temperature,
        }
        data = self._post("/chat/completions", body)
        message = ((data.get("choices") or [{}])[0].get("message") or {})
        calls: list[ToolCallRequest] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            args_raw = fn.get("arguments") or "{}"
            try:
                arguments = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                ToolCallRequest(
                    id=str(raw.get("id") or uuid4()),
                    name=str(fn.get("name") or ""),
                    arguments=arguments,
                )
            )
        return LLMResponse(
            content=message.get("content"),
            tool_calls=calls,
            model=data.get("model") or self._model,
            raw=data,
        )

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=body)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("LLM request timed out") from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(f"LLM request failed: {exc}") from exc
        if response.status_code >= 400:
            detail = response.text.strip()
            if len(detail) > 300:
                detail = detail[:300] + "..."
            message = f"LLM provider returned HTTP {response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            raise LLMProviderError(message)
        try:
            return response.json()
        except ValueError as exc:
            raise LLMProviderError("LLM provider returned non-JSON") from exc


class ConfigurableLLMProvider(LLMProvider):
    """Selects a concrete provider from settings without coupling the agent to an SDK."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        provider = self._settings.llm_provider.strip().lower()
        if provider in {"", "stub", "none", "mock"}:
            self._inner: LLMProvider = StubLLMProvider(model=self._settings.llm_model)
        elif provider in {"openai", "openai_compatible", "deepseek"}:
            name = "openai" if provider == "openai_compatible" else provider
            self._inner = OpenAICompatibleProvider(self._settings, provider_name=name)
        else:
            raise ConfigurationError(
                f"LLM provider '{self._settings.llm_provider}' is not supported. "
                "Use LLM_PROVIDER=stub, openai, or deepseek."
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
