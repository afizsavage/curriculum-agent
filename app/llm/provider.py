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
from app.llm.deepseek import DeepSeekResponsesProvider
from app.llm.json_utils import parse_llm_json
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
        title = str(schema.get("title") or "")
        if title == "VerificationResult":
            return {
                "passed": True,
                "score": 0.9,
                "issues": [],
                "unsupported_claims": [],
                "missing_evidence": [],
                "incorrect_claims": [],
                "recommendation": "accept",
                "claims": [],
                "notes": "Stub verifier default accept.",
            }
        if title == "GroundedAnswer":
            return {
                "answer": "[stub] Grounded curriculum answer.",
                "summary": "Stub summary",
                "evidence": [],
                "limitations": [],
                "confidence": "medium",
            }
        return {"provider": "stub", "schema_title": title, "ok": True}

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


# Base URLs that belong to another provider. OpenRouter replaces these so
# LLM_PROVIDER can change without also editing LLM_BASE_URL.
_OTHER_PROVIDER_BASE_URLS = {
    "https://api.openai.com/v1",
    "https://api.openai.com",
    "https://api.deepseek.com",
    "https://api.deepseek.com/v1",
}
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# Free-model router: picks a free model that supports the request's features.
OPENROUTER_FREE_MODEL = "openrouter/free"


def resolve_openrouter_connection(
    settings: Settings,
) -> tuple[str, str, str, dict[str, str]]:
    """API key, base URL, model, and optional ranking headers for OpenRouter.

    ``OPENROUTER_API_KEY`` wins over ``LLM_API_KEY`` and ``OPENROUTER_MODEL``
    wins over ``LLM_MODEL``, so the active provider can flip without replacing
    the openai/deepseek key or model. The OpenAI default base URL is replaced
    with OpenRouter's. A blank OpenRouter model becomes the free router, unless
    ``LLM_MODEL`` is an explicit non-stub name.
    """
    api_key = (settings.openrouter_api_key or settings.llm_api_key or "").strip()
    if not api_key:
        raise ConfigurationError(
            "LLM_API_KEY or OPENROUTER_API_KEY is required for openrouter provider"
        )
    base_url = (settings.llm_base_url or "").rstrip("/")
    if not base_url or base_url in _OTHER_PROVIDER_BASE_URLS:
        base_url = OPENROUTER_DEFAULT_BASE_URL
    model = (settings.openrouter_model or "").strip()
    if not model:
        configured = (settings.llm_model or "").strip()
        model = configured if configured and configured != "stub-model" else OPENROUTER_FREE_MODEL
    headers: dict[str, str] = {}
    referer = (settings.openrouter_http_referer or "").strip()
    title = (settings.openrouter_app_title or "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-OpenRouter-Title"] = title
    return api_key, base_url, model, headers


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI Chat Completions API (native tool/function calling)."""

    def __init__(
        self,
        settings: Settings,
        *,
        provider_name: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        extra_headers: dict[str, str] | None = None,
        require_parameters: bool = False,
    ) -> None:
        key = settings.llm_api_key if api_key is None else api_key
        if not key:
            raise ConfigurationError(
                f"LLM_API_KEY is required for {provider_name} provider"
            )
        self._settings = settings
        self._provider_name = provider_name
        self._model = settings.llm_model if model is None else model
        self._require_parameters = require_parameters
        resolved_base = (
            settings.llm_base_url if base_url is None else base_url
        ).rstrip("/") or "https://api.openai.com/v1"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        self._client = httpx.Client(
            base_url=resolved_base,
            timeout=httpx.Timeout(settings.llm_timeout_seconds),
            headers=headers,
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
        data = self._post("/chat/completions", body, require_parameters=False)
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
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_api_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": 4096,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("title") or "response",
                    "schema": schema,
                },
            },
        }
        data = self._post("/chat/completions", body, require_parameters=True)
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
        if not content:
            raise LLMProviderError("Empty structured response from LLM")
        try:
            return parse_llm_json(content)
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
                    "parameters": tool.get("parameters")
                    or {"type": "object", "properties": {}},
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
        data = self._post("/chat/completions", body, require_parameters=True)
        message = ((data.get("choices") or [{}])[0].get("message") or {})
        calls: list[ToolCallRequest] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            args_raw = fn.get("arguments") or "{}"
            try:
                arguments = (
                    json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
                )
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

    def _post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        require_parameters: bool = False,
    ) -> dict[str, Any]:
        payload = dict(body)
        # OpenRouter treats response_format and tools as soft preferences unless
        # this flag is set, which would drop structured output on some free routes.
        if require_parameters and self._require_parameters:
            prefs = dict(payload.get("provider") or {})
            prefs["require_parameters"] = True
            payload["provider"] = prefs
        try:
            response = self._client.post(path, json=payload)
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
        elif provider == "deepseek":
            self._inner = DeepSeekResponsesProvider(self._settings)
        elif provider in {"openai", "openai_compatible"}:
            name = "openai" if provider == "openai_compatible" else provider
            self._inner = OpenAICompatibleProvider(self._settings, provider_name=name)
        elif provider == "openrouter":
            api_key, base_url, model, headers = resolve_openrouter_connection(
                self._settings
            )
            self._inner = OpenAICompatibleProvider(
                self._settings,
                provider_name="openrouter",
                api_key=api_key,
                base_url=base_url,
                model=model,
                extra_headers=headers,
                require_parameters=True,
            )
        else:
            raise ConfigurationError(
                f"LLM provider '{self._settings.llm_provider}' is not supported. "
                "Use LLM_PROVIDER=stub, openai, deepseek, or openrouter."
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
