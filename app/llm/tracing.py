"""LLM provider wrapper that emits agent.llm.* diagnostic events."""

from __future__ import annotations

import time
from typing import Any

from app.agent.trace import get_current_trace
from app.llm.base import LLMMessage, LLMProvider, LLMResponse


class TracingLLMProvider(LLMProvider):
    """Delegates to an inner provider; records start/end without changing outputs."""

    def __init__(self, inner: LLMProvider, *, default_node: str = "other") -> None:
        self._inner = inner
        self.default_node = default_node
        self.active_node: str = default_node

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def inner(self) -> LLMProvider:
        return self._inner

    def set_active_node(self, node: str) -> None:
        self.active_node = node

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return self._wrap(
            kind="generate",
            call=lambda: self._inner.generate(
                messages, temperature=temperature, max_tokens=max_tokens
            ),
        )

    def generate_structured(
        self,
        messages: list[LLMMessage],
        *,
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        title = str((schema or {}).get("title") or "")
        kind = "verification" if title == "VerificationResult" else "generation"
        if title == "VerificationResult":
            node = "verify_answer"
        elif title == "GroundedAnswer":
            node = "generate_answer"
        else:
            node = self.active_node

        def call():
            return self._inner.generate_structured(
                messages, schema=schema, temperature=temperature
            )

        return self._wrap(kind=kind, call=call, node_override=node, schema_title=title)

    def generate_with_tools(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> LLMResponse:
        def call():
            return self._inner.generate_with_tools(
                messages, tools=tools, temperature=temperature
            )

        return self._wrap(
            kind="tools",
            call=call,
            node_override="retrieve",
            tools_available=len(tools),
        )

    def _wrap(
        self,
        *,
        kind: str,
        call,
        node_override: str | None = None,
        schema_title: str | None = None,
        tools_available: int | None = None,
    ):
        trace = get_current_trace()
        node = node_override or self.active_node or self.default_node
        iteration = None
        if trace is not None:
            # Best-effort: last known iteration from final-ish state is not here;
            # callers may set active iteration via emit fields only.
            iteration = _guess_iteration(trace)
            trace.emit(
                "agent.llm.start",
                node=node,
                iteration=iteration,
                model=self.model,
                kind=kind,
                schema_title=schema_title,
                tools_available=tools_available,
            )
        started = time.perf_counter()
        success = False
        error = None
        tool_calls_requested = None
        structured_valid = None
        try:
            result = call()
            success = True
            if isinstance(result, LLMResponse):
                tool_calls_requested = len(result.tool_calls or [])
            elif isinstance(result, dict):
                structured_valid = True
            return result
        except Exception as exc:
            error = str(exc)
            if isinstance(exc, Exception) and "json" in str(exc).lower():
                structured_valid = False
            raise
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            if trace is not None:
                trace.record_llm(
                    node=node,
                    iteration=iteration,
                    model=self.model,
                    kind=kind,
                    duration_ms=duration_ms,
                    success=success,
                    error=error,
                    tool_calls_requested=tool_calls_requested,
                    structured_output_valid=structured_valid,
                )


def _guess_iteration(trace) -> int | None:
    if not trace.iterations:
        return None
    return max(trace.iterations.keys())


def wrap_llm(provider: LLMProvider) -> TracingLLMProvider:
    if isinstance(provider, TracingLLMProvider):
        return provider
    return TracingLLMProvider(provider)
