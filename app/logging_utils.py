"""Structured logging helpers for agent requests.

Never log API keys, tokens, or other secrets.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator
from uuid import uuid4

from app.config import settings


_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "llm_api_key",
        "authorization",
        "token",
        "password",
        "secret",
    }
)


def configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_logger(name: str = "curriculum_agent") -> logging.Logger:
    return logging.getLogger(name)


def new_request_id() -> str:
    return str(uuid4())


def _safe_extra(extra: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in extra.items():
        if key.lower() in _SENSITIVE_KEYS or any(
            part in key.lower() for part in ("api_key", "token", "secret", "password")
        ):
            cleaned[key] = "[redacted]"
        else:
            cleaned[key] = value
    return cleaned


def log_agent_event(
    logger: logging.Logger,
    event: str,
    *,
    request_id: str | None = None,
    conversation_id: str | None = None,
    agent_name: str = "curriculum_qa",
    question: str | None = None,
    status: str | None = None,
    iteration: int | None = None,
    tool_calls: int | None = None,
    model: str | None = None,
    latency_ms: float | None = None,
    error: str | None = None,
    **extra: Any,
) -> None:
    """Emit a structured agent log line for later step-level tracing."""
    payload = _safe_extra(
        {
            "event": event,
            "request_id": request_id,
            "conversation_id": conversation_id,
            "agent_name": agent_name,
            "question": question,
            "status": status,
            "iteration": iteration,
            "tool_calls": tool_calls,
            "model": model,
            "latency_ms": latency_ms,
            "error": error,
            **extra,
        }
    )
    # Drop Nones for readability.
    payload = {k: v for k, v in payload.items() if v is not None}
    parts = " ".join(f"{key}={value!r}" for key, value in payload.items())
    logger.info(parts)


@contextmanager
def timed_request(
    logger: logging.Logger,
    *,
    request_id: str,
    conversation_id: str | None,
    question: str,
    model: str | None,
) -> Generator[dict[str, Any], None, None]:
    started = time.perf_counter()
    ctx: dict[str, Any] = {
        "request_id": request_id,
        "conversation_id": conversation_id,
        "status": "received",
        "iteration": 0,
        "tool_calls": 0,
        "error": None,
    }
    log_agent_event(
        logger,
        "agent.request.start",
        request_id=request_id,
        conversation_id=conversation_id,
        question=question,
        status="received",
        model=model,
    )
    try:
        yield ctx
        latency_ms = (time.perf_counter() - started) * 1000
        log_agent_event(
            logger,
            "agent.request.end",
            request_id=request_id,
            conversation_id=ctx.get("conversation_id") or conversation_id,
            question=question,
            status=ctx.get("status"),
            iteration=ctx.get("iteration"),
            tool_calls=ctx.get("tool_calls"),
            model=model,
            latency_ms=round(latency_ms, 2),
            error=ctx.get("error"),
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        log_agent_event(
            logger,
            "agent.request.error",
            request_id=request_id,
            conversation_id=ctx.get("conversation_id") or conversation_id,
            question=question,
            status="error",
            iteration=ctx.get("iteration"),
            tool_calls=ctx.get("tool_calls"),
            model=model,
            latency_ms=round(latency_ms, 2),
            error=str(exc),
        )
        raise
