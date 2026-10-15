"""Trace-only details of what the verifier model returned.

These helpers do not change parsing, normalization, or routing. When no
agent trace is active, recording is a no-op.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from app.schemas.verification import VERIFICATION_RESULT_JSON_SCHEMA

_SECRET = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+\S+|api[_-]?key[\"']?\s*[:=]\s*[\"']?\S+)",
    re.IGNORECASE,
)
_TRACKED_FIELDS = (
    "passed",
    "score",
    "issues",
    "unsupported_claims",
    "incorrect_claims",
    "missing_evidence",
    "recommendation",
    "notes",
)


def fingerprint(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def redact_text(text: str | None) -> str | None:
    if text is None:
        return None
    return _SECRET.sub("[redacted]", text)


def field_presence(raw: dict[str, Any] | None, key: str) -> str:
    """Distinguish a missing key from null, an empty string, and an empty list."""
    if raw is None or key not in raw:
        return "missing"
    value = raw[key]
    if value is None:
        return "null"
    if value == "":
        return "empty_string"
    if value == []:
        return "empty_array"
    if value == {}:
        return "empty_object"
    return "present"


def field_report(raw: dict[str, Any] | None) -> dict[str, str]:
    return {key: field_presence(raw, key) for key in _TRACKED_FIELDS}


def schema_problems(raw: Any) -> list[str]:
    """Local check of the verifier JSON schema. Does not reject the payload."""
    if not isinstance(raw, dict):
        return ["not_object"]
    problems: list[str] = []
    required = list(VERIFICATION_RESULT_JSON_SCHEMA.get("required") or [])
    properties = VERIFICATION_RESULT_JSON_SCHEMA.get("properties") or {}
    for key in required:
        if key not in raw:
            problems.append(f"missing:{key}")
    recommendation = raw.get("recommendation")
    allowed = (properties.get("recommendation") or {}).get("enum") or []
    if "recommendation" in raw and recommendation not in allowed:
        problems.append("recommendation_not_in_enum")
    if "passed" in raw and not isinstance(raw.get("passed"), bool):
        problems.append("passed_not_boolean")
    if "score" in raw and not isinstance(raw.get("score"), (int, float)):
        problems.append("score_not_number")
    elif isinstance(raw.get("score"), (int, float)) and not 0 <= float(raw["score"]) <= 1:
        problems.append("score_out_of_range")
    for key in ("issues", "unsupported_claims", "incorrect_claims", "missing_evidence"):
        if key in raw and not isinstance(raw.get(key), list):
            problems.append(f"{key}_not_array")
    extra = sorted(set(raw) - set(properties))
    for key in extra:
        problems.append(f"unexpected:{key}")
    return problems


def sanitize_exception(exc: BaseException) -> dict[str, str]:
    message = redact_text(f"{type(exc).__name__}: {exc}") or type(exc).__name__
    return {"category": type(exc).__name__, "message": message[:400]}


def stash_verifier_model_text(
    content: str | None,
    *,
    provider: str | None,
    model: str | None,
    temperature: float | None,
    max_output_tokens: int | None,
    json_extracted: bool,
    error: BaseException | None = None,
) -> None:
    """Remember the raw verifier text on the active trace. No log line."""
    from app.agent.trace import get_current_trace

    trace = get_current_trace()
    if trace is None:
        return
    redacted = redact_text(content)
    trace.verifier_model_stash = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "raw_response_present": content is not None and content != "",
        "raw_response_length": len(content or ""),
        "raw_response_fingerprint": fingerprint(redacted),
        "raw_response": redacted,
        "json_extracted": json_extracted,
        "json_error": sanitize_exception(error) if error is not None else None,
    }
