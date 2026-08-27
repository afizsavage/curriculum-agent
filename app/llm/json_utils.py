"""Helpers for parsing structured LLM output."""

from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*([\s\S]*?)\s*```",
    re.MULTILINE,
)


def parse_llm_json(content: str) -> dict[str, Any]:
    """Parse a JSON object from model text that may include fences or chatter."""
    text = (content or "").strip()
    if not text:
        raise json.JSONDecodeError("Empty content", text, 0)

    candidates: list[str] = [text]
    for match in _FENCE_RE.finditer(text):
        candidates.append(match.group(1).strip())

    # Raw object/array slice if the model added prose around JSON.
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end > obj_start:
        candidates.append(text[obj_start : obj_end + 1])
    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start != -1 and arr_end > arr_start:
        candidates.append(text[arr_start : arr_end + 1])

    seen: set[str] = set()
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            repaired = _try_repair_truncated(candidate)
            if repaired is None:
                continue
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError as repair_exc:
                last_error = repair_exc
                continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            # Wrap unexpected list payloads for callers that expect an object.
            return {"items": parsed}
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _try_repair_truncated(text: str) -> str | None:
    """Best-effort close of truncated JSON objects/arrays/strings."""
    if not text or text[0] not in "{[":
        return None
    # If already valid, caller would have succeeded.
    in_string = False
    escape = False
    stack: list[str] = []
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                return None
    if not stack and not in_string:
        return None
    repaired = text
    if in_string:
        repaired += '"'
    # Remove trailing comma before closing.
    repaired = re.sub(r",\s*$", "", repaired)
    repaired += "".join(reversed(stack))
    return repaired
