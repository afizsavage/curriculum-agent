"""Deterministic detection of SSS stream subject questions.

Recognizes narrow question shapes that ask which subjects belong to a named
Senior Secondary stream. Grade, topic, and other curriculum questions stay on
their existing routes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.curriculum.codes import normalize_grade_code

INTENT_SSS_STREAM_SUBJECTS = "SSS_STREAM_SUBJECTS"
TOOL_GET_SSS_STREAM_SUBJECTS = "get_sss_stream_subjects"

_THE = r"(?:the\s+)?"
_NAME = r"(?P<name>.+?)"
_OPTIONAL_STREAM = r"(?:\s+stream)?"

# Structural patterns only. Each one asks for subjects of a named stream.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"^(?:which|what)\s+subjects?\s+are(?:\s+(?:taught|offered|included))?\s+"
        rf"in\s+{_THE}{_NAME}{_OPTIONAL_STREAM}$",
        re.I,
    ),
    re.compile(
        rf"^(?:which|what)\s+subjects?\s+belong\s+to\s+{_THE}{_NAME}{_OPTIONAL_STREAM}$",
        re.I,
    ),
    re.compile(
        rf"^what\s+are\s+(?:the\s+)?subjects?\s+in\s+{_THE}{_NAME}{_OPTIONAL_STREAM}$",
        re.I,
    ),
    re.compile(
        rf"^(?:list|show|give)(?:\s+me)?\s+(?:the\s+)?subjects?\s+"
        rf"in\s+{_THE}{_NAME}{_OPTIONAL_STREAM}$",
        re.I,
    ),
    re.compile(
        rf"^subjects?\s+in\s+{_THE}{_NAME}\s+stream$",
        re.I,
    ),
    re.compile(
        rf"^(?:which|what)\s+subjects?\s+does\s+{_THE}{_NAME}\s+stream\s+"
        rf"(?:contain|include|have)$",
        re.I,
    ),
)

_REJECT_KEYS = frozenset(
    {
        "curriculum",
        "the curriculum",
        "mbsse",
        "mbsse curriculum",
        "senior secondary",
        "senior secondary school",
        "school",
        "this",
        "that",
        "it",
        "stream",
    }
)


@dataclass(frozen=True)
class SSSStreamSubjectsIntent:
    intent: str
    stream_name: str


def stream_lookup_key(value: str) -> str:
    """Normalize a stream title for comparison.

    Case, extra whitespace, punctuation, and ``&`` versus ``and`` do not
    change the key. The word ``stream`` at the end is ignored.
    """
    text = value.strip().lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.endswith(" stream"):
        text = text[: -len(" stream")].strip()
    return text


def match_sss_stream(streams: list[dict], stream_name: str) -> dict | None:
    """Return the unique stream whose name matches ``stream_name``.

    Matching uses the normalized name from the SSS stream records. Stream ids
    are never assumed by the caller.
    """
    wanted = stream_lookup_key(stream_name)
    if not wanted:
        return None
    matches = [
        stream
        for stream in streams
        if stream_lookup_key(str(stream.get("name") or "")) == wanted
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def detect_sss_stream_subjects(question: str | None) -> SSSStreamSubjectsIntent | None:
    """Return the stream-subject intent when the question shape is confident."""
    if not question or not question.strip():
        return None
    prepared = _prepare_question(question)
    if not prepared:
        return None
    for pattern in _PATTERNS:
        match = pattern.match(prepared)
        if match is None:
            continue
        stream_name = _clean_stream_name(match.group("name"))
        if not _confident_stream_name(stream_name, prepared):
            continue
        return SSSStreamSubjectsIntent(
            intent=INTENT_SSS_STREAM_SUBJECTS,
            stream_name=stream_name,
        )
    return None


def _prepare_question(question: str) -> str:
    text = question.strip()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n?.!;")


def _clean_stream_name(raw: str) -> str:
    name = " ".join(raw.split()).strip(" \t?.!,;:")
    name = re.sub(r"^(?:the)\s+", "", name, count=1, flags=re.I)
    name = re.sub(r"\s+stream$", "", name, count=1, flags=re.I)
    return " ".join(name.split()).strip(" \t?.!,;:")


def _confident_stream_name(name: str, question: str) -> bool:
    if not name:
        return False
    # Grade questions stay on the curriculum-structure route.
    if normalize_grade_code(name) or normalize_grade_code(question):
        return False
    key = stream_lookup_key(name)
    if not key or key in _REJECT_KEYS or len(key) < 3:
        return False
    has_stream_word = re.search(r"\bstreams?\b", question, re.I) is not None
    has_connector = re.search(r"&|\band\b", name, re.I) is not None
    # A compound title or an explicit stream word. Single-word curriculum
    # subjects without "stream" are left for the existing routers.
    return has_stream_word or has_connector
