"""Deterministic purpose of a curriculum question.

This is separate from ``state.intent``. It does not route retrieval, select
evidence, or decide verification. Unrecognized values mean ``unknown``.
"""

from __future__ import annotations

import re

PURPOSE_LEARNING_OUTCOME_LIST = "learning_outcome_list"
PURPOSE_TOPIC_LIST = "topic_list"
PURPOSE_UNKNOWN = "unknown"

QUESTION_PURPOSES = frozenset(
    {
        PURPOSE_LEARNING_OUTCOME_LIST,
        PURPOSE_TOPIC_LIST,
        PURPOSE_UNKNOWN,
    }
)

# A request that asks for both lists has no single purpose.
_MIXED_LIST = re.compile(
    r"""
    ^(?:what|which|list|show|give)(?:\s+\w+){0,6}\s+
    (?:topics?|outcomes?|learning\s+(?:outcomes?|objectives?))
    \s+and\s+
    (?:the\s+)?(?:learning\s+)?(?:topics?|outcomes?|objectives?)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Clear outcome-list or single-outcome identification frames.
# "learning objectives" is the same request as learning outcomes.
_OUTCOME_LIST = re.compile(
    r"""
    ^(?:
        (?:what|which)\s+(?:learning\s+)?outcomes?\s+are\s+
            (?:published|listed|covered|taught|included)\b
        |(?:what|which)\s+are\s+(?:all\s+|every\s+)?the\s+learning\s+
            (?:outcomes?|objectives?)\b
        |(?:what|which)\s+is\s+the\s+learning\s+
            (?:outcome|objective)\b
        |(?:list|show|give)(?:\s+me)?\s+
            (?:the\s+|all\s+|every\s+)?
            (?:published\s+)?
            learning\s+(?:outcomes?|objectives?)\b
        |(?:list|show|give)(?:\s+me)?\s+
            (?:the\s+|all\s+|every\s+)?
            (?:published\s+)?
            outcomes?\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_TOPIC_LIST = re.compile(
    r"""
    ^(?:
        (?:what|which)\s+topics?\s+are\s+
            (?:listed|covered|taught|published|included|in)\b
        |(?:what|which)\s+are\s+the\s+topics?\b
        |(?:list|show|give)(?:\s+me)?\s+
            (?:the\s+|all\s+|every\s+)?
            (?:published\s+)?
            topics?\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _prepare(question: str | None) -> str:
    return " ".join((question or "").split()).casefold().rstrip(" ?.!\"'")


def classify_question_purpose(question: str | None) -> str:
    """Return one purpose. Mixed or unrecognized questions are ``unknown``."""
    text = _prepare(question)
    if not text or _MIXED_LIST.search(text):
        return PURPOSE_UNKNOWN
    outcome = _OUTCOME_LIST.search(text) is not None
    topic = _TOPIC_LIST.search(text) is not None
    if outcome and topic:
        return PURPOSE_UNKNOWN
    if outcome:
        return PURPOSE_LEARNING_OUTCOME_LIST
    if topic:
        return PURPOSE_TOPIC_LIST
    return PURPOSE_UNKNOWN


def coerce_question_purpose(value: object) -> str:
    """Map a missing or unrecognized stored value to ``unknown``."""
    if isinstance(value, str) and value in QUESTION_PURPOSES:
        return value
    return PURPOSE_UNKNOWN
