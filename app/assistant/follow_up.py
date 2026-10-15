"""Resolve a short follow-up from the current owned thread.

The stored user text stays unchanged. Dispatch receives a shorter question
that keeps grade, subject, or topic words already present in the prior turn.
It does not invent curriculum facts, and it does not read another teacher's
messages. There is no long-term memory beyond this thread.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.assistant.contract import AssistantRoute
from app.assistant.router import _GRADE_LEVEL, heuristic_route
from app.assistant.threads import QaOutcomeSummary, ThreadMessage

_PURE_FOLLOW_UPS = frozenset(
    {
        "tell me more",
        "what about that",
        "and the other one",
        "continue",
        "go on",
        "more detail",
    }
)

_SUBJECT_CUES = (
    "integrated science",
    "social studies",
    "mathematics",
    "maths",
    "english",
    "science",
    "french",
    "fractions",
    "multiplication",
)

_CURRICULUM_CUES = (
    "topics are taught",
    "what topics",
    "what units",
    "learning outcome",
    "learning objective",
    "scheme of work",
    "syllabus",
    "curriculum",
)

_ADVICE_CUES = (
    "settle a noisy",
    "teaching strategy",
    "teaching activity",
    "classroom management",
    "lesson idea",
    "group work",
    "mixed-ability",
    "how should i teach",
    "how do i teach",
    "how can i teach",
)

_GAP_STATUSES = frozenset({"insufficient_evidence", "failed", "error", "needs_clarification"})
_GAP_EVIDENCE = frozenset({"partial", "not_found", "error"})
_GAP_COVERAGE = frozenset({"reject", "unavailable"})


@dataclass(frozen=True)
class FollowUpResolution:
    """Question passed to the router and handlers."""

    dispatch_question: str
    used_prior: bool


def resolve_context_follow_up(
    question: str,
    messages: list[ThreadMessage],
    qa_outcome: QaOutcomeSummary | None = None,
) -> FollowUpResolution:
    """Use the latest substantive user message on this thread, when one exists."""
    text = _normalize(question)
    if text not in _PURE_FOLLOW_UPS:
        return FollowUpResolution(question.strip(), used_prior=False)
    prior = _latest_substantive_user_text(messages)
    if prior is None:
        return FollowUpResolution(question.strip(), used_prior=False)
    prior_route, _reason = heuristic_route(prior)
    if prior_route is AssistantRoute.CLARIFY:
        return FollowUpResolution(question.strip(), used_prior=False)
    if _curriculum_gap(qa_outcome):
        cues = _curriculum_cues(prior)
    elif prior_route is AssistantRoute.GENERAL:
        cues = _advice_cues(prior)
    else:
        cues = _curriculum_cues(prior) or _advice_cues(prior)
    if not cues:
        return FollowUpResolution(question.strip(), used_prior=False)
    return FollowUpResolution(f"Tell me more about {cues}", used_prior=True)


def _latest_substantive_user_text(messages: list[ThreadMessage]) -> str | None:
    for message in reversed(messages):
        if message.role != "user":
            continue
        text = _normalize(message.content)
        if text in _PURE_FOLLOW_UPS:
            continue
        if text:
            return message.content.strip()
    return None


def _curriculum_gap(outcome: QaOutcomeSummary | None) -> bool:
    if outcome is None:
        return False
    if outcome.technical_failure:
        return True
    if (outcome.status or "") in _GAP_STATUSES:
        return True
    if (outcome.evidence_status or "") in _GAP_EVIDENCE:
        return True
    return (outcome.coverage_decision or "") in _GAP_COVERAGE


def _curriculum_cues(prior: str) -> str:
    text = _normalize(prior)
    bits: list[str] = []
    for match in _GRADE_LEVEL.finditer(text):
        _add(bits, match.group(0))
    for cue in _SUBJECT_CUES:
        if cue in text:
            _add(bits, cue)
    for cue in _CURRICULUM_CUES:
        if cue in text:
            _add(bits, cue)
            break
    return " ".join(bits)


def _advice_cues(prior: str) -> str:
    text = _normalize(prior)
    for cue in _ADVICE_CUES:
        if cue in text:
            return cue
    return ""


def _add(bits: list[str], value: str) -> None:
    if value not in bits:
        bits.append(value)


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())
