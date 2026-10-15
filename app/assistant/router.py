"""Deterministic route selection for one assistant turn.

A classifier may propose a route. The proposal is accepted only when it is a
member of the closed set. Anything else raises ``InvalidAssistantRoute``.
"""

from __future__ import annotations

from collections.abc import Callable

from app.assistant.contract import (
    AssistantRoute,
    AssistantTurnRequest,
    InvalidAssistantRoute,
    RoutingResult,
)

RouteClassifier = Callable[[AssistantTurnRequest], object]

CLARIFICATION_PROMPT = (
    "I need a bit more detail before I can help. "
    "Are you asking about the official MBSSE curriculum, "
    "or about general teaching practice? "
    "If it is curriculum, which grade and subject should I use?"
)

_CURRICULUM_PHRASES = (
    "curriculum",
    "syllabus",
    "learning outcome",
    "learning objective",
    "mbsse",
    "scheme of work",
    "what topics",
    "what units",
    "topics are taught",
    "topics taught",
)

_ADVICE_PHRASES = (
    "how should i teach",
    "how do i teach",
    "how can i teach",
    "how should i",
    "teaching strategy",
    "classroom management",
    "lesson idea",
    "group work",
    "mixed-ability",
    "settle a noisy",
)

_CLARIFY_PHRASES = (
    "tell me more",
    "what about that",
    "and the other one",
    "can you help",
    "help me",
    "what should i do",
)

_CLARIFY_EXACT = frozenset(
    {
        "help",
        "hi",
        "hello",
        "hey",
        "continue",
        "?",
    }
)


def parse_closed_route(raw: object) -> AssistantRoute:
    """Accept only the four route names. Case and surrounding space are ignored."""
    if isinstance(raw, AssistantRoute):
        return raw
    if not isinstance(raw, str):
        raise InvalidAssistantRoute(raw)
    cleaned = raw.strip().lower()
    try:
        return AssistantRoute(cleaned)
    except ValueError:
        raise InvalidAssistantRoute(raw) from None


def heuristic_route(question: str) -> tuple[AssistantRoute, str]:
    """Classify without a model. Curriculum markers win over vague phrasing."""
    text = " ".join(question.strip().lower().split())
    curriculum = _contains_any(text, _CURRICULUM_PHRASES)
    advice = _contains_any(text, _ADVICE_PHRASES)
    if curriculum and advice:
        return AssistantRoute.MIXED, "curriculum_and_advice"
    if curriculum:
        return AssistantRoute.CURRICULUM, "curriculum_markers"
    if advice:
        return AssistantRoute.GENERAL, "advice_markers"
    if text in _CLARIFY_EXACT or len(text) < 12:
        return AssistantRoute.CLARIFY, "underspecified"
    if _contains_any(text, _CLARIFY_PHRASES):
        return AssistantRoute.CLARIFY, "underspecified"
    return AssistantRoute.GENERAL, "general_default"


def resolve_route(
    request: AssistantTurnRequest,
    *,
    classifier: RouteClassifier | None = None,
) -> RoutingResult:
    """Resolve a route. Invalid classifier output is not rewritten."""
    if classifier is None:
        route, reason = heuristic_route(request.question)
        return RoutingResult(route=route, source="heuristic", reason=reason)
    raw = classifier(request)
    route = parse_closed_route(raw)
    raw_text = raw.value if isinstance(raw, AssistantRoute) else str(raw)
    return RoutingResult(
        route=route,
        source="classifier",
        reason="classifier_closed_set",
        raw_classification=raw_text,
    )


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)
