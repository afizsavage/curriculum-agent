"""Constrained conditional routing for the Curriculum Q&A graph.

The graph — not the LLM — owns legal destinations.
"""

from __future__ import annotations

from typing import Literal

from app.agent.graph_state import ALLOWED_ROUTES, GraphState
from app.config import Settings
from app.logging_utils import get_logger, log_agent_event
from app.schemas.verification import VerificationRecommendation

logger = get_logger(__name__)

RouteName = Literal["finish", "retrieve_more", "clarify", "fallback"]
AfterPrepare = Literal["retrieve", "fallback"]


def route_after_prepare(graph_state: GraphState) -> AfterPrepare:
    """After prepare_cycle: either enter retrieve or go straight to fallback."""
    if graph_state.get("max_iterations_hit"):
        return "fallback"
    return "retrieve"


def route_after_verification(
    graph_state: GraphState,
    *,
    settings: Settings,
) -> RouteName:
    """Map verification outcome to a validated orchestration transition."""
    qa = graph_state["qa"]
    result = qa.verification
    route: RouteName

    if result is None:
        route = "fallback"
    elif result.passed or result.recommendation == VerificationRecommendation.ACCEPT:
        route = "finish"
    elif result.recommendation == VerificationRecommendation.CLARIFY:
        route = "clarify"
    elif result.recommendation == VerificationRecommendation.FALLBACK:
        route = "fallback"
    elif result.recommendation == VerificationRecommendation.RETRIEVE_MORE:
        if _can_retrieve_more(qa, settings):
            route = "retrieve_more"
        else:
            route = "fallback"
    else:
        route = "fallback"

    if route not in ALLOWED_ROUTES:
        route = "fallback"

    log_agent_event(
        logger,
        "agent.route",
        request_id=graph_state.get("request_id"),
        conversation_id=qa.conversation_id,
        question=qa.question,
        route=route,
        retrieval_rounds=qa.retrieval_rounds,
        verification_attempts=qa.verification_attempts,
        iteration=qa.iteration,
        tool_calls=qa.tool_calls,
    )
    return route


def _can_retrieve_more(qa, settings: Settings) -> bool:
    if qa.retrieval_rounds >= settings.agent_max_retrieval_rounds:
        return False
    if qa.iteration >= settings.agent_max_iterations:
        return False
    if qa.tool_calls >= settings.agent_max_tool_calls:
        return False
    return True


def validate_route(route: str) -> RouteName:
    if route not in ALLOWED_ROUTES:
        raise ValueError(f"Invalid graph route: {route!r}")
    return route  # type: ignore[return-value]
