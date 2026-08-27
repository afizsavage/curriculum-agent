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
    reason: str | None = None

    if result is None:
        route = "fallback"
        reason = "missing_verification_result"
    elif result.passed or result.recommendation == VerificationRecommendation.ACCEPT:
        route = "finish"
        reason = "verification_passed"
    elif result.recommendation == VerificationRecommendation.CLARIFY:
        route = "clarify"
        reason = "needs_clarification"
    elif result.recommendation == VerificationRecommendation.FALLBACK:
        route = "fallback"
        reason = "verification_fallback"
    elif result.recommendation == VerificationRecommendation.RETRIEVE_MORE:
        if _can_retrieve_more(qa, settings):
            route = "retrieve_more"
            reason = "missing_evidence"
        else:
            route = "fallback"
            reason = "retrieve_more_limits_exhausted"
    else:
        route = "fallback"
        reason = "unknown_recommendation"

    if route not in ALLOWED_ROUTES:
        route = "fallback"
        reason = "invalid_route"

    next_node = {
        "finish": "finish",
        "retrieve_more": "prepare_cycle",
        "clarify": "clarify",
        "fallback": "fallback",
    }[route]

    from app.agent.trace import get_current_trace

    trace = get_current_trace()
    if trace is not None:
        it = trace.ensure_iteration(qa.iteration)
        route_row = {
            "from_node": "verify_answer",
            "decision": route,
            "next_node": next_node,
            "iteration": qa.iteration,
            "reason": reason,
            "missing_evidence": [
                m.model_dump() if hasattr(m, "model_dump") else m
                for m in ((result.missing_evidence if result else []) or [])
            ][:10],
        }
        it["route"] = route_row
        trace.routes.append(route_row)
        trace.emit(
            "agent.route",
            **route_row,
            retrieval_rounds=qa.retrieval_rounds,
            verification_attempts=qa.verification_attempts,
            tool_calls=qa.tool_calls,
        )

    log_agent_event(
        logger,
        "agent.route",
        request_id=graph_state.get("request_id"),
        conversation_id=qa.conversation_id,
        question=qa.question,
        route=route,
        from_node="verify_answer",
        next_node=next_node,
        reason=reason,
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
