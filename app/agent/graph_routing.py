"""Constrained conditional routing for the Curriculum Q&A graph.

The graph — not the LLM — owns legal destinations.
"""

from __future__ import annotations

from typing import Literal

from app.agent.graph_state import ALLOWED_ROUTES, GraphState
from app.agent.context_boundary import (
    context_boundary_experiment_enabled,
    get_boundary,
    missing_evidence_covered_by_boundary,
)
from app.agent.retrieval_state import (
    has_credible_retrieval_path,
    is_incomplete_source_gap,
)
from app.config import Settings
from app.logging_utils import get_logger, log_agent_event
from app.schemas.verification import VerificationRecommendation

logger = get_logger(__name__)

RouteName = Literal["finish", "retrieve_more", "regenerate", "clarify", "fallback"]
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
        from app.agent.v24_diagnostics import emit_v24_route_event, single_pass_enabled

        if single_pass_enabled(qa):
            route = "fallback"
            reason = "v23_single_pass_complete"
            qa.metadata["termination_reason"] = "v23_single_pass"
        elif not _can_retrieve_more(qa, settings):
            route = "fallback"
            reason = "retrieve_more_limits_exhausted"
        elif _should_regenerate_with_boundary(qa, settings):
            route = "regenerate"
            reason = "evidence_already_present"
            qa.metadata["regeneration_without_retrieval"] = True
            qa.metadata["conservative_regeneration"] = True
            qa.metadata["evidence_already_present"] = True
            qa.metadata["retrieve_more_reason"] = (
                (result.issues or [None])[0]
                if result.issues
                else "verifier_requested_more"
            )
            qa.pending_missing_evidence = []
        elif not _has_retrieval_path(qa):
            # Do not burn another round on duplicates / incomplete re-fetch.
            qa.retrieval_state.no_progress = True
            qa.retrieval_state.no_progress_reason = (
                qa.retrieval_state.no_progress_reason or "no_credible_retrieval_path"
            )
            qa.metadata["no_retrieval_progress"] = True
            qa.metadata["fallback_reason"] = "no_retrieval_progress"
            if is_incomplete_source_gap(
                result.missing_evidence or qa.pending_missing_evidence,
                verification_issues=list(result.issues or []),
            ) and (qa.draft_answer or qa.final_answer):
                # Evidence-aware finalize with limitations (not max_rounds).
                route = "fallback"
                reason = "no_retrieval_progress_incomplete_source"
            else:
                route = "fallback"
                reason = "no_retrieval_progress"
        else:
            route = "retrieve_more"
            reason = "missing_evidence"
    else:
        route = "fallback"
        reason = "unknown_recommendation"

    if route not in ALLOWED_ROUTES:
        route = "fallback"
        reason = "invalid_route"

    if reason and reason.startswith("no_retrieval_progress"):
        qa.metadata["fallback_reason"] = "no_retrieval_progress"
        qa.metadata["termination_reason"] = reason

    next_node = {
        "finish": "finish",
        "retrieve_more": "prepare_cycle",
        "regenerate": "generate_answer",
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
            "retrieval_metrics": qa.retrieval_state.metrics_snapshot(),
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
    if qa.metadata.get("v24_experiment_arm"):
        from app.agent.v24_diagnostics import emit_v24_route_event

        emit_v24_route_event(qa, route=route, reason=reason, result=result)
    return route


def _should_regenerate_with_boundary(qa, settings: Settings) -> bool:
    if not context_boundary_experiment_enabled(settings, qa):
        return False
    boundary = get_boundary(qa.retrieval_state)
    pending = qa.pending_missing_evidence or (
        qa.verification.missing_evidence if qa.verification else []
    )
    if not missing_evidence_covered_by_boundary(
        pending,
        boundary=boundary,
        evidence=qa.evidence,
    ):
        return False
    return bool(boundary and boundary.context_resolved)


def _can_retrieve_more(qa, settings: Settings) -> bool:
    if qa.retrieval_rounds >= settings.agent_max_retrieval_rounds:
        return False
    if qa.iteration >= settings.agent_max_iterations:
        return False
    if qa.tool_calls >= settings.agent_max_tool_calls:
        return False
    return True


def _has_retrieval_path(qa) -> bool:
    """Whether retrieve_more can execute a non-duplicate, goal-directed call."""
    available = set(qa.selected_tools or []) | {
        "resolve_curriculum_context",
        "search_curriculum",
        "get_curriculum_structure",
        "get_topic",
        "get_learning_objectives",
        "get_subject",
    }
    pending = qa.pending_missing_evidence or (
        qa.verification.missing_evidence if qa.verification else []
    )
    issues = list(qa.verification.issues) if qa.verification else []
    # Without a structured gap, do not burn another LLM-planned round.
    if not pending:
        return False
    return has_credible_retrieval_path(
        retrieval_state=qa.retrieval_state,
        pending_missing=pending,
        available_tools=available,
        grade=qa.grade,
        subject=qa.subject or qa.retrieval_state.resolved_subject,
        topic=qa.topic,
        verification_issues=issues,
    )


def validate_route(route: str) -> RouteName:
    if route not in ALLOWED_ROUTES:
        raise ValueError(f"Invalid graph route: {route!r}")
    return route  # type: ignore[return-value]
