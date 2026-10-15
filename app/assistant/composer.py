"""Compose an assistant turn without merging provenance.

Curriculum sentences stay on the curriculum channel. General teaching help
stays on the general channel and is labelled as general AI assistance.
"""

from __future__ import annotations

from app.assistant.contract import (
    AssistantRoute,
    AssistantTurnResponse,
    CurriculumChannel,
    CurriculumOutcome,
    GeneralChannel,
    GeneralExecutionFailure,
    RoutingResult,
)
from app.assistant.router import CLARIFICATION_PROMPT
from app.enums import AgentStatus

_GAP_STATUSES = frozenset({AgentStatus.INSUFFICIENT_EVIDENCE.value})
_TECHNICAL_STATUSES = frozenset({AgentStatus.FAILED.value, AgentStatus.ERROR.value})
_INCOMPLETE_COVERAGE = frozenset({"reject", "unavailable"})


def blocks_general_advice(outcome: CurriculumOutcome | None) -> str | None:
    """Reason general prose must not be attached, or None when it may be."""
    if outcome is None:
        return "missing_curriculum_result"
    if outcome.failure is not None:
        return f"technical_failure:{outcome.failure.kind}"
    response = outcome.response
    if response is None:
        return "missing_curriculum_result"
    evidence_status = (response.metadata.evidence_status or "").strip().lower()
    if evidence_status == "error":
        return "evidence_status_error"
    if response.status in _TECHNICAL_STATUSES:
        return "technical_failure"
    if response.status in _GAP_STATUSES:
        return "insufficient_evidence"
    if response.status == AgentStatus.NEEDS_CLARIFICATION.value:
        return "needs_clarification"
    if outcome.coverage_decision in _INCOMPLETE_COVERAGE:
        return "incomplete_coverage"
    if evidence_status == "partial":
        return "partial_evidence"
    if evidence_status == "not_found":
        return "evidence_not_found"
    if response.status != AgentStatus.COMPLETED.value:
        return f"curriculum_status:{response.status}"
    if evidence_status != "found":
        return "evidence_not_complete"
    return None


def compose_turn(
    *,
    routing: RoutingResult,
    qa: CurriculumOutcome | None = None,
    general_text: str | None = None,
    general_failure: GeneralExecutionFailure | None = None,
    clarification: str | None = None,
) -> AssistantTurnResponse:
    """Build a response for a route that has already been validated."""
    route = routing.route
    if route == AssistantRoute.CLARIFY:
        suppressed = "route_clarify" if _has_text(general_text) or qa is not None else None
        return AssistantTurnResponse(
            route=route,
            routing=routing,
            clarification=clarification or CLARIFICATION_PROMPT,
            general_suppressed_reason=suppressed,
        )

    if route == AssistantRoute.GENERAL:
        return AssistantTurnResponse(
            route=route,
            routing=routing,
            general=_general_channel(general_text) if general_failure is None else None,
            general_failure=general_failure,
            general_suppressed_reason=_general_route_suppressed(
                general_text, general_failure, qa
            ),
        )

    curriculum = _curriculum_channel(qa) if qa is not None else None
    clarification_out = _qa_clarification(qa)
    if route == AssistantRoute.CURRICULUM:
        suppressed = "route_curriculum" if _has_text(general_text) else None
        return AssistantTurnResponse(
            route=route,
            routing=routing,
            curriculum=curriculum,
            clarification=clarification_out,
            general_failure=None,
            general_suppressed_reason=suppressed,
        )

    blocked = blocks_general_advice(qa)
    if blocked is not None:
        return AssistantTurnResponse(
            route=route,
            routing=routing,
            curriculum=curriculum,
            clarification=clarification_out,
            general_failure=general_failure,
            general_suppressed_reason=blocked,
        )
    if general_failure is not None:
        return AssistantTurnResponse(
            route=route,
            routing=routing,
            curriculum=curriculum,
            clarification=clarification_out,
            general_failure=general_failure,
            general_suppressed_reason="general_generation_failed",
        )
    return AssistantTurnResponse(
        route=route,
        routing=routing,
        curriculum=curriculum,
        general=_general_channel(general_text),
        clarification=clarification_out,
        general_suppressed_reason=(
            None if _has_text(general_text) else "empty_general_answer"
        ),
    )


def _curriculum_channel(outcome: CurriculumOutcome) -> CurriculumChannel:
    response = (
        outcome.response.model_copy(deep=True) if outcome.response is not None else None
    )
    failure = (
        outcome.failure.model_copy(deep=True) if outcome.failure is not None else None
    )
    evidence_status = ""
    status = ""
    if response is not None:
        evidence_status = (response.metadata.evidence_status or "").strip().lower()
        status = response.status
    technical = failure is not None or evidence_status == "error" or status in _TECHNICAL_STATUSES
    http_status = outcome.http_status
    if http_status is None and failure is not None:
        http_status = failure.http_status
    return CurriculumChannel(
        response=response,
        failure=failure,
        coverage_decision=outcome.coverage_decision,
        http_status=http_status,
        technical_failure=technical,
    )


def _general_channel(text: str | None) -> GeneralChannel | None:
    if not _has_text(text):
        return None
    return GeneralChannel(text=text.strip())


def _general_route_suppressed(
    general_text: str | None,
    general_failure: GeneralExecutionFailure | None,
    qa: CurriculumOutcome | None,
) -> str | None:
    if qa is not None:
        return "route_general"
    if general_failure is not None:
        return "general_generation_failed"
    if not _has_text(general_text):
        return "empty_general_answer"
    return None


def _qa_clarification(outcome: CurriculumOutcome | None) -> str | None:
    if outcome is None or outcome.response is None:
        return None
    if outcome.response.status != AgentStatus.NEEDS_CLARIFICATION.value:
        return None
    return outcome.response.clarification


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())
