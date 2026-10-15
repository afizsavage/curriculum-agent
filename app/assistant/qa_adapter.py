"""In-process adapter over CurriculumQAAgent.ask.

The adapter does not call the agent HTTP API. A returned state is mapped with
the same response mapper as POST /api/v1/agent/ask, so the embedded
AskResponse keeps that route's answer, status, and metadata.

AskResponse.verification has no metadata field. coverage_decision is stored
only on VerificationResult.metadata (accept, reject, or unavailable). This
adapter copies that value onto CurriculumOutcome when it is present. It does
not add the field to AskResponse.

Working-context fields on AssistantTurnRequest are not ask() parameters.
Grade, subject, and task are left for the agent to recover from the question
and its own conversation checkpoint. Experiment flags are not forwarded.
"""

from __future__ import annotations

from typing import Protocol

from app.agent.response_mapper import map_graph_result_to_response
from app.agent.state import CurriculumQAState
from app.assistant.contract import (
    AssistantTurnRequest,
    CoverageDecision,
    CurriculumOutcome,
)
from app.llm.base import LLMProvider

_COVERAGE_DECISIONS = frozenset({"accept", "reject", "unavailable"})


class CurriculumAskAgent(Protocol):
    """The slice of CurriculumQAAgent this adapter calls."""

    llm: LLMProvider

    def ask(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        request_id: str | None = None,
    ) -> CurriculumQAState: ...


class CurriculumQAAdapter:
    """QAHandler that invokes an existing curriculum QA agent in process."""

    def __init__(self, agent: CurriculumAskAgent) -> None:
        self.agent = agent

    def __call__(self, request: AssistantTurnRequest) -> CurriculumOutcome:
        state = self.agent.ask(
            request.question,
            conversation_id=request.conversation_id,
            request_id=request.request_id,
        )
        return outcome_from_state(state, llm=self.agent.llm)


def outcome_from_state(
    state: CurriculumQAState,
    *,
    llm: LLMProvider,
) -> CurriculumOutcome:
    """Map a completed ask() state. Exceptions are left to the caller."""
    response = map_graph_result_to_response(
        qa=state,
        llm=llm,
        graph_state=public_ask_graph_state(state),
    )
    return CurriculumOutcome(
        response=response,
        coverage_decision=coverage_decision_from_state(state),
        http_status=200,
    )


def public_ask_graph_state(state: CurriculumQAState) -> dict:
    """Envelope matching app.api.v1.agent.ask's mapper arguments."""
    return {
        "qa": state,
        "graph_run_id": state.metadata.get("graph_run_id"),
        "visited_nodes": list(state.metadata.get("visited_nodes") or []),
        "route": state.metadata.get("route"),
        "max_iterations_hit": bool(state.metadata.get("max_iterations_hit")),
    }


def coverage_decision_from_state(
    state: CurriculumQAState,
) -> CoverageDecision | None:
    """Return a known coverage decision, or None when the verifier did not set one."""
    verification = state.verification
    if verification is None:
        return None
    raw = (verification.metadata or {}).get("coverage_decision")
    if raw in _COVERAGE_DECISIONS:
        return raw  # type: ignore[return-value]
    return None
