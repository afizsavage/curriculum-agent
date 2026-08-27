"""Map LangGraph final state to the public Agent API response shape."""

from __future__ import annotations

from typing import Any

from app.agent.graph_state import GraphState
from app.agent.state import CurriculumQAState
from app.enums import AgentStatus
from app.llm.base import LLMProvider
from app.schemas.agent import (
    AnswerEvidenceSummary,
    AskMetadata,
    AskResponse,
    EvidenceSummary,
    VerificationSummary,
)


def map_graph_result_to_response(
    *,
    qa: CurriculumQAState,
    llm: LLMProvider,
    graph_state: GraphState | None = None,
) -> AskResponse:
    """Convert domain + graph envelope state into AskResponse."""
    verification_summary = None
    if qa.verification is not None:
        verification_summary = VerificationSummary(
            passed=qa.verification.passed,
            score=qa.verification.score,
            recommendation=qa.verification.recommendation.value,
            issues=list(qa.verification.issues[:5]),
        )

    visited = list((graph_state or {}).get("visited_nodes") or [])
    graph_meta = {
        "graph_run_id": (graph_state or {}).get("graph_run_id"),
        "visited_nodes": visited,
        "route": (graph_state or {}).get("route"),
        "max_iterations_hit": bool((graph_state or {}).get("max_iterations_hit")),
    }
    # Keep graph internals out of the typed metadata fields; stash under tools/meta.
    tools_used = list(qa.metadata.get("tools_used") or qa.selected_tools)

    return AskResponse(
        conversation_id=qa.conversation_id or "",
        question=qa.question,
        answer=(
            None
            if qa.status == AgentStatus.NEEDS_CLARIFICATION
            else (qa.final_answer or qa.draft_answer)
        ),
        status=qa.status.value,
        clarification=qa.clarification,
        evidence=[
            EvidenceSummary(
                entity_type=item.entity_type,
                entity_id=item.entity_id,
                name=item.name,
                grade=item.grade,
                subject=item.subject,
                topic=item.topic,
            )
            for item in qa.evidence
        ],
        answer_evidence=[
            AnswerEvidenceSummary(
                entity_id=ref.entity_id,
                entity_type=ref.entity_type,
                claim=ref.claim,
                name=ref.name,
                grade=ref.grade,
                subject=ref.subject,
                topic=ref.topic,
            )
            for ref in qa.answer_evidence
        ],
        confidence=qa.answer_confidence,
        limitations=qa.answer_limitations,
        verification=verification_summary,
        metadata=AskMetadata(
            iterations=qa.iteration,
            tool_calls=qa.tool_calls,
            tools_used=tools_used,
            evidence_status=qa.evidence_status.value,
            evidence_count=len(qa.evidence),
            model=llm.model,
            provider=llm.name,
            retrieval_rounds=qa.retrieval_rounds,
            verification_attempts=qa.verification_attempts,
            verification_status=qa.verification_status.value,
            graph_run_id=graph_meta["graph_run_id"],
            visited_nodes=visited,
            agent_run_id=qa.metadata.get("agent_run_id"),
            termination_reason=qa.metadata.get("termination_reason")
            or (graph_state or {}).get("fallback_reason"),
        ),
        error=qa.error,
    )


def attach_graph_metadata(qa: CurriculumQAState, graph_state: GraphState) -> CurriculumQAState:
    """Copy safe graph execution fields onto CurriculumQAState.metadata."""
    qa.metadata["graph_run_id"] = graph_state.get("graph_run_id")
    qa.metadata["visited_nodes"] = list(graph_state.get("visited_nodes") or [])
    qa.metadata["route"] = graph_state.get("route")
    qa.metadata["max_iterations_hit"] = bool(graph_state.get("max_iterations_hit"))
    if graph_state.get("fallback_reason"):
        qa.metadata["fallback_reason"] = graph_state.get("fallback_reason")
        qa.metadata["termination_reason"] = graph_state.get("fallback_reason")
    return qa


def graph_execution_summary(graph_state: GraphState) -> dict[str, Any]:
    qa = graph_state["qa"]
    return {
        "graph_run_id": graph_state.get("graph_run_id"),
        "conversation_id": qa.conversation_id,
        "iteration": qa.iteration,
        "visited_nodes": list(graph_state.get("visited_nodes") or []),
        "tool_calls": qa.tool_calls,
        "retrieval_rounds": qa.retrieval_rounds,
        "verification_attempts": qa.verification_attempts,
        "final_status": qa.status.value,
        "route": graph_state.get("route"),
    }
