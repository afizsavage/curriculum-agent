"""Explicit graph nodes — thin adapters over existing domain services.

LangGraph coordinates; domain logic stays in RetrievalNode, AnswerGenerationNode,
VerificationNode, and the understand helpers.
"""

from __future__ import annotations

from typing import Any

from app.agent.answer import AnswerGenerationNode
from app.agent.context import ConversationContext
from app.agent.graph_state import GraphState, mark_visited
from app.agent.retrieve import RetrievalNode
from app.agent.state import CurriculumQAState
from app.agent.verify import VerificationNode
from app.config import Settings
from app.curriculum.codes import extract_filters_from_question, normalize_grade_code
from app.enums import AgentStatus
from app.logging_utils import get_logger, log_agent_event
from app.schemas.answer import AnswerConfidence
from app.schemas.verification import (
    VerificationRecommendation,
    VerificationResult,
    VerificationStatus,
)

logger = get_logger(__name__)

FALLBACK_ANSWER = (
    "I couldn't find sufficient MBSSE curriculum evidence to answer "
    "this reliably. The available curriculum records did not clearly "
    "establish the specific information requested."
)


class GraphNodes:
    """Node callables bound to injected domain services."""

    def __init__(
        self,
        *,
        settings: Settings,
        retrieval: RetrievalNode,
        answer_node: AnswerGenerationNode,
        verification_node: VerificationNode,
        conversation: ConversationContext | None = None,
    ) -> None:
        self.settings = settings
        self.retrieval = retrieval
        self.answer_node = answer_node
        self.verification_node = verification_node
        self.conversation = conversation

    def bind_conversation(self, conversation: ConversationContext | None) -> None:
        self.conversation = conversation

    def understand(self, graph_state: GraphState) -> dict[str, Any]:
        qa = graph_state["qa"]
        qa.status = AgentStatus.UNDERSTANDING
        filters = extract_filters_from_question(qa.question)
        prior = graph_state.get("prior_filters") or {}

        if prior:
            for field in ("grade", "subject", "level", "topic"):
                new_value = filters.get(field)
                if new_value:
                    setattr(qa, field, new_value)
                elif prior.get(field):
                    setattr(qa, field, prior[field])
        else:
            qa.grade = filters.get("grade") or qa.grade
            qa.subject = filters.get("subject") or qa.subject
            qa.level = filters.get("level") or qa.level
            qa.topic = filters.get("topic") or qa.topic

        explicit_grade = normalize_grade_code(qa.question)
        if explicit_grade:
            qa.grade = explicit_grade
            qa.level = filters.get("level") or qa.level

        qa.intent = qa.intent or "retrieve_curriculum"
        return {"qa": qa, "visited_nodes": mark_visited(graph_state, "understand")}

    def prepare_cycle(self, graph_state: GraphState) -> dict[str, Any]:
        """Bump iteration counters before retrieve; flag limit exhaustion."""
        qa = graph_state["qa"]
        settings = self.settings
        hit = False
        reason: str | None = None

        if qa.retrieval_rounds >= settings.agent_max_retrieval_rounds:
            hit = True
            reason = "max_retrieval_rounds"
        elif qa.iteration >= settings.agent_max_iterations:
            hit = True
            reason = "max_iterations"
        elif qa.tool_calls >= settings.agent_max_tool_calls:
            hit = True
            reason = "max_tool_calls"
        else:
            qa.bump_iteration()
            qa.retrieval_rounds += 1

        return {
            "qa": qa,
            "max_iterations_hit": hit,
            "fallback_reason": reason,
            "visited_nodes": mark_visited(graph_state, "prepare_cycle"),
        }

    def retrieve(self, graph_state: GraphState) -> dict[str, Any]:
        qa = self.retrieval.run(
            graph_state["qa"],
            request_id=graph_state.get("request_id"),
        )
        return {"qa": qa, "visited_nodes": mark_visited(graph_state, "retrieve")}

    def generate_answer(self, graph_state: GraphState) -> dict[str, Any]:
        qa = self.answer_node.run(
            graph_state["qa"],
            conversation=self.conversation,
            request_id=graph_state.get("request_id"),
        )
        return {
            "qa": qa,
            "visited_nodes": mark_visited(graph_state, "generate_answer"),
        }

    def verify_answer(self, graph_state: GraphState) -> dict[str, Any]:
        qa = self.verification_node.run(
            graph_state["qa"],
            request_id=graph_state.get("request_id"),
        )
        return {
            "qa": qa,
            "visited_nodes": mark_visited(graph_state, "verify_answer"),
        }

    def clarify(self, graph_state: GraphState) -> dict[str, Any]:
        qa = apply_clarification(graph_state["qa"])
        return {
            "qa": qa,
            "visited_nodes": mark_visited(graph_state, "clarify"),
            "route": "clarify",
            "prior_filters": filters_from_state(qa),
        }

    def fallback(self, graph_state: GraphState) -> dict[str, Any]:
        reason = graph_state.get("fallback_reason") or _infer_fallback_reason(
            graph_state["qa"], self.settings
        )
        qa = apply_fallback(
            graph_state["qa"],
            reason=reason,
            request_id=graph_state.get("request_id"),
        )
        hit = graph_state.get("max_iterations_hit", False) or reason in (
            "max_iterations",
            "max_retrieval_rounds",
            "max_tool_calls",
        )
        return {
            "qa": qa,
            "visited_nodes": mark_visited(graph_state, "fallback"),
            "route": "fallback",
            "fallback_reason": reason,
            "max_iterations_hit": hit,
            "prior_filters": filters_from_state(qa),
        }

    def finish(self, graph_state: GraphState) -> dict[str, Any]:
        qa = graph_state["qa"]
        qa.status = AgentStatus.COMPLETED
        qa.verification_status = VerificationStatus.PASSED
        return {
            "qa": qa,
            "visited_nodes": mark_visited(graph_state, "finish"),
            "route": "finish",
            "prior_filters": filters_from_state(qa),
        }


def _infer_fallback_reason(qa: CurriculumQAState, settings: Settings) -> str:
    if qa.retrieval_rounds >= settings.agent_max_retrieval_rounds:
        return "max_retrieval_rounds"
    if qa.iteration >= settings.agent_max_iterations:
        return "max_iterations"
    if qa.tool_calls >= settings.agent_max_tool_calls:
        return "max_tool_calls"
    return "verification_fallback"


def apply_clarification(state: CurriculumQAState) -> CurriculumQAState:
    clarification = None
    if state.verification and state.verification.clarification:
        clarification = state.verification.clarification
    clarification = clarification or (
        "Which grade or level would you like me to check?"
    )
    state.clarification = clarification
    state.final_answer = None
    state.draft_answer = None
    state.answer_confidence = AnswerConfidence.LOW
    state.answer_limitations = list(
        dict.fromkeys(
            (state.answer_limitations or [])
            + ["Question requires clarification before a grounded answer."]
        )
    )
    state.status = AgentStatus.NEEDS_CLARIFICATION
    state.verification_status = VerificationStatus.NEEDS_CLARIFICATION
    return state


def apply_fallback(
    state: CurriculumQAState,
    *,
    reason: str,
    request_id: str | None = None,
) -> CurriculumQAState:
    limitations = list(state.answer_limitations or [])
    if state.verification:
        limitations.extend(state.verification.issues)
        for item in state.verification.missing_evidence:
            if isinstance(item, str):
                limitations.append(item)
            elif item.detail:
                limitations.append(item.detail)
    limitations.append(
        "Available MBSSE curriculum records were insufficient for a reliable answer."
    )
    limitations = list(dict.fromkeys(x for x in limitations if x))

    found_hint = ""
    names = [e.name for e in state.evidence if e.name][:5]
    if names:
        found_hint = (
            f" I found related records ({', '.join(names)}), but they did not "
            "clearly establish the specific placement or claim requested."
        )

    state.final_answer = FALLBACK_ANSWER + found_hint
    state.draft_answer = state.draft_answer or state.final_answer
    state.answer_confidence = AnswerConfidence.LOW
    state.answer_limitations = limitations
    state.status = AgentStatus.INSUFFICIENT_EVIDENCE
    if reason in ("max_iterations", "max_retrieval_rounds", "max_tool_calls"):
        state.verification_status = VerificationStatus.MAX_ITERATIONS
    else:
        state.verification_status = VerificationStatus.INSUFFICIENT_EVIDENCE
    state.metadata["fallback_reason"] = reason

    if state.verification is None:
        state.verification = VerificationResult(
            passed=False,
            score=0.0,
            issues=limitations[:3],
            recommendation=VerificationRecommendation.FALLBACK,
            metadata={"reason": reason},
        )
        state.verification_history.append(state.verification)

    log_agent_event(
        logger,
        "agent.fallback",
        request_id=request_id,
        conversation_id=state.conversation_id,
        question=state.question,
        status=state.status.value,
        reason=reason,
        verification_status=state.verification_status.value,
        retrieval_rounds=state.retrieval_rounds,
        iteration=state.iteration,
        tool_calls=state.tool_calls,
    )
    return state


def filters_from_state(state: CurriculumQAState) -> dict[str, Any]:
    return {
        "grade": state.grade,
        "subject": state.subject,
        "level": state.level,
        "topic": state.topic,
    }
