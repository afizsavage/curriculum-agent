"""Explicit graph nodes — thin adapters over existing domain services.

LangGraph coordinates; domain logic stays in RetrievalNode, AnswerGenerationNode,
VerificationNode, and the understand helpers.
"""

from __future__ import annotations

import time
from typing import Any

from app.agent.answer import AnswerGenerationNode
from app.agent.context import ConversationContext
from app.agent.graph_state import GraphState, mark_visited
from app.agent.retrieve import RetrievalNode
from app.agent.state import CurriculumQAState
from app.agent.trace import get_current_trace, timed_ms
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
        started = time.perf_counter()
        trace = get_current_trace()
        if trace is not None:
            trace.emit("agent.node.start", node="understand", iteration=0)
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
        duration_ms = timed_ms(started)
        intent_payload = {
            "intent": qa.intent,
            "level": qa.level,
            "grade": qa.grade,
            "subject": qa.subject,
            "topic": qa.topic,
            "requested_information": filters,
            "raw_filters": filters,
            "prior_filters": prior,
        }
        if trace is not None:
            trace.add_node_timing("understand", duration_ms)
            trace.emit(
                "agent.node.end",
                node="understand",
                iteration=0,
                duration_ms=round(duration_ms, 2),
                **intent_payload,
            )
        return {"qa": qa, "visited_nodes": mark_visited(graph_state, "understand")}

    def prepare_cycle(self, graph_state: GraphState) -> dict[str, Any]:
        """Bump iteration counters before retrieve; flag limit exhaustion."""
        started = time.perf_counter()
        trace = get_current_trace()
        if trace is not None:
            trace.emit(
                "agent.node.start",
                node="prepare_cycle",
                iteration=graph_state["qa"].iteration,
            )
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

        if trace is not None:
            trace.ensure_iteration(qa.iteration)
            trace.emit(
                "agent.loop",
                iteration=qa.iteration,
                max_iterations=settings.agent_max_iterations,
                tool_calls_so_far=qa.tool_calls,
                max_tool_calls=settings.agent_max_tool_calls,
                retrieval_round=qa.retrieval_rounds,
                max_retrieval_rounds=settings.agent_max_retrieval_rounds,
                max_iterations_hit=hit,
                fallback_reason=reason,
            )
            duration_ms = timed_ms(started)
            trace.add_node_timing("prepare_cycle", duration_ms)
            trace.emit(
                "agent.node.end",
                node="prepare_cycle",
                iteration=qa.iteration,
                duration_ms=round(duration_ms, 2),
                max_iterations_hit=hit,
                fallback_reason=reason,
            )

        return {
            "qa": qa,
            "max_iterations_hit": hit,
            "fallback_reason": reason,
            "visited_nodes": mark_visited(graph_state, "prepare_cycle"),
        }

    def retrieve(self, graph_state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        trace = get_current_trace()
        qa = graph_state["qa"]
        if trace is not None:
            trace.emit(
                "agent.node.start", node="retrieve", iteration=qa.iteration
            )
        qa = self.retrieval.run(
            qa,
            request_id=graph_state.get("request_id"),
        )
        if trace is not None:
            duration_ms = timed_ms(started)
            trace.add_node_timing("retrieve", duration_ms)
            it = trace.ensure_iteration(qa.iteration)
            it["nodes"].append("retrieve")
            trace.emit(
                "agent.node.end",
                node="retrieve",
                iteration=qa.iteration,
                duration_ms=round(duration_ms, 2),
                evidence_count=len(qa.evidence),
                tool_calls=qa.tool_calls,
            )
        return {"qa": qa, "visited_nodes": mark_visited(graph_state, "retrieve")}

    def generate_answer(self, graph_state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        trace = get_current_trace()
        qa = graph_state["qa"]
        if hasattr(self.answer_node.generator.llm, "set_active_node"):
            self.answer_node.generator.llm.set_active_node("generate_answer")
        if trace is not None:
            trace.emit(
                "agent.node.start",
                node="generate_answer",
                iteration=qa.iteration,
            )
            trace.emit(
                "agent.generation.start",
                iteration=qa.iteration,
                evidence_count=len(qa.evidence),
                question=qa.question,
            )
        qa = self.answer_node.run(
            qa,
            conversation=self.conversation,
            request_id=graph_state.get("request_id"),
        )
        if trace is not None:
            duration_ms = timed_ms(started)
            trace.add_node_timing("generate_answer", duration_ms)
            it = trace.ensure_iteration(qa.iteration)
            it["nodes"].append("generate_answer")
            gen = {
                "duration_ms": round(duration_ms, 2),
                "answer_length": len(qa.final_answer or qa.draft_answer or ""),
                "confidence": (
                    qa.answer_confidence.value if qa.answer_confidence else None
                ),
                "evidence_references": [
                    ref.entity_id for ref in (qa.answer_evidence or [])
                ],
                "retrieved_evidence_count": len(qa.evidence),
                "generation_evidence_count": qa.metadata.get(
                    "generation_evidence_count"
                ),
                "generation_evidence_ids": qa.metadata.get("generation_evidence_ids"),
            }
            it["generation"] = gen
            trace.emit("agent.generation.end", iteration=qa.iteration, **gen)
            trace.emit(
                "agent.node.end",
                node="generate_answer",
                iteration=qa.iteration,
                duration_ms=round(duration_ms, 2),
            )
        return {
            "qa": qa,
            "visited_nodes": mark_visited(graph_state, "generate_answer"),
        }

    def verify_answer(self, graph_state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        trace = get_current_trace()
        qa = graph_state["qa"]
        if hasattr(self.verification_node.verifier.llm, "set_active_node"):
            self.verification_node.verifier.llm.set_active_node("verify_answer")
        if trace is not None:
            evidence_ids = [e.entity_id for e in qa.evidence if e.entity_id]
            trace.emit(
                "agent.node.start",
                node="verify_answer",
                iteration=qa.iteration,
            )
            trace.emit(
                "agent.verification.start",
                iteration=qa.iteration,
                answer=(qa.final_answer or qa.draft_answer or "")[:2000],
                evidence_count=len(qa.evidence),
                evidence_ids=evidence_ids[:40],
                evidence_summary=[
                    {
                        "id": e.entity_id,
                        "type": e.entity_type,
                        "name": e.name,
                        "grade": e.grade,
                        "subject": e.subject,
                    }
                    for e in qa.evidence[:20]
                ],
                question=qa.question,
            )
        qa = self.verification_node.run(
            qa,
            request_id=graph_state.get("request_id"),
        )
        if trace is not None:
            duration_ms = timed_ms(started)
            trace.add_node_timing("verify_answer", duration_ms)
            it = trace.ensure_iteration(qa.iteration)
            it["nodes"].append("verify_answer")
            result = qa.verification
            ver = {
                "passed": result.passed if result else None,
                "score": result.score if result else None,
                "status": qa.verification_status.value,
                "recommendation": (
                    result.recommendation.value if result else None
                ),
                "issues": list(result.issues) if result else [],
                "unsupported_claims": list(result.unsupported_claims)
                if result
                else [],
                "incorrect_claims": list(result.incorrect_claims) if result else [],
                "missing_evidence": [
                    m.model_dump() if hasattr(m, "model_dump") else m
                    for m in (result.missing_evidence if result else [])
                ],
                "duration_ms": round(duration_ms, 2),
            }
            it["verification"] = ver
            trace.emit("agent.verification.end", iteration=qa.iteration, **ver)
            trace.emit(
                "agent.node.end",
                node="verify_answer",
                iteration=qa.iteration,
                duration_ms=round(duration_ms, 2),
            )
        return {
            "qa": qa,
            "visited_nodes": mark_visited(graph_state, "verify_answer"),
        }

    def clarify(self, graph_state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        trace = get_current_trace()
        if trace is not None:
            trace.emit(
                "agent.node.start",
                node="clarify",
                iteration=graph_state["qa"].iteration,
            )
        qa = apply_clarification(graph_state["qa"])
        if trace is not None:
            duration_ms = timed_ms(started)
            trace.add_node_timing("clarify", duration_ms)
            trace.emit(
                "agent.node.end",
                node="clarify",
                iteration=qa.iteration,
                duration_ms=round(duration_ms, 2),
            )
        return {
            "qa": qa,
            "visited_nodes": mark_visited(graph_state, "clarify"),
            "route": "clarify",
            "prior_filters": filters_from_state(qa),
        }

    def fallback(self, graph_state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        trace = get_current_trace()
        if trace is not None:
            trace.emit(
                "agent.node.start",
                node="fallback",
                iteration=graph_state["qa"].iteration,
            )
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
        if trace is not None:
            duration_ms = timed_ms(started)
            trace.add_node_timing("fallback", duration_ms)
            trace.emit(
                "agent.node.end",
                node="fallback",
                iteration=qa.iteration,
                duration_ms=round(duration_ms, 2),
                fallback_reason=reason,
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
        started = time.perf_counter()
        trace = get_current_trace()
        if trace is not None:
            trace.emit(
                "agent.node.start",
                node="finish",
                iteration=graph_state["qa"].iteration,
            )
        qa = graph_state["qa"]
        qa.status = AgentStatus.COMPLETED
        qa.verification_status = VerificationStatus.PASSED
        if trace is not None:
            duration_ms = timed_ms(started)
            trace.add_node_timing("finish", duration_ms)
            trace.emit(
                "agent.node.end",
                node="finish",
                iteration=qa.iteration,
                duration_ms=round(duration_ms, 2),
            )
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
