"""Curriculum Q&A agent — LangGraph orchestration over domain services."""

from __future__ import annotations

import time
from typing import Any

from app.agent.answer import AnswerGenerationNode
from app.agent.context import ConversationStore
from app.agent.graph import build_curriculum_qa_graph, graph_ascii, graph_mermaid
from app.agent.graph_nodes import (
    FALLBACK_ANSWER,
    GraphNodes,
    apply_clarification,
    apply_fallback,
    filters_from_state,
)
from app.agent.graph_routing import route_after_verification as graph_route_after_verification
from app.agent.graph_state import GraphState, initial_graph_input
from app.agent.memory import build_checkpointer, thread_config
from app.agent.metrics import get_metrics
from app.agent.response_mapper import attach_graph_metadata
from app.agent.retrieve import RetrievalNode
from app.agent.state import CurriculumQAState
from app.agent.trace import (
    bind_conversation_id,
    finish_agent_run,
    start_agent_run,
)
from app.agent.verify import VerificationNode
from app.config import Settings, get_settings
from app.curriculum.evidence import EvidenceStatus
from app.exceptions import AgentError, AgentExecutionError, InvalidRequestError
from app.llm.base import LLMProvider
from app.llm.provider import build_llm_provider
from app.llm.tracing import wrap_llm
from app.logging_utils import get_logger, log_agent_event
from app.tools.registry import ToolRegistry, build_default_registry

logger = get_logger(__name__)

__all__ = ["CurriculumQAAgent", "FALLBACK_ANSWER"]


class CurriculumQAAgent:
    """Invokes the compiled Curriculum Q&A LangGraph for each ask turn."""

    name = "curriculum_qa"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm: LLMProvider | None = None,
        verifier_llm: LLMProvider | None = None,
        tools: ToolRegistry | None = None,
        conversations: ConversationStore | None = None,
        verification_node: VerificationNode | None = None,
        checkpointer: Any | None = ...,
        compiled_graph: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        base_llm = llm or build_llm_provider(self.settings)
        self.llm = wrap_llm(base_llm)

        verifier_model = (self.settings.verifier_llm_model or "").strip()
        if verifier_llm is not None:
            self.verifier_llm = wrap_llm(verifier_llm)
        elif not verifier_model or verifier_model == self.settings.llm_model:
            self.verifier_llm = self.llm
        else:
            override = self.settings.model_copy(update={"llm_model": verifier_model})
            self.verifier_llm = wrap_llm(build_llm_provider(override))

        self.tools = (
            tools
            if tools is not None
            else build_default_registry(settings=self.settings, include_echo=False)
        )
        self.conversations = conversations or ConversationStore()
        self.retrieval = RetrievalNode(
            llm=self.llm, tools=self.tools, settings=self.settings
        )
        self.answer_node = AnswerGenerationNode(llm=self.llm, settings=self.settings)
        self.verification_node = verification_node or VerificationNode(
            llm=self.verifier_llm, settings=self.settings
        )
        self.nodes = GraphNodes(
            settings=self.settings,
            retrieval=self.retrieval,
            answer_node=self.answer_node,
            verification_node=self.verification_node,
        )
        if checkpointer is ...:
            self.checkpointer = build_checkpointer(self.settings)
        else:
            self.checkpointer = checkpointer
        self.graph = compiled_graph or build_curriculum_qa_graph(
            nodes=self.nodes,
            settings=self.settings,
            checkpointer=self.checkpointer,
        )

    def _build_verifier_llm(self) -> LLMProvider:
        """Legacy helper — prefers VERIFIER_LLM_MODEL when set."""
        verifier_model = (self.settings.verifier_llm_model or "").strip()
        if not verifier_model or verifier_model == self.settings.llm_model:
            return self.llm
        override = self.settings.model_copy(update={"llm_model": verifier_model})
        return wrap_llm(build_llm_provider(override))

    def ask(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        request_id: str | None = None,
        context_boundary_experiment: bool | None = None,
        v23_diagnostic_experiment: bool | None = None,
        generation_mode: str | None = None,
        v24_experiment_arm: str | None = None,
        v25_experiment_arm: str | None = None,
    ) -> CurriculumQAState:
        cleaned = (question or "").strip()
        if not cleaned:
            raise InvalidRequestError("question must not be blank")

        started = time.perf_counter()
        max_iterations_hit = False
        with start_agent_run(
            question=cleaned,
            request_id=request_id,
            conversation_id=conversation_id,
            agent_name=self.name,
        ) as trace:
            try:
                context = self.conversations.get_or_create(conversation_id)
                bind_conversation_id(trace, context.conversation_id)
                prior_state = context.current_state
                prior_filters = filters_from_state(prior_state) if prior_state else {}
                if self.checkpointer is not None and context.conversation_id:
                    prior_filters = self._prior_filters_from_checkpoint(
                        context.conversation_id, fallback=prior_filters
                    )

                state = CurriculumQAState.initial(
                    question=cleaned,
                    conversation_id=context.conversation_id,
                )
                if context_boundary_experiment is not None:
                    state.metadata["context_boundary_experiment"] = (
                        context_boundary_experiment
                    )
                from app.agent.v23_diagnostics import configure_v23_experiment

                configure_v23_experiment(
                    state,
                    settings=self.settings,
                    generation_mode=generation_mode,
                    enabled=v23_diagnostic_experiment,
                )
                from app.agent.v24_diagnostics import configure_v24_experiment, finalize_v24_diagnostics

                if v24_experiment_arm is not None:
                    configure_v24_experiment(
                        state,
                        settings=self.settings,
                        arm=v24_experiment_arm,  # type: ignore[arg-type]
                    )
                if v25_experiment_arm is not None:
                    from app.agent.v25_experiment import (
                        configure_v25_experiment,
                        finalize_v25_diagnostics,
                    )

                    configure_v25_experiment(
                        state,
                        settings=self.settings,
                        arm=v25_experiment_arm,  # type: ignore[arg-type]
                    )
                context.append_user(cleaned)
                self.nodes.bind_conversation(context)

                graph_input = initial_graph_input(
                    qa=state,
                    request_id=request_id,
                    prior_filters=prior_filters,
                )
                invoke_kwargs: dict[str, Any] = {}
                if self.checkpointer is not None:
                    invoke_kwargs["config"] = thread_config(
                        context.conversation_id, request_id=request_id
                    )

                result: GraphState = self.graph.invoke(graph_input, **invoke_kwargs)
                state = attach_graph_metadata(result["qa"], result)
                max_iterations_hit = bool(result.get("max_iterations_hit"))
                state.metadata["agent_run_id"] = trace.agent_run_id

                context.set_state(state)
                if state.final_answer:
                    context.append_assistant(state.final_answer)
                elif state.clarification:
                    context.append_assistant(state.clarification)
                self.conversations.save(context)

                latency_ms = (time.perf_counter() - started) * 1000
                verification_passed = (
                    state.verification.passed if state.verification else None
                )
                get_metrics().record_request(
                    status=state.status.value,
                    iterations=state.iteration,
                    tool_calls=state.tool_calls,
                    latency_ms=latency_ms,
                    verification_passed=verification_passed,
                    max_iterations=max_iterations_hit,
                    retrieval_failed=state.evidence_status == EvidenceStatus.ERROR,
                )

                visited = list(state.metadata.get("visited_nodes") or [])
                termination_reason = _termination_reason(
                    state=state,
                    result=result,
                    max_iterations_hit=max_iterations_hit,
                    settings=self.settings,
                )
                state.metadata["termination_reason"] = termination_reason
                if state.metadata.get("v24_experiment_arm"):
                    finalize_v24_diagnostics(state)
                if state.metadata.get("v25_experiment_arm"):
                    from app.agent.v25_experiment import finalize_v25_diagnostics

                    finalize_v25_diagnostics(state)
                finish_agent_run(
                    trace,
                    status=state.status.value,
                    final_node=visited[-1] if visited else None,
                    iteration=state.iteration,
                    tool_calls=state.tool_calls,
                    retrieval_rounds=state.retrieval_rounds,
                    verification_attempts=state.verification_attempts,
                    evidence_count=len(state.evidence),
                    latency_ms=latency_ms,
                    termination_reason=termination_reason,
                    verification_score=(
                        state.verification.score if state.verification else None
                    ),
                    visited_nodes=visited,
                    max_iterations=self.settings.agent_max_iterations,
                    max_tool_calls=self.settings.agent_max_tool_calls,
                    max_retrieval_rounds=self.settings.agent_max_retrieval_rounds,
                )

                log_agent_event(
                    logger,
                    "agent.turn.complete",
                    request_id=request_id,
                    conversation_id=state.conversation_id,
                    question=state.question,
                    status=state.status.value,
                    iteration=state.iteration,
                    tool_calls=state.tool_calls,
                    model=self.llm.model,
                    evidence_count=len(state.evidence),
                    evidence_status=state.evidence_status.value,
                    confidence=(
                        state.answer_confidence.value
                        if state.answer_confidence
                        else None
                    ),
                    verification_status=state.verification_status.value,
                    verification_attempts=state.verification_attempts,
                    retrieval_rounds=state.retrieval_rounds,
                    visited_nodes=visited,
                    graph_run_id=state.metadata.get("graph_run_id"),
                    agent_run_id=trace.agent_run_id,
                    termination_reason=termination_reason,
                    llm_counts=trace.llm_counts,
                    phase_timings_ms=trace.phase_timings_ms,
                    latency_ms=round(latency_ms, 2),
                )
                return state
            except AgentError as exc:
                get_metrics().record_request(
                    status="error",
                    iterations=0,
                    tool_calls=0,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
                trace.emit("agent.error", error=str(exc), error_type=type(exc).__name__)
                raise
            except Exception as exc:
                get_metrics().record_request(
                    status="error",
                    iterations=0,
                    tool_calls=0,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
                trace.emit("agent.error", error=str(exc), error_type=type(exc).__name__)
                raise AgentExecutionError(
                    "Agent failed while processing the question"
                ) from exc

    def _prior_filters_from_checkpoint(
        self, conversation_id: str, *, fallback: dict
    ) -> dict:
        try:
            snapshot = self.graph.get_state(thread_config(conversation_id))
            values = getattr(snapshot, "values", None) or {}
            prior = values.get("prior_filters")
            if isinstance(prior, dict) and any(prior.values()):
                return prior
        except Exception:
            logger.debug(
                "agent.checkpoint.prior_filters_unavailable",
                exc_info=True,
            )
        return fallback

    # --- Domain helpers retained for unit tests / direct node access ---

    def understand(
        self,
        state: CurriculumQAState,
        *,
        prior_state: CurriculumQAState | None = None,
    ) -> CurriculumQAState:
        prior_filters = filters_from_state(prior_state) if prior_state else {}
        result = self.nodes.understand(
            {
                "qa": state,
                "prior_filters": prior_filters,
                "visited_nodes": [],
            }
        )
        return result["qa"]

    def retrieve(
        self,
        state: CurriculumQAState,
        *,
        request_id: str | None = None,
    ) -> CurriculumQAState:
        return self.retrieval.run(state, request_id=request_id)

    def answer(
        self,
        state: CurriculumQAState,
        *,
        conversation=None,
        request_id: str | None = None,
    ) -> CurriculumQAState:
        return self.answer_node.run(
            state, conversation=conversation, request_id=request_id
        )

    def verify(
        self,
        state: CurriculumQAState,
        *,
        request_id: str | None = None,
    ) -> CurriculumQAState:
        return self.verification_node.run(state, request_id=request_id)

    def route(self, state: CurriculumQAState) -> str:
        return graph_route_after_verification(
            {"qa": state, "visited_nodes": []},
            settings=self.settings,
        )

    def within_limits(self, state: CurriculumQAState) -> bool:
        return (
            state.iteration < self.settings.agent_max_iterations
            and state.tool_calls < self.settings.agent_max_tool_calls
            and state.retrieval_rounds < self.settings.agent_max_retrieval_rounds
        )

    def can_retrieve_more(self, state: CurriculumQAState) -> bool:
        if state.retrieval_rounds >= self.settings.agent_max_retrieval_rounds:
            return False
        if state.iteration >= self.settings.agent_max_iterations:
            return False
        if state.tool_calls >= self.settings.agent_max_tool_calls:
            return False
        return True

    def _apply_clarification(self, state: CurriculumQAState) -> CurriculumQAState:
        return apply_clarification(state)

    def _apply_fallback(
        self,
        state: CurriculumQAState,
        *,
        reason: str,
        request_id: str | None = None,
    ) -> CurriculumQAState:
        return apply_fallback(state, reason=reason, request_id=request_id)

    def inspect_graph(self) -> str:
        """ASCII + Mermaid inspection for development."""
        return f"{graph_ascii(self.graph)}\n\n{graph_mermaid(self.graph)}"


def _termination_reason(
    *,
    state: CurriculumQAState,
    result: GraphState,
    max_iterations_hit: bool,
    settings: Settings,
) -> str:
    if state.status.value == "completed":
        return "verification_passed"
    if state.status.value == "needs_clarification":
        return "clarify"
    reason = result.get("fallback_reason") or state.metadata.get("fallback_reason")
    if reason:
        return str(reason)
    if max_iterations_hit:
        if state.tool_calls >= settings.agent_max_tool_calls:
            return "max_tool_calls"
        if state.retrieval_rounds >= settings.agent_max_retrieval_rounds:
            return "max_retrieval_rounds"
        return "max_iterations"
    return "insufficient_evidence"
