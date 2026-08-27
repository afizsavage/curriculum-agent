"""Curriculum Q&A agent orchestrator with bounded verify loop."""

from __future__ import annotations

import time

from app.agent.answer import AnswerGenerationNode
from app.agent.context import ConversationStore
from app.agent.metrics import get_metrics
from app.agent.retrieve import RetrievalNode
from app.agent.state import CurriculumQAState
from app.agent.verify import VerificationNode
from app.config import Settings, get_settings
from app.curriculum.codes import extract_filters_from_question, normalize_grade_code
from app.curriculum.evidence import EvidenceStatus
from app.enums import AgentStatus
from app.exceptions import AgentError, AgentExecutionError, InvalidRequestError
from app.llm.base import LLMProvider
from app.llm.provider import build_llm_provider
from app.logging_utils import get_logger, log_agent_event
from app.schemas.answer import AnswerConfidence
from app.schemas.verification import (
    VerificationRecommendation,
    VerificationResult,
    VerificationStatus,
)
from app.tools.registry import ToolRegistry, build_default_registry

logger = get_logger(__name__)

FALLBACK_ANSWER = (
    "I couldn't find sufficient MBSSE curriculum evidence to answer "
    "this reliably. The available curriculum records did not clearly "
    "establish the specific information requested."
)


class CurriculumQAAgent:
    """Orchestrates Understand → Retrieve → Generate → Verify (bounded loop)."""

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
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or build_llm_provider(self.settings)
        self.verifier_llm = verifier_llm or self._build_verifier_llm()
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

    def _build_verifier_llm(self) -> LLMProvider:
        """Use VERIFIER_LLM_MODEL when set; otherwise reuse the main provider."""
        verifier_model = (self.settings.verifier_llm_model or "").strip()
        if not verifier_model or verifier_model == self.settings.llm_model:
            return self.llm
        # Same provider/credentials, distinct model name when configured.
        override = self.settings.model_copy(update={"llm_model": verifier_model})
        return build_llm_provider(override)

    def ask(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        request_id: str | None = None,
    ) -> CurriculumQAState:
        cleaned = (question or "").strip()
        if not cleaned:
            raise InvalidRequestError("question must not be blank")

        started = time.perf_counter()
        max_iterations_hit = False
        try:
            context = self.conversations.get_or_create(conversation_id)
            prior_state = context.current_state
            state = CurriculumQAState.initial(
                question=cleaned,
                conversation_id=context.conversation_id,
            )
            context.append_user(cleaned)

            state = self.understand(state, prior_state=prior_state)

            while True:
                if state.retrieval_rounds >= self.settings.agent_max_retrieval_rounds:
                    max_iterations_hit = True
                    state = self._apply_fallback(
                        state,
                        reason="max_retrieval_rounds",
                        request_id=request_id,
                    )
                    break

                if state.iteration >= self.settings.agent_max_iterations:
                    max_iterations_hit = True
                    state = self._apply_fallback(
                        state,
                        reason="max_iterations",
                        request_id=request_id,
                    )
                    break

                state.bump_iteration()
                state.retrieval_rounds += 1
                state = self.retrieve(state, request_id=request_id)
                state = self.answer(
                    state,
                    conversation=context,
                    request_id=request_id,
                )
                state = self.verify(state, request_id=request_id)

                route = self.route(state)
                log_agent_event(
                    logger,
                    "agent.route",
                    request_id=request_id,
                    conversation_id=state.conversation_id,
                    question=state.question,
                    route=route,
                    retrieval_rounds=state.retrieval_rounds,
                    verification_attempts=state.verification_attempts,
                    iteration=state.iteration,
                    tool_calls=state.tool_calls,
                )

                if route == "finish":
                    state.status = AgentStatus.COMPLETED
                    state.verification_status = VerificationStatus.PASSED
                    break

                if route == "clarify":
                    state = self._apply_clarification(state)
                    break

                if route == "fallback":
                    state = self._apply_fallback(
                        state,
                        reason="verification_fallback",
                        request_id=request_id,
                    )
                    break

                # route == retrieve_more
                if not self.can_retrieve_more(state):
                    max_iterations_hit = True
                    state = self._apply_fallback(
                        state,
                        reason="max_iterations",
                        request_id=request_id,
                    )
                    break
                # Loop continues with pending_missing_evidence guiding retrieval.

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
                    state.answer_confidence.value if state.answer_confidence else None
                ),
                verification_status=state.verification_status.value,
                verification_attempts=state.verification_attempts,
                retrieval_rounds=state.retrieval_rounds,
                latency_ms=round(latency_ms, 2),
            )
            return state
        except AgentError:
            get_metrics().record_request(
                status="error",
                iterations=0,
                tool_calls=0,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        except Exception as exc:
            get_metrics().record_request(
                status="error",
                iterations=0,
                tool_calls=0,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            raise AgentExecutionError(
                "Agent failed while processing the question"
            ) from exc

    def understand(
        self,
        state: CurriculumQAState,
        *,
        prior_state: CurriculumQAState | None = None,
    ) -> CurriculumQAState:
        """Extract filters; inherit conversation context when appropriate."""
        state.status = AgentStatus.UNDERSTANDING
        filters = extract_filters_from_question(state.question)

        if prior_state:
            for field in ("grade", "subject", "level", "topic"):
                new_value = filters.get(field)
                if new_value:
                    setattr(state, field, new_value)
                elif getattr(prior_state, field, None):
                    setattr(state, field, getattr(prior_state, field))
        else:
            state.grade = filters.get("grade") or state.grade
            state.subject = filters.get("subject") or state.subject
            state.level = filters.get("level") or state.level
            state.topic = filters.get("topic") or state.topic

        # Explicit grade change in follow-up overrides inherited grade.
        explicit_grade = normalize_grade_code(state.question)
        if explicit_grade:
            state.grade = explicit_grade
            state.level = filters.get("level") or state.level

        state.intent = state.intent or "retrieve_curriculum"
        return state

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
        """Map verification outcome to a constrained orchestration transition."""
        result = state.verification
        if result is None:
            return "fallback"
        if result.passed or result.recommendation == VerificationRecommendation.ACCEPT:
            return "finish"
        if result.recommendation == VerificationRecommendation.CLARIFY:
            return "clarify"
        if result.recommendation == VerificationRecommendation.FALLBACK:
            return "fallback"
        if result.recommendation == VerificationRecommendation.RETRIEVE_MORE:
            return "retrieve_more"
        return "fallback"

    def within_limits(self, state: CurriculumQAState) -> bool:
        return (
            state.iteration < self.settings.agent_max_iterations
            and state.tool_calls < self.settings.agent_max_tool_calls
            and state.retrieval_rounds < self.settings.agent_max_retrieval_rounds
        )

    def can_retrieve_more(self, state: CurriculumQAState) -> bool:
        """Whether another retrieve→generate→verify cycle is allowed."""
        if state.retrieval_rounds >= self.settings.agent_max_retrieval_rounds:
            return False
        if state.iteration >= self.settings.agent_max_iterations:
            return False
        if state.tool_calls >= self.settings.agent_max_tool_calls:
            return False
        return True

    def _apply_clarification(self, state: CurriculumQAState) -> CurriculumQAState:
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

    def _apply_fallback(
        self,
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
        # Deduplicate while preserving order
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
        if reason == "max_iterations" or reason == "max_retrieval_rounds":
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
