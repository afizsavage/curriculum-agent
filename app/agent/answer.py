"""Answer generation node: evidence → grounded response."""

from __future__ import annotations

from app.agent.answer_generator import AnswerGenerator
from app.agent.context import ConversationContext
from app.agent.evidence_snapshot import generation_to_verifier_overlap
from app.agent.state import CurriculumQAState
from app.agent.v23_diagnostics import build_generation_diagnostics
from app.config import Settings
from app.enums import AgentStatus
from app.llm.base import LLMProvider
from app.logging_utils import get_logger, log_agent_event

logger = get_logger(__name__)


class AnswerGenerationNode:
    """GENERATE_ANSWER step — separate from retrieval."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        settings: Settings,
    ) -> None:
        self.settings = settings
        self.generator = AnswerGenerator(llm)

    def run(
        self,
        state: CurriculumQAState,
        *,
        conversation: ConversationContext | None = None,
        request_id: str | None = None,
    ) -> CurriculumQAState:
        state.status = AgentStatus.ANSWERING

        grounded = self.generator.generate(
            state,
            conversation=conversation,
            request_id=request_id,
        )

        state.draft_answer = grounded.answer
        state.final_answer = grounded.answer
        state.answer_evidence = grounded.evidence
        state.answer_confidence = grounded.confidence
        state.answer_limitations = grounded.limitations

        state.metadata["answer_confidence"] = grounded.confidence.value
        state.metadata["answer_evidence_count"] = len(grounded.evidence)
        state.metadata["answer_limitations"] = grounded.limitations
        if grounded.summary:
            state.metadata["answer_summary"] = grounded.summary
        state.metadata.update(
            build_generation_diagnostics(
                state,
                generation_latency_ms=state.metadata.get("generation_latency_ms"),
            )
        )
        state.metadata.update(generation_to_verifier_overlap(state))

        # Final status is decided by the orchestrator after verification.
        state.status = AgentStatus.ANSWERING

        log_agent_event(
            logger,
            "agent.answer.end",
            request_id=request_id,
            conversation_id=state.conversation_id,
            question=state.question,
            status=state.status.value,
            model=self.generator.llm.model,
            evidence_count=len(state.evidence),
            answer_evidence_count=len(grounded.evidence),
            confidence=grounded.confidence.value,
        )
        return state
