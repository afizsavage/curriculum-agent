"""Verification node: answer + evidence → VerificationResult."""

from __future__ import annotations

import time

from app.agent.state import CurriculumQAState
from app.agent.verifier import AnswerVerifier
from app.config import Settings
from app.enums import AgentStatus
from app.llm.base import LLMProvider
from app.logging_utils import get_logger, log_agent_event
from app.schemas.verification import VerificationStatus

logger = get_logger(__name__)


class VerificationNode:
    """VERIFY step — separate from answer generation."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        settings: Settings,
        verifier: AnswerVerifier | None = None,
    ) -> None:
        self.settings = settings
        self.verifier = verifier or AnswerVerifier(llm, settings=settings)

    def run(
        self,
        state: CurriculumQAState,
        *,
        request_id: str | None = None,
    ) -> CurriculumQAState:
        state.status = AgentStatus.VERIFYING
        started = time.perf_counter()
        result = self.verifier.verify(state, request_id=request_id)
        latency_ms = (time.perf_counter() - started) * 1000

        state.verification = result
        state.verification_attempts += 1
        state.verification_history.append(result)
        state.pending_missing_evidence = list(result.missing_evidence)

        if result.passed:
            state.verification_status = VerificationStatus.PASSED
        elif result.recommendation.value == "clarify":
            state.verification_status = VerificationStatus.NEEDS_CLARIFICATION
        else:
            state.verification_status = VerificationStatus.FAILED

        # Confidence must follow verification, not unrestricted LLM labels.
        state.answer_confidence = self.verifier.confidence_from_verification(
            result, previous=state.answer_confidence
        )
        state.metadata["verification_status"] = state.verification_status.value
        state.metadata["verification_score"] = result.score
        state.metadata["verification_attempts"] = state.verification_attempts
        state.metadata["verification_recommendation"] = result.recommendation.value
        state.metadata["answer_confidence"] = (
            state.answer_confidence.value if state.answer_confidence else None
        )

        log_agent_event(
            logger,
            "agent.verify.end",
            request_id=request_id,
            conversation_id=state.conversation_id,
            question=state.question,
            status=state.status.value,
            iteration=state.iteration,
            tool_calls=state.tool_calls,
            verification_status=state.verification_status.value,
            verification_score=result.score,
            verification_attempts=state.verification_attempts,
            recommendation=result.recommendation.value,
            latency_ms=round(latency_ms, 2),
            model=self.verifier.llm.model,
        )
        return state
