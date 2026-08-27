"""Typed agent state for the Curriculum Q&A loop."""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.agent.retrieval_state import RetrievalState
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus, ToolCallRecord
from app.enums import AgentStatus
from app.schemas.answer import AnswerConfidence, AnswerEvidenceRef
from app.schemas.verification import (
    MissingEvidenceRequest,
    VerificationResult,
    VerificationStatus,
)


class PlanStep(BaseModel):
    """One planned step in a future understand/retrieve/answer plan."""

    id: str
    description: str
    status: str = "pending"


class RetrievedContextItem(BaseModel):
    """A single piece of curriculum context retrieved by a tool."""

    source: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CurriculumQAState(BaseModel):
    """Explicit, extensible state for CurriculumQAAgent turns."""

    question: str
    conversation_id: Optional[str] = None

    intent: Optional[str] = None
    level: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    topic: Optional[str] = None

    plan: Optional[list[PlanStep]] = None
    retrieved_context: list[RetrievedContextItem] = Field(default_factory=list)
    evidence: list[CurriculumEvidence] = Field(default_factory=list)
    evidence_status: EvidenceStatus = EvidenceStatus.NOT_FOUND
    retrieval_history: list[ToolCallRecord] = Field(default_factory=list)
    selected_tools: list[str] = Field(default_factory=list)
    executed_tool_keys: list[str] = Field(default_factory=list)
    retrieval_state: RetrievalState = Field(default_factory=RetrievalState)

    draft_answer: Optional[str] = None
    final_answer: Optional[str] = None
    answer_evidence: list[AnswerEvidenceRef] = Field(default_factory=list)
    answer_confidence: Optional[AnswerConfidence] = None
    answer_limitations: list[str] = Field(default_factory=list)
    clarification: Optional[str] = None

    verification: Optional[VerificationResult] = None
    verification_attempts: int = 0
    verification_status: VerificationStatus = VerificationStatus.PENDING
    verification_history: list[VerificationResult] = Field(default_factory=list)
    retrieval_rounds: int = 0
    pending_missing_evidence: list[MissingEvidenceRequest | str] = Field(
        default_factory=list
    )

    iteration: int = 0
    tool_calls: int = 0
    status: AgentStatus = AgentStatus.RECEIVED
    error: Optional[str] = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def initial(
        cls,
        *,
        question: str,
        conversation_id: str | None = None,
    ) -> "CurriculumQAState":
        return cls(
            question=question.strip(),
            conversation_id=conversation_id or str(uuid4()),
            iteration=0,
            tool_calls=0,
            status=AgentStatus.RECEIVED,
            retrieved_context=[],
            evidence=[],
            evidence_status=EvidenceStatus.NOT_FOUND,
            retrieval_history=[],
            selected_tools=[],
            executed_tool_keys=[],
            retrieval_state=RetrievalState(),
            plan=None,
            draft_answer=None,
            final_answer=None,
            answer_evidence=[],
            answer_confidence=None,
            answer_limitations=[],
            clarification=None,
            verification=None,
            verification_attempts=0,
            verification_status=VerificationStatus.PENDING,
            verification_history=[],
            retrieval_rounds=0,
            pending_missing_evidence=[],
            error=None,
        )

    def bump_iteration(self) -> None:
        self.iteration += 1

    def bump_tool_calls(self, count: int = 1) -> None:
        self.tool_calls += count

    def mark_failed(self, message: str) -> None:
        self.status = AgentStatus.FAILED
        self.error = message
