"""HTTP/API schemas for the Curriculum Q&A agent."""

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Natural-language question about the MBSSE curriculum.",
        examples=["What topics are taught in Primary 4 Mathematics?"],
    )
    conversation_id: Optional[UUID] = Field(
        default=None,
        description="Optional existing conversation id for multi-turn context.",
    )

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class AskMetadata(BaseModel):
    iterations: int = 0
    tool_calls: int = 0
    tools_used: list[str] = Field(default_factory=list)
    evidence_status: Optional[str] = None
    evidence_count: int = 0
    model: Optional[str] = None
    provider: Optional[str] = None


class EvidenceSummary(BaseModel):
    entity_type: str
    entity_id: Optional[str] = None
    name: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    topic: Optional[str] = None


class AskResponse(BaseModel):
    conversation_id: str
    question: str
    answer: Optional[str] = Field(
        default=None,
        description="Final answer. Null until Phase 3 answer generation.",
    )
    status: str = Field(
        ...,
        description="Agent turn status. Phase 2 returns `retrieved` after tool use.",
        examples=["retrieved"],
    )
    evidence: list[EvidenceSummary] = Field(default_factory=list)
    metadata: AskMetadata = Field(default_factory=AskMetadata)
    error: Optional[str] = None


class ErrorBody(BaseModel):
    detail: str
    code: str
    request_id: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)
