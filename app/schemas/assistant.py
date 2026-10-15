"""HTTP schemas for the teacher assistant. AskResponse is reused, not changed."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.agent import AskResponse


class AssistantChatRequest(BaseModel):
    """A teacher turn. Identity is not a field on this body."""

    model_config = ConfigDict(extra="ignore")

    question: str = Field(..., min_length=1, max_length=4000)
    thread_id: Optional[str] = None

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("thread_id")
    @classmethod
    def strip_thread_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class AssistantChatResponse(BaseModel):
    thread_id: str
    route: str
    curriculum: Optional[AskResponse] = None
    curriculum_technical_failure: bool = False
    general_text: Optional[str] = None
    general_label: Optional[str] = None
    clarification: Optional[str] = None
    persistence_warning: Optional[str] = None
