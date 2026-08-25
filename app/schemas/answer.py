"""Structured answer models for grounded curriculum Q&A."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AnswerConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AnswerEvidenceRef(BaseModel):
    """A curriculum claim linked to retrieved evidence."""

    entity_id: str
    entity_type: str
    claim: str
    name: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    topic: Optional[str] = None


class GroundedAnswer(BaseModel):
    """Structured LLM output for Sprint 3 answer generation."""

    answer: str
    summary: Optional[str] = None
    evidence: list[AnswerEvidenceRef] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    confidence: AnswerConfidence = AnswerConfidence.MEDIUM


GROUNDED_ANSWER_JSON_SCHEMA: dict = {
    "title": "GroundedAnswer",
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Grounded curriculum answer for teachers and education officers.",
        },
        "summary": {
            "type": "string",
            "description": "Optional one-line summary of the answer.",
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                    "entity_type": {"type": "string"},
                    "claim": {"type": "string"},
                },
                "required": ["entity_id", "entity_type", "claim"],
                "additionalProperties": False,
            },
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
    },
    "required": ["answer", "evidence", "limitations", "confidence"],
    "additionalProperties": False,
}
