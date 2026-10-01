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
            "description": (
                "User-facing curriculum answer: a natural-language synthesis for "
                "pupils and teachers. Do not include internal curriculum identifiers "
                "unless the question asks for them."
            ),
        },
        "summary": {
            "type": "string",
            "description": "Optional one-line summary of the answer.",
        },
        "refs": {
            "type": "array",
            "description": (
                "Entity IDs from the supplied evidence that this answer actually "
                "used. Do not invent IDs. Omit retrieved records the answer does "
                "not use. Include a record used only for a curriculum evidence note."
            ),
            "items": {"type": "string"},
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


# Experimental only. Not part of the production generation schema: exact claim
# text is too brittle to require from the live model until that failure rate
# is measured. The parser accepts this shape when a response includes it.
EXPERIMENTAL_CLAIM_MAPPING_SCHEMA: dict = {
    "title": "ExperimentalClaimMapping",
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Exact span copied from the answer.",
                    },
                    "refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Supplied evidence ids that support that span.",
                    },
                },
                "required": ["text", "refs"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "claims"],
    "additionalProperties": False,
}


def shadow_claim_generation_schema() -> dict:
    """Production grounded-answer schema plus an optional claims array.

    Used only by the shadow experiment. `claims` is not required, and this
    schema is not sent by the production generator.
    """
    schema = {
        "title": "ShadowClaimAnswer",
        "type": "object",
        "properties": dict(GROUNDED_ANSWER_JSON_SCHEMA["properties"]),
        "required": list(GROUNDED_ANSWER_JSON_SCHEMA["required"]),
        "additionalProperties": False,
    }
    schema["properties"]["claims"] = EXPERIMENTAL_CLAIM_MAPPING_SCHEMA["properties"]["claims"]
    return schema
