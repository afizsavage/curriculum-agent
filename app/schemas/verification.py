"""Verification schemas for Sprint 4 claim-level evidence checking."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class VerificationRecommendation(str, Enum):
    ACCEPT = "accept"
    RETRIEVE_MORE = "retrieve_more"
    CLARIFY = "clarify"
    FALLBACK = "fallback"


class VerificationStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    MAX_ITERATIONS = "max_iterations"
    NEEDS_CLARIFICATION = "needs_clarification"


class ClaimVerdict(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    MISSING = "missing"


class ClaimAssessment(BaseModel):
    """One claim extracted from the generated answer."""

    claim: str
    verdict: ClaimVerdict = ClaimVerdict.UNSUPPORTED
    evidence_ids: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


class MissingEvidenceRequest(BaseModel):
    """Targeted retrieval hint produced by the verifier."""

    type: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    topic: Optional[str] = None
    query: Optional[str] = None
    detail: Optional[str] = None


class VerificationResult(BaseModel):
    """Structured outcome of verify_answer()."""

    passed: bool = False
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_evidence: list[MissingEvidenceRequest | str] = Field(default_factory=list)
    incorrect_claims: list[str] = Field(default_factory=list)
    recommendation: VerificationRecommendation = VerificationRecommendation.FALLBACK
    claims: list[ClaimAssessment] = Field(default_factory=list)
    clarification: Optional[str] = None
    notes: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


VERIFICATION_RESULT_JSON_SCHEMA: dict = {
    "title": "VerificationResult",
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "issues": {"type": "array", "items": {"type": "string"}},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "missing_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "grade": {"type": "string"},
                    "subject": {"type": "string"},
                    "topic": {"type": "string"},
                    "query": {"type": "string"},
                    "detail": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "incorrect_claims": {"type": "array", "items": {"type": "string"}},
        "recommendation": {
            "type": "string",
            "enum": ["accept", "retrieve_more", "clarify", "fallback"],
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": [
                            "supported",
                            "unsupported",
                            "contradicted",
                            "missing",
                        ],
                    },
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "notes": {"type": "string"},
                },
                "required": ["claim", "verdict"],
                "additionalProperties": False,
            },
        },
        "clarification": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": [
        "passed",
        "score",
        "issues",
        "unsupported_claims",
        "missing_evidence",
        "incorrect_claims",
        "recommendation",
    ],
    "additionalProperties": False,
}
