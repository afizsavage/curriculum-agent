"""Normalized curriculum evidence shared by all retrieval tools."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EvidenceStatus(str, Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    ERROR = "error"
    PARTIAL = "partial"


class CurriculumEvidence(BaseModel):
    """Canonical evidence item for answer generation (Phase 3)."""

    source: str = "curriculum_api"
    entity_type: str
    entity_id: Optional[str] = None
    name: Optional[str] = None
    level: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    topic: Optional[str] = None
    content: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_reference: Optional[str] = None


class ToolCallRecord(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: str = "success"
    error: Optional[str] = None
    evidence_count: int = 0
    latency_ms: Optional[float] = None
    curriculum_api_status: Optional[int] = None


class SearchHit(BaseModel):
    """Structured search result for search_curriculum."""

    id: str
    type: str
    name: str
    level: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
