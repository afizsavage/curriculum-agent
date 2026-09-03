"""V2.13B framework-neutral hybrid semantic retrieval contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.agent.v213_document_contract import DocumentPassage, DocumentProvenance


class RetrievalVariant(str, Enum):
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    CONTEXT_HYBRID = "context_hybrid"


@dataclass(frozen=True)
class CurriculumContext:
    """Explicit curriculum context supplied to retrieval (never invented)."""

    curriculum_id: str | None = None
    curriculum_version_id: str | None = None
    grade: str | None = None
    subject: str | None = None
    unit: str | None = None
    topic: str | None = None
    resolved: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "curriculum_id": self.curriculum_id,
            "curriculum_version_id": self.curriculum_version_id,
            "grade": self.grade,
            "subject": self.subject,
            "unit": self.unit,
            "topic": self.topic,
            "resolved": self.resolved,
        }


@dataclass(frozen=True)
class RetrievedPassageHit:
    """Ranked document passage with retrieval provenance."""

    passage: DocumentPassage
    retrieval_method: str
    retrieval_score: float
    retrieval_rank: int
    curriculum_context: CurriculumContext
    metadata_valid: bool = True
    lexical_score: float | None = None
    semantic_score: float | None = None
    context_boost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        p = self.passage.to_dict()
        p.update(
            {
                "retrieval_method": self.retrieval_method,
                "retrieval_score": self.retrieval_score,
                "retrieval_rank": self.retrieval_rank,
                "lexical_score": self.lexical_score,
                "semantic_score": self.semantic_score,
                "context_boost": self.context_boost,
                "curriculum_context": self.curriculum_context.to_dict(),
                "metadata_valid": self.metadata_valid,
            }
        )
        return p


@dataclass
class HybridRetrievalDiagnostics:
    query: str
    variant: str
    documents_searched: int = 0
    passages_scanned: int = 0
    passages_matched: int = 0
    lexical_candidates: int = 0
    semantic_candidates: int = 0
    filters_applied: dict[str, Any] = field(default_factory=dict)
    rejected_wrong_grade: int = 0
    rejected_wrong_subject: int = 0
    rejected_wrong_source: int = 0
    rejected_wrong_version: int = 0
    unresolved_context: bool = False
    embedding_model: str = ""
    index_passages: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "variant": self.variant,
            "documents_searched": self.documents_searched,
            "passages_scanned": self.passages_scanned,
            "passages_matched": self.passages_matched,
            "lexical_candidates": self.lexical_candidates,
            "semantic_candidates": self.semantic_candidates,
            "filters_applied": dict(self.filters_applied),
            "rejected_wrong_grade": self.rejected_wrong_grade,
            "rejected_wrong_subject": self.rejected_wrong_subject,
            "rejected_wrong_source": self.rejected_wrong_source,
            "rejected_wrong_version": self.rejected_wrong_version,
            "unresolved_context": self.unresolved_context,
            "embedding_model": self.embedding_model,
            "index_passages": self.index_passages,
            "latency_ms": self.latency_ms,
        }


@dataclass
class HybridSearchResult:
    hits: list[RetrievedPassageHit]
    diagnostics: HybridRetrievalDiagnostics
    provenance: list[DocumentProvenance] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": [h.to_dict() for h in self.hits],
            "diagnostics": self.diagnostics.to_dict(),
            "provenance": [p.to_dict() for p in self.provenance],
        }


def hits_to_provenance(hits: list[RetrievedPassageHit]) -> list[DocumentProvenance]:
    return [
        DocumentProvenance(
            source_id=h.passage.source_id,
            source_name=str(h.passage.metadata.get("source_name") or ""),
            document_id=h.passage.document_id,
            document_version=h.passage.metadata.get("document_version"),
            source_url=h.passage.source_url,
            page_number=h.passage.page_number,
            section=h.passage.section,
            heading=h.passage.heading,
            block_id=h.passage.metadata.get("block_id"),
            content_hash=h.passage.metadata.get("document_content_hash") or h.passage.content_hash,
        )
        for h in hits
    ]


__all__ = [
    "CurriculumContext",
    "HybridRetrievalDiagnostics",
    "HybridSearchResult",
    "RetrievalVariant",
    "RetrievedPassageHit",
    "hits_to_provenance",
]
