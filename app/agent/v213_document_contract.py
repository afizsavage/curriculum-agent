"""V2.13A framework-neutral document evidence contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_EXPERIMENT_NAME = "v2.13a_document_evidence"


class AssociationMethod(str, Enum):
    HEADING_MATCH = "heading_match"
    KNOWN_PAGE_RANGE = "known_page_range"
    SOURCE_METADATA = "source_metadata"
    STRUCTURE_ENTITY = "structure_entity"
    UNRESOLVED = "unresolved"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    ACQUIRED = "acquired"
    PARSED = "parsed"
    FAILED = "failed"
    HASH_CONFLICT = "hash_conflict"


@dataclass(frozen=True)
class DocumentProvenance:
    source_id: str
    source_name: str
    document_id: str
    document_version: str | None
    source_url: str
    page_number: int | None = None
    section: str | None = None
    heading: str | None = None
    block_id: str | None = None
    content_hash: str = ""
    retrieved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "document_id": self.document_id,
            "document_version": self.document_version,
            "source_url": self.source_url,
            "page_number": self.page_number,
            "section": self.section,
            "heading": self.heading,
            "block_id": self.block_id,
            "content_hash": self.content_hash,
            "retrieved_at": self.retrieved_at,
        }


@dataclass(frozen=True)
class DocumentPassage:
    """Authoritative document passage with curriculum context."""

    passage_id: str
    document_id: str
    source_id: str
    curriculum_id: str | None
    curriculum_version_id: str | None
    page_number: int
    section: str | None
    heading: str | None
    text: str
    source_url: str
    content_hash: str
    grade: str | None = None
    subject: str | None = None
    unit: str | None = None
    topic: str | None = None
    association_method: AssociationMethod = AssociationMethod.UNRESOLVED
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "document_id": self.document_id,
            "source_id": self.source_id,
            "curriculum_id": self.curriculum_id,
            "curriculum_version_id": self.curriculum_version_id,
            "page_number": self.page_number,
            "section": self.section,
            "heading": self.heading,
            "text": self.text,
            "source_url": self.source_url,
            "content_hash": self.content_hash,
            "grade": self.grade,
            "subject": self.subject,
            "unit": self.unit,
            "topic": self.topic,
            "association_method": self.association_method.value,
            "metadata": dict(self.metadata),
        }


@dataclass
class DocumentRecord:
    document_id: str
    source_id: str
    source_url: str
    document_version: str | None
    content_hash: str
    content_type: str
    file_size: int
    retrieved_at: str
    status: DocumentStatus
    curriculum_id: str | None = None
    page_count: int = 0
    passage_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "document_version": self.document_version,
            "content_hash": self.content_hash,
            "content_type": self.content_type,
            "file_size": self.file_size,
            "retrieved_at": self.retrieved_at,
            "status": self.status.value,
            "curriculum_id": self.curriculum_id,
            "page_count": self.page_count,
            "passage_count": self.passage_count,
            "metadata": dict(self.metadata),
        }


@dataclass
class DocumentRetrievalDiagnostics:
    query: str
    documents_searched: int
    passages_scanned: int
    passages_matched: int
    filters_applied: dict[str, Any] = field(default_factory=dict)
    rejected_wrong_grade: int = 0
    rejected_wrong_subject: int = 0
    rejected_wrong_source: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "documents_searched": self.documents_searched,
            "passages_scanned": self.passages_scanned,
            "passages_matched": self.passages_matched,
            "filters_applied": dict(self.filters_applied),
            "rejected_wrong_grade": self.rejected_wrong_grade,
            "rejected_wrong_subject": self.rejected_wrong_subject,
            "rejected_wrong_source": self.rejected_wrong_source,
        }


@dataclass
class DocumentSearchResult:
    passages: list[DocumentPassage]
    diagnostics: DocumentRetrievalDiagnostics
    provenance: list[DocumentProvenance] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passages": [p.to_dict() for p in self.passages],
            "diagnostics": self.diagnostics.to_dict(),
            "provenance": [p.to_dict() for p in self.provenance],
        }


def passage_content_hash(text: str, *, page_number: int, passage_index: int) -> str:
    payload = json.dumps(
        {"page": page_number, "index": passage_index, "text": text},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def document_id_for_source(*, source_id: str, document_version: str | None) -> str:
    version = document_version or "unknown"
    digest = hashlib.sha256(f"{source_id}:{version}".encode()).hexdigest()[:12]
    return f"doc-{digest}"


def passage_id_for(*, document_id: str, page_number: int, passage_index: int) -> str:
    digest = hashlib.sha256(
        f"{document_id}:{page_number}:{passage_index}".encode()
    ).hexdigest()[:12]
    return f"passage-{digest}"


__all__ = [
    "AssociationMethod",
    "DocumentPassage",
    "DocumentProvenance",
    "DocumentRecord",
    "DocumentRetrievalDiagnostics",
    "DocumentSearchResult",
    "DocumentStatus",
    "document_id_for_source",
    "passage_content_hash",
    "passage_id_for",
]
