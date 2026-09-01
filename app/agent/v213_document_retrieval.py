"""V2.13A lexical document evidence retrieval with curriculum filters."""

from __future__ import annotations

import re
from typing import Any

from app.agent.v213_document_contract import (
    DocumentPassage,
    DocumentProvenance,
    DocumentRetrievalDiagnostics,
    DocumentSearchResult,
)
from app.agent.v213_passage_builder import PassageBuilder
from app.agent.v213_document_store import DocumentStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _lexical_score(query_tokens: set[str], passage: DocumentPassage) -> float:
    if not query_tokens:
        return 0.0
    passage_tokens = _tokenize(passage.text)
    if not passage_tokens:
        return 0.0
    overlap = len(query_tokens & passage_tokens)
    return overlap / len(query_tokens)


class DocumentRetrievalService:
    """Deterministic lexical search over cached document passages."""

    def __init__(
        self,
        store: DocumentStore | None = None,
        builder: PassageBuilder | None = None,
    ) -> None:
        self.store = store or DocumentStore()
        self.builder = builder or PassageBuilder(self.store)

    def list_document_ids(self) -> list[str]:
        root = self.store.root
        if not root.exists():
            return []
        return sorted(
            p.name for p in root.iterdir() if p.is_dir() and (p / "metadata.json").exists()
        )

    def search_document_evidence(
        self,
        *,
        query: str,
        curriculum_id: str | None = None,
        curriculum_version_id: str | None = None,
        grade: str | None = None,
        subject: str | None = None,
        topic: str | None = None,
        source_id: str | None = None,
        limit: int = 5,
    ) -> DocumentSearchResult:
        query_tokens = _tokenize(query)
        filters = {
            "curriculum_id": curriculum_id,
            "curriculum_version_id": curriculum_version_id,
            "grade": grade,
            "subject": subject,
            "topic": topic,
            "source_id": source_id,
        }
        diagnostics = DocumentRetrievalDiagnostics(
            query=query,
            documents_searched=0,
            passages_scanned=0,
            passages_matched=0,
            filters_applied={k: v for k, v in filters.items() if v},
        )
        ranked: list[tuple[float, DocumentPassage]] = []

        for document_id in self.list_document_ids():
            record = self.store.load_record(document_id)
            if not record:
                continue
            if curriculum_id and record.curriculum_id and record.curriculum_id != curriculum_id:
                diagnostics.rejected_wrong_source += 1
                continue
            if source_id and record.source_id != source_id:
                diagnostics.rejected_wrong_source += 1
                continue
            diagnostics.documents_searched += 1
            passages = self.builder.load_passages(document_id)
            for passage in passages:
                diagnostics.passages_scanned += 1
                if not self._passes_filters(
                    passage,
                    grade=grade,
                    subject=subject,
                    topic=topic,
                    curriculum_version_id=curriculum_version_id,
                    diagnostics=diagnostics,
                ):
                    continue
                score = _lexical_score(query_tokens, passage)
                if score <= 0:
                    continue
                diagnostics.passages_matched += 1
                ranked.append((score, passage))

        ranked.sort(key=lambda item: (-item[0], item[1].page_number))
        top = [p for _, p in ranked[:limit]]
        provenance = [
            DocumentProvenance(
                source_id=p.source_id,
                source_name=str(p.metadata.get("source_name") or ""),
                document_id=p.document_id,
                document_version=p.metadata.get("document_version"),
                source_url=p.source_url,
                page_number=p.page_number,
                section=p.section,
                heading=p.heading,
                block_id=p.metadata.get("block_id"),
                content_hash=p.metadata.get("document_content_hash") or p.content_hash,
            )
            for p in top
        ]
        return DocumentSearchResult(
            passages=top,
            diagnostics=diagnostics,
            provenance=provenance,
        )

    @staticmethod
    def _passes_filters(
        passage: DocumentPassage,
        *,
        grade: str | None,
        subject: str | None,
        topic: str | None,
        curriculum_version_id: str | None,
        diagnostics: DocumentRetrievalDiagnostics,
    ) -> bool:
        if curriculum_version_id and passage.curriculum_version_id:
            if passage.curriculum_version_id != curriculum_version_id:
                return False
        if grade and passage.grade and passage.grade != grade:
            diagnostics.rejected_wrong_grade += 1
            return False
        if subject and passage.subject and passage.subject != subject:
            diagnostics.rejected_wrong_subject += 1
            return False
        if topic and passage.topic and passage.topic.lower() != topic.lower():
            return False
        return True


__all__ = ["DocumentRetrievalService"]
