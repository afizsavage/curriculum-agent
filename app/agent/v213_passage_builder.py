"""V2.13A curriculum-aware passage construction from parsed documents."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.agent.v213_document_contract import (
    AssociationMethod,
    DocumentPassage,
    DocumentRecord,
    passage_content_hash,
    passage_id_for,
)
from app.agent.v213_document_parser import ParsedDocument, ParsedPage
from app.agent.v213_document_store import DocumentStore

_HEADING_RULES: list[tuple[re.Pattern[str], dict[str, str | None]]] = [
    (re.compile(r"\bclass\s*4\b|\bprimary\s*4\b|\bp\.?\s*4\b", re.I), {"grade": "CLASS_4"}),
    (re.compile(r"\bclass\s*5\b|\bprimary\s*5\b", re.I), {"grade": "CLASS_5"}),
    (re.compile(r"\bmathematics\b|\bmaths\b", re.I), {"subject": "MATHEMATICS"}),
    (re.compile(r"\bscience\b", re.I), {"subject": "SCIENCE"}),
    (re.compile(r"\benglish\b|\blanguage arts\b", re.I), {"subject": "ENGLISH"}),
    (
        re.compile(r"everyday arithmetic money|c4-u18", re.I),
        {"unit": "Everyday Arithmetic Money", "topic": "money"},
    ),
    (re.compile(r"\bfractions?\b", re.I), {"topic": "fractions"}),
]


class PassageBuilder:
    """Build curriculum-context passages without LLM inference."""

    def __init__(self, store: DocumentStore | None = None) -> None:
        self.store = store or DocumentStore()

    def build_passages(
        self,
        *,
        parsed: ParsedDocument,
        record: DocumentRecord,
        structure_hints: dict[str, Any] | None = None,
    ) -> list[DocumentPassage]:
        hints = dict(structure_hints or {})
        passages: list[DocumentPassage] = []
        active_section: str | None = None
        active_heading: str | None = None
        context: dict[str, str | None] = {
            "grade": hints.get("grade"),
            "subject": hints.get("subject"),
            "unit": hints.get("unit"),
            "topic": hints.get("topic"),
        }

        for page in parsed.pages:
            self._update_context_from_page(page, context)
            for index, block in enumerate(page.blocks):
                if block.block_type == "heading":
                    active_heading = block.text
                    active_section = block.text
                    self._apply_heading_rules(block.text, context)
                    if len(block.text.strip()) < 60:
                        continue
                elif block.block_type != "paragraph":
                    continue
                if len(block.text.strip()) < 20:
                    continue
                association = self._association_method(context, hints)
                passage = DocumentPassage(
                    passage_id=passage_id_for(
                        document_id=record.document_id,
                        page_number=page.page_number,
                        passage_index=index,
                    ),
                    document_id=record.document_id,
                    source_id=record.source_id,
                    curriculum_id=record.curriculum_id,
                    curriculum_version_id=hints.get("curriculum_version_id"),
                    page_number=page.page_number,
                    section=active_section,
                    heading=active_heading,
                    text=block.text,
                    source_url=record.source_url,
                    content_hash=passage_content_hash(
                        block.text,
                        page_number=page.page_number,
                        passage_index=index,
                    ),
                    grade=context.get("grade"),
                    subject=context.get("subject"),
                    unit=context.get("unit"),
                    topic=context.get("topic"),
                    association_method=association,
                    metadata={
                        "block_id": block.block_id,
                        "document_content_hash": record.content_hash,
                    },
                )
                passages.append(passage)
        return passages

    def persist_passages(self, document_id: str, passages: list[DocumentPassage]) -> None:
        payload = [p.to_dict() for p in passages]
        self.store.passages_path(document_id).write_text(json.dumps(payload, indent=2))

    def load_passages(self, document_id: str) -> list[DocumentPassage]:
        path = self.store.passages_path(document_id)
        if not path.exists():
            return []
        rows = json.loads(path.read_text())
        out: list[DocumentPassage] = []
        for row in rows:
            out.append(
                DocumentPassage(
                    passage_id=row["passage_id"],
                    document_id=row["document_id"],
                    source_id=row["source_id"],
                    curriculum_id=row.get("curriculum_id"),
                    curriculum_version_id=row.get("curriculum_version_id"),
                    page_number=int(row["page_number"]),
                    section=row.get("section"),
                    heading=row.get("heading"),
                    text=row["text"],
                    source_url=row["source_url"],
                    content_hash=row["content_hash"],
                    grade=row.get("grade"),
                    subject=row.get("subject"),
                    unit=row.get("unit"),
                    topic=row.get("topic"),
                    association_method=AssociationMethod(
                        row.get("association_method", AssociationMethod.UNRESOLVED.value)
                    ),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
        return out

    @staticmethod
    def _update_context_from_page(page: ParsedPage, context: dict[str, str | None]) -> None:
        for heading in page.headings:
            PassageBuilder._apply_heading_rules(heading, context)

    @staticmethod
    def _apply_heading_rules(text: str, context: dict[str, str | None]) -> None:
        for pattern, updates in _HEADING_RULES:
            if pattern.search(text):
                for key, value in updates.items():
                    if value:
                        context[key] = value

    @staticmethod
    def _association_method(
        context: dict[str, str | None],
        hints: dict[str, Any],
    ) -> AssociationMethod:
        if hints.get("association_method"):
            try:
                return AssociationMethod(str(hints["association_method"]))
            except ValueError:
                pass
        if context.get("grade") or context.get("subject") or context.get("unit"):
            if hints.get("known_page_range"):
                return AssociationMethod.KNOWN_PAGE_RANGE
            return AssociationMethod.HEADING_MATCH
        if hints.get("source_metadata"):
            return AssociationMethod.SOURCE_METADATA
        return AssociationMethod.UNRESOLVED


__all__ = ["PassageBuilder"]
