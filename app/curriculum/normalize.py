"""Helpers to flatten syllabus trees and map API payloads to evidence."""

from __future__ import annotations

from typing import Any, Iterator

from app.curriculum.evidence import CurriculumEvidence, SearchHit


def iter_content_nodes(
    nodes: list[dict[str, Any]] | dict[str, Any],
    *,
    parent_id: str | None = None,
) -> Iterator[tuple[dict[str, Any], str | None]]:
    if isinstance(nodes, dict):
        nodes = [nodes]
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        yield node, parent_id
        children = node.get("children") or []
        node_id = str(node.get("id")) if node.get("id") is not None else None
        yield from iter_content_nodes(children, parent_id=node_id)


def node_to_search_hit(
    node: dict[str, Any],
    *,
    parent_id: str | None,
    grade: str | None,
    subject: str | None,
    level: str | None,
) -> SearchHit:
    content_type = (node.get("content_type") or node.get("type") or "topic").lower()
    return SearchHit(
        id=str(node.get("id") or node.get("code") or node.get("name")),
        type=content_type,
        name=str(node.get("name") or node.get("title") or ""),
        level=level,
        grade=grade,
        subject=subject,
        parent_id=parent_id,
        metadata={
            "code": node.get("code"),
            "description": node.get("description"),
            "display_order": node.get("display_order"),
        },
    )


def match_query(node: dict[str, Any], query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return True
    haystacks = [
        str(node.get("name") or ""),
        str(node.get("code") or ""),
        str(node.get("description") or ""),
        str(node.get("content_type") or ""),
    ]
    text = " ".join(haystacks).lower()
    return all(token in text for token in q.split())


def evidence_from_hit(
    hit: SearchHit, *, source_reference: str = "syllabus.content.tree"
) -> CurriculumEvidence:
    metadata = dict(hit.metadata or {})
    # The tree walk already knows the published parent. Keep that id on the
    # evidence record; do not invent a parent from a shared display name.
    if hit.parent_id and "parent_id" not in metadata:
        metadata["parent_id"] = hit.parent_id
    return CurriculumEvidence(
        entity_type=hit.type,
        entity_id=hit.id,
        name=hit.name,
        level=hit.level,
        grade=hit.grade,
        subject=hit.subject,
        topic=hit.name if hit.type in {"topic", "subtopic", "unit"} else None,
        content=hit.name,
        metadata=metadata,
        source_reference=source_reference,
    )


def evidence_from_subject(subject: dict[str, Any], *, grade: str | None = None) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="subject",
        entity_id=str(subject.get("id")) if subject.get("id") else None,
        name=subject.get("name"),
        grade=grade,
        subject=subject.get("code") or subject.get("name"),
        content=subject.get("description") or subject.get("name"),
        metadata={
            "code": subject.get("code"),
            "category": subject.get("category"),
            "status": subject.get("status"),
            "classification": subject.get("classification"),
        },
        source_reference="subjects",
    )


def evidence_from_outcome(
    outcome: dict[str, Any],
    *,
    topic_id: str | None = None,
    grade: str | None = None,
    subject: str | None = None,
) -> CurriculumEvidence:
    text = (
        outcome.get("description")
        or outcome.get("text")
        or outcome.get("statement")
        or ""
    )
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id=str(outcome.get("id")) if outcome.get("id") else None,
        name=outcome.get("code") or None,
        grade=grade,
        subject=subject,
        topic=topic_id,
        content=str(text),
        metadata={
            "code": outcome.get("code"),
            "sequence": outcome.get("display_order") or outcome.get("sequence"),
            "knowledge": outcome.get("knowledge"),
            "skills": outcome.get("skills"),
        },
        source_reference="learning_outcomes",
    )


def learning_outcome_summaries(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Outcome id, code, and description already present on a content node.

    These fields are copied. No text is invented, and knowledge or skills
    blobs are left off the evidence row.
    """
    rows: list[dict[str, Any]] = []
    for outcome in node.get("learning_outcomes") or []:
        if not isinstance(outcome, dict):
            continue
        text = (
            outcome.get("description")
            or outcome.get("text")
            or outcome.get("statement")
            or ""
        )
        outcome_id = outcome.get("id")
        if not outcome_id and not str(text).strip():
            continue
        rows.append(
            {
                "id": str(outcome_id) if outcome_id else None,
                "code": outcome.get("code"),
                "description": str(text),
            }
        )
    return rows


def evidence_from_structure_node(
    node: dict[str, Any],
    *,
    entity_type: str,
    grade: str | None = None,
    subject: str | None = None,
) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type=entity_type,
        entity_id=str(node.get("id")) if node.get("id") else None,
        name=node.get("name"),
        grade=grade,
        subject=subject,
        content=node.get("description") or node.get("name"),
        metadata={"code": node.get("code")},
        source_reference="curricula.structure",
    )
