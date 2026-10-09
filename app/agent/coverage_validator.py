"""Deterministic checks for SSS syllabus-coverage answers.

The reviewed prototype lives under ``data/diagnostics/sss_coverage_validator``.
This module keeps that parent-id contract and differs in three ways:

* Source rows are the evidence records already copied from the content tree.
  The validator does not walk Markdown and does not invent a parent id.
* Answer blocks come from the renderer. Their placement is checked against
  ``parent_id``. A shared display name is not a relationship.
* Learning-outcome parent checks are implemented here, but outcome questions
  are not routed through this decision. The renderer still prints outcomes as
  a flat list, so those answers do not carry parent placement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.agent.state import CurriculumQAState
from app.curriculum.evidence import CurriculumEvidence
from app.curriculum.sss_stream_intent import (
    FOCUS_COVERAGE,
    FOCUS_OUTCOMES,
    FOCUS_TOPICS,
    INTENT_SSS_STREAM_SUBJECTS,
)

SCOPE = {
    FOCUS_COVERAGE: frozenset({"THEME", "TOPIC"}),
    FOCUS_TOPICS: frozenset({"THEME", "TOPIC"}),
    FOCUS_OUTCOMES: frozenset({"THEME", "TOPIC", "LEARNING_OUTCOME"}),
}
ROUTED_FOCUSES = frozenset({FOCUS_COVERAGE, FOCUS_TOPICS})


@dataclass(frozen=True)
class SourceRecord:
    entity_id: str
    entity_type: str
    name: str
    parent_id: str | None
    parent_entity_type: str | None
    parent_in_evidence: bool | None
    section_id: str | None = None
    section_name: str | None = None
    omitted_child_ids: tuple[str, ...] = ()
    omitted_child_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoverageBlock:
    entity_id: str | None
    entity_type: str
    name: str
    source_parent_id: str | None = None
    parent_entity_type: str | None = None
    parent_display_name: str | None = None
    section_id: str | None = None
    section_name: str | None = None
    children: tuple["CoverageBlock", ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "name": self.name,
            "source_parent_id": self.source_parent_id,
            "parent_entity_type": self.parent_entity_type,
            "parent_display_name": self.parent_display_name,
            "section_id": self.section_id,
            "section_name": self.section_name,
            "children": [child.as_dict() for child in self.children],
        }


@dataclass(frozen=True)
class StructuralError:
    code: str
    entity_id: str | None
    message: str


@dataclass(frozen=True)
class StructuralResult:
    passed: bool
    errors: tuple[StructuralError, ...]
    latency_ms: float


@dataclass(frozen=True)
class CoverageGate:
    """Whether this question is inside the authoritative coverage check."""

    status: str
    reason: str | None = None
    result: StructuralResult | None = None


def coverage_block_from_evidence(
    item: CurriculumEvidence,
    children: tuple[CoverageBlock, ...] = (),
) -> CoverageBlock:
    """Copy identifiers from the evidence row. Do not read a display-name parent."""
    metadata = item.metadata or {}
    parent_id = metadata.get("parent_id") if "parent_id" in metadata else None
    return CoverageBlock(
        entity_id=str(item.entity_id) if item.entity_id else None,
        entity_type=(item.entity_type or "").lower(),
        name=str(item.name or ""),
        source_parent_id=str(parent_id) if parent_id else None,
        parent_entity_type=metadata.get("parent_entity_type"),
        parent_display_name=metadata.get("parent_display_name"),
        section_id=metadata.get("section_id"),
        section_name=metadata.get("section_name"),
        children=children,
    )


def blocks_from_payload(payload: list[dict[str, Any]] | None) -> list[CoverageBlock]:
    return [_block_from_dict(item) for item in payload or []]


def source_records(
    evidence: list[CurriculumEvidence],
    *,
    focus: str,
) -> list[SourceRecord] | None:
    """In-scope rows, or None when a row has no stable id or parent field.

    ``None`` means the check cannot run. It is not a structural pass.
    A present ``parent_id`` of null is a genuinely parentless row.
    """
    scope = SCOPE[focus]
    rows: list[SourceRecord] = []
    for item in evidence:
        kind = (item.entity_type or "").upper()
        if kind not in scope or not item.name:
            continue
        metadata = item.metadata or {}
        if not item.entity_id or "parent_id" not in metadata:
            return None
        raw_parent = metadata.get("parent_id")
        omitted = metadata.get("omitted_child_ids") or []
        omitted_types = metadata.get("omitted_child_types") or []
        rows.append(
            SourceRecord(
                entity_id=str(item.entity_id),
                entity_type=kind,
                name=str(item.name),
                parent_id=str(raw_parent) if raw_parent else None,
                parent_entity_type=metadata.get("parent_entity_type"),
                parent_in_evidence=metadata.get("parent_in_evidence"),
                section_id=metadata.get("section_id"),
                section_name=metadata.get("section_name"),
                omitted_child_ids=tuple(str(value) for value in omitted),
                omitted_child_types=tuple(str(value) for value in omitted_types),
            )
        )
    return rows


def validate(
    records: list[SourceRecord],
    blocks: list[CoverageBlock],
    *,
    focus: str = FOCUS_COVERAGE,
) -> StructuralResult:
    started = time.perf_counter()
    scope = SCOPE[focus]
    source = {row.entity_id: row for row in records if row.entity_type in scope}
    names: dict[str, list[str]] = {}
    for row in source.values():
        names.setdefault(row.name, []).append(row.entity_id)
    placed: dict[str, str | None] = {}
    errors: list[StructuralError] = []

    def walk(nodes: list[CoverageBlock] | tuple[CoverageBlock, ...], parent_id: str | None) -> None:
        for node in nodes:
            if not node.entity_id:
                errors.append(
                    StructuralError(
                        "unknown_entity",
                        None,
                        "An answer block has no entity id.",
                    )
                )
                continue
            if node.entity_id in placed:
                errors.append(
                    StructuralError(
                        "duplicate_entity",
                        node.entity_id,
                        f"{node.entity_id} appears more than once in the answer.",
                    )
                )
                continue
            if node.entity_id not in source:
                errors.append(
                    StructuralError(
                        "unknown_entity",
                        node.entity_id,
                        f"{node.entity_id} is not an in-scope source record.",
                    )
                )
                continue
            placed[node.entity_id] = parent_id
            walk(node.children, node.entity_id)

    walk(blocks, None)
    for entity_id, row in source.items():
        if entity_id in placed:
            continue
        repeated = len(names.get(row.name, [])) > 1 and row.entity_type == "THEME"
        errors.append(
            StructuralError(
                "missing_theme_occurrence" if repeated else "missing_entity",
                entity_id,
                (
                    f"Theme occurrence {entity_id} ({row.section_name or 'no section'}) "
                    f"is missing. Another record shares the name {row.name!r}."
                    if repeated
                    else f"{row.entity_type} {entity_id} is missing from the answer."
                ),
            )
        )
    for entity_id, answer_parent in placed.items():
        row = source[entity_id]
        if row.entity_type == "THEME":
            if answer_parent is not None:
                errors.append(
                    StructuralError(
                        "incorrect_parent",
                        entity_id,
                        f"Theme {entity_id} is nested under {answer_parent}.",
                    )
                )
            continue
        expected = _expected_parent(row, source)
        if answer_parent == expected:
            continue
        errors.append(_placement_error(row, expected, answer_parent))
    elapsed_ms = (time.perf_counter() - started) * 1000
    return StructuralResult(passed=not errors, errors=tuple(errors), latency_ms=elapsed_ms)


def assess_coverage(state: CurriculumQAState) -> CoverageGate:
    """Route only explicit SSS coverage and topic questions.

    Outcome questions stay on the generic verifier. The outcome parent
    contract is tested directly, and the rendered answer does not carry it.
    """
    if state.intent != INTENT_SSS_STREAM_SUBJECTS:
        return CoverageGate("not_applicable", "not_sss_stream_subjects")
    focus = state.metadata.get("sss_focus") or state.metadata.get("focus")
    if focus == FOCUS_OUTCOMES:
        return CoverageGate("not_applicable", "outcomes_hierarchy_not_represented")
    if focus not in ROUTED_FOCUSES:
        return CoverageGate("not_applicable", "focus_outside_coverage_contract")
    in_scope = [
        item
        for item in state.evidence
        if (item.entity_type or "").upper() in SCOPE[focus] and item.name
    ]
    if not in_scope:
        return CoverageGate("not_applicable", "no_in_scope_records")
    payload = state.metadata.get("sss_coverage_blocks")
    if not isinstance(payload, list):
        return CoverageGate("unavailable", "structured_blocks_unavailable")
    records = source_records(state.evidence, focus=focus)
    if records is None:
        return CoverageGate("unavailable", "source_relationships_unresolved")
    result = validate(records, blocks_from_payload(payload), focus=focus)
    return CoverageGate("authoritative", None, result)


def classify_verifier_payload(raw: Any) -> str:
    """Separate a schema envelope from a verdict. Does not change parsing."""
    if not isinstance(raw, dict):
        return "invalid"
    properties = raw.get("properties")
    if (
        raw.get("title") == "VerificationResult"
        and raw.get("type") == "object"
        and isinstance(properties, dict)
        and "passed" not in raw
        and "recommendation" not in raw
    ):
        return "schema_envelope"
    if "passed" in raw or "recommendation" in raw:
        return "verdict"
    return "unrecognized"


def _expected_parent(row: SourceRecord, source: dict[str, SourceRecord]) -> str | None:
    if not row.parent_id:
        return None
    parent = source.get(row.parent_id)
    if parent is None:
        return None
    if row.entity_type == "LEARNING_OUTCOME" and parent.entity_type in {
        "THEME",
        "TOPIC",
        "LEARNING_OUTCOME",
    }:
        return parent.entity_id
    if parent.entity_type != "THEME":
        return None
    if row.parent_entity_type not in {None, "THEME"}:
        return None
    return parent.entity_id


def _placement_error(
    row: SourceRecord,
    expected: str | None,
    actual: str | None,
) -> StructuralError:
    actual_text = actual or "top level"
    if row.parent_id and row.parent_entity_type == "SECTION":
        section = f" ({row.section_name})" if row.section_name else ""
        return StructuralError(
            "incorrect_section",
            row.entity_id,
            (
                f"{row.entity_id} is assigned to {actual_text}. "
                f"Its source parent is section {row.parent_id}{section}, "
                "which is omitted from the answer rows. The relationship is known, "
                "so the topic stays at the top level."
            ),
        )
    if row.parent_id and expected is None:
        return StructuralError(
            "incorrect_parent",
            row.entity_id,
            (
                f"{row.entity_id} is assigned to {actual_text}. "
                f"Its source parent {row.parent_id} is absent from the evidence list, "
                "so it must not be attached by display name."
            ),
        )
    if expected is None:
        return StructuralError(
            "incorrect_parent",
            row.entity_id,
            f"{row.entity_id} is genuinely parentless and is assigned to {actual_text}.",
        )
    return StructuralError(
        "incorrect_parent",
        row.entity_id,
        (
            f"{row.entity_id} is assigned to {actual or 'top level'}. "
            f"Its source parent is {expected}."
        ),
    )


def _block_from_dict(payload: dict[str, Any]) -> CoverageBlock:
    children = tuple(_block_from_dict(child) for child in payload.get("children") or [])
    return CoverageBlock(
        entity_id=payload.get("entity_id"),
        entity_type=str(payload.get("entity_type") or ""),
        name=str(payload.get("name") or ""),
        source_parent_id=payload.get("source_parent_id"),
        parent_entity_type=payload.get("parent_entity_type"),
        parent_display_name=payload.get("parent_display_name"),
        section_id=payload.get("section_id"),
        section_name=payload.get("section_name"),
        children=children,
    )
