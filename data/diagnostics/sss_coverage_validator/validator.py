"""Deterministic check for syllabus-coverage answers.

Isolated proof of concept. The agent does not import this module.
Relationships come only from stored parent ids. A shared display name is
never a parent.
"""

from __future__ import annotations

from dataclasses import dataclass


# Coverage and topic questions list themes and topics. Sections are context.
# Subtopics are outside that scope. Outcome questions also require outcomes.
SCOPE = {
    "coverage": frozenset({"THEME", "TOPIC"}),
    "topics": frozenset({"THEME", "TOPIC"}),
    "outcomes": frozenset({"THEME", "TOPIC", "LEARNING_OUTCOME"}),
}
EMITTED = frozenset({"THEME", "TOPIC", "LEARNING_OUTCOME"})


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
class AnswerBlock:
    entity_id: str
    entity_type: str
    children: tuple["AnswerBlock", ...] = ()


@dataclass(frozen=True)
class StructuralError:
    code: str
    entity_id: str | None
    message: str


@dataclass(frozen=True)
class StructuralResult:
    passed: bool
    errors: tuple[StructuralError, ...]


def records_from_tree(tree: list) -> list[SourceRecord]:
    """Copy parent ids from an API content tree. Do not invent any."""
    rows: list[SourceRecord] = []

    def walk(nodes, parent_node, section) -> None:
        for node in nodes or []:
            if not isinstance(node, dict):
                continue
            content_type = str(node.get("content_type") or "").upper()
            name = node.get("name") or node.get("statement") or node.get("description")
            node_id = str(node["id"]) if node.get("id") is not None else None
            current = {
                "id": node_id,
                "content_type": content_type,
                "name": str(name) if name else None,
            }
            next_section = section
            if content_type == "SECTION" and name and node_id:
                next_section = {"id": node_id, "name": str(name)}
            if content_type in EMITTED and name and node_id:
                raw_parent = node.get("parent_id")
                parent_id = str(raw_parent) if raw_parent else None
                parent_type = None
                parent_in_evidence = None
                if parent_id and parent_node and parent_node.get("id") == parent_id:
                    parent_type = parent_node.get("content_type") or None
                    parent_in_evidence = parent_type in EMITTED
                elif parent_id:
                    parent_in_evidence = False
                omitted = []
                for child in node.get("children") or []:
                    if not isinstance(child, dict):
                        continue
                    child_type = str(child.get("content_type") or "").upper()
                    if not child_type or child_type in EMITTED or child_type == "SECTION":
                        continue
                    child_id = child.get("id")
                    if child_id is not None:
                        omitted.append((str(child_id), child_type))
                rows.append(
                    SourceRecord(
                        entity_id=node_id,
                        entity_type=content_type,
                        name=str(name),
                        parent_id=parent_id,
                        parent_entity_type=parent_type,
                        parent_in_evidence=parent_in_evidence,
                        section_id=section["id"] if section else None,
                        section_name=section["name"] if section else None,
                        omitted_child_ids=tuple(item[0] for item in omitted),
                        omitted_child_types=tuple(sorted({item[1] for item in omitted})),
                    )
                )
            walk(node.get("children") or [], current, next_section)

    walk(tree if isinstance(tree, list) else [tree], None, None)
    return rows


def faithful_blocks(records: list[SourceRecord], *, focus: str = "coverage") -> list[AnswerBlock]:
    """Group by parent id. This is the structured answer, before text."""
    scope = SCOPE[focus]
    scoped = [row for row in records if row.entity_type in scope]
    theme_ids = {row.entity_id for row in scoped if row.entity_type == "THEME"}
    children: dict[str, list[AnswerBlock]] = {theme_id: [] for theme_id in theme_ids}
    top_level_topics: list[AnswerBlock] = []
    for row in scoped:
        if row.entity_type == "THEME":
            continue
        block = AnswerBlock(row.entity_id, row.entity_type.lower())
        if row.parent_id in theme_ids and row.parent_entity_type == "THEME":
            children[row.parent_id].append(block)
        else:
            top_level_topics.append(block)
    blocks = [
        AnswerBlock(
            row.entity_id,
            "theme",
            tuple(children[row.entity_id]),
        )
        for row in scoped
        if row.entity_type == "THEME"
    ]
    blocks.extend(top_level_topics)
    return blocks


def validate(
    records: list[SourceRecord],
    blocks: list[AnswerBlock],
    *,
    focus: str = "coverage",
) -> StructuralResult:
    scope = SCOPE[focus]
    source = {row.entity_id: row for row in records if row.entity_type in scope}
    names: dict[str, list[str]] = {}
    for row in source.values():
        names.setdefault(row.name, []).append(row.entity_id)

    placed: dict[str, str | None] = {}
    errors: list[StructuralError] = []

    def walk(nodes: list[AnswerBlock] | tuple[AnswerBlock, ...], parent_id: str | None) -> None:
        for node in nodes:
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
        errors.append(
            StructuralError(
                "incorrect_parent",
                entity_id,
                _parent_message(row, expected, answer_parent),
            )
        )

    return StructuralResult(passed=not errors, errors=tuple(errors))


def combined_decision(structural: StructuralResult, llm_passed: bool | None) -> str:
    """Coverage scope is decided by the structural check.

    A generic verifier's completeness objection does not override a
    structural pass. A structural failure is a rejection even when the
    generic verifier accepts.
    """
    if structural.passed:
        return "accept"
    return "reject"


def _expected_parent(row: SourceRecord, source: dict[str, SourceRecord]) -> str | None:
    if not row.parent_id:
        return None
    parent = source.get(row.parent_id)
    if parent is None or parent.entity_type != "THEME":
        # Section, omitted row, or a parent type outside the evidence list.
        return None
    if row.parent_entity_type not in {None, "THEME"}:
        return None
    return parent.entity_id


def _parent_message(row: SourceRecord, expected: str | None, actual: str | None) -> str:
    actual_text = actual or "top level"
    if row.parent_id and row.parent_entity_type == "SECTION":
        return (
            f"{row.entity_id} is assigned to {actual_text}. "
            f"Its source parent is section {row.parent_id}"
            f"{f' ({row.section_name})' if row.section_name else ''}, "
            "which is omitted from the evidence list. The relationship is known, "
            "so the topic stays at the top level."
        )
    if row.parent_id and expected is None:
        return (
            f"{row.entity_id} is assigned to {actual_text}. "
            f"Its source parent {row.parent_id} is absent from the evidence list, "
            "so it must not be attached by display name."
        )
    if expected is None:
        return (
            f"{row.entity_id} is genuinely parentless and is assigned to {actual_text}."
        )
    return (
        f"{row.entity_id} is assigned to {actual or 'top level'}. "
        f"Its source parent is {expected}."
    )
