"""Structural fidelity of the deterministic SSS theme/topic renderer."""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from app.agent.answer_generator import (
    _render_sss_grade_coverage,
    _sss_parentless_topics,
    _sss_theme_occurrences,
)
from app.agent.state import CurriculumQAState
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus

MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "diagnostics"
    / "sss_grade_stream_experiment"
    / "manifest.json"
)

def _item(kind: str, name: str, parent: str | None = None) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type=kind,
        name=name,
        metadata={"parent_name": parent},
    )


def _parent(item: CurriculumEvidence) -> str | None:
    raw = (item.metadata or {}).get("parent_name")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


@lru_cache(maxsize=1)
def _syllabi() -> dict[tuple[str, str], list[CurriculumEvidence]]:
    manifest = json.loads(MANIFEST.read_text())
    found: dict[tuple[str, str], list[CurriculumEvidence]] = {}
    for row in manifest["syllabi"]:
        content = []
        for item in row.get("content") or []:
            kind = str(item.get("type") or "").lower()
            name = item.get("name")
            if kind not in {"theme", "topic", "learning_outcome"} or not name:
                continue
            content.append(
                CurriculumEvidence(
                    entity_type=kind,
                    name=str(name),
                    grade=row["grade"],
                    subject=row["subject_name"],
                    metadata={"parent_name": item.get("parent")},
                    source_reference=row.get("source_reference"),
                )
            )
        found[(row["subject_name"], row["grade"])] = content
    return found


def _content(subject: str, grade: str) -> list[CurriculumEvidence]:
    key = (subject, grade)
    if key not in _syllabi():
        raise AssertionError(f"missing frozen syllabus for {subject} {grade}")
    return _syllabi()[key]


def _render(subject: str, grade: str, content: list[CurriculumEvidence]) -> str:
    state = CurriculumQAState(
        question=f"What does {subject} cover?",
        intent="SSS_STREAM_SUBJECTS",
        grade=grade,
        evidence=content,
        evidence_status=EvidenceStatus.FOUND,
        metadata={
            "subject_name": subject,
            "grade": grade,
            "resolved_stream_name": "Mathematics & Numeracy",
            "sss_focus": "coverage",
        },
    )
    text, _limitations, _used = _render_sss_grade_coverage(
        state,
        content,
        "Mathematics & Numeracy",
        grade.replace("_", " "),
        "coverage",
    )
    return text


def _sections(answer: str) -> list[dict]:
    sections: list[dict] = []
    current = None
    for line in answer.splitlines():
        match = re.match(r"^(\s*)\*\s+(.+?)\s*$", line)
        if not match:
            continue
        indent = len(match.group(1))
        text = match.group(2)
        if indent == 0:
            current = {"name": text, "topics": []}
            sections.append(current)
        elif current is None:
            raise AssertionError(f"child bullet before a heading: {text}")
        else:
            current["topics"].append(text)
    return sections


def _source_parentless(content: list[CurriculumEvidence]) -> list[str]:
    return [
        item.name
        for item in content
        if (item.entity_type or "").lower() == "topic" and item.name and _parent(item) is None
    ]


def _source_occurrences(content: list[CurriculumEvidence]) -> list[tuple[str, list[str]]]:
    """Assign topics from the frozen records, without using the renderer."""
    themes = [
        (index, item)
        for index, item in enumerate(content)
        if (item.entity_type or "").lower() == "theme" and item.name
    ]
    counts = Counter(item.name for _index, item in themes)
    assigned: dict[int, list[str]] = {index: [] for index, _item in themes}
    first_index: dict[str, int] = {}
    last_seen: dict[str, int] = {}
    for index, item in themes:
        first_index.setdefault(item.name, index)
    for index, item in enumerate(content):
        kind = (item.entity_type or "").lower()
        if kind == "theme" and item.name:
            last_seen[item.name] = index
            continue
        if kind != "topic" or not item.name:
            continue
        parent = _parent(item)
        if not parent or counts.get(parent, 0) <= 1:
            continue
        owner = last_seen.get(parent, first_index.get(parent))
        if owner is not None:
            assigned[owner].append(item.name)
    for index, item in themes:
        if counts[item.name] == 1:
            assigned[index] = [
                topic.name
                for topic in content
                if (topic.entity_type or "").lower() == "topic"
                and topic.name
                and _parent(topic) == item.name
            ]
    return [(item.name, assigned[index]) for index, item in themes]


def assert_rendered_structure(content: list[CurriculumEvidence], answer: str) -> None:
    """Fail when the answer drops, invents, merges, or reorders syllabus records."""
    expected_themes = _source_occurrences(content)
    expected_parentless = _source_parentless(content)
    sections = _sections(answer)
    theme_sections = sections[: len(expected_themes)]
    tail = sections[len(expected_themes) :]
    assert [section["name"] for section in theme_sections] == [
        name for name, _topics in expected_themes
    ]
    assert [section["topics"] for section in theme_sections] == [
        topics for _name, topics in expected_themes
    ]
    assert [section["name"] for section in tail] == expected_parentless
    assert all(section["topics"] == [] for section in tail)
    source_names = {
        item.name
        for item in content
        if (item.entity_type or "").lower() in {"theme", "topic"} and item.name
    }
    for section in sections:
        assert section["name"] in source_names
        for topic in section["topics"]:
            assert topic in source_names
    by_name: dict[str, list[list[str]]] = {}
    for name, topics in expected_themes:
        by_name.setdefault(name, []).append(list(topics))
    for name, groups in by_name.items():
        if len(groups) < 2:
            continue
        flat = [topic for group in groups for topic in group]
        parented = [
            item.name
            for item in content
            if (item.entity_type or "").lower() == "topic" and item.name and _parent(item) == name
        ]
        assert flat == parented


def test_parentless_topics_stay_in_source_order_and_are_not_nested():
    content = [
        _item("theme", "Letters"),
        _item("topic", "Official letter", "Letters"),
        _item("topic", "Colloquial French"),
        _item("topic", "Debate"),
    ]
    names = [item.name for item in _sss_parentless_topics(content)]
    assert names == ["Colloquial French", "Debate"]
    occurrences = _sss_theme_occurrences(content)
    assert [item.name for item in occurrences[0][1]] == ["Official letter"]
    assert all(item.name != "Debate" for _theme, children in occurrences for item in children)


def test_repeated_theme_records_keep_their_own_children():
    content = [
        _item("theme", "Heat"),
        _item("topic", "A1", "Heat"),
        _item("topic", "A2", "Heat"),
        _item("theme", "Heat"),
        _item("topic", "B1", "Heat"),
        _item("topic", "B2", "Heat"),
    ]
    occurrences = _sss_theme_occurrences(content)
    assert [item.name for item, _children in occurrences] == ["Heat", "Heat"]
    assert [child.name for child in occurrences[0][1]] == ["A1", "A2"]
    assert [child.name for child in occurrences[1][1]] == ["B1", "B2"]
    answer = _render("Engineering Science", "SSS_2", content)
    assert_rendered_structure(content, answer)
    assert "Heat 1" not in answer
    assert "Heat 2" not in answer


def test_empty_repeated_theme_occurrence_remains():
    content = [
        _item("theme", "Workshop"),
        _item("topic", "Draft", "Workshop"),
        _item("theme", "Workshop"),
        _item("theme", "Talk"),
    ]
    occurrences = _sss_theme_occurrences(content)
    assert [len(children) for _theme, children in occurrences] == [1, 0, 0]
    answer = _render("Creative Writing", "SSS_3", content)
    sections = _sections(answer)
    assert [section["name"] for section in sections] == ["Workshop", "Workshop", "Talk"]
    assert sections[1]["topics"] == []


def test_unique_theme_still_collects_its_parented_topics():
    content = [
        _item("theme", "Geology"),
        _item("theme", "Basement"),
        _item("topic", "The intrusive", "Geology"),
    ]
    occurrences = _sss_theme_occurrences(content)
    assert [child.name for child in occurrences[0][1]] == ["The intrusive"]
    assert occurrences[1][1] == []


def test_french_parentless_topics_are_rendered():
    _assert_parentless_case("French as a Foreign Language", "SSS_3")


def test_fundamentals_parentless_topics_are_rendered():
    _assert_parentless_case("Fundamentals of Mathematics", "SSS_1")


def test_calculus_parentless_topics_are_rendered():
    _assert_parentless_case("Calculus", "SSS_1")


def test_statistics_parentless_topics_are_rendered():
    _assert_parentless_case("Statistics and Probability", "SSS_2")


def _assert_parentless_case(subject: str, grade: str) -> None:
    content = _content(subject, grade)
    parentless = _source_parentless(content)
    assert parentless, f"{subject} {grade} should contain parentless topics"
    assert [item.name for item in _sss_parentless_topics(content)] == parentless
    answer = _render(subject, grade, content)
    assert f"{subject} covers:" in answer
    assert_rendered_structure(content, answer)
    assert "parentless" not in answer.casefold()
    assert "theme occurrence" not in answer.casefold()
    assert "entity_id" not in answer


def test_creative_writing_repeated_themes_stay_separate():
    _assert_repeated_case("Creative Writing", "SSS_3")


def test_music_repeated_themes_stay_separate():
    _assert_repeated_case("Music", "SSS_1")


def test_steamm_sss1_repeated_themes_stay_separate():
    _assert_repeated_case("Mathematics for STEAMM", "SSS_1")


def test_steamm_sss2_repeated_themes_stay_separate():
    _assert_repeated_case("Mathematics for STEAMM", "SSS_2")


def test_engineering_repeated_themes_stay_separate():
    _assert_repeated_case("Engineering Science", "SSS_2")


def _assert_repeated_case(subject: str, grade: str) -> None:
    content = _content(subject, grade)
    expected = _source_occurrences(content)
    counts = Counter(name for name, _topics in expected)
    assert any(count > 1 for count in counts.values())
    rendered_records = _sss_theme_occurrences(content)
    rendered = [
        (theme.name, [child.name for child in children]) for theme, children in rendered_records
    ]
    assert rendered == expected
    attached = [child for _theme, children in rendered_records for child in children]
    assert len(attached) == len({id(child) for child in attached})
    answer = _render(subject, grade, content)
    assert_rendered_structure(content, answer)
    assert "Heat 1" not in answer
    assert "theme occurrence" not in answer.casefold()


def test_music_renders_parentless_topics_and_separate_theme_occurrences():
    content = _content("Music", "SSS_1")
    parentless = _source_parentless(content)
    assert parentless == [
        "The place of music in human development in the region",
        "The place of music in human development in the world",
    ]
    western = [
        topics
        for name, topics in _source_occurrences(content)
        if name == "Western tradition"
    ]
    assert [len(topics) for topics in western] == [3, 1, 1, 1]
    assert western[0] != western[0] + western[1]
    answer = _render("Music", "SSS_1", content)
    assert_rendered_structure(content, answer)
    sections = _sections(answer)
    assert [section["name"] for section in sections[-2:]] == parentless
    for section in sections[:-2]:
        assert "The place of music in human development in the world" not in section["topics"]
