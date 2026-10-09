"""Structural coverage checks. These do not call the verifier."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data/diagnostics/sss_coverage_validator"))

from validator import (  # noqa: E402
    AnswerBlock,
    faithful_blocks,
    records_from_tree,
    validate,
)

THEME_NAME = "Research, group or independent work on reading and revision"
TERM_1 = "e38f5d99-7a16-4403-9281-26c8afa0e697"
TERM_3 = "78ec3c1c-86ee-4b0e-aab7-6d435060e86e"
THEME_1 = "192d2900-d79a-4e1f-b775-6c03203fb146"
THEME_3 = "e366dff5-acc3-4ecd-a108-db8f9637ec6e"
WHOLE_CLASS = "446af2a3-9000-464c-88fd-211afcf86c42"
NUCLEAR = "d3d5d312-e768-44b0-ad60-9d638c7211ab"
ENV_TERM_2 = "5436cc23-aa8e-4919-ba94-4eae3e3cf8d7"
SOURCES = "44d4e2b2-3511-48fd-8191-dffab1a52f1b"
REVISION = "878307d7-fe08-45af-8085-978600b40227"


def _node(node_id, content_type, name, parent_id=None, children=None):
    return {
        "id": node_id,
        "parent_id": parent_id,
        "content_type": content_type,
        "name": name,
        "children": children or [],
    }


def _literature():
    def block(section_id, theme_id, term, third_name, third_id):
        return _node(
            section_id,
            "SECTION",
            term,
            children=[
                _node(
                    theme_id,
                    "THEME",
                    THEME_NAME,
                    section_id,
                    [
                        _node(f"{theme_id}-review", "TOPIC", "Review", theme_id),
                        _node(third_id, "TOPIC", third_name, theme_id),
                        _node(f"{theme_id}-exam", "TOPIC", "Exam Preparation", theme_id),
                    ],
                )
            ],
        )

    return [
        block(TERM_1, THEME_1, "Term 1", "Teacher introduces the research task", f"{THEME_1}-task"),
        block("35b5740e-058a-4812-aaa6-c97233352c50", "1cc9ab80-311a-4644-bd3f-45a051f3aa7f", "Term 2", "Teacher introduces the research task", "1cc9ab80-task"),
        block(TERM_3, THEME_3, "Term 3", "Teacher introduces the whole class task", WHOLE_CLASS),
    ]


def _environment():
    return [
        _node(
            "85423a42-f5ec-4c42-b0a7-5f1d955704e4",
            "SECTION",
            "Term 1",
            children=[
                _node(
                    NUCLEAR,
                    "THEME",
                    "Nuclear Energy",
                    "85423a42-f5ec-4c42-b0a7-5f1d955704e4",
                    [
                        _node(
                            SOURCES,
                            "TOPIC",
                            "Sources:",
                            NUCLEAR,
                            [
                                _node("920f1a1f-8833-46ea-ac17-ef2cb02a0ba9", "SUBTOPIC", "Nuclear fuel processing", SOURCES),
                                _node("2d8eb662-6172-472e-91b0-a8d8f7e9165c", "SUBTOPIC", "Nuclear power plant", SOURCES),
                                _node("5f3fc3a1-627f-449e-8339-bbd6e9743b78", "SUBTOPIC", "Benefits of Nuclear energy.", SOURCES),
                                _node("100a8db9-08ea-4588-8320-410bfca2a269", "SUBTOPIC", "Draw backs of nuclear energy.", SOURCES),
                            ],
                        )
                    ],
                )
            ],
        ),
        _node(
            ENV_TERM_2,
            "SECTION",
            "Term 2",
            children=[_node(REVISION, "TOPIC", "Revision of the entire syllabus", ENV_TERM_2)],
        ),
    ]


def _mutate_m1(blocks):
    first, second, third = blocks
    merged = AnswerBlock(first.entity_id, first.entity_type, first.children + third.children)
    return [merged, second]


def _mutate_m2(blocks):
    theme, revision = blocks
    return [
        AnswerBlock(
            theme.entity_id,
            theme.entity_type,
            theme.children + (revision,),
        )
    ]


def test_literature_accepts_distinct_theme_occurrences():
    records = records_from_tree(_literature())
    result = validate(records, faithful_blocks(records))
    assert result.passed
    whole = next(row for row in records if row.entity_id == WHOLE_CLASS)
    assert whole.parent_id == THEME_3
    assert whole.section_id == TERM_3
    assert whole.parent_id != THEME_1


def test_m1_reports_the_missing_theme_and_the_wrong_parent():
    records = records_from_tree(_literature())
    result = validate(records, _mutate_m1(faithful_blocks(records)))
    assert not result.passed
    codes = {error.code for error in result.errors}
    assert "missing_theme_occurrence" in codes
    assert "incorrect_parent" in codes
    missing = next(error for error in result.errors if error.entity_id == THEME_3)
    assert missing.code == "missing_theme_occurrence"
    parent = next(error for error in result.errors if error.entity_id == WHOLE_CLASS)
    assert THEME_3 in parent.message
    assert THEME_1 in parent.message


def test_environment_accepts_a_section_parent_and_ignores_subtopics():
    records = records_from_tree(_environment())
    result = validate(records, faithful_blocks(records))
    assert result.passed
    sources = next(row for row in records if row.entity_id == SOURCES)
    assert sources.omitted_child_types == ("SUBTOPIC",)
    assert len(sources.omitted_child_ids) == 4
    revision = next(row for row in records if row.entity_id == REVISION)
    assert revision.parent_id == ENV_TERM_2
    assert revision.parent_entity_type == "SECTION"
    assert revision.parent_in_evidence is False


def test_m2_reports_the_revision_topic_under_nuclear_energy():
    records = records_from_tree(_environment())
    result = validate(records, _mutate_m2(faithful_blocks(records)))
    assert not result.passed
    error = next(item for item in result.errors if item.entity_id == REVISION)
    assert error.code == "incorrect_parent"
    assert NUCLEAR in error.message
    assert ENV_TERM_2 in error.message
    assert "Sources:" not in error.message


def test_parentless_topic_is_not_attached_by_a_shared_name():
    shared = "Workshop"
    records = records_from_tree(
        [
            _node("theme-1", "THEME", shared),
            _node("loose", "TOPIC", "Draft"),
        ]
    )
    assert validate(records, faithful_blocks(records)).passed
    attached = [
        AnswerBlock("theme-1", "theme", (AnswerBlock("loose", "topic"),)),
    ]
    result = validate(records, attached)
    assert not result.passed
    assert result.errors[0].code == "incorrect_parent"
    assert "genuinely parentless" in result.errors[0].message


def test_omitted_theme_parent_is_not_replaced_by_the_same_display_name():
    records = records_from_tree(
        [
            _node("present", "THEME", "Heat"),
            _node("topic-1", "TOPIC", "Conduction", "absent-theme"),
        ]
    )
    # The walked parent is absent, so the topic's parent type stays unknown.
    topic = next(row for row in records if row.entity_id == "topic-1")
    assert topic.parent_id == "absent-theme"
    assert topic.parent_entity_type is None
    assert validate(records, faithful_blocks(records)).passed
    stolen = [AnswerBlock("present", "theme", (AnswerBlock("topic-1", "topic"),))]
    result = validate(records, stolen)
    assert result.errors[0].code == "incorrect_parent"
    assert "absent-theme" in result.errors[0].message


def test_reordering_blocks_does_not_change_the_verdict():
    records = records_from_tree(_literature())
    blocks = faithful_blocks(records)
    assert validate(records, list(reversed(blocks))).passed
