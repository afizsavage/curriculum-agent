"""Verifier evidence keeps API parent ids after ranking."""

from app.agent.answer_generator import format_evidence_for_prompt
from app.agent.state import CurriculumQAState
from app.curriculum.evidence import EvidenceStatus
from app.tools.curriculum import _content_evidence

TERM_1 = "e38f5d99-7a16-4403-9281-26c8afa0e697"
TERM_2 = "35b5740e-058a-4812-aaa6-c97233352c50"
TERM_3 = "78ec3c1c-86ee-4b0e-aab7-6d435060e86e"
THEME_1 = "192d2900-d79a-4e1f-b775-6c03203fb146"
THEME_2 = "1cc9ab80-311a-4644-bd3f-45a051f3aa7f"
THEME_3 = "e366dff5-acc3-4ecd-a108-db8f9637ec6e"
WHOLE_CLASS = "446af2a3-9000-464c-88fd-211afcf86c42"
NUCLEAR = "d3d5d312-e768-44b0-ad60-9d638c7211ab"
ENV_TERM_1 = "85423a42-f5ec-4c42-b0a7-5f1d955704e4"
ENV_TERM_2 = "5436cc23-aa8e-4919-ba94-4eae3e3cf8d7"
SOURCES = "44d4e2b2-3511-48fd-8191-dffab1a52f1b"
REVISION = "878307d7-fe08-45af-8085-978600b40227"
THEME_NAME = "Research, group or independent work on reading and revision"


def _node(node_id, content_type, name, parent_id=None, children=None):
    return {
        "id": node_id,
        "parent_id": parent_id,
        "content_type": content_type,
        "name": name,
        "children": children or [],
    }


def _topic(node_id, name, parent_id):
    return _node(node_id, "TOPIC", name, parent_id)


def _literature_tree():
    def block(section_id, theme_id, term, third):
        topics = [
            _topic(f"{theme_id}-review", "Review knowledge and skills covered in SS1 & SS2", theme_id),
            _topic(third, "Teacher introduces the whole class task", theme_id)
            if third == WHOLE_CLASS
            else _topic(f"{theme_id}-task", "Teacher introduces the research and independent / group task", theme_id),
            _topic(f"{theme_id}-exam", "Exam Preparation", theme_id),
        ]
        return _node(
            section_id,
            "SECTION",
            term,
            children=[_node(theme_id, "THEME", THEME_NAME, section_id, topics)],
        )

    return [
        block(TERM_1, THEME_1, "Term 1", None),
        block(TERM_2, THEME_2, "Term 2", None),
        block(TERM_3, THEME_3, "Term 3", WHOLE_CLASS),
    ]


def _environment_tree():
    return [
        _node(
            ENV_TERM_1,
            "SECTION",
            "Term 1",
            children=[
                _node(
                    NUCLEAR,
                    "THEME",
                    "Nuclear Energy",
                    ENV_TERM_1,
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
            children=[
                _topic(
                    REVISION,
                    "Revision of the entire syllabus and preparation for WASSCE exams",
                    ENV_TERM_2,
                )
            ],
        ),
    ]


def _rows(tree, subject):
    return _content_evidence(
        tree,
        grade="SSS_3",
        subject=subject,
        stream_name="Languages & Literatures",
        source_reference="syllabus.pdf",
    )


def _by_id(rows):
    return {row.entity_id: row for row in rows}


def _record_text(block: str, entity_id: str) -> str:
    marker = f"Entity ID: {entity_id}"
    start = block.index(marker)
    next_record = block.find("\n--- Record ", start)
    return block[start:] if next_record < 0 else block[start:next_record]


def test_literature_themes_keep_distinct_ids_and_term_context():
    rows = _by_id(_rows(_literature_tree(), "African Literature"))
    assert [rows[THEME_1].metadata["section_name"], rows[THEME_2].metadata["section_name"], rows[THEME_3].metadata["section_name"]] == [
        "Term 1",
        "Term 2",
        "Term 3",
    ]
    assert rows[THEME_1].metadata["section_id"] == TERM_1
    assert rows[THEME_3].metadata["section_id"] == TERM_3
    assert rows[WHOLE_CLASS].metadata["parent_id"] == THEME_3
    assert rows[WHOLE_CLASS].metadata["parent_id"] != THEME_1
    assert rows[WHOLE_CLASS].metadata["parent_in_evidence"] is True
    assert rows[WHOLE_CLASS].metadata["section_name"] == "Term 3"
    block = format_evidence_for_prompt(list(reversed(list(rows.values()))), question="cover")
    record = _record_text(block, WHOLE_CLASS)
    assert f"Parent ID: {THEME_3}" in record
    assert f"Parent ID: {THEME_1}" not in record
    assert "Section: Term 3" in record
    assert "Section: Term 1" not in record


def test_environment_revision_stays_with_term_2_and_sources_stay_with_nuclear():
    rows = _by_id(_rows(_environment_tree(), "Environmental Science"))
    assert rows[SOURCES].metadata["parent_id"] == NUCLEAR
    assert rows[SOURCES].metadata["parent_entity_type"] == "THEME"
    assert rows[SOURCES].metadata["parent_in_evidence"] is True
    assert rows[REVISION].metadata["parent_id"] == ENV_TERM_2
    assert rows[REVISION].metadata["parent_entity_type"] == "SECTION"
    assert rows[REVISION].metadata["parent_in_evidence"] is False
    assert rows[REVISION].metadata["parent_id"] != NUCLEAR
    assert rows[REVISION].metadata["section_id"] == ENV_TERM_2
    # Direct subtopics are real source records. They are not evidence rows.
    assert rows[SOURCES].metadata["omitted_child_count"] == 4
    assert rows[SOURCES].metadata["omitted_child_types"] == ["SUBTOPIC"]
    assert "920f1a1f-8833-46ea-ac17-ef2cb02a0ba9" in rows[SOURCES].metadata["omitted_child_ids"]
    assert all(row.entity_type != "subtopic" for row in rows.values())
    assert all(row.entity_type != "section" for row in rows.values())
    block = format_evidence_for_prompt(list(rows.values()), question="cover")
    revision = _record_text(block, REVISION)
    assert f"Parent ID: {ENV_TERM_2}" in revision
    assert "Parent in evidence: no" in revision
    assert f"Parent ID: {NUCLEAR}" not in revision
    sources = _record_text(block, SOURCES)
    assert f"Parent ID: {NUCLEAR}" in sources
    assert "Omitted child records: 4 SUBTOPIC not included in this payload" in sources


def test_parentless_topic_is_not_assigned_by_a_matching_name():
    shared = "Workshop"
    tree = [
        _node("theme-1", "THEME", shared, children=[]),
        _node("loose", "TOPIC", "Draft"),
    ]
    rows = _by_id(
        _content_evidence(
            tree,
            grade="SSS_1",
            subject="Fula",
            stream_name="Languages & Literatures",
            source_reference=None,
        )
    )
    assert rows["loose"].metadata["parent_id"] is None
    assert "parent_display_name" not in rows["loose"].metadata
    assert rows["loose"].metadata["parent_in_evidence"] is None
    block = format_evidence_for_prompt(list(rows.values()), question="cover")
    loose = _record_text(block, "loose")
    assert "Parent ID: none" in loose
    assert "Parent name:" not in loose


def test_reordering_does_not_change_parent_association():
    rows = _rows(_literature_tree(), "African Literature")
    forward = format_evidence_for_prompt(rows, question="African Literature cover")
    backward = format_evidence_for_prompt(list(reversed(rows)), question="African Literature cover")
    assert f"Parent ID: {THEME_3}" in _record_text(forward, WHOLE_CLASS)
    assert f"Parent ID: {THEME_3}" in _record_text(backward, WHOLE_CLASS)


def test_coverage_answer_grouping_is_unchanged():
    from app.agent.answer_generator import _render_sss_grade_coverage

    literature = _rows(_literature_tree(), "African Literature")
    state = CurriculumQAState(
        question="What does African Literature cover?",
        intent="SSS_STREAM_SUBJECTS",
        grade="SSS_3",
        evidence=literature,
        evidence_status=EvidenceStatus.FOUND,
        metadata={"subject_name": "African Literature", "sss_focus": "coverage"},
    )
    answer, _limitations, _used = _render_sss_grade_coverage(
        state, literature, "Languages & Literatures", "SSS 3", "coverage"
    )
    assert answer.count(THEME_NAME) == 3
    assert "Teacher introduces the whole class task" in answer
    third = answer.split(f"* {THEME_NAME}")[-1]
    assert "Teacher introduces the whole class task" in third
    environment = _rows(_environment_tree(), "Environmental Science")
    env_state = state.model_copy(
        update={
            "evidence": environment,
            "metadata": {"subject_name": "Environmental Science", "sss_focus": "coverage"},
        }
    )
    env_answer, _limitations, _used = _render_sss_grade_coverage(
        env_state, environment, "Sciences & Technologies", "SSS 3", "coverage"
    )
    assert "* Nuclear Energy\n  * Sources:" in env_answer
    assert "\n* Revision of the entire syllabus and preparation for WASSCE exams" in env_answer
    nuclear = env_answer.split("* Nuclear Energy", 1)[1].split("\n* Revision", 1)[0]
    assert "Revision of the entire syllabus" not in nuclear
