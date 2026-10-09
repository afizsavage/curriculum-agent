"""SSS coverage validation uses structured blocks, not the generic verifier."""

from __future__ import annotations

from app.agent.answer_generator import _render_sss_grade_coverage
from app.agent.coverage_validator import (
    CoverageBlock,
    SourceRecord,
    assess_coverage,
    classify_verifier_payload,
    source_records,
    validate,
)
from app.agent.state import CurriculumQAState
from app.agent.verifier import AnswerVerifier
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.schemas.verification import VERIFICATION_RESULT_JSON_SCHEMA
from app.tools.curriculum import _content_evidence, coverage_expected_from_tree

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

# Observed shape: a completed response whose verdict sits under properties.
SCHEMA_ENVELOPE = {
    "title": "VerificationResult",
    "type": "object",
    "properties": {
        "passed": False,
        "score": 0.35,
        "recommendation": "retrieve_more",
        "issues": [
            "The answer repeats the same three bullets three times, which is not supported by the evidence."
        ],
    },
}


class _ScriptedLLM:
    name = "scripted"
    model = "scripted"

    def __init__(self, payload: dict):
        self.payload = payload

    def generate_structured(self, messages, *, schema, temperature: float = 0.0):
        assert messages
        assert schema is VERIFICATION_RESULT_JSON_SCHEMA
        assert temperature == 0.0
        return self.payload


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
        block(
            "35b5740e-058a-4812-aaa6-c97233352c50",
            "1cc9ab80-311a-4644-bd3f-45a051f3aa7f",
            "Term 2",
            "Teacher introduces the research task",
            "1cc9ab80-task",
        ),
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
                                _node(
                                    "920f1a1f-8833-46ea-ac17-ef2cb02a0ba9",
                                    "SUBTOPIC",
                                    "Nuclear fuel processing",
                                    SOURCES,
                                )
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


def _repeated_themes():
    def theme(theme_id, topic_id, topic_name):
        return _node(
            theme_id,
            "THEME",
            "Life histories of religious leaders",
            children=[_node(topic_id, "TOPIC", topic_name, theme_id)],
        )

    return [
        theme("theme-a", "topic-a", "The life of Jesus Christ"),
        theme("theme-b", "topic-b", "Christianity/Islam in West Africa"),
        theme("theme-c", "topic-c", "The lives of the leaders"),
    ]


def _parentless_topics():
    return [
        _node(f"topic-{index}", "TOPIC", f"Fula topic {index}")
        for index in range(1, 4)
    ]


def _evidence(tree, subject, stream):
    return _content_evidence(
        tree,
        grade="SSS_3",
        subject=subject,
        stream_name=stream,
        source_reference="syllabus.pdf",
    )


def _state(question, evidence, *, focus="coverage", intent="SSS_STREAM_SUBJECTS", subject="Subject"):
    return CurriculumQAState(
        question=question,
        intent=intent,
        grade="SSS_3",
        evidence=evidence,
        evidence_status=EvidenceStatus.FOUND,
        metadata={
            "subject_name": subject,
            "sss_focus": focus,
            "stream_name": "Languages & Literatures",
        },
    )


def _render(state, stream, grade_label="SSS 3"):
    text, _limitations, _used = _render_sss_grade_coverage(
        state,
        [
            item
            for item in state.evidence
            if (item.entity_type or "").lower() in {"theme", "topic", "learning_outcome"}
        ],
        stream,
        grade_label,
        state.metadata["sss_focus"],
    )
    return text


def _bind_expected(
    state,
    tree,
    *,
    subject,
    stream,
    grade="SSS_3",
    source_reference="syllabus.pdf",
    grade_curriculum_id="gc-test",
):
    state.grade = grade
    state.metadata["subject_name"] = subject
    state.metadata["stream_name"] = stream
    state.metadata["resolved_stream_name"] = stream
    state.metadata["source_reference"] = source_reference
    state.metadata["grade_curriculum_id"] = grade_curriculum_id
    state.metadata["sss_coverage_expected"] = coverage_expected_from_tree(
        tree,
        grade=grade,
        subject=subject,
        stream_name=stream,
        source_reference=source_reference,
        grade_curriculum_id=grade_curriculum_id,
    )


def _verify(state, payload):
    if not (state.final_answer or state.draft_answer):
        state.final_answer = "Syllabus coverage answer."
        state.draft_answer = state.final_answer
    return AnswerVerifier(_ScriptedLLM(payload)).verify(state)


def test_d1_accepts_distinct_themes_and_keeps_the_rendered_wording():
    evidence = _evidence(_literature(), "African Literature", "Languages & Literatures")
    state = _state("What does African Literature cover?", evidence, subject="African Literature")
    text = _render(state, "Languages & Literatures")
    _bind_expected(state, _literature(), subject="African Literature", stream="Languages & Literatures", grade_curriculum_id="gc-d1")
    result = _verify(state, {"passed": False, "score": 0.25, "recommendation": "retrieve_more", "issues": ["incomplete"]})
    assert result.passed
    assert result.metadata["coverage_decision"] == "accept"
    assert result.metadata["coverage_reason_codes"] == []
    assert result.metadata["llm_authoritative"] is False
    assert result.metadata["llm_verification"]["recommendation"] == "retrieve_more"
    assert result.metadata["coverage_latency_ms"] < 5
    blocks = state.metadata["sss_coverage_blocks"]
    assert [block["entity_id"] for block in blocks] == [
        THEME_1,
        "1cc9ab80-311a-4644-bd3f-45a051f3aa7f",
        THEME_3,
    ]
    whole = next(
        child
        for block in blocks
        for child in block["children"]
        if child["entity_id"] == WHOLE_CLASS
    )
    assert whole["source_parent_id"] == THEME_3
    assert whole["section_id"] == TERM_3
    assert "Teacher introduces the whole class task" in text
    third = text.split(f"* {THEME_NAME}")[-1]
    assert "Teacher introduces the whole class task" in third
    assert text.split(f"* {THEME_NAME}")[1].count("Teacher introduces the whole class task") == 0


def test_d2_accepts_the_section_parent_and_ignores_the_subtopic():
    evidence = _evidence(_environment(), "Environmental Science", "Sciences & Technologies")
    state = _state(
        "What does Environmental Science cover?",
        evidence,
        subject="Environmental Science",
    )
    text = _render(state, "Sciences & Technologies")
    _bind_expected(
        state,
        _environment(),
        subject="Environmental Science",
        stream="Sciences & Technologies",
        grade_curriculum_id="gc-d2",
    )
    result = _verify(state, {"passed": False, "score": 0.3, "recommendation": "retrieve_more", "issues": ["incomplete"]})
    assert result.passed
    assert result.metadata["coverage_decision"] == "accept"
    sources = next(row for row in source_records(evidence, focus="coverage") if row.entity_id == SOURCES)
    assert sources.omitted_child_ids == ("920f1a1f-8833-46ea-ac17-ef2cb02a0ba9",)
    assert "Nuclear fuel processing" not in text
    assert "\n* Revision of the entire syllabus" in text
    nuclear = text.split("* Nuclear Energy", 1)[1].split("\n* Revision", 1)[0]
    assert "Revision of the entire syllabus" not in nuclear


def test_d3_and_d4_accept_repeated_names_and_parentless_topics():
    repeated = _evidence(_repeated_themes(), "Derivatives of Religious and Moral Education", "Social & Cultural Studies")
    repeated_state = _state("What does it cover?", repeated)
    _render(repeated_state, "Social & Cultural Studies")
    _bind_expected(
        repeated_state,
        _repeated_themes(),
        subject="Derivatives of Religious and Moral Education",
        stream="Social & Cultural Studies",
        grade_curriculum_id="gc-d3",
    )
    repeated_result = _verify(repeated_state, {"passed": True, "score": 0.95, "recommendation": "accept"})
    assert repeated_result.passed
    assert len({block["entity_id"] for block in repeated_state.metadata["sss_coverage_blocks"]}) == 3

    parentless = _evidence(_parentless_topics(), "Fula", "Languages & Literatures")
    parentless_state = _state("What does Fula cover?", parentless, subject="Fula")
    _render(parentless_state, "Languages & Literatures")
    _bind_expected(
        parentless_state,
        _parentless_topics(),
        subject="Fula",
        stream="Languages & Literatures",
        grade_curriculum_id="gc-d4",
    )
    parentless_result = _verify(parentless_state, {"passed": True, "score": 1, "recommendation": "accept"})
    assert parentless_result.passed
    assert all(block["children"] == [] for block in parentless_state.metadata["sss_coverage_blocks"])
    assert all(block["source_parent_id"] is None for block in parentless_state.metadata["sss_coverage_blocks"])


def test_m1_rejects_the_missing_term_3_theme_and_the_moved_topics():
    evidence = _evidence(_literature(), "African Literature", "Languages & Literatures")
    state = _state("What does African Literature cover?", evidence, subject="African Literature")
    _render(state, "Languages & Literatures")
    _bind_expected(state, _literature(), subject="African Literature", stream="Languages & Literatures", grade_curriculum_id="gc-d1")
    first, second, third = state.metadata["sss_coverage_blocks"]
    first["children"] = first["children"] + third["children"]
    state.metadata["sss_coverage_blocks"] = [first, second]
    result = _verify(state, {"passed": True, "score": 0.95, "recommendation": "accept", "issues": []})
    assert not result.passed
    assert result.metadata["coverage_decision"] == "reject"
    assert "missing_theme_occurrence" in result.metadata["coverage_reason_codes"]
    assert result.metadata["coverage_reason_codes"].count("incorrect_parent") == 3
    errors = {item["entity_id"]: item for item in result.metadata["coverage_errors"]}
    assert errors[THEME_3]["code"] == "missing_theme_occurrence"
    assert "Term 3" in errors[THEME_3]["message"]
    assert errors[WHOLE_CLASS]["code"] == "incorrect_parent"
    assert THEME_1 in errors[WHOLE_CLASS]["message"]
    assert THEME_3 in errors[WHOLE_CLASS]["message"]
    assert result.metadata["llm_verification"]["recommendation"] == "accept"


def test_m2_rejects_the_revision_topic_under_nuclear_energy():
    evidence = _evidence(_environment(), "Environmental Science", "Sciences & Technologies")
    state = _state("What does Environmental Science cover?", evidence, subject="Environmental Science")
    _render(state, "Sciences & Technologies")
    _bind_expected(
        state,
        _environment(),
        subject="Environmental Science",
        stream="Sciences & Technologies",
        grade_curriculum_id="gc-d2",
    )
    theme, revision = state.metadata["sss_coverage_blocks"]
    theme["children"] = theme["children"] + [revision]
    state.metadata["sss_coverage_blocks"] = [theme]
    result = _verify(state, {"passed": True, "score": 0.9, "recommendation": "accept"})
    assert not result.passed
    error = result.metadata["coverage_errors"][0]
    assert error["code"] == "incorrect_section"
    assert error["entity_id"] == REVISION
    assert NUCLEAR in error["message"]
    assert ENV_TERM_2 in error["message"]
    assert "Sources:" not in error["message"]


def test_reordering_blocks_does_not_change_parent_associations():
    evidence = _evidence(_literature(), "African Literature", "Languages & Literatures")
    state = _state("What does African Literature cover?", evidence)
    _render(state, "Languages & Literatures")
    _bind_expected(state, _literature(), subject="African Literature", stream="Languages & Literatures", grade_curriculum_id="gc-d1")
    state.metadata["sss_coverage_blocks"] = list(reversed(state.metadata["sss_coverage_blocks"]))
    result = _verify(state, {"passed": False, "score": 0.2, "recommendation": "retrieve_more"})
    assert result.passed
    assert result.metadata["coverage_reason_codes"] == []


def test_missing_topic_is_rejected_and_a_subtopic_is_not_required():
    evidence = _evidence(_environment(), "Environmental Science", "Sciences & Technologies")
    state = _state("What does Environmental Science cover?", evidence)
    _render(state, "Sciences & Technologies")
    _bind_expected(
        state,
        _environment(),
        subject="Environmental Science",
        stream="Sciences & Technologies",
        grade_curriculum_id="gc-d2",
    )
    theme, _revision = state.metadata["sss_coverage_blocks"]
    theme["children"] = []
    state.metadata["sss_coverage_blocks"] = [theme, _revision]
    result = _verify(state, {"passed": True, "score": 1, "recommendation": "accept"})
    assert not result.passed
    assert result.metadata["coverage_errors"][0]["code"] == "missing_entity"
    assert result.metadata["coverage_errors"][0]["entity_id"] == SOURCES


def test_outcome_parent_contract_is_checked_but_not_routed():
    theme = SourceRecord(
        "theme-1",
        "THEME",
        "Workshop",
        None,
        None,
        None,
    )
    outcome = SourceRecord(
        "outcome-1",
        "LEARNING_OUTCOME",
        "Explain the workshop",
        "theme-1",
        "THEME",
        True,
    )
    placed = [
        CoverageBlock(
            "theme-1",
            "theme",
            "Workshop",
            children=(CoverageBlock("outcome-1", "learning_outcome", "Explain the workshop"),),
        )
    ]
    assert validate([theme, outcome], placed, focus="outcomes").passed
    flat = [
        CoverageBlock("theme-1", "theme", "Workshop"),
        CoverageBlock("outcome-1", "learning_outcome", "Explain the workshop"),
    ]
    rejected = validate([theme, outcome], flat, focus="outcomes")
    assert rejected.errors[0].code == "incorrect_parent"
    assert rejected.errors[0].entity_id == "outcome-1"

    evidence = [
        CurriculumEvidence(
            entity_type="learning_outcome",
            entity_id="outcome-1",
            name="Explain the workshop",
            metadata={"parent_id": "theme-1", "parent_entity_type": "THEME"},
        )
    ]
    state = _state("What are the learning outcomes?", evidence, focus="outcomes")
    gate = assess_coverage(state)
    assert gate.status == "not_applicable"
    assert gate.reason == "outcomes_hierarchy_not_represented"
    result = _verify(state, {"passed": False, "score": 0.4, "recommendation": "retrieve_more", "issues": ["generic"]})
    assert result.metadata.get("source") != "coverage_validator"
    assert result.recommendation.value == "retrieve_more"


def test_ordinary_and_non_sss_questions_use_the_generic_verifier():
    evidence = [CurriculumEvidence(entity_type="topic", entity_id="t1", name="Fractions", metadata={"parent_id": None})]
    ordinary = _state("What is a fraction?", evidence, intent="CURRICULUM_QA", focus=None)
    ordinary.metadata.pop("sss_focus")
    result = _verify(ordinary, {"passed": False, "score": 0.4, "recommendation": "clarify", "issues": ["which grade"]})
    assert result.metadata.get("source") != "coverage_validator"
    assert result.recommendation.value == "clarify"

    subjects = _state("Which subjects are in the stream?", evidence, focus="subjects")
    listed = _verify(subjects, {"passed": True, "score": 0.9, "recommendation": "accept"})
    assert listed.metadata.get("coverage_decision") is None
    assert listed.passed


def test_missing_structured_blocks_fail_closed_without_a_structural_code():
    evidence = _evidence(_literature(), "African Literature", "Languages & Literatures")
    state = _state("What does African Literature cover?", evidence)
    result = _verify(state, {"passed": True, "score": 0.99, "recommendation": "accept"})
    assert not result.passed
    assert result.metadata["coverage_decision"] == "unavailable"
    assert result.metadata["coverage_reason_codes"] == ["structured_blocks_unavailable"]
    assert "incorrect_parent" not in result.metadata["coverage_reason_codes"]
    assert result.metadata["llm_verification"]["passed"] is True


def test_schema_envelope_is_an_invalid_verifier_result_not_a_structure_error():
    shape = classify_verifier_payload(SCHEMA_ENVELOPE)
    assert shape == "schema_envelope"
    parsed = AnswerVerifier(_ScriptedLLM(SCHEMA_ENVELOPE))._parse(SCHEMA_ENVELOPE)
    assert parsed.passed is False
    assert parsed.score == 0.0
    assert parsed.recommendation.value == "fallback"
    assert parsed.issues == []

    evidence = [CurriculumEvidence(entity_type="topic", entity_id="t1", name="Fractions", grade="SSS_3", subject="Mathematics")]
    state = _state("How is a fraction introduced?", evidence, intent="CURRICULUM_QA")
    state.metadata.pop("sss_focus")
    result = _verify(state, SCHEMA_ENVELOPE)
    assert result.metadata["verifier_payload_shape"] == "schema_envelope"
    assert result.metadata["verifier_result_invalid"] is True
    assert result.metadata["verifier_invalid_reason"] == "schema_envelope"
    assert result.metadata.get("coverage_decision") is None
    assert "incorrect_parent" not in result.issues


def _prepared_literature():
    evidence = _evidence(_literature(), "African Literature", "Languages & Literatures")
    state = _state("What does African Literature cover?", evidence, subject="African Literature")
    _render(state, "Languages & Literatures")
    _bind_expected(
        state,
        _literature(),
        subject="African Literature",
        stream="Languages & Literatures",
        grade_curriculum_id="gc-d1",
    )
    return state


def test_removing_a_required_theme_or_topic_is_evidence_incomplete():
    accepting = {"passed": True, "score": 0.99, "recommendation": "accept", "issues": []}
    theme_state = _prepared_literature()
    theme_state.evidence = [item for item in theme_state.evidence if item.entity_id != THEME_3]
    theme_state.metadata["sss_coverage_blocks"] = [
        block for block in theme_state.metadata["sss_coverage_blocks"] if block["entity_id"] != THEME_3
    ]
    theme_result = _verify(theme_state, accepting)
    assert not theme_result.passed
    assert theme_result.metadata["coverage_decision"] == "unavailable"
    assert theme_result.metadata["coverage_reason_codes"] == ["evidence_incomplete"]
    assert theme_result.metadata["llm_verification"]["recommendation"] == "accept"

    topic_state = _prepared_literature()
    topic_state.evidence = [item for item in topic_state.evidence if item.entity_id != WHOLE_CLASS]
    for block in topic_state.metadata["sss_coverage_blocks"]:
        block["children"] = [child for child in block["children"] if child["entity_id"] != WHOLE_CLASS]
    topic_result = _verify(topic_state, accepting)
    assert not topic_result.passed
    assert topic_result.metadata["coverage_reason_codes"] == ["evidence_incomplete"]


def test_a_census_from_another_syllabus_cannot_validate_this_request():
    state = _prepared_literature()
    expected = dict(state.metadata["sss_coverage_expected"])
    accepting = {"passed": True, "score": 1, "recommendation": "accept"}
    for field, foreign in (
        ("subject", "Biology"),
        ("grade", "SSS_1"),
        ("source_reference", "SSS-Syllabus-Biology.pdf"),
        ("grade_curriculum_id", "gc-other"),
    ):
        state.metadata["sss_coverage_expected"] = {**expected, field: foreign}
        result = _verify(state, accepting)
        assert not result.passed
        assert result.metadata["coverage_decision"] == "unavailable"
        assert result.metadata["coverage_reason_codes"] == ["evidence_scope_mismatch"]
    state.metadata["sss_coverage_expected"] = expected


def test_omitted_sections_and_subtopics_do_not_make_evidence_incomplete():
    evidence = _evidence(_environment(), "Environmental Science", "Sciences & Technologies")
    state = _state("What does Environmental Science cover?", evidence, subject="Environmental Science")
    _render(state, "Sciences & Technologies")
    _bind_expected(
        state,
        _environment(),
        subject="Environmental Science",
        stream="Sciences & Technologies",
        grade_curriculum_id="gc-d2",
    )
    census = set(state.metadata["sss_coverage_expected"]["entity_ids"])
    assert "85423a42-f5ec-4c42-b0a7-5f1d955704e4" not in census
    assert "920f1a1f-8833-46ea-ac17-ef2cb02a0ba9" not in census
    result = _verify(state, {"passed": False, "score": 0.2, "recommendation": "retrieve_more", "issues": ["incomplete"]})
    assert result.passed
    assert result.metadata["coverage_reason_codes"] == []


def test_duplicate_source_ids_are_unavailable_even_when_the_model_accepts():
    shared = "same-id"
    evidence = [
        CurriculumEvidence(
            entity_type="theme",
            entity_id="theme-1",
            name="Heat",
            grade="SSS_3",
            subject="African Literature",
            metadata={"parent_id": None, "stream_name": "Languages & Literatures"},
            source_reference="syllabus.pdf",
        ),
        CurriculumEvidence(
            entity_type="topic",
            entity_id=shared,
            name="Alpha",
            grade="SSS_3",
            subject="African Literature",
            metadata={"parent_id": "theme-1", "parent_entity_type": "THEME", "parent_in_evidence": True, "stream_name": "Languages & Literatures"},
            source_reference="syllabus.pdf",
        ),
        CurriculumEvidence(
            entity_type="topic",
            entity_id=shared,
            name="Beta",
            grade="SSS_3",
            subject="African Literature",
            metadata={"parent_id": "theme-1", "parent_entity_type": "THEME", "parent_in_evidence": True, "stream_name": "Languages & Literatures"},
            source_reference="syllabus.pdf",
        ),
    ]
    state = _state("What does African Literature cover?", evidence, subject="African Literature")
    state.metadata["stream_name"] = "Languages & Literatures"
    state.metadata["resolved_stream_name"] = "Languages & Literatures"
    state.metadata["sss_coverage_blocks"] = [
        {
            "entity_id": "theme-1",
            "entity_type": "theme",
            "name": "Heat",
            "children": [{"entity_id": shared, "entity_type": "topic", "name": "Alpha", "children": []}],
        }
    ]
    state.metadata["sss_coverage_expected"] = {
        "grade": "SSS_3",
        "subject": "African Literature",
        "stream_name": "Languages & Literatures",
        "source_reference": "syllabus.pdf",
        "grade_curriculum_id": "gc-d1",
        "entity_ids": ["theme-1", shared],
    }
    state.metadata["source_reference"] = "syllabus.pdf"
    state.metadata["grade_curriculum_id"] = "gc-d1"
    result = _verify(state, {"passed": True, "score": 0.95, "recommendation": "accept", "issues": []})
    assert not result.passed
    assert result.metadata["coverage_decision"] == "unavailable"
    assert result.metadata["coverage_reason_codes"] == ["duplicate_source_id"]
    assert result.metadata["llm_authoritative"] is False
    assert result.metadata["llm_verification"]["recommendation"] == "accept"
    assert "duplicate_entity" not in result.metadata["coverage_reason_codes"]
