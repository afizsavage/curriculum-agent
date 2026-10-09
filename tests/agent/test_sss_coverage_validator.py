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
from app.agent.retrieve import RetrievalNode
from app.agent.state import CurriculumQAState
from app.agent.verifier import AnswerVerifier
from app.llm.base import ToolCallRequest
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
    state = _state(
        "What are the learning outcomes for African Literature for SSS 3 Languages & Literatures?",
        evidence,
        focus="outcomes",
    )
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

    subjects = _state(
        "What subjects are in the Languages & Literatures stream?",
        evidence,
        focus="subjects",
    )
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


ACCEPTING = {"passed": True, "score": 0.99, "recommendation": "accept", "issues": []}
COVERAGE_QUESTION = "What does African Literature cover for SSS 3 Languages & Literatures?"


def _coverage_state():
    evidence = _evidence(_literature(), "African Literature", "Languages & Literatures")
    state = _state(COVERAGE_QUESTION, evidence, subject="African Literature")
    _render(state, "Languages & Literatures")
    _bind_expected(
        state,
        _literature(),
        subject="African Literature",
        stream="Languages & Literatures",
        grade_curriculum_id="gc-d1",
    )
    return state


def _drop_theme(state):
    state.evidence = [item for item in state.evidence if item.entity_id != THEME_3]
    state.metadata["sss_coverage_blocks"] = [
        block for block in state.metadata["sss_coverage_blocks"] if block["entity_id"] != THEME_3
    ]


def test_missing_or_invalid_focus_cannot_accept_incomplete_coverage():
    missing = _coverage_state()
    _drop_theme(missing)
    missing.metadata.pop("sss_focus")
    missing.metadata.pop("focus", None)
    missing_gate = assess_coverage(missing)
    missing_result = _verify(missing, ACCEPTING)
    assert missing_gate.status == "unavailable"
    assert missing_gate.reason == "evidence_incomplete"
    assert not missing_result.passed
    assert missing_result.metadata["coverage_decision"] == "unavailable"
    assert missing_result.metadata["coverage_reason_codes"] == ["evidence_incomplete"]
    assert missing_result.metadata["llm_authoritative"] is False

    invalid = _coverage_state()
    _drop_theme(invalid)
    invalid.metadata["sss_focus"] = "banana"
    invalid.metadata["focus"] = "coverage"
    invalid_gate = assess_coverage(invalid)
    invalid_result = _verify(invalid, ACCEPTING)
    assert invalid_gate.status == "unavailable"
    assert invalid_gate.reason == "evidence_incomplete"
    assert not invalid_result.passed
    assert invalid_result.metadata["coverage_reason_codes"] == ["evidence_incomplete"]

    copied = _coverage_state()
    _drop_theme(copied)
    copied.question = "What does African Literature cover?"
    copied.metadata.pop("sss_focus")
    copied.metadata["focus"] = "coverage"
    copied_gate = assess_coverage(copied)
    copied_result = _verify(copied, ACCEPTING)
    assert copied_gate.reason == "evidence_incomplete"
    assert not copied_result.passed
    assert copied_result.metadata["coverage_reason_codes"] == ["evidence_incomplete"]


def test_reclassification_keeps_outcomes_subjects_and_ordinary_questions_generic():
    outcomes = _state(
        "What are the learning outcomes for African Literature for SSS 3 Languages & Literatures?",
        _evidence(_literature(), "African Literature", "Languages & Literatures"),
        subject="African Literature",
    )
    outcomes.metadata.pop("sss_focus")
    outcomes.metadata.pop("focus", None)
    outcomes_gate = assess_coverage(outcomes)
    outcomes_result = _verify(
        outcomes,
        {"passed": False, "score": 0.4, "recommendation": "retrieve_more", "issues": ["generic"]},
    )
    assert outcomes_gate.status == "not_applicable"
    assert outcomes_gate.reason == "outcomes_hierarchy_not_represented"
    assert outcomes_result.metadata.get("coverage_decision") is None
    assert outcomes_result.recommendation.value == "retrieve_more"

    subjects = _state(
        "What subjects are in the Languages & Literatures stream?",
        [CurriculumEvidence(entity_type="subject", entity_id="subj", name="African Literature")],
        focus=None,
    )
    subjects.metadata.pop("sss_focus")
    subjects_gate = assess_coverage(subjects)
    subjects_result = _verify(subjects, ACCEPTING)
    assert subjects_gate.status == "not_applicable"
    assert subjects_gate.reason == "focus_outside_coverage_contract"
    assert subjects_result.metadata.get("coverage_decision") is None
    assert subjects_result.passed

    ordinary = _state(
        "What is a fraction?",
        [CurriculumEvidence(entity_type="topic", entity_id="t1", name="Fractions", metadata={"parent_id": None})],
        intent="CURRICULUM_QA",
        focus=None,
    )
    ordinary.metadata.pop("sss_focus")
    ordinary_gate = assess_coverage(ordinary)
    ordinary_result = _verify(
        ordinary,
        {"passed": False, "score": 0.4, "recommendation": "clarify", "issues": ["which grade"]},
    )
    assert ordinary_gate.reason == "not_sss_stream_subjects"
    assert ordinary_result.metadata.get("coverage_decision") is None
    assert ordinary_result.recommendation.value == "clarify"


def test_recognized_but_wrong_focus_cannot_bypass_coverage():
    subjects = _coverage_state()
    _drop_theme(subjects)
    subjects.metadata["sss_focus"] = "subjects"
    subjects.metadata["focus"] = "subjects"
    subjects_gate = assess_coverage(subjects)
    subjects_result = _verify(subjects, ACCEPTING)
    assert subjects_gate.status == "unavailable"
    assert subjects_gate.reason == "evidence_incomplete"
    assert not subjects_result.passed
    assert subjects_result.metadata["coverage_decision"] == "unavailable"
    assert subjects_result.metadata["coverage_reason_codes"] == ["evidence_incomplete"]
    assert subjects_result.metadata["llm_authoritative"] is False

    for stored in ("outcomes", "membership", "streams"):
        state = _coverage_state()
        _drop_theme(state)
        state.metadata["sss_focus"] = stored
        state.metadata["focus"] = stored
        gate = assess_coverage(state)
        result = _verify(state, ACCEPTING)
        assert gate.reason == "evidence_incomplete", stored
        assert not result.passed
        assert result.metadata["coverage_reason_codes"] == ["evidence_incomplete"]


def test_classified_non_coverage_questions_stay_outside_the_contract():
    listed = _state(
        "What subjects are in the Languages & Literatures stream?",
        [CurriculumEvidence(entity_type="subject", entity_id="subj", name="African Literature")],
        focus="subjects",
    )
    listed_gate = assess_coverage(listed)
    listed_result = _verify(listed, ACCEPTING)
    assert listed_gate.status == "not_applicable"
    assert listed_gate.reason == "focus_outside_coverage_contract"
    assert listed_result.metadata.get("coverage_decision") is None
    assert listed_result.passed

    outcomes = _state(
        "What are the learning outcomes for African Literature for SSS 3 Languages & Literatures?",
        _evidence(_literature(), "African Literature", "Languages & Literatures"),
        focus="coverage",
    )
    outcomes_gate = assess_coverage(outcomes)
    outcomes_result = _verify(
        outcomes,
        {"passed": False, "score": 0.4, "recommendation": "retrieve_more", "issues": ["generic"]},
    )
    assert outcomes_gate.reason == "outcomes_hierarchy_not_represented"
    assert outcomes_result.metadata.get("coverage_decision") is None
    assert outcomes_result.recommendation.value == "retrieve_more"

    membership = _state(
        "Is African Literature part of the Languages & Literatures stream?",
        [CurriculumEvidence(entity_type="subject", entity_id="subj", name="African Literature")],
        focus="coverage",
    )
    membership_gate = assess_coverage(membership)
    membership_result = _verify(membership, ACCEPTING)
    assert membership_gate.reason == "focus_outside_coverage_contract"
    assert membership_result.metadata.get("coverage_decision") is None
    assert membership_result.passed

    streams = _state(
        "What streams are available at SSS?",
        [CurriculumEvidence(entity_type="sss_stream", entity_id="stream-1", name="Languages & Literatures")],
        focus="coverage",
    )
    streams_gate = assess_coverage(streams)
    streams_result = _verify(streams, ACCEPTING)
    assert streams_gate.reason == "focus_outside_coverage_contract"
    assert streams_result.metadata.get("coverage_decision") is None
    assert streams_result.passed


def test_unclassified_question_with_a_non_coverage_focus_fails_closed():
    state = _coverage_state()
    _drop_theme(state)
    state.question = "Explain this syllabus"
    state.metadata["sss_focus"] = "subjects"
    state.metadata["focus"] = "subjects"
    gate = assess_coverage(state)
    result = _verify(state, ACCEPTING)
    assert gate.status == "unavailable"
    assert gate.reason == "coverage_focus_unresolved"
    assert not result.passed
    assert result.metadata["coverage_reason_codes"] == ["coverage_focus_unresolved"]
    assert result.metadata["llm_verification"]["recommendation"] == "accept"


def test_unresolved_focus_cannot_authorize_generic_acceptance():
    state = _coverage_state()
    _drop_theme(state)
    state.question = "Explain this syllabus"
    state.metadata.pop("sss_focus")
    state.metadata.pop("focus", None)
    gate = assess_coverage(state)
    result = _verify(state, ACCEPTING)
    assert gate.status == "unavailable"
    assert gate.reason == "coverage_focus_unresolved"
    assert not result.passed
    assert result.metadata["coverage_decision"] == "unavailable"
    assert result.metadata["coverage_reason_codes"] == ["coverage_focus_unresolved"]
    assert result.metadata["llm_verification"]["recommendation"] == "accept"


def test_subject_row_with_a_census_is_evidence_incomplete():
    state = _coverage_state()
    state.evidence = [
        CurriculumEvidence(
            entity_type="subject",
            entity_id="subj-1",
            name="African Literature",
            grade="SSS_3",
            subject="African Literature",
        )
    ]
    state.metadata["sss_coverage_blocks"] = []
    gate = assess_coverage(state)
    result = _verify(state, ACCEPTING)
    assert gate.status == "unavailable"
    assert gate.reason == "evidence_incomplete"
    assert not result.passed
    assert result.metadata["coverage_reason_codes"] == ["evidence_incomplete"]

    emptied = _coverage_state()
    emptied.evidence = []
    emptied.metadata["sss_coverage_blocks"] = []
    emptied.evidence_status = EvidenceStatus.NOT_FOUND
    empty_result = _verify(emptied, ACCEPTING)
    assert not empty_result.passed
    assert empty_result.recommendation.value == "retrieve_more"
    assert empty_result.metadata.get("source") == "deterministic"
    assert empty_result.metadata.get("no_evidence") is True

    removed = _coverage_state()
    removed.metadata.pop("sss_coverage_expected")
    removed_result = _verify(removed, ACCEPTING)
    assert not removed_result.passed
    assert removed_result.metadata["coverage_reason_codes"] == ["evidence_incomplete"]


def test_syllabus_identifiers_must_agree_on_both_sides():
    def decide(state):
        gate = assess_coverage(state)
        result = _verify(state, ACCEPTING)
        return gate, result

    crossed_source = _coverage_state()
    crossed_source.metadata.pop("grade_curriculum_id")
    crossed_source.metadata["sss_coverage_expected"] = {
        **crossed_source.metadata["sss_coverage_expected"],
        "source_reference": None,
        "grade_curriculum_id": "gc-other",
    }
    gate, result = decide(crossed_source)
    assert gate.reason == "evidence_scope_mismatch"
    assert not result.passed
    assert result.metadata["coverage_reason_codes"] == ["evidence_scope_mismatch"]

    crossed_id = _coverage_state()
    crossed_id.metadata["source_reference"] = None
    crossed_id.metadata["sss_coverage_expected"] = {
        **crossed_id.metadata["sss_coverage_expected"],
        "source_reference": "other.pdf",
        "grade_curriculum_id": None,
    }
    gate, result = decide(crossed_id)
    assert not result.passed
    assert result.metadata["coverage_reason_codes"] == ["evidence_scope_mismatch"]

    contradictory_source = _coverage_state()
    contradictory_source.metadata["sss_coverage_expected"] = {
        **contradictory_source.metadata["sss_coverage_expected"],
        "source_reference": "other.pdf",
    }
    _, result = decide(contradictory_source)
    assert not result.passed
    assert result.metadata["coverage_reason_codes"] == ["evidence_scope_mismatch"]

    contradictory_id = _coverage_state()
    contradictory_id.metadata["sss_coverage_expected"] = {
        **contradictory_id.metadata["sss_coverage_expected"],
        "grade_curriculum_id": "gc-other",
    }
    _, result = decide(contradictory_id)
    assert not result.passed
    assert result.metadata["coverage_reason_codes"] == ["evidence_scope_mismatch"]

    missing_grade = _coverage_state()
    missing_grade.grade = None
    missing_grade.metadata.pop("grade", None)
    _, result = decide(missing_grade)
    assert not result.passed
    assert result.metadata["coverage_reason_codes"] == ["evidence_scope_mismatch"]

    missing_stream = _coverage_state()
    missing_stream.metadata["sss_coverage_expected"] = {
        **missing_stream.metadata["sss_coverage_expected"],
        "stream_name": None,
    }
    _, result = decide(missing_stream)
    assert not result.passed
    assert result.metadata["coverage_reason_codes"] == ["evidence_scope_mismatch"]

    source_only = _coverage_state()
    source_only.metadata["grade_curriculum_id"] = None
    source_only.metadata["sss_coverage_expected"] = {
        **source_only.metadata["sss_coverage_expected"],
        "grade_curriculum_id": None,
    }
    _, result = decide(source_only)
    assert result.passed
    assert result.metadata["coverage_decision"] == "accept"

    identifier_only = _coverage_state()
    identifier_only.metadata["source_reference"] = None
    identifier_only.metadata["sss_coverage_expected"] = {
        **identifier_only.metadata["sss_coverage_expected"],
        "source_reference": None,
    }
    _, result = decide(identifier_only)
    assert result.passed
    assert result.metadata["coverage_decision"] == "accept"

    matched = _coverage_state()
    _, result = decide(matched)
    assert result.passed
    assert result.metadata["coverage_reason_codes"] == []


def _record(state, observability):
    RetrievalNode._record_sss_stream_outcome(
        RetrievalNode.__new__(RetrievalNode),
        state,
        "get_sss_stream_subjects",
        observability,
        request_id="audit",
    )


def test_a_later_tool_result_cannot_reuse_an_earlier_census():
    stale = _coverage_state()
    _record(
        stale,
        {
            "sss_stream_resolution": "found",
            "subject_count": 1,
            "stream_name": "Languages & Literatures",
            "focus": "coverage",
        },
    )
    assert "sss_coverage_expected" not in stale.metadata
    stale_result = _verify(stale, ACCEPTING)
    assert not stale_result.passed
    assert stale_result.metadata["coverage_reason_codes"] == ["evidence_incomplete"]

    replaced = _coverage_state()
    _record(
        replaced,
        {
            "sss_stream_resolution": "found",
            "subject_count": 1,
            "stream_name": "Sciences & Technologies",
            "subject_name": "Environmental Science",
            "grade": "SSS_1",
            "focus": "coverage",
            "source_reference": "env.pdf",
            "grade_curriculum_id": "gc-env",
            "coverage_expected": {
                "grade": "SSS_1",
                "subject": "Environmental Science",
                "stream_name": "Sciences & Technologies",
                "source_reference": "env.pdf",
                "grade_curriculum_id": "gc-env",
                "entity_ids": ["only-env"],
            },
        },
    )
    assert replaced.metadata["sss_coverage_expected"]["grade_curriculum_id"] == "gc-env"
    replaced_result = _verify(replaced, ACCEPTING)
    assert not replaced_result.passed
    assert replaced_result.metadata["coverage_reason_codes"] == ["evidence_scope_mismatch"]

    not_in_stream = _coverage_state()
    _record(
        not_in_stream,
        {
            "sss_stream_resolution": "subject_not_in_stream",
            "subject_count": 0,
            "stream_name": "Languages & Literatures",
            "requested_subject": "Biology",
            "grade": "SSS_3",
        },
    )
    assert "sss_coverage_expected" not in not_in_stream.metadata
    not_in_stream_result = _verify(not_in_stream, ACCEPTING)
    assert not not_in_stream_result.passed
    assert not_in_stream_result.metadata["coverage_reason_codes"] == ["evidence_incomplete"]

    missing_grade = _coverage_state()
    _record(
        missing_grade,
        {
            "sss_stream_resolution": "grade_content_missing",
            "subject_count": 1,
            "stream_name": "Languages & Literatures",
            "subject_name": "African Literature",
            "grade": "SSS_2",
            "focus": "coverage",
            "source_reference": None,
            "grade_curriculum_id": None,
            "coverage_expected": None,
        },
    )
    assert "sss_coverage_expected" not in missing_grade.metadata
    missing_result = _verify(missing_grade, ACCEPTING)
    assert not missing_result.passed
    assert missing_result.metadata["coverage_decision"] == "unavailable"

    kept = _coverage_state()
    _record(kept, {"tool": "get_topic"})
    assert kept.metadata["sss_coverage_expected"]["grade_curriculum_id"] == "gc-d1"
    kept_result = _verify(kept, {"passed": False, "score": 0.2, "recommendation": "retrieve_more"})
    assert kept_result.passed
    assert kept_result.metadata["coverage_decision"] == "accept"

    fresh_a = CurriculumQAState.initial(question=COVERAGE_QUESTION)
    fresh_b = CurriculumQAState.initial(
        question="What does Environmental Science cover for SSS 1 Sciences & Technologies?"
    )
    fresh_a.metadata["sss_coverage_expected"] = {"entity_ids": ["theme-a"]}
    assert "sss_coverage_expected" not in fresh_b.metadata
    assert fresh_a.metadata is not fresh_b.metadata


def test_a_failed_syllabus_load_clears_the_previous_census():
    class _Boom:
        def execute(self, name, **kwargs):
            raise RuntimeError("syllabus load failed")

    state = _coverage_state()
    node = RetrievalNode.__new__(RetrievalNode)
    node.tools = _Boom()
    node.settings = None
    node._execute_call(
        state,
        ToolCallRequest(id="load", name="get_sss_stream_subjects", arguments={}),
        request_id="audit",
    )
    assert "sss_coverage_expected" not in state.metadata
    result = _verify(state, ACCEPTING)
    assert not result.passed
    assert result.metadata["coverage_reason_codes"] == ["evidence_incomplete"]
