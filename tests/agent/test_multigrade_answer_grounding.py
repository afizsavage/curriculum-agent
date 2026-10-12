"""Grade-faithful answers for multi-grade curriculum evidence.

Fixtures are synthetic. Nothing here is specific to one subject or question
phrasing: the checks follow grades, record titles, outcome text, and
retrieval status.
"""

from __future__ import annotations

import re

import pytest

from app.agent.answer_generator import (
    AnswerGenerator,
    assess_topic_grounding,
    dedupe_units_for_generation,
    mentioned_grade_codes,
    select_evidence_for_prompt,
)
from app.agent.state import CurriculumQAState
from app.curriculum.codes import extract_filters_from_question
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus, ToolCallRecord
from app.curriculum.sss_stream_intent import detect_sss_stream_subjects
from app.llm.provider import StubLLMProvider


def _unit(
    entity_id: str,
    name: str,
    grade: str,
    *,
    subject: str = "MATHEMATICS",
    **metadata: object,
) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="unit",
        entity_id=entity_id,
        name=name,
        grade=grade,
        subject=subject,
        topic=name,
        content=name,
        metadata=dict(metadata),
        source_reference="grade_curriculum.content",
    )


def _outcome(
    entity_id: str,
    text: str,
    grade: str,
    *,
    subject: str = "MATHEMATICS",
) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id=entity_id,
        name=entity_id,
        grade=grade,
        subject=subject,
        topic=text,
        content=text,
        source_reference="grade_curriculum.content",
    )


def _generate(
    question: str,
    evidence: list[CurriculumEvidence],
    *,
    status: EvidenceStatus = EvidenceStatus.FOUND,
    history: list[ToolCallRecord] | None = None,
    grade: str | None = None,
    subject: str | None = "MATHEMATICS",
    llm: object | None = None,
):
    state = CurriculumQAState.initial(question=question)
    state.evidence = evidence
    state.evidence_status = status
    state.retrieval_history = history or []
    state.grade = grade
    state.subject = subject
    provider = llm or StubLLMProvider()
    return AnswerGenerator(provider).generate(state)


def _section(answer: str, heading: str) -> str:
    parts = re.split(r"(?m)^#{1,3}\s+", answer)
    for part in parts:
        title, _, body = part.partition("\n")
        if heading.lower() in title.lower():
            return body
    return ""


def _bullets(text: str) -> list[str]:
    return re.findall(r"(?m)^\s*[-*]\s+(.*\S)\s*$", text)


class _ExplodingLLM:
    name = "live"
    model = "live"

    def generate_structured(self, *args: object, **kwargs: object) -> dict:
        raise AssertionError("title-only evidence must not be rewritten by the model")


def test_three_grades_stay_in_separate_sections():
    evidence = [
        _unit("c1", "Counting", "CLASS_1"),
        _unit("c2", "Time", "CLASS_2"),
        _unit("c3", "Shapes", "CLASS_3"),
    ]
    result = _generate(
        "What does Mathematics cover in classes 1, 2, and 3?",
        evidence,
    )
    assert "### Primary 1" in result.answer
    assert "### Primary 2" in result.answer
    assert "### Primary 3" in result.answer
    assert "Counting" in _section(result.answer, "Primary 1")
    assert "Time" in _section(result.answer, "Primary 2")
    assert "Shapes" in _section(result.answer, "Primary 3")
    assert assess_topic_grounding(result.answer, evidence) == []


def test_a_topic_is_not_attributed_to_another_grade():
    evidence = [
        _unit("c1", "Counting", "CLASS_1"),
        _unit("c2", "Time", "CLASS_2"),
    ]
    result = _generate("What does Mathematics cover in classes 1 and 2?", evidence)
    assert "Time" not in _section(result.answer, "Primary 1")
    assert "Counting" not in _section(result.answer, "Primary 2")
    assert assess_topic_grounding(result.answer, evidence) == []


def test_the_same_title_in_two_grades_stays_distinct():
    evidence = [
        _unit("c1-shapes", "Shapes", "CLASS_1"),
        _unit("c2-shapes", "Shapes", "CLASS_2"),
    ]
    prepared = dedupe_units_for_generation(evidence)
    assert {item.entity_id for item in prepared} == {"c1-shapes", "c2-shapes"}
    result = _generate("What does Mathematics cover in classes 1 and 2?", evidence)
    assert "Shapes" in _section(result.answer, "Primary 1")
    assert "Shapes" in _section(result.answer, "Primary 2")
    assert {ref.entity_id for ref in result.evidence} == {"c1-shapes", "c2-shapes"}


def test_unit_titles_are_not_rewritten_as_learning_outcomes():
    evidence = [
        _unit("c1-count", "Number and Numeration Counting", "CLASS_1"),
        _unit("c1-write", "Number and Numeration writing numbers", "CLASS_1"),
        _unit("c1-every", "Everyday Arithmetic", "CLASS_1"),
    ]
    result = _generate("What does Mathematics cover in Class 1?", evidence, grade="CLASS_1")
    lowered = result.answer.lower()
    assert "pupils learn to" not in lowered
    assert "pupils will be able" not in lowered
    assert "Number and Numeration Counting" in result.answer
    assert "Number and Numeration writing numbers" in result.answer
    assert "Everyday Arithmetic" in result.answer
    assert "learning outcomes" in lowered
    assert assess_topic_grounding(result.answer, evidence) == []


def test_explicit_outcomes_can_say_what_pupils_learn():
    evidence = [
        _outcome("lo-half", "identify a half", "CLASS_1"),
        _outcome("lo-quarter", "identify a quarter", "CLASS_1"),
        _outcome("lo-time", "tell the time", "CLASS_2"),
    ]
    result = _generate(
        "What does Mathematics cover in classes 1 and 2?",
        evidence,
    )
    primary_1 = _section(result.answer, "Primary 1")
    primary_2 = _section(result.answer, "Primary 2")
    assert "Pupils learn to:" in primary_1
    assert "identify a half" in primary_1.lower()
    assert "tell the time" not in primary_1.lower()
    assert "tell the time" in primary_2.lower()


def test_source_parent_heading_is_kept():
    evidence = [
        _unit("unit-length", "Length", "CLASS_1", section_name="Measurement"),
        _unit("unit-area", "Area", "CLASS_1", section_name="Measurement"),
        _unit("unit-circles", "Circles", "CLASS_1"),
    ]
    result = _generate("What does Mathematics cover in Class 1?", evidence, grade="CLASS_1")
    assert "**Measurement**" in result.answer
    assert "Length" in result.answer
    assert "Area" in result.answer
    assert "Circles" in result.answer


def test_missing_parent_heading_is_not_invented():
    evidence = [
        _unit("unit-counting", "Counting", "CLASS_1"),
        _unit("unit-shapes", "Shapes", "CLASS_1"),
    ]
    result = _generate("What does Mathematics cover in Class 1?", evidence, grade="CLASS_1")
    assert "Number and Numeration" not in result.answer
    assert "Everyday Arithmetic" not in result.answer
    assert "**" not in result.answer
    assert _bullets(result.answer) == ["Counting", "Shapes"]


def test_partial_retrieval_names_the_failed_grade_as_an_error():
    evidence = [
        _unit("c1", "Counting", "CLASS_1"),
        _unit("c2", "Time", "CLASS_2"),
    ]
    history = [
        ToolCallRecord(
            tool="resolve_curriculum_context",
            arguments={"grade": "CLASS_3", "subject": "MATHEMATICS"},
            status="error",
            error="Curriculum API request timed out",
            curriculum_api_status=504,
        )
    ]
    result = _generate(
        "What does Mathematics cover in classes 1, 2, and 3?",
        evidence,
        status=EvidenceStatus.PARTIAL,
        history=history,
    )
    text = result.answer + " " + " ".join(result.limitations)
    assert "### Primary 1" in result.answer
    assert "### Primary 2" in result.answer
    assert "CLASS_3" in text
    assert "could not be completed" in text
    assert "No syllabus was found for CLASS_3" not in text
    assert "Shapes" not in result.answer


def test_confirmed_absence_is_not_a_timeout():
    evidence = [_unit("c1", "Counting", "CLASS_1")]
    history = [
        ToolCallRecord(
            tool="get_curriculum_structure",
            arguments={"grade": "CLASS_2", "subject": "MATHEMATICS"},
            status="not_found",
            error="No syllabus found for the given subject/grade filters",
            curriculum_api_status=404,
        )
    ]
    result = _generate(
        "What does Mathematics cover in Class 1 and Class 2?",
        evidence,
        history=history,
    )
    text = result.answer + " " + " ".join(result.limitations)
    assert "No syllabus was found for CLASS_2" in text
    assert "could not be checked" not in text.lower()
    assert "Time" not in result.answer


def test_timeout_with_no_records_is_not_a_missing_syllabus():
    history = [
        ToolCallRecord(
            tool="resolve_curriculum_context",
            arguments={"grade": "CLASS_1"},
            status="error",
            error="Curriculum API request timed out",
            curriculum_api_status=504,
        )
    ]
    result = _generate(
        "What does Mathematics cover in Class 1?",
        [],
        status=EvidenceStatus.ERROR,
        history=history,
        grade="CLASS_1",
    )
    assert "could not be checked" in result.answer.lower()
    assert "no syllabus" not in result.answer.lower()
    assert "couldn't find sufficient" not in result.answer.lower()


def test_confirmed_missing_syllabus_does_not_invent_topics():
    history = [
        ToolCallRecord(
            tool="get_curriculum_structure",
            arguments={"grade": "CLASS_1", "subject": "HISTORY"},
            status="not_found",
            error="No syllabus found",
            curriculum_api_status=404,
        )
    ]
    result = _generate(
        "What does History cover in Class 1?",
        [],
        status=EvidenceStatus.NOT_FOUND,
        history=history,
        grade="CLASS_1",
        subject="HISTORY",
    )
    assert "couldn't find sufficient" in result.answer.lower()
    assert "could not be checked" not in result.answer.lower()
    assert "Counting" not in result.answer


def test_truncated_child_records_are_named():
    evidence = [
        _unit(
            "unit-shapes",
            "Shapes",
            "CLASS_2",
            omitted_child_count=3,
            omitted_child_types=["LEARNING_OUTCOME"],
        )
    ]
    result = _generate("What does Mathematics cover in Class 2?", evidence, grade="CLASS_2")
    assert "Shapes" in result.answer
    assert "3 child records (LEARNING_OUTCOME)" in result.answer
    assert "verified complete syllabus" not in result.answer or "not a verified complete syllabus" in result.answer


def test_every_topic_bullet_maps_to_a_retrieved_title():
    evidence = [
        _unit("c1", "Counting", "CLASS_1"),
        _unit("c2", "Time", "CLASS_2"),
        _unit("c3", "Shapes", "CLASS_3"),
    ]
    result = _generate(
        "What does Mathematics cover in classes 1, 2, and 3?",
        evidence,
    )
    titles = {item.name for item in evidence}
    assert _bullets(result.answer)
    assert set(_bullets(result.answer)) <= titles
    assert {ref.entity_id for ref in result.evidence} == {"c1", "c2", "c3"}


def test_single_grade_title_question_still_lists_its_records():
    evidence = [
        _unit("one", "Counting", "CLASS_4"),
        _unit("two", "Money", "CLASS_4"),
    ]
    result = _generate(
        "What are the topics in Primary 4 Mathematics?",
        evidence,
        grade="CLASS_4",
    )
    assert result.answer.startswith("# Primary 4 Mathematics")
    assert "Counting" in result.answer
    assert "Money" in result.answer
    assert "pupils learn to" not in result.answer.lower()
    assert "### Primary 1" not in result.answer


def test_title_only_generation_does_not_call_the_model():
    evidence = [_unit("c1", "Counting", "CLASS_1"), _unit("c2", "Time", "CLASS_2")]
    result = _generate(
        "What does Mathematics cover in classes 1 and 2?",
        evidence,
        llm=_ExplodingLLM(),
    )
    assert "Counting" in result.answer
    assert "Time" in result.answer


def test_subject_codes_for_ict_and_ict_literacy_stay_distinct():
    ict = extract_filters_from_question("What does ICT cover in Class 1?")
    literacy = extract_filters_from_question(
        "What does ICT Literacy cover in classes 1, 2, and 3?"
    )
    assert ict["subject"] == "ICT"
    assert ict["grade"] == "CLASS_1"
    assert literacy["subject"] == "ICT_LITERACY"
    assert literacy["grade"] is None


def test_senior_secondary_stream_routing_is_unchanged():
    detected = detect_sss_stream_subjects(
        "What does Biology cover for SSS1 Science and Technology?"
    )
    assert detected is not None
    assert detected.grade == "SSS_1"
    assert detected.stream_name == "Science and Technology"
    assert detected.intent == "SSS_STREAM_SUBJECTS"


def test_grade_mentions_do_not_swallow_a_following_stream_name():
    assert mentioned_grade_codes(
        "What does Mathematics cover in classes 1, 2, and 3?"
    ) == ["CLASS_1", "CLASS_2", "CLASS_3"]
    assert mentioned_grade_codes("What does Mathematics cover in classes 1-3?") == [
        "CLASS_1",
        "CLASS_2",
        "CLASS_3",
    ]
    assert mentioned_grade_codes("What does ICT cover in Class 1 and Class 2?") == [
        "CLASS_1",
        "CLASS_2",
    ]
    assert mentioned_grade_codes(
        "What does ICT cover in classes 1 and the Science stream?"
    ) == ["CLASS_1"]


def test_capped_window_keeps_every_grade():
    evidence = [
        _unit(f"c1-{index}", f"Class 1 topic {index}", "CLASS_1") for index in range(20)
    ] + [
        _unit(f"c2-{index}", f"Class 2 topic {index}", "CLASS_2") for index in range(20)
    ]
    selected, _ids = select_evidence_for_prompt(
        evidence,
        question="What does Mathematics cover in classes 1 and 2?",
    )
    assert len(selected) == 24
    assert {item.grade for item in selected} == {"CLASS_1", "CLASS_2"}


def test_model_overstatement_is_rejected_by_the_grade_check():
    evidence = [
        _unit("c1", "Counting", "CLASS_1"),
        _unit("c2", "Time", "CLASS_2"),
    ]
    invented = (
        "# Mathematics\n\n"
        "### Primary 1\n"
        "Pupils learn to:\n"
        "* Count objects.\n"
        "* Tell the time.\n"
        "### Primary 2\n"
        "* Time\n"
    )
    issues = assess_topic_grounding(invented, evidence)
    assert any("learning outcomes" in issue for issue in issues)
    assert any("Tell the time" in issue for issue in issues)


@pytest.mark.parametrize(
    "question,grade",
    [
        ("What does Science cover in Class 2?", "CLASS_2"),
        ("What does English Language cover in JSS 1?", "JSS_1"),
    ],
)
def test_single_grade_filter_remains_one_grade(question: str, grade: str):
    assert extract_filters_from_question(question)["grade"] == grade
