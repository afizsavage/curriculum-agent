"""Primary class lists stay off the SSS stream route.

A conjunction inside "classes 1, 2, and 3" must not make that list a stream
title. Official stream titles that contain "and" stay on the SSS route.
"""

from __future__ import annotations

import pytest

from app.agent.graph_nodes import GraphNodes
from app.agent.state import CurriculumQAState
from app.config import Settings
from app.curriculum.codes import extract_filters_from_question, normalize_grade_code
from app.curriculum.sss_stream_intent import (
    INTENT_SSS_STREAM_SUBJECTS,
    detect_sss_stream_subjects,
)
from app.llm.base import LLMMessage
from app.llm.tool_selection import select_tool_calls


def _understand(question: str) -> CurriculumQAState:
    nodes = GraphNodes(
        settings=Settings(llm_provider="stub"),
        retrieval=None,  # understand() does not call retrieval
        answer_node=None,
        verification_node=None,
    )
    result = nodes.understand({"qa": CurriculumQAState(question=question), "visited_nodes": []})
    return result["qa"]


@pytest.mark.parametrize(
    "question",
    [
        "What does the Science stream cover in SSS 1?",
        "What does the Sciences and Technologies stream cover in SSS 1?",
        "What does Biology cover for SSS1 Science and Technology?",
        "What does ICT Literacy cover for SSS1 Science and Technology stream?",
        "Which subjects are in the Sciences & Technologies stream?",
    ],
)
def test_valid_stream_names_stay_on_the_sss_route(question: str):
    detected = detect_sss_stream_subjects(question)
    assert detected is not None
    assert detected.intent == INTENT_SSS_STREAM_SUBJECTS
    assert detected.stream_name
    assert "class" not in detected.stream_name.lower()


def test_science_stream_keeps_sss1_grade():
    detected = detect_sss_stream_subjects(
        "What does the Science stream cover in SSS 1?"
    )
    assert detected is not None
    assert detected.stream_name == "Science"
    assert detected.grade == "SSS_1"


def test_sciences_and_technologies_keeps_the_conjunction():
    detected = detect_sss_stream_subjects(
        "What does the Sciences and Technologies stream cover in SSS 1?"
    )
    assert detected is not None
    assert detected.stream_name == "Sciences and Technologies"
    assert detected.grade == "SSS_1"


@pytest.mark.parametrize(
    "question",
    [
        "What does ICT Literacy cover in classes 1, 2, and 3?",
        "What does ICT Literacy cover in Classes 1, 2, and 3?",
        "What does Mathematics cover in Class 1 and Class 2?",
        "What does Science cover in classes 1-3?",
        "What does Science cover in classes 1–3?",
        "What does English cover in Primary 1, Primary 2, and Primary 3?",
        "What does English cover in class 4?",
    ],
)
def test_primary_class_lists_are_not_stream_names(question: str):
    assert detect_sss_stream_subjects(question) is None


@pytest.mark.parametrize(
    "question",
    [
        "What does History cover in classes 1 and 7?",
        "What does History cover in classes 7 and 8?",
        "What does History cover in class 0 and class 3?",
    ],
)
def test_malformed_class_lists_are_not_stream_names(question: str):
    assert detect_sss_stream_subjects(question) is None


def test_mixed_class_and_stream_wording_is_not_treated_as_a_pure_class_list():
    """A title that is not only a class list keeps the existing stream rule."""
    detected = detect_sss_stream_subjects(
        "What does ICT cover in classes 1 and the Science stream?"
    )
    assert detected is not None
    assert detected.intent == INTENT_SSS_STREAM_SUBJECTS
    assert detected.stream_name == "classes 1 and the Science"


def test_combined_ict_question_reaches_the_primary_planner_intent():
    state = _understand("What does ICT Literacy cover in classes 1, 2, and 3?")
    assert state.intent == "retrieve_curriculum"
    assert state.level is None
    assert state.grade is None
    assert state.subject == "ICT_LITERACY"
    assert "stream_name" not in state.metadata


def test_single_class_questions_keep_one_primary_grade():
    for question, grade in (
        ("What does ICT cover in Class 1?", "CLASS_1"),
        ("What does ICT cover in Class 2?", "CLASS_2"),
        ("What does ICT cover in Class 3?", "CLASS_3"),
    ):
        state = _understand(question)
        assert state.intent == "retrieve_curriculum"
        assert state.grade == grade
        assert state.subject == "ICT"


@pytest.mark.parametrize(
    ("question", "grade", "subject"),
    [
        ("What does ICT cover in Class 1?", "CLASS_1", "ICT"),
        ("What does ICT cover in Class 2?", "CLASS_2", "ICT"),
        ("What does ICT cover in Class 3?", "CLASS_3", "ICT"),
        ("What does ICT cover in classes 1, 2, and 3?", None, "ICT"),
        ("What does Mathematics cover in Class 1 and Class 2?", "CLASS_1", "MATHEMATICS"),
        ("What does ICT cover in classes 1–3?", None, "ICT"),
    ],
)
def test_grade_filter_is_a_single_optional_value(question: str, grade: str | None, subject: str):
    filters = extract_filters_from_question(question)
    assert filters["grade"] == grade
    assert filters["subject"] == subject
    assert "grades" not in filters


@pytest.mark.parametrize(
    "question",
    [
        "What does ICT cover in Class 7?",
        "What does ICT cover in classes 1 and 9?",
        "What does ICT cover in class 0?",
    ],
)
def test_unavailable_primary_grades_are_not_stream_names(question: str):
    assert detect_sss_stream_subjects(question) is None
    filters = extract_filters_from_question(question)
    assert filters["grade"] is None


def test_two_grade_filter_does_not_relabel_the_second_grade_as_the_first():
    """The single-grade filter keeps Class 1 and does not invent a merged grade."""
    filters = extract_filters_from_question(
        "What does Mathematics cover in Class 1 and Class 2?"
    )
    assert filters["grade"] == "CLASS_1"
    assert filters["grade"] != "CLASS_2"
    assert "CLASS_1_CLASS_2" not in (filters["grade"] or "")


def test_partial_grade_evidence_is_not_relabeled_by_the_question():
    """A missing class is not filled in from the question text.

    ``normalize_grade_code`` is the same helper the answer post-processor uses.
    The combined question has no single grade, so that check cannot invent
    CLASS_3 when only some classes were retrieved.
    """
    question = "What does ICT Literacy cover in classes 1, 2, and 3?"
    question_grade = normalize_grade_code(question)
    retrieved = {"CLASS_1", "CLASS_2"}
    assert question_grade is None
    assert "CLASS_3" not in retrieved
    assert question_grade not in retrieved


def test_stub_planner_does_not_call_the_sss_tool_for_a_class_list():
    calls = select_tool_calls(
        [
            LLMMessage(
                role="user",
                content="Question: What does ICT Literacy cover in classes 1, 2, and 3?",
            )
        ],
        [
            {"name": "get_sss_stream_subjects"},
            {"name": "resolve_curriculum_context"},
            {"name": "search_curriculum"},
            {"name": "get_curriculum_structure"},
        ],
    )
    assert calls
    assert all(call.name != "get_sss_stream_subjects" for call in calls)
    assert len({call.arguments.get("grade") for call in calls}) == 1
    assert calls[0].arguments.get("grade") is None
