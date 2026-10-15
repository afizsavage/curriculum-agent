import pytest
from pydantic import ValidationError

from app.assistant.contract import AssistantRoute, AssistantTurnRequest, InvalidAssistantRoute
from app.assistant.router import heuristic_route, parse_closed_route, resolve_route


def _request(question: str) -> AssistantTurnRequest:
    return AssistantTurnRequest(question=question)


def test_heuristic_routes_curriculum_general_mixed_and_clarify():
    cases = [
        (
            "What topics are taught in Primary 4 Mathematics?",
            AssistantRoute.CURRICULUM,
            "curriculum_markers",
        ),
        (
            "How should I settle a noisy class before the lesson starts?",
            AssistantRoute.GENERAL,
            "advice_markers",
        ),
        (
            "What does the Class 4 Mathematics syllabus say about fractions, "
            "and how should I teach it?",
            AssistantRoute.MIXED,
            "curriculum_and_advice",
        ),
        ("Tell me more", AssistantRoute.CLARIFY, "underspecified"),
    ]
    for question, route, reason in cases:
        resolved = resolve_route(_request(question))
        assert resolved.route is route
        assert resolved.reason == reason
        assert resolved.source == "heuristic"
        assert resolved.raw_classification is None


def test_short_or_deictic_questions_clarify_even_with_working_context():
    request = AssistantTurnRequest(
        question="What about that",
        grade="CLASS_4",
        subject="MATHEMATICS",
    )
    resolved = resolve_route(request)
    assert resolved.route is AssistantRoute.CLARIFY
    assert heuristic_route("help")[0] is AssistantRoute.CLARIFY


def test_curriculum_marker_is_not_treated_as_clarify():
    resolved = resolve_route(
        _request("Tell me more about the Class 4 Mathematics syllabus")
    )
    assert resolved.route is AssistantRoute.CURRICULUM


def test_classifier_output_must_be_a_closed_route():
    assert parse_closed_route(" Curriculum ") is AssistantRoute.CURRICULUM
    assert parse_closed_route(AssistantRoute.GENERAL) is AssistantRoute.GENERAL

    for raw in ("lesson_plan", "general knowledge", "", "official", None, 3):
        with pytest.raises(InvalidAssistantRoute) as raised:
            parse_closed_route(raw)
        assert raised.value.raw == raw


def test_invalid_classifier_is_not_rewritten_to_general():
    request = _request("How should I settle a noisy class?")

    def classify(_request: AssistantTurnRequest) -> str:
        return "lesson_plan"

    with pytest.raises(InvalidAssistantRoute):
        resolve_route(request, classifier=classify)


def test_valid_classifier_is_normalized_and_recorded():
    request = _request("anything")
    resolved = resolve_route(request, classifier=lambda _request: " MiXeD ")
    assert resolved.route is AssistantRoute.MIXED
    assert resolved.source == "classifier"
    assert resolved.raw_classification == " MiXeD "
    assert resolved.reason == "classifier_closed_set"


def test_blank_question_is_rejected():
    with pytest.raises(ValidationError):
        AssistantTurnRequest(question="   ")
