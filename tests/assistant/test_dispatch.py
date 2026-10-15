import pytest

from app.assistant.contract import AssistantRoute, AssistantTurnRequest, InvalidAssistantRoute
from app.assistant.dispatch import AssistantDispatchError, dispatch_turn
from app.curriculum.errors import CurriculumTimeoutError
from app.exceptions import AgentExecutionError, LLMProviderError, LLMTimeoutError
from tests.assistant.qa_fixtures import (
    completed_with_evidence,
    evidence_status_error_http_200,
    insufficient_evidence,
    needs_clarification,
    partial_incomplete_coverage,
)

_ADVICE = "Partner talk is one classroom technique. It is not curriculum text."


class _Calls:
    def __init__(self) -> None:
        self.qa = 0
        self.general = 0


def _handlers(qa_result, calls: _Calls):
    def qa(_request: AssistantTurnRequest):
        calls.qa += 1
        if isinstance(qa_result, Exception):
            raise qa_result
        return qa_result

    def general(_request: AssistantTurnRequest) -> str:
        calls.general += 1
        return _ADVICE

    return qa, general


def test_each_route_calls_only_its_handlers():
    request = AssistantTurnRequest(question="unused when a classifier is injected")
    expectations = [
        (AssistantRoute.CURRICULUM, 1, 0),
        (AssistantRoute.GENERAL, 0, 1),
        (AssistantRoute.MIXED, 1, 1),
        (AssistantRoute.CLARIFY, 0, 0),
    ]
    for route, qa_calls, general_calls in expectations:
        calls = _Calls()
        qa, general = _handlers(completed_with_evidence(), calls)
        composed = dispatch_turn(
            request,
            qa=qa,
            general=general,
            classifier=lambda _request, chosen=route: chosen.value,
        )
        assert composed.route is route
        assert calls.qa == qa_calls
        assert calls.general == general_calls


def test_clarify_does_not_call_handlers_and_asks_for_detail():
    calls = _Calls()
    qa, general = _handlers(completed_with_evidence(), calls)
    composed = dispatch_turn(
        AssistantTurnRequest(question="Tell me more"),
        qa=qa,
        general=general,
    )
    assert composed.route is AssistantRoute.CLARIFY
    assert calls.qa == 0
    assert calls.general == 0
    assert composed.curriculum is None
    assert composed.general is None
    assert "official MBSSE curriculum" in (composed.clarification or "")


def test_invalid_classification_calls_nothing():
    calls = _Calls()
    qa, general = _handlers(completed_with_evidence(), calls)
    with pytest.raises(InvalidAssistantRoute):
        dispatch_turn(
            AssistantTurnRequest(question="How should I settle a noisy class?"),
            qa=qa,
            general=general,
            classifier=lambda _request: "lesson_plan",
        )
    assert calls.qa == 0
    assert calls.general == 0


def test_mixed_gap_and_technical_results_do_not_call_general_generation():
    for outcome in (
        insufficient_evidence(),
        partial_incomplete_coverage(),
        evidence_status_error_http_200(),
    ):
        calls = _Calls()
        qa, general = _handlers(outcome, calls)
        composed = dispatch_turn(
            AssistantTurnRequest(question="unused"),
            qa=qa,
            general=general,
            classifier=lambda _request: "mixed",
        )
        assert calls.qa == 1
        assert calls.general == 0
        assert composed.general is None
        assert composed.curriculum is not None
        assert composed.official_curriculum_answer == (
            outcome.response.answer if outcome.response else None
        )


def test_qa_timeout_and_execution_errors_do_not_call_general_generation():
    errors = [
        LLMTimeoutError("LLM request timed out"),
        CurriculumTimeoutError("Curriculum API request timed out"),
        AgentExecutionError("Agent failed while processing the question"),
    ]
    for error in errors:
        calls = _Calls()
        qa, general = _handlers(error, calls)
        composed = dispatch_turn(
            AssistantTurnRequest(question="unused"),
            qa=qa,
            general=general,
            classifier=lambda _request: "mixed",
        )
        assert calls.general == 0
        assert composed.general is None
        assert composed.curriculum is not None
        assert composed.curriculum.technical_failure is True
        assert composed.curriculum.failure.code == error.code
        assert composed.official_curriculum_answer is None


def test_provider_error_is_not_rewritten_as_a_curriculum_answer():
    calls = _Calls()
    qa, general = _handlers(LLMProviderError("LLM provider returned HTTP 502"), calls)
    composed = dispatch_turn(
        AssistantTurnRequest(question="unused"),
        qa=qa,
        general=general,
        classifier=lambda _request: "curriculum",
    )
    assert composed.curriculum.failure.kind == "provider_error"
    assert composed.curriculum.failure.code == "LLM_PROVIDER_FAILURE"
    assert composed.official_curriculum_answer is None
    assert calls.general == 0


def test_curriculum_clarification_does_not_generate_general_advice():
    calls = _Calls()
    qa, general = _handlers(needs_clarification(), calls)
    composed = dispatch_turn(
        AssistantTurnRequest(question="unused"),
        qa=qa,
        general=general,
        classifier=lambda _request: "curriculum",
    )
    assert calls.general == 0
    assert composed.general is None
    assert composed.clarification == "Which grade or level would you like me to check?"
    assert composed.official_curriculum_answer is None


def test_general_handler_failure_does_not_become_curriculum_text():
    def general(_request: AssistantTurnRequest) -> str:
        raise LLMProviderError("general model unavailable")

    composed = dispatch_turn(
        AssistantTurnRequest(question="unused"),
        qa=lambda _request: completed_with_evidence(),
        general=general,
        classifier=lambda _request: "mixed",
    )
    assert composed.official_curriculum_answer
    assert composed.general is None
    assert composed.general_failure is not None
    assert composed.general_failure.code == "LLM_PROVIDER_FAILURE"
    assert composed.general_failure.message not in composed.official_curriculum_answer


def test_missing_handler_is_an_explicit_error():
    with pytest.raises(AssistantDispatchError):
        dispatch_turn(
            AssistantTurnRequest(question="What topics are taught in Primary 4 Mathematics?"),
        )
