from app.assistant.contract import (
    GENERAL_LABEL,
    GENERAL_SOURCE,
    CURRICULUM_SOURCE,
    AssistantRoute,
    RoutingResult,
)
from app.assistant.composer import blocks_general_advice, compose_turn
from app.assistant.router import CLARIFICATION_PROMPT
from tests.assistant.qa_fixtures import (
    completed_with_evidence,
    evidence_status_error_http_200,
    execution_failure,
    insufficient_evidence,
    needs_clarification,
    partial_incomplete_coverage,
    timeout_failure,
)

_ADVICE = (
    "General teaching note: ask pupils to explain their method to a partner. "
    "This is not an MBSSE curriculum statement."
)


def _routing(route: AssistantRoute) -> RoutingResult:
    return RoutingResult(route=route, source="heuristic", reason="test")


def test_completed_curriculum_text_is_copied_and_labelled():
    qa = completed_with_evidence()
    composed = compose_turn(routing=_routing(AssistantRoute.CURRICULUM), qa=qa)

    assert composed.curriculum is not None
    assert composed.curriculum.source == CURRICULUM_SOURCE
    assert composed.curriculum.response is not None
    assert composed.curriculum.response.answer == qa.response.answer
    assert composed.curriculum.response.status == "completed"
    assert composed.curriculum.response.metadata.evidence_status == "found"
    assert composed.curriculum.coverage_decision == "accept"
    assert composed.curriculum.technical_failure is False
    assert composed.general is None
    assert composed.official_curriculum_answer == qa.response.answer
    assert composed.curriculum.response is not qa.response


def test_insufficient_evidence_drops_general_advice():
    qa = insufficient_evidence()
    composed = compose_turn(
        routing=_routing(AssistantRoute.MIXED),
        qa=qa,
        general_text=_ADVICE,
    )

    assert composed.curriculum is not None
    assert composed.curriculum.response.answer == qa.response.answer
    assert composed.curriculum.response.status == "insufficient_evidence"
    assert composed.general is None
    assert composed.general_suppressed_reason == "insufficient_evidence"
    assert composed.official_curriculum_answer == qa.response.answer
    assert _ADVICE not in (composed.official_curriculum_answer or "")


def test_completed_partial_evidence_does_not_receive_general_advice():
    qa = completed_with_evidence()
    qa.response.metadata.evidence_status = "partial"
    qa.coverage_decision = "accept"
    assert blocks_general_advice(qa) == "partial_evidence"
    composed = compose_turn(
        routing=_routing(AssistantRoute.MIXED),
        qa=qa,
        general_text=_ADVICE,
    )

    assert composed.curriculum.response.status == "completed"
    assert composed.curriculum.response.metadata.evidence_status == "partial"
    assert composed.curriculum.coverage_decision == "accept"
    assert composed.general is None
    assert composed.general_suppressed_reason == "partial_evidence"
    assert composed.official_curriculum_answer == qa.response.answer
    assert _ADVICE not in composed.official_curriculum_answer


def test_adjacent_evidence_statuses_keep_the_general_channel_closed():
    missing = completed_with_evidence()
    missing.response.metadata.evidence_status = "not_found"
    missing.coverage_decision = None
    assert blocks_general_advice(missing) == "evidence_not_found"

    partial_without_coverage = completed_with_evidence()
    partial_without_coverage.response.metadata.evidence_status = "partial"
    partial_without_coverage.coverage_decision = None
    assert blocks_general_advice(partial_without_coverage) == "partial_evidence"

    found_without_coverage = completed_with_evidence()
    found_without_coverage.coverage_decision = None
    assert blocks_general_advice(found_without_coverage) is None
    composed = compose_turn(
        routing=_routing(AssistantRoute.MIXED),
        qa=found_without_coverage,
        general_text=_ADVICE,
    )
    assert composed.general is not None
    assert composed.general.label == "General AI assistance"
    assert composed.official_curriculum_answer == found_without_coverage.response.answer
    assert _ADVICE not in composed.official_curriculum_answer


def test_incomplete_coverage_is_not_filled_by_general_advice():
    qa = partial_incomplete_coverage()
    assert blocks_general_advice(qa) == "insufficient_evidence"
    composed = compose_turn(
        routing=_routing(AssistantRoute.MIXED),
        qa=qa,
        general_text=_ADVICE,
    )

    assert composed.curriculum.response.metadata.evidence_status == "partial"
    assert composed.curriculum.coverage_decision == "unavailable"
    assert composed.curriculum.response.verification.passed is False
    assert composed.general is None
    assert "complete topic list" in composed.official_curriculum_answer
    assert _ADVICE not in composed.official_curriculum_answer


def test_evidence_status_error_on_http_200_is_a_technical_failure():
    qa = evidence_status_error_http_200()
    composed = compose_turn(
        routing=_routing(AssistantRoute.MIXED),
        qa=qa,
        general_text=_ADVICE,
    )

    assert composed.curriculum.http_status == 200
    assert composed.curriculum.response.metadata.evidence_status == "error"
    assert composed.curriculum.technical_failure is True
    assert composed.general is None
    assert composed.general_suppressed_reason == "evidence_status_error"
    assert composed.official_curriculum_answer is None


def test_execution_failure_and_timeout_suppress_general_advice():
    for qa, kind, code in (
        (execution_failure(), "execution_failure", "AGENT_EXECUTION_FAILURE"),
        (timeout_failure(), "timeout", "LLM_TIMEOUT"),
    ):
        composed = compose_turn(
            routing=_routing(AssistantRoute.MIXED),
            qa=qa,
            general_text=_ADVICE,
        )
        assert composed.curriculum.response is None
        assert composed.curriculum.failure.kind == kind
        assert composed.curriculum.failure.code == code
        assert composed.curriculum.technical_failure is True
        assert composed.general is None
        assert composed.general_suppressed_reason == f"technical_failure:{kind}"
        assert composed.official_curriculum_answer is None


def test_mixed_completed_answer_keeps_provenance_separate():
    qa = completed_with_evidence()
    composed = compose_turn(
        routing=_routing(AssistantRoute.MIXED),
        qa=qa,
        general_text=_ADVICE,
    )

    assert composed.curriculum.source == CURRICULUM_SOURCE
    assert composed.general is not None
    assert composed.general.source == GENERAL_SOURCE
    assert composed.general.label == GENERAL_LABEL
    assert composed.general.text == _ADVICE
    assert composed.official_curriculum_answer == qa.response.answer
    assert composed.official_curriculum_answer != composed.general.text
    assert composed.general.text not in composed.official_curriculum_answer


def test_curriculum_route_does_not_keep_general_prose():
    composed = compose_turn(
        routing=_routing(AssistantRoute.CURRICULUM),
        qa=completed_with_evidence(),
        general_text=_ADVICE,
    )
    assert composed.general is None
    assert composed.general_suppressed_reason == "route_curriculum"


def test_general_route_does_not_present_a_curriculum_answer():
    composed = compose_turn(
        routing=_routing(AssistantRoute.GENERAL),
        qa=completed_with_evidence(),
        general_text=_ADVICE,
    )
    assert composed.curriculum is None
    assert composed.official_curriculum_answer is None
    assert composed.general.text == _ADVICE
    assert composed.general_suppressed_reason == "route_general"


def test_clarification_composition_drops_both_channels():
    composed = compose_turn(
        routing=_routing(AssistantRoute.CLARIFY),
        qa=completed_with_evidence(),
        general_text=_ADVICE,
    )
    assert composed.curriculum is None
    assert composed.general is None
    assert composed.official_curriculum_answer is None
    assert composed.clarification == CLARIFICATION_PROMPT
    assert composed.general_suppressed_reason == "route_clarify"


def test_qa_clarification_is_copied_from_the_qa_result():
    qa = needs_clarification()
    composed = compose_turn(routing=_routing(AssistantRoute.CURRICULUM), qa=qa)
    assert composed.curriculum.response.answer is None
    assert composed.curriculum.response.status == "needs_clarification"
    assert composed.clarification == "Which grade or level would you like me to check?"
    assert composed.general is None
