"""In-process QA adapter tests. No live model and no agent HTTP route."""

from __future__ import annotations

import httpx
import pytest

from app.agent.context import ConversationStore
from app.agent.orchestrator import CurriculumQAAgent
from app.agent.response_mapper import map_graph_result_to_response
from app.agent.state import CurriculumQAState
from app.assistant.contract import AssistantTurnRequest
from app.assistant.dispatch import dispatch_turn
from app.assistant.qa_adapter import (
    CurriculumQAAdapter,
    outcome_from_state,
    public_ask_graph_state,
)
from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.enums import AgentStatus
from app.exceptions import AgentExecutionError, LLMProviderError, LLMTimeoutError
from app.llm.provider import StubLLMProvider
from app.schemas.answer import AnswerConfidence
from app.schemas.verification import (
    VerificationRecommendation,
    VerificationResult,
    VerificationStatus,
)
from app.tools.registry import build_default_registry
from tests.tools.test_curriculum_tools import _router

_ADVICE = "Ask pupils to explain one step aloud. This is not an MBSSE record."


class FakeQAAgent:
    def __init__(self, result: CurriculumQAState | Exception) -> None:
        self.llm = StubLLMProvider(model="stub-model")
        self.result = result
        self.calls: list[dict] = []

    def ask(self, question: str, *, conversation_id=None, request_id=None):
        self.calls.append(
            {
                "question": question,
                "conversation_id": conversation_id,
                "request_id": request_id,
            }
        )
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _state(
    *,
    answer: str | None,
    status: AgentStatus,
    evidence_status: EvidenceStatus,
    verification_status: VerificationStatus,
    recommendation: VerificationRecommendation,
    passed: bool,
    coverage_decision: str | None = None,
    error: str | None = None,
    evidence: list[CurriculumEvidence] | None = None,
) -> CurriculumQAState:
    state = CurriculumQAState.initial(
        question="What topics are taught in Primary 4 Mathematics?",
        conversation_id="conv-adapter",
    )
    state.status = status
    state.final_answer = answer
    state.draft_answer = answer
    state.evidence = list(evidence or [])
    state.evidence_status = evidence_status
    state.verification_status = verification_status
    state.answer_confidence = AnswerConfidence.LOW if not passed else AnswerConfidence.HIGH
    state.error = error
    state.iteration = 1
    state.tool_calls = 1
    state.selected_tools = ["get_curriculum_structure"]
    state.metadata.update(
        {
            "graph_run_id": "run-adapter",
            "visited_nodes": ["understand", "finish"],
            "route": "finish" if passed else "fallback",
            "termination_reason": "verification_passed" if passed else "verification_fallback",
            "tools_used": ["get_curriculum_structure"],
        }
    )
    metadata = {}
    if coverage_decision is not None:
        metadata["coverage_decision"] = coverage_decision
        metadata["source"] = "coverage_validator"
    state.verification = VerificationResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        recommendation=recommendation,
        issues=[] if passed else ["Evidence did not establish the claim."],
        metadata=metadata,
    )
    return state


def _completed() -> CurriculumQAState:
    return _state(
        answer="Primary 4 Mathematics includes Number and Numeration.",
        status=AgentStatus.COMPLETED,
        evidence_status=EvidenceStatus.FOUND,
        verification_status=VerificationStatus.PASSED,
        recommendation=VerificationRecommendation.ACCEPT,
        passed=True,
        coverage_decision="accept",
        evidence=[
            CurriculumEvidence(
                entity_type="topic",
                entity_id="topic-number",
                name="Number and Numeration",
                grade="CLASS_4",
                subject="MATHEMATICS",
            )
        ],
    )


def _request(**kwargs) -> AssistantTurnRequest:
    payload = {
        "question": "What topics are taught in Primary 4 Mathematics?",
        "conversation_id": "conv-adapter",
        "request_id": "req-adapter",
    }
    payload.update(kwargs)
    return AssistantTurnRequest(**payload)


def test_adapter_copies_completed_ask_response_verbatim():
    state = _completed()
    agent = FakeQAAgent(state)
    outcome = CurriculumQAAdapter(agent)(_request(grade="CLASS_4", subject="MATHEMATICS"))

    expected = map_graph_result_to_response(
        qa=state,
        llm=agent.llm,
        graph_state=public_ask_graph_state(state),
    )
    assert outcome.http_status == 200
    assert outcome.failure is None
    assert outcome.response is not None
    assert outcome.response.model_dump() == expected.model_dump()
    assert outcome.response.answer == "Primary 4 Mathematics includes Number and Numeration."
    assert outcome.response.status == "completed"
    assert outcome.response.metadata.evidence_status == "found"
    assert outcome.response.metadata.model == "stub-model"
    assert outcome.response.metadata.provider == "stub"
    assert outcome.coverage_decision == "accept"
    assert "coverage_decision" not in outcome.response.verification.model_dump()
    assert agent.calls == [
        {
            "question": "What topics are taught in Primary 4 Mathematics?",
            "conversation_id": "conv-adapter",
            "request_id": "req-adapter",
        }
    ]


def test_adapter_preserves_insufficient_partial_and_evidence_error_bodies():
    cases = [
        (
            _state(
                answer="I couldn't find sufficient MBSSE curriculum evidence to answer this reliably.",
                status=AgentStatus.INSUFFICIENT_EVIDENCE,
                evidence_status=EvidenceStatus.NOT_FOUND,
                verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                recommendation=VerificationRecommendation.FALLBACK,
                passed=False,
            ),
            "insufficient_evidence",
            "not_found",
            False,
            None,
        ),
        (
            _state(
                answer="Some Class 4 topics were found, but the list is not complete.",
                status=AgentStatus.INSUFFICIENT_EVIDENCE,
                evidence_status=EvidenceStatus.PARTIAL,
                verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                recommendation=VerificationRecommendation.FALLBACK,
                passed=False,
                coverage_decision="reject",
                evidence=[
                    CurriculumEvidence(
                        entity_type="topic",
                        entity_id="topic-number",
                        name="Number and Numeration",
                    )
                ],
            ),
            "insufficient_evidence",
            "partial",
            False,
            "reject",
        ),
        (
            _state(
                answer=None,
                status=AgentStatus.INSUFFICIENT_EVIDENCE,
                evidence_status=EvidenceStatus.ERROR,
                verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                recommendation=VerificationRecommendation.FALLBACK,
                passed=False,
                error="Curriculum API request timed out",
            ),
            "insufficient_evidence",
            "error",
            True,
            None,
        ),
    ]
    for state, status, evidence_status, technical, coverage in cases:
        outcome = outcome_from_state(state, llm=StubLLMProvider())
        assert outcome.http_status == 200
        assert outcome.response.status == status
        assert outcome.response.answer == state.final_answer
        assert outcome.response.metadata.evidence_status == evidence_status
        assert outcome.response.error == state.error
        assert outcome.coverage_decision == coverage
        composed = dispatch_turn(
            _request(),
            qa_agent=FakeQAAgent(state),
            general=lambda _request: _ADVICE,
            classifier=lambda _request: "mixed",
        )
        assert composed.general is None
        assert composed.official_curriculum_answer == state.final_answer
        assert composed.curriculum.technical_failure is technical


def test_raised_qa_failures_stay_failures_and_skip_general_generation():
    calls = {"general": 0}

    def general(_request: AssistantTurnRequest) -> str:
        calls["general"] += 1
        return _ADVICE

    cases = [
        (LLMTimeoutError("LLM request timed out"), "timeout", "LLM_TIMEOUT", 504),
        (
            LLMProviderError("LLM provider returned HTTP 502"),
            "provider_error",
            "LLM_PROVIDER_FAILURE",
            502,
        ),
        (
            AgentExecutionError("Agent failed while processing the question"),
            "execution_failure",
            "AGENT_EXECUTION_FAILURE",
            500,
        ),
    ]
    for error, kind, code, http_status in cases:
        agent = FakeQAAgent(error)
        with pytest.raises(type(error)):
            CurriculumQAAdapter(agent)(_request())
        composed = dispatch_turn(
            _request(),
            qa_agent=agent,
            general=general,
            classifier=lambda _request: "mixed",
        )
        assert composed.general is None
        assert composed.official_curriculum_answer is None
        assert composed.curriculum.failure.kind == kind
        assert composed.curriculum.failure.code == code
        assert composed.curriculum.failure.http_status == http_status
        assert composed.curriculum.technical_failure is True
        assert _ADVICE not in (composed.curriculum.failure.message)
    assert calls["general"] == 0


def test_mixed_completed_outcome_may_include_separate_general_advice():
    state = _completed()
    calls = {"general": 0}

    def general(_request: AssistantTurnRequest) -> str:
        calls["general"] += 1
        return _ADVICE

    composed = dispatch_turn(
        _request(),
        qa_agent=FakeQAAgent(state),
        general=general,
        classifier=lambda _request: "mixed",
    )
    assert calls["general"] == 1
    assert composed.official_curriculum_answer == state.final_answer
    assert composed.general is not None
    assert composed.general.text == _ADVICE
    assert composed.general.text not in composed.official_curriculum_answer
    assert composed.curriculum.source == "mbsse_curriculum"
    assert composed.general.source == "general_ai_assistance"


def test_unavailable_coverage_blocks_general_generation_even_when_status_completed():
    state = _completed()
    state.verification.metadata["coverage_decision"] = "unavailable"
    calls = {"general": 0}

    def general(_request: AssistantTurnRequest) -> str:
        calls["general"] += 1
        return _ADVICE

    composed = dispatch_turn(
        _request(),
        qa_agent=FakeQAAgent(state),
        general=general,
        classifier=lambda _request: "mixed",
    )
    assert calls["general"] == 0
    assert composed.general is None
    assert composed.general_suppressed_reason == "incomplete_coverage"
    assert composed.curriculum.coverage_decision == "unavailable"
    assert composed.official_curriculum_answer == state.final_answer


def test_clarify_does_not_call_the_qa_agent():
    agent = FakeQAAgent(_completed())
    composed = dispatch_turn(
        AssistantTurnRequest(question="Tell me more"),
        qa_agent=agent,
        general=lambda _request: _ADVICE,
    )
    assert agent.calls == []
    assert composed.curriculum is None
    assert composed.general is None


def test_real_agent_is_in_process_and_preserves_the_stub_answer():
    seen: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return _router(request)

    settings = Settings(
        curriculum_api_base_url="http://curriculum.test",
        agent_checkpointing_enabled=False,
        llm_provider="stub",
    )
    api = CurriculumAPIClient(settings=settings, transport=httpx.MockTransport(transport))
    agent = CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=api),
        conversations=ConversationStore(),
        checkpointer=None,
    )
    captured: dict[str, CurriculumQAState] = {}
    real_ask = agent.ask

    def spy(question, *, conversation_id=None, request_id=None, **kwargs):
        assert kwargs == {}
        state = real_ask(
            question,
            conversation_id=conversation_id,
            request_id=request_id,
        )
        captured["state"] = state
        return state

    agent.ask = spy
    request = AssistantTurnRequest(
        question="What topics are taught in Primary 4 Mathematics?",
        request_id="req-live-stub",
    )
    outcome = CurriculumQAAdapter(agent)(request)
    state = captured["state"]
    expected = map_graph_result_to_response(
        qa=state,
        llm=agent.llm,
        graph_state=public_ask_graph_state(state),
    )

    assert seen
    assert all("/agent/ask" not in path for path in seen)
    assert outcome.response.model_dump() == expected.model_dump()
    assert outcome.response.answer == state.final_answer
    assert outcome.response.status == state.status.value
    assert outcome.http_status == 200
    assert outcome.coverage_decision is None or outcome.coverage_decision in {
        "accept",
        "reject",
        "unavailable",
    }
