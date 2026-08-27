"""Sprint 4 verification + bounded loop tests."""

from __future__ import annotations

import httpx
import pytest

from app.agent.context import ConversationStore
from app.agent.metrics import get_metrics
from app.agent.orchestrator import CurriculumQAAgent
from app.agent.state import CurriculumQAState
from app.agent.verification_checks import run_deterministic_checks
from app.agent.verifier import AnswerVerifier
from app.agent.verify import VerificationNode
from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.enums import AgentStatus
from app.llm.base import LLMMessage, LLMProvider, LLMResponse, ToolCallRequest
from app.llm.provider import StubLLMProvider
from app.schemas.answer import AnswerConfidence, AnswerEvidenceRef
from app.schemas.verification import (
    MissingEvidenceRequest,
    VerificationRecommendation,
    VerificationResult,
    VerificationStatus,
)
from app.tools.registry import build_default_registry
from tests.tools.test_curriculum_tools import _router


@pytest.fixture
def settings() -> Settings:
    return Settings(
        curriculum_api_base_url="http://curriculum.test",
        agent_max_iterations=3,
        agent_max_tool_calls=10,
        agent_max_retrieval_rounds=3,
    )


@pytest.fixture
def agent(settings: Settings) -> CurriculumQAAgent:
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    return CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
    )


class AlwaysRetrieveMoreVerifier(AnswerVerifier):
    def verify(self, state, *, request_id=None):  # type: ignore[override]
        return VerificationResult(
            passed=False,
            score=0.2,
            issues=["forced retrieve_more"],
            unsupported_claims=["forced"],
            missing_evidence=[
                MissingEvidenceRequest(
                    type="learning_objective",
                    grade=state.grade or "CLASS_4",
                    subject=state.subject or "MATHEMATICS",
                    topic=state.topic or "Fractions",
                    query="fractions",
                )
            ],
            incorrect_claims=[],
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            metadata={"source": "test"},
        )


class AlwaysClarifyVerifier(AnswerVerifier):
    def verify(self, state, *, request_id=None):  # type: ignore[override]
        return VerificationResult(
            passed=False,
            score=0.4,
            issues=["ambiguous question"],
            recommendation=VerificationRecommendation.CLARIFY,
            clarification="Which grade or level would you like me to check?",
            metadata={"source": "test"},
        )


class PassThenFailVerifier(AnswerVerifier):
    """Pass on second attempt to exercise retrieve→verify→retrieve→verify."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def verify(self, state, *, request_id=None):  # type: ignore[override]
        self.calls += 1
        if self.calls >= 2:
            return VerificationResult(
                passed=True,
                score=0.95,
                recommendation=VerificationRecommendation.ACCEPT,
                metadata={"source": "test", "attempt": self.calls},
            )
        return VerificationResult(
            passed=False,
            score=0.4,
            issues=["need more evidence"],
            missing_evidence=[
                MissingEvidenceRequest(
                    type="topic",
                    grade="CLASS_4",
                    subject="MATHEMATICS",
                    topic="Fractions",
                )
            ],
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            metadata={"source": "test", "attempt": self.calls},
        )


def test_happy_path_verify_pass(agent):
    state = agent.ask("What topics are taught in Primary 4 Mathematics?")
    assert state.status == AgentStatus.COMPLETED
    assert state.verification is not None
    assert state.verification.passed is True
    assert state.verification_status == VerificationStatus.PASSED
    assert state.verification_attempts >= 1
    assert state.final_answer


def test_loop_terminates_when_verifier_always_retrieve_more(settings):
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    verifier = AlwaysRetrieveMoreVerifier(StubLLMProvider(), settings=settings)
    agent = CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
        verification_node=VerificationNode(
            llm=StubLLMProvider(), settings=settings, verifier=verifier
        ),
    )
    state = agent.ask("What topics are taught in Primary 4 Mathematics?")
    assert state.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert state.verification_status == VerificationStatus.MAX_ITERATIONS
    assert state.retrieval_rounds == settings.agent_max_retrieval_rounds
    assert state.verification_attempts == settings.agent_max_retrieval_rounds
    assert state.answer_confidence == AnswerConfidence.LOW
    assert "couldn't find sufficient" in (state.final_answer or "").lower()


def test_clarify_path(settings):
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    agent = CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
        verification_node=VerificationNode(
            llm=StubLLMProvider(),
            settings=settings,
            verifier=AlwaysClarifyVerifier(StubLLMProvider(), settings=settings),
        ),
    )
    state = agent.ask("What does the curriculum say about fractions?")
    assert state.status == AgentStatus.NEEDS_CLARIFICATION
    assert state.clarification
    assert "grade" in state.clarification.lower()


def test_multi_step_retrieve_then_pass(settings):
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    verifier = PassThenFailVerifier(StubLLMProvider(), settings=settings)
    agent = CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
        verification_node=VerificationNode(
            llm=StubLLMProvider(), settings=settings, verifier=verifier
        ),
    )
    state = agent.ask("What topics are taught in Primary 4 Mathematics?")
    assert state.status == AgentStatus.COMPLETED
    assert verifier.calls == 2
    assert state.retrieval_rounds == 2
    assert state.verification_attempts == 2


def test_deterministic_wrong_grade_fails():
    state = CurriculumQAState.initial(
        question="Are fractions taught in Primary 4 Mathematics?"
    )
    state.grade = "CLASS_4"
    state.subject = "MATHEMATICS"
    state.final_answer = "Fractions are taught in Primary 4 Mathematics."
    state.evidence = [
        CurriculumEvidence(
            entity_type="topic",
            entity_id="t1",
            name="Fractions",
            grade="CLASS_5",
            subject="MATHEMATICS",
        )
    ]
    state.evidence_status = EvidenceStatus.FOUND
    state.answer_evidence = [
        AnswerEvidenceRef(
            entity_id="t1",
            entity_type="topic",
            claim="Fractions are taught in Primary 4",
            grade="CLASS_5",
            subject="MATHEMATICS",
        )
    ]
    result = run_deterministic_checks(state)
    assert result.passed is False
    assert result.incorrect_claims or result.issues
    assert result.recommendation == VerificationRecommendation.RETRIEVE_MORE


def test_deterministic_hallucinated_entity_id():
    state = CurriculumQAState.initial(question="What about fractions?")
    state.grade = "CLASS_4"
    state.subject = "MATHEMATICS"
    state.final_answer = "Learners identify equivalent fractions."
    state.evidence = [
        CurriculumEvidence(
            entity_type="topic",
            entity_id="real-1",
            name="Fractions",
            grade="CLASS_4",
            subject="MATHEMATICS",
        )
    ]
    state.evidence_status = EvidenceStatus.FOUND
    state.answer_evidence = [
        AnswerEvidenceRef(
            entity_id="invented-id",
            entity_type="learning_outcome",
            claim="Learners identify equivalent fractions.",
        )
    ]
    result = run_deterministic_checks(state)
    assert result.passed is False
    assert result.unsupported_claims


def test_deterministic_no_evidence_clarify_or_retrieve():
    state = CurriculumQAState.initial(
        question="What does the curriculum say about fractions?"
    )
    state.final_answer = "Fractions appear somewhere."
    state.evidence_status = EvidenceStatus.NOT_FOUND
    result = run_deterministic_checks(state)
    assert result.passed is False
    assert result.recommendation in {
        VerificationRecommendation.CLARIFY,
        VerificationRecommendation.RETRIEVE_MORE,
    }


def test_duplicate_tool_calls_are_skipped(settings):
    class RepeatSearchLLM(LLMProvider):
        def __init__(self):
            self.calls = 0

        @property
        def name(self) -> str:
            return "stub"

        @property
        def model(self) -> str:
            return "repeat-search"

        def generate(self, messages, *, temperature=0.0, max_tokens=None):
            return LLMResponse(content="ok", model=self.model)

        def generate_structured(self, messages, *, schema, temperature=0.0):
            return {
                "passed": True,
                "score": 0.95,
                "issues": [],
                "unsupported_claims": [],
                "missing_evidence": [],
                "incorrect_claims": [],
                "recommendation": "accept",
                "claims": [],
            }

        def generate_with_tools(self, messages, *, tools, temperature=0.0):
            self.calls += 1
            if self.calls > 2:
                return LLMResponse(content="done", tool_calls=[], model=self.model)
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id=f"call-{self.calls}",
                        name="get_curriculum_structure",
                        arguments={"grade": "Primary 4", "subject": "Mathematics"},
                    )
                ],
                model=self.model,
            )

    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    llm = RepeatSearchLLM()
    agent = CurriculumQAAgent(
        settings=settings,
        llm=llm,
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
    )
    state = agent.ask("What topics are taught in Primary 4 Mathematics?")
    successful = [
        r
        for r in state.retrieval_history
        if r.tool == "get_curriculum_structure" and r.status == "success"
    ]
    assert len(successful) == 1
    assert state.status in {
        AgentStatus.COMPLETED,
        AgentStatus.INSUFFICIENT_EVIDENCE,
    }


def test_tool_call_limit_enforced(settings):
    limited = settings.model_copy(
        update={
            "agent_max_tool_calls": 1,
            "agent_max_retrieval_rounds": 3,
            "agent_max_iterations": 3,
        }
    )
    client = CurriculumAPIClient(
        settings=limited, transport=httpx.MockTransport(_router)
    )
    agent = CurriculumQAAgent(
        settings=limited,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=limited, client=client),
        conversations=ConversationStore(),
        verification_node=VerificationNode(
            llm=StubLLMProvider(),
            settings=limited,
            verifier=AlwaysRetrieveMoreVerifier(StubLLMProvider(), settings=limited),
        ),
    )
    state = agent.ask("What topics are taught in Primary 4 Mathematics?")
    assert state.tool_calls <= 1
    assert state.status == AgentStatus.INSUFFICIENT_EVIDENCE


def test_metrics_snapshot_updated(agent):
    metrics = get_metrics()
    before = metrics.total_requests
    agent.ask("What topics are taught in Primary 4 Mathematics?")
    snap = metrics.snapshot()
    assert snap["total_requests"] >= before + 1
    assert "verification_pass_rate" in snap
