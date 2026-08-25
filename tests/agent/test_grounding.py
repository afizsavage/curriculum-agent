import httpx
import pytest

from app.agent.context import ConversationStore
from app.agent.orchestrator import CurriculumQAAgent
from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.llm.provider import StubLLMProvider
from app.schemas.answer import AnswerConfidence
from app.tools.registry import build_default_registry
from tests.tools.test_curriculum_tools import _router


@pytest.fixture
def agent() -> CurriculumQAAgent:
    settings = Settings(curriculum_api_base_url="http://curriculum.test")
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    return CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
    )


def test_primary4_math_question_gets_grounded_answer(agent):
    state = agent.ask("What topics are taught in Primary 4 Mathematics?")
    assert state.final_answer
    assert state.status.value == "completed"
    assert state.evidence
    assert state.answer_confidence in AnswerConfidence


def test_hallucination_grade_mismatch():
    """Evidence for CLASS_5 must not support a Primary 4 claim."""
    from app.agent.answer_generator import AnswerGenerator
    from app.agent.state import CurriculumQAState

    qa_state = CurriculumQAState.initial(
        question="Is topic X taught in Primary 4 Mathematics?"
    )
    qa_state.grade = "CLASS_4"
    qa_state.subject = "MATHEMATICS"
    qa_state.evidence = [
        CurriculumEvidence(
            entity_type="topic",
            entity_id="topic-x",
            name="Topic X",
            grade="CLASS_5",
            subject="MATHEMATICS",
            content="Topic X appears in Primary 5 only.",
        )
    ]
    qa_state.evidence_status = EvidenceStatus.FOUND

    result = AnswerGenerator(StubLLMProvider()).generate(qa_state)
    assert result.confidence == AnswerConfidence.LOW
    assert any("CLASS_5" in note for note in result.limitations)
    assert "Primary 4" not in result.answer or "CLASS_5" in " ".join(result.limitations)


def test_follow_up_inherits_context(agent):
    first = agent.ask("What topics are in Primary 4 Mathematics?")
    second = agent.ask(
        "Tell me more about fractions.",
        conversation_id=first.conversation_id,
    )
    assert second.conversation_id == first.conversation_id
    assert second.grade == "CLASS_4"
    assert second.final_answer
    assert any(
        e.name and "Fraction" in e.name for e in second.evidence
    ) or "Fraction" in (second.final_answer or "")


def test_explicit_grade_change_overrides_context(agent):
    first = agent.ask("What topics are in Primary 4 Mathematics?")
    second = agent.ask(
        "What about Primary 5 Mathematics?",
        conversation_id=first.conversation_id,
    )
    assert second.grade == "CLASS_5"


def test_end_to_end_api_style_flow(agent):
    state = agent.ask("What are the learning objectives for fractions in Primary 4?")
    assert state.status.value == "completed"
    assert state.final_answer
    assert state.tool_calls >= 1
    assert state.answer_evidence or state.evidence
