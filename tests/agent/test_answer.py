import pytest

from app.agent.answer import AnswerGenerationNode
from app.agent.state import CurriculumQAState
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.enums import AgentStatus
from app.llm.provider import StubLLMProvider
from app.schemas.answer import AnswerConfidence


@pytest.fixture
def node() -> AnswerGenerationNode:
    return AnswerGenerationNode(llm=StubLLMProvider(), settings=Settings())


def test_answer_node_sets_answering_state(node):
    state = CurriculumQAState.initial(question="What is in Primary 4 Mathematics?")
    state.status = AgentStatus.RETRIEVED
    state.evidence = [
        CurriculumEvidence(
            entity_type="subject",
            entity_id="s1",
            name="Mathematics",
            grade="CLASS_4",
            subject="MATHEMATICS",
        )
    ]
    state.evidence_status = EvidenceStatus.FOUND

    result = node.run(state)
    assert result.status == AgentStatus.ANSWERING
    assert result.final_answer
    assert result.draft_answer == result.final_answer
    assert result.answer_confidence in AnswerConfidence
    assert result.metadata["answer_confidence"]


def test_answer_node_no_evidence_still_answers(node):
    state = CurriculumQAState.initial(question="What is topic Zebra?")
    state.status = AgentStatus.RETRIEVED
    state.evidence_status = EvidenceStatus.NOT_FOUND

    result = node.run(state)
    assert result.status == AgentStatus.ANSWERING
    assert "couldn't find sufficient" in (result.final_answer or "").lower()
    assert result.answer_confidence == AnswerConfidence.LOW
