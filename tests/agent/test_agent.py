import httpx
import pytest

from app.agent.context import ConversationStore
from app.agent.orchestrator import CurriculumQAAgent
from app.agent.state import CurriculumQAState
from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.enums import AgentStatus
from app.exceptions import InvalidRequestError
from app.llm.provider import StubLLMProvider
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


def test_agent_accepts_question_and_completes(agent):
    state = agent.ask("What topics are taught in Primary 4 Mathematics?")
    assert state.question.startswith("What topics")
    assert state.conversation_id
    assert state.status == AgentStatus.COMPLETED
    assert state.iteration >= 1
    assert state.tool_calls >= 1
    assert state.final_answer
    assert state.draft_answer


def test_agent_reuses_conversation(agent):
    first = agent.ask("What topics are in Primary 4 Mathematics?")
    second = agent.ask(
        "Which one comes before fractions?",
        conversation_id=first.conversation_id,
    )
    assert second.conversation_id == first.conversation_id


def test_agent_rejects_blank_question(agent):
    with pytest.raises(InvalidRequestError):
        agent.ask("   ")


def test_understand_and_retrieve_nodes(agent):
    state = CurriculumQAState.initial(
        question="What topics are in Primary 4 Mathematics?"
    )
    state = agent.understand(state)
    assert state.status == AgentStatus.UNDERSTANDING
    assert state.grade == "CLASS_4"
    state = agent.retrieve(state)
    assert state.status == AgentStatus.RETRIEVED
