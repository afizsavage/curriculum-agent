import pytest

from app.agent.orchestrator import CurriculumQAAgent
from app.agent.context import ConversationStore
from app.enums import AgentStatus
from app.exceptions import InvalidRequestError
from app.llm.provider import StubLLMProvider
from app.tools.registry import ToolRegistry


def test_agent_accepts_question_and_creates_state():
    agent = CurriculumQAAgent(
        llm=StubLLMProvider(),
        tools=ToolRegistry(),
        conversations=ConversationStore(),
    )
    state = agent.ask("What topics are taught in Primary 4 Mathematics?")
    assert state.question.startswith("What topics")
    assert state.conversation_id
    assert state.status == AgentStatus.RECEIVED
    assert state.iteration == 0
    assert state.tool_calls == 0
    assert state.draft_answer is None


def test_agent_reuses_conversation():
    store = ConversationStore()
    agent = CurriculumQAAgent(
        llm=StubLLMProvider(),
        tools=ToolRegistry(),
        conversations=store,
    )
    first = agent.ask("What topics are in Primary 4 Mathematics?")
    second = agent.ask(
        "Which one comes before fractions?",
        conversation_id=first.conversation_id,
    )
    assert second.conversation_id == first.conversation_id
    ctx = store.get(first.conversation_id)
    assert ctx is not None
    assert len(ctx.messages) == 2
    assert ctx.current_question == "Which one comes before fractions?"


def test_agent_rejects_blank_question():
    agent = CurriculumQAAgent(
        llm=StubLLMProvider(),
        tools=ToolRegistry(),
        conversations=ConversationStore(),
    )
    with pytest.raises(InvalidRequestError):
        agent.ask("   ")


def test_stub_nodes_update_status():
    agent = CurriculumQAAgent(
        llm=StubLLMProvider(),
        tools=ToolRegistry(),
        conversations=ConversationStore(),
    )
    state = agent.ask("Q")
    state = agent.understand(state)
    assert state.status == AgentStatus.UNDERSTANDING
    assert state.iteration == 1
    state = agent.retrieve(state)
    assert state.status == AgentStatus.RETRIEVING
    state = agent.answer(state)
    assert state.status == AgentStatus.ANSWERING
    state = agent.verify(state)
    assert state.status == AgentStatus.VERIFYING
