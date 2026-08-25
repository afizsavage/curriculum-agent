import httpx

from app.agent.context import ConversationStore
from app.agent.orchestrator import CurriculumQAAgent
from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.enums import AgentStatus
from app.llm.provider import StubLLMProvider
from app.tools.registry import build_default_registry
from tests.tools.test_curriculum_tools import _router


def test_retrieval_flow_updates_state():
    settings = Settings(
        llm_provider="stub",
        curriculum_api_base_url="http://curriculum.test",
        agent_max_iterations=3,
        agent_max_tool_calls=5,
    )
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    agent = CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
    )
    state = agent.ask("What topics are taught in Primary 4 Mathematics?")
    assert state.status == AgentStatus.RETRIEVED
    assert state.tool_calls >= 1
    assert "get_curriculum_structure" in state.selected_tools
    assert state.evidence_status.value in {"found", "partial"}
    assert state.retrieval_history
    assert state.grade == "CLASS_4"
    assert state.subject == "MATHEMATICS"


def test_tool_call_limit_stops():
    settings = Settings(
        llm_provider="stub",
        curriculum_api_base_url="http://curriculum.test",
        agent_max_iterations=5,
        agent_max_tool_calls=1,
    )
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    agent = CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
    )
    state = agent.ask("Find curriculum content related to fractions in Primary 4 Mathematics.")
    assert state.tool_calls <= 1
    assert state.status == AgentStatus.RETRIEVED
