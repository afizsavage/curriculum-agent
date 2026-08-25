import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent.context import ConversationStore
from app.agent.orchestrator import CurriculumQAAgent
from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.deps import get_agent, reset_singletons
from app.exceptions import AgentExecutionError
from app.llm.provider import StubLLMProvider
from app.main import create_app
from app.tools.registry import build_default_registry
from tests.tools.test_curriculum_tools import _router


@pytest.fixture
def client():
    reset_singletons()
    settings = Settings(curriculum_api_base_url="http://curriculum.test")
    api = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    agent = CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=api),
        conversations=ConversationStore(),
    )
    application = create_app()
    application.dependency_overrides[get_agent] = lambda: agent
    with TestClient(application, raise_server_exceptions=False) as test_client:
        yield test_client
    application.dependency_overrides.clear()
    reset_singletons()


def test_ask_valid_request(client):
    response = client.post(
        "/api/v1/agent/ask",
        json={
            "question": "What topics are taught in Primary 4 Mathematics?",
            "conversation_id": None,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"]
    assert body["status"] == "completed"
    assert body["question"].startswith("What topics")
    assert body["conversation_id"]
    assert body["metadata"]["tool_calls"] >= 1
    assert "get_curriculum_structure" in body["metadata"]["tools_used"]
    assert isinstance(body["evidence"], list)
    assert body["confidence"] in {"high", "medium", "low"}
    assert isinstance(body["limitations"], list)
    assert response.headers.get("X-Request-ID")


def test_ask_invalid_request(client):
    response = client.post("/api/v1/agent/ask", json={"question": "  "})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "INVALID_REQUEST"


def test_ask_missing_question(client):
    response = client.post("/api/v1/agent/ask", json={})
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_REQUEST"


def test_unexpected_agent_errors_are_safe(client):
    class BoomAgent(CurriculumQAAgent):
        def ask(self, question, *, conversation_id=None, request_id=None):
            raise AgentExecutionError("Agent failed while processing the question")

    app = create_app()
    app.dependency_overrides[get_agent] = lambda: BoomAgent(
        llm=StubLLMProvider(),
        tools=ToolRegistrySafe(),
        conversations=ConversationStore(),
    )
    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post(
            "/api/v1/agent/ask",
            json={"question": "What topics are taught in Primary 4 Mathematics?"},
        )
    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "AGENT_EXECUTION_FAILURE"
    assert "traceback" not in body


class ToolRegistrySafe:
    def llm_tool_specs(self):
        return []

    def execute(self, *args, **kwargs):
        raise RuntimeError("unused")

    def list(self):
        return []


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
