import pytest
from fastapi.testclient import TestClient

from app.agent.orchestrator import CurriculumQAAgent
from app.deps import get_agent, reset_singletons
from app.exceptions import AgentExecutionError
from app.main import create_app


@pytest.fixture
def client():
    reset_singletons()
    application = create_app()
    with TestClient(application, raise_server_exceptions=False) as test_client:
        yield test_client
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
    assert body["answer"] is None
    assert body["status"] == "received"
    assert body["question"].startswith("What topics")
    assert body["conversation_id"]
    assert body["metadata"]["iterations"] == 0
    assert body["metadata"]["tool_calls"] == 0
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
            raise AgentExecutionError("Agent failed while accepting the question")

    app = create_app()
    app.dependency_overrides[get_agent] = lambda: BoomAgent()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post(
            "/api/v1/agent/ask",
            json={"question": "What topics are taught in Primary 4 Mathematics?"},
        )
    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "AGENT_EXECUTION_FAILURE"
    assert "traceback" not in body
    assert "Traceback" not in body["detail"]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
