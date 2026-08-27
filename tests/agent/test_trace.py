"""Instrumentation tests — observability must not change agent behavior."""

from __future__ import annotations

import httpx
import pytest

from app.agent.context import ConversationStore
from app.agent.orchestrator import CurriculumQAAgent
from app.agent.trace import get_trace_store
from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.enums import AgentStatus
from app.llm.provider import StubLLMProvider
from app.tools.registry import build_default_registry
from tests.tools.test_curriculum_tools import _router


@pytest.fixture
def agent() -> CurriculumQAAgent:
    get_trace_store().clear()
    settings = Settings(
        curriculum_api_base_url="http://curriculum.test",
        agent_checkpointing_enabled=False,
    )
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    return CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
        checkpointer=None,
    )


def test_ask_records_agent_run_id_and_trace(agent):
    state = agent.ask("What topics are taught in Primary 4 Mathematics?")
    run_id = state.metadata.get("agent_run_id")
    assert run_id
    assert state.metadata.get("termination_reason")
    trace = get_trace_store().get(run_id)
    assert trace is not None
    events = [e["event"] for e in trace.events]
    assert "agent.request.start" in events
    assert "agent.request.end" in events
    assert "agent.node.start" in events
    assert "agent.tool.start" in events or state.tool_calls == 0
    assert "agent.generation.start" in events or "agent.generation.end" in events
    assert "agent.verification.start" in events or "agent.verification.end" in events
    assert "agent.route" in events
    assert "agent.loop" in events
    assert any(e["event"] == "agent.llm.start" for e in trace.events)
    assert any(e["event"] == "agent.llm.end" for e in trace.events)
    assert trace.final.get("status") == state.status.value
    assert trace.final.get("latency_ms") is not None
    assert trace.final.get("termination_reason")


def test_tool_calls_have_sequence_numbers(agent):
    state = agent.ask("What topics are taught in Primary 4 Mathematics?")
    trace = get_trace_store().get(state.metadata["agent_run_id"])
    assert trace is not None
    assert trace.tool_calls
    numbers = [t["tool_call_number"] for t in trace.tool_calls]
    assert numbers == sorted(numbers)
    assert all("arguments" in t for t in trace.tool_calls)
    assert all("duration_ms" in t for t in trace.tool_calls)


def test_iterations_and_evidence_recorded(agent):
    state = agent.ask("What topics are taught in Primary 4 Mathematics?")
    trace = get_trace_store().get(state.metadata["agent_run_id"])
    assert trace is not None
    assert trace.iterations
    for it in trace.iterations.values():
        assert "evidence" in it
        assert "total_after" in it["evidence"]
    assert state.status == AgentStatus.COMPLETED


def test_debug_endpoint_returns_trace(agent):
    from fastapi.testclient import TestClient

    from app.deps import get_agent
    from app.main import create_app

    state = agent.ask("What topics are taught in Primary 4 Mathematics?")
    run_id = state.metadata["agent_run_id"]
    app = create_app()
    app.dependency_overrides[get_agent] = lambda: agent
    with TestClient(app) as client:
        response = client.get(f"/api/v1/agent/debug/runs/{run_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["agent_run_id"] == run_id
        assert body["final"]["status"] == "completed"
    app.dependency_overrides.clear()
