"""V2.1 grade-resolution integration: agent tool returns Fractions context."""

from __future__ import annotations

import httpx
import pytest

from app.agent.state import CurriculumQAState
from app.agent.retrieve import RetrievalNode
from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.llm.provider import StubLLMProvider
from app.tools.registry import build_default_registry

GOLDEN = "What are the learning objectives for fractions in Primary 4?"


def _fractions_v2_router(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/api/v2/curriculum/context/resolve"):
        return httpx.Response(
            200,
            json={
                "curriculum": {
                    "id": "74a9310e-4aed-47c5-9981-a0a65ba27093",
                    "code": "MBSSE-BEC",
                    "version": "2020",
                },
                "grade": {"id": "g4-gc", "code": "CLASS_4", "name": "Class 4"},
                "subject": {
                    "id": "sub-math",
                    "code": "MATHEMATICS",
                    "name": "Mathematics",
                },
                "grade_curriculum_id": "gc-math-p4",
                "units": [
                    {"id": "u4", "code": "C4-U04", "name": "FRACTION", "content_type": "UNIT"},
                    {"id": "u5", "code": "C4-U05", "name": "OPERATION ON FRACTIONS", "content_type": "UNIT"},
                    {"id": "u6", "code": "C4-U06", "name": "Fraction Multiplication", "content_type": "UNIT"},
                ],
                "learning_outcomes": [
                    {
                        "id": f"lo-{i}",
                        "code": f"C4U0{i}-LO01",
                        "description": f"Outcome {i}",
                        "parent_content_id": f"u{i}",
                    }
                    for i in range(4, 14)
                ],
                "resolution": {
                    "status": "resolved",
                    "matched_by": {
                        "grade": "curriculum_grade_code+subject",
                        "subject": "code",
                        "topic": "name_or_code",
                    },
                    "authority": "grade_curriculum",
                    "query_timing_ms": 12.0,
                    "diagnostics": {
                        "requested_grade": "CLASS_4",
                        "resolved_grade_code": "CLASS_4",
                        "grade_curriculum_id": "gc-math-p4",
                        "grade_strategy": "curriculum_grade_code+subject",
                    },
                },
            },
        )
    if path.endswith("/api/v1/curricula"):
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "74a9310e-4aed-47c5-9981-a0a65ba27093",
                        "code": "MBSSE-BEC",
                        "version": "2020",
                    }
                ],
                "total": 1,
                "limit": 20,
                "offset": 0,
            },
        )
    return httpx.Response(404, json={"detail": f"unhandled {path}"})


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_provider="stub",
        curriculum_api_base_url="http://curriculum.test",
        agent_max_iterations=2,
        agent_max_retrieval_rounds=1,
        agent_max_tool_calls=5,
    )


@pytest.fixture
def tools(settings: Settings):
    client = CurriculumAPIClient(
        settings=settings,
        transport=httpx.MockTransport(_fractions_v2_router),
    )
    return build_default_registry(settings=settings, client=client)


def test_golden_question_resolver_returns_three_units_and_ten_los(settings, tools):
    """Agent retrieval uses resolve_curriculum_context with full Fractions evidence."""
    node = RetrievalNode(
        llm=StubLLMProvider(), tools=tools, settings=settings
    )
    state = CurriculumQAState.initial(question=GOLDEN)
    state.grade = "CLASS_4"
    state.subject = "MATHEMATICS"
    state.topic = "fractions"
    state = node.run(state)

    resolve_calls = [
        r for r in state.retrieval_history if r.tool == "resolve_curriculum_context"
    ]
    assert resolve_calls, "expected resolve_curriculum_context in retrieval history"

    result = tools.execute(
        "resolve_curriculum_context",
        grade="CLASS_4",
        subject="MATHEMATICS",
        topic="fractions",
    )
    assert result.success
    data = result.data or {}
    assert data["resolution"]["status"] == "resolved"
    assert len(data["units"]) == 3
    assert len(data["learning_outcomes"]) == 10
    obs = data.get("observability") or {}
    assert obs.get("resolution_status") == "resolved"
    assert obs.get("grade_strategy") == "curriculum_grade_code+subject"
    assert obs.get("learning_outcome_count") == 10
