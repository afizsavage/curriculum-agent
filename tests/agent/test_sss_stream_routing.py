"""Deterministic routing for SSS stream subject questions."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pytest

from app.agent.context import ConversationStore
from app.agent.orchestrator import CurriculumQAAgent
from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.curriculum.sss_stream_intent import (
    INTENT_SSS_STREAM_SUBJECTS,
    detect_sss_stream_subjects,
    match_sss_stream,
)
from app.enums import AgentStatus
from app.llm.base import LLMMessage
from app.llm.provider import StubLLMProvider
from app.llm.tool_selection import select_tool_calls
from app.tools.curriculum import GetSSSStreamSubjectsTool
from app.tools.registry import build_default_registry
from tests.tools.test_curriculum_tools import _router as structure_router

# A lookup id returned by the mocked stream list. Not a production stream id.
STREAM_ID = "lookup-stream-id"
CURRICULUM_ID = "curr-ssc"
KNOWN_PRODUCTION_STREAM_ID = "ea3b4aa0-f010-4742-97aa-f350a3fb500d"

SUBJECTS = [
    {
        "id": "assign-chem",
        "subject_id": "subject-chem",
        "subject_type": "CORE",
        "display_order": 1,
        "subject": {"id": "subject-chem", "name": "Chemistry", "code": "CHEMISTRY", "status": "ACTIVE"},
    },
    {
        "id": "assign-bio",
        "subject_id": "subject-bio",
        "subject_type": "CORE",
        "display_order": 2,
        "subject": {"id": "subject-bio", "name": "Biology", "code": "BIOLOGY", "status": "ACTIVE"},
    },
    {
        "id": "assign-phy",
        "subject_id": "subject-phy",
        "subject_type": "APPLIED",
        "display_order": 3,
        "subject": {"id": "subject-phy", "name": "Physics", "code": "PHYSICS", "status": "ACTIVE"},
    },
]
SUBJECT_NAMES = ["Chemistry", "Biology", "Physics"]

VARIATIONS = [
    "What subjects are in the Sciences & Technologies stream?",
    "What are the subjects in Sciences & Technologies?",
    "List the subjects in the Sciences & Technologies stream.",
    "Which subjects belong to the Sciences & Technologies stream?",
    "What subjects does the Sciences & Technologies stream contain?",
]


class PlannerSpy(StubLLMProvider):
    """Counts planner calls. Stream questions must not reach this method."""

    def __init__(self) -> None:
        super().__init__()
        self.planner_calls = 0

    def generate_with_tools(self, messages, *, tools, temperature=0.0):
        self.planner_calls += 1
        return super().generate_with_tools(
            messages, tools=tools, temperature=temperature
        )


def _streams_page() -> dict:
    return {
        "items": [
            {
                "id": STREAM_ID,
                "name": "Sciences & Technologies",
                "code": "SCIENCES_TECHNOLOGIES",
                "curriculum_id": CURRICULUM_ID,
                "status": "ACTIVE",
            },
            {
                "id": "lookup-math-stream",
                "name": "Mathematics & Numeracy",
                "code": "MATHEMATICS_NUMERACY",
                "curriculum_id": CURRICULUM_ID,
                "status": "ACTIVE",
            },
        ],
        "total": 2,
        "limit": 200,
        "offset": 0,
    }


def _stream_router(subjects: list[dict] | None):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        assert KNOWN_PRODUCTION_STREAM_ID not in path
        if path.endswith("/api/v1/curricula"):
            assert request.url.params.get("code") == "MBSSE-SSC"
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": CURRICULUM_ID,
                            "code": "MBSSE-SSC",
                            "version": "2021",
                            "name": "SSC",
                        }
                    ],
                    "total": 1,
                },
            )
        if path.endswith(f"/api/v1/curricula/{CURRICULUM_ID}/sss-streams"):
            return httpx.Response(200, json=_streams_page())
        if path.endswith(f"/api/v1/sss-streams/{STREAM_ID}/subjects"):
            rows = subjects or []
            return httpx.Response(
                200,
                json={"items": rows, "total": len(rows), "limit": 200, "offset": 0},
            )
        if path.endswith(f"/api/v1/sss-streams/{STREAM_ID}"):
            return httpx.Response(
                200,
                json={
                    "id": STREAM_ID,
                    "name": "Sciences & Technologies",
                    "code": "SCIENCES_TECHNOLOGIES",
                    "description": "Specialist SSS stream.",
                    "curriculum_id": CURRICULUM_ID,
                },
            )
        return httpx.Response(404, json={"detail": f"unhandled {path}"})

    return calls, handler


def _agent(handler, spy: PlannerSpy) -> CurriculumQAAgent:
    settings = Settings(
        llm_provider="stub",
        curriculum_api_base_url="http://curriculum.test",
        agent_max_iterations=3,
        agent_max_tool_calls=5,
    )
    client = CurriculumAPIClient(settings=settings, transport=httpx.MockTransport(handler))
    return CurriculumQAAgent(
        settings=settings,
        llm=spy,
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
    )


def test_exact_question_detects_stream_subject_intent():
    detected = detect_sss_stream_subjects(
        "Which subjects are in the Sciences & Technologies stream?"
    )
    assert detected is not None
    assert detected.intent == INTENT_SSS_STREAM_SUBJECTS
    assert detected.stream_name == "Sciences & Technologies"


@pytest.mark.parametrize("question", VARIATIONS)
def test_wording_variations_share_the_stream_subject_intent(question: str):
    detected = detect_sss_stream_subjects(question)
    assert detected is not None
    assert detected.intent == INTENT_SSS_STREAM_SUBJECTS
    assert detected.stream_name == "Sciences & Technologies"


@pytest.mark.parametrize(
    "question",
    [
        "What are the topics in Primary 3 Mathematics?",
        "What subjects are available in Primary 4?",
        "Which subjects are in Primary 4?",
        "Tell me about the Sciences & Technologies stream",
        "Which stream should a pupil choose?",
    ],
)
def test_unrelated_questions_are_not_stream_subject_intent(question: str):
    assert detect_sss_stream_subjects(question) is None


def test_stream_lookup_resolves_the_listed_name():
    matched = match_sss_stream(
        _streams_page()["items"],
        "sciences and technologies",
    )
    assert matched is not None
    assert matched["id"] == STREAM_ID
    assert matched["name"] == "Sciences & Technologies"
    assert KNOWN_PRODUCTION_STREAM_ID not in str(matched["id"])


def test_tool_loads_subjects_through_the_stream_lookup():
    calls, handler = _stream_router(SUBJECTS)
    settings = Settings(curriculum_api_base_url="http://curriculum.test")
    client = CurriculumAPIClient(settings=settings, transport=httpx.MockTransport(handler))
    result = GetSSSStreamSubjectsTool(client).execute(
        stream_name="Sciences & Technologies"
    )
    assert result.success
    assert result.data["observability"]["sss_stream_resolution"] == "found"
    assert result.data["observability"]["subject_count"] == 3
    subject_evidence = [
        row for row in result.data["evidence"] if row["entity_type"] == "subject"
    ]
    assert [row["name"] for row in subject_evidence] == SUBJECT_NAMES
    assert all(
        row["metadata"]["source_type"] == "sss_stream_subject" for row in subject_evidence
    )
    assert any(row["entity_type"] == "sss_stream" for row in result.data["evidence"])
    assert any(path.endswith(f"/sss-streams/{STREAM_ID}") for path in calls)
    assert any(path.endswith(f"/sss-streams/{STREAM_ID}/subjects") for path in calls)
    assert all(KNOWN_PRODUCTION_STREAM_ID not in path for path in calls)
    assert "structure" not in " ".join(calls)


def test_agent_answers_from_stream_subjects_without_the_planner(caplog):
    caplog.set_level(logging.INFO)
    calls, handler = _stream_router(SUBJECTS)
    spy = PlannerSpy()
    state = _agent(handler, spy).ask(
        "Which subjects are in the Sciences & Technologies stream?"
    )
    assert state.intent == INTENT_SSS_STREAM_SUBJECTS
    assert state.metadata["stream_name"] == "Sciences & Technologies"
    assert state.metadata["retrieval_plan_source"] == "heuristic"
    assert state.metadata["sss_stream_resolution"] == "found"
    assert state.metadata["subject_count"] == 3
    assert state.selected_tools == ["get_sss_stream_subjects"]
    assert "get_curriculum_structure" not in state.selected_tools
    assert spy.planner_calls == 0
    assert state.status == AgentStatus.COMPLETED
    assert state.verification is not None and state.verification.passed
    answer = state.final_answer or ""
    assert "couldn't find sufficient" not in answer.lower()
    for name in SUBJECT_NAMES:
        assert name in answer
    assert "Sciences & Technologies" in answer
    assert KNOWN_PRODUCTION_STREAM_ID not in answer
    assert "retrieval_plan_source='heuristic'" in caplog.text
    assert "intent='SSS_STREAM_SUBJECTS'" in caplog.text
    assert "stream_name='Sciences & Technologies'" in caplog.text
    assert "tool='get_sss_stream_subjects'" in caplog.text
    assert "sss_stream_resolution='found'" in caplog.text
    assert "subject_count=3" in caplog.text
    assert all("get_curriculum_structure" not in path and "structure" not in path for path in calls)


@pytest.mark.parametrize("question", VARIATIONS)
def test_wording_variations_skip_the_planner(question: str):
    _calls, handler = _stream_router(SUBJECTS)
    spy = PlannerSpy()
    state = _agent(handler, spy).ask(question)
    assert state.intent == INTENT_SSS_STREAM_SUBJECTS
    assert state.metadata["stream_name"] == "Sciences & Technologies"
    assert state.selected_tools == ["get_sss_stream_subjects"]
    assert "get_curriculum_structure" not in state.selected_tools
    assert spy.planner_calls == 0
    assert state.status == AgentStatus.COMPLETED
    for name in SUBJECT_NAMES:
        assert name in (state.final_answer or "")


def test_select_tool_calls_prefers_the_stream_tool():
    messages = [
        LLMMessage(
            role="user",
            content=(
                "Question: Which subjects are in the Sciences & Technologies stream?\n"
                "Known filters: {}"
            ),
        )
    ]
    calls = select_tool_calls(
        messages,
        [
            {"name": "get_curriculum_structure"},
            {"name": "get_sss_stream_subjects"},
            {"name": "search_curriculum"},
        ],
    )
    assert len(calls) == 1
    assert calls[0].name == "get_sss_stream_subjects"
    assert calls[0].arguments["stream_name"] == "Sciences & Technologies"


def test_unknown_stream_does_not_invent_subjects():
    calls, handler = _stream_router(SUBJECTS)
    spy = PlannerSpy()
    state = _agent(handler, spy).ask(
        "Which subjects are in the Imaginary Sciences stream?"
    )
    assert state.intent == INTENT_SSS_STREAM_SUBJECTS
    assert state.metadata["sss_stream_resolution"] == "not_found"
    assert state.metadata["subject_count"] == 0
    assert state.evidence == []
    assert spy.planner_calls == 0
    assert "get_curriculum_structure" not in state.selected_tools
    assert state.status == AgentStatus.INSUFFICIENT_EVIDENCE
    answer = (state.final_answer or "").lower()
    assert "couldn't find sufficient" in answer
    for name in (*SUBJECT_NAMES, "Imaginary Physics"):
        assert name.lower() not in answer
    assert not any(path.endswith("/subjects") for path in calls)


def test_stream_with_no_subjects_is_distinct_from_not_found():
    _calls, handler = _stream_router([])
    spy = PlannerSpy()
    state = _agent(handler, spy).ask(
        "Which subjects are in the Sciences & Technologies stream?"
    )
    assert state.metadata["sss_stream_resolution"] == "no_subjects"
    assert state.metadata["subject_count"] == 0
    assert state.status == AgentStatus.COMPLETED
    assert spy.planner_calls == 0
    answer = state.final_answer or ""
    assert "no associated subjects" in answer.lower()
    assert "Sciences & Technologies" in answer
    assert "couldn't find sufficient" not in answer.lower()
    for name in SUBJECT_NAMES:
        assert name not in answer


def test_primary3_mathematics_still_uses_curriculum_structure():
    spy = PlannerSpy()
    settings = Settings(
        llm_provider="stub",
        curriculum_api_base_url="http://curriculum.test",
        agent_max_iterations=3,
        agent_max_tool_calls=5,
    )
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(structure_router)
    )
    state = CurriculumQAAgent(
        settings=settings,
        llm=spy,
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
    ).ask("What are the topics in Primary 3 Mathematics?")
    assert state.grade == "CLASS_3"
    assert state.subject == "MATHEMATICS"
    assert "get_curriculum_structure" in state.selected_tools
    assert "get_sss_stream_subjects" not in state.selected_tools
    assert spy.planner_calls >= 1
    assert state.intent != INTENT_SSS_STREAM_SUBJECTS


def test_production_code_does_not_hardcode_the_known_stream_id():
    app_root = Path(__file__).resolve().parents[2] / "app"
    for path in app_root.rglob("*.py"):
        assert KNOWN_PRODUCTION_STREAM_ID not in path.read_text(encoding="utf-8")
