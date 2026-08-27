"""State-aware retrieval: fingerprints, targeting, no-progress guards."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from app.agent.retrieval_state import (
    RetrievalState,
    build_retrieval_objective,
    has_credible_retrieval_path,
    targeted_tool_calls_from_missing,
    tool_fingerprint,
)
from app.agent.retrieve import RetrievalNode
from app.agent.state import CurriculumQAState
from app.agent.trace import AgentRunTrace, _current_trace
from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.curriculum.evidence import CurriculumEvidence
from app.llm.base import LLMResponse, ToolCallRequest
from app.llm.provider import StubLLMProvider
from app.schemas.verification import MissingEvidenceRequest
from app.tools.base import ToolResult
from app.tools.registry import build_default_registry
from tests.tools.test_curriculum_tools import _router

DIAG = Path(__file__).resolve().parents[2] / "data" / "diagnostics"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_provider="stub",
        curriculum_api_base_url="http://curriculum.test",
        agent_max_iterations=3,
        agent_max_retrieval_rounds=3,
        agent_max_tool_calls=10,
    )


@pytest.fixture
def tools(settings: Settings):
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    return build_default_registry(settings=settings, client=client)


def test_a_exact_duplicate_tool_call_skipped(settings, tools):
    """Test A — same tool + same args → second call skipped."""
    trace = AgentRunTrace(
        agent_run_id="test-dup",
        request_id="r1",
        conversation_id="c1",
        agent_name="curriculum_qa",
        question="q",
    )
    token = _current_trace.set(trace)
    try:
        llm = MagicMock()
        llm.model = "stub"
        llm.set_active_node = MagicMock()
        call = ToolCallRequest(
            id="1",
            name="search_curriculum",
            arguments={"query": "fractions", "grade": "CLASS_4"},
        )
        llm.generate_with_tools.side_effect = [
            LLMResponse(content="", model="stub", tool_calls=[call]),
            LLMResponse(content="", model="stub", tool_calls=[call]),
            LLMResponse(content="done", model="stub", tool_calls=[]),
        ]
        node = RetrievalNode(llm=llm, tools=tools, settings=settings)
        state = CurriculumQAState.initial(question="fractions Primary 4")
        state.grade = "CLASS_4"
        state.retrieval_rounds = 1
        state.iteration = 1
        state = node.run(state)
        assert state.retrieval_state.duplicate_tool_calls_prevented >= 1
        skip_events = [
            e for e in trace.events if e.get("event") == "agent.tool.skip"
        ]
        assert skip_events
        assert skip_events[0]["reason"] == "duplicate_call"
        searches = [r for r in state.retrieval_history if r.tool == "search_curriculum"]
        assert len(searches) == 1
    finally:
        _current_trace.reset(token)


def test_b_different_valid_query_not_exact_duplicate():
    """Test B — search(fractions) vs search(fractions, Mathematics) differ."""
    a = tool_fingerprint(
        "search_curriculum", {"query": "fractions"}
    )
    b = tool_fingerprint(
        "search_curriculum",
        {"query": "fractions", "subject": "Mathematics"},
    )
    assert a != b


def test_c_duplicate_evidence_through_different_tools(settings, tools):
    """Test C — same entity_id from two tools → one evidence record."""
    state = CurriculumQAState.initial(question="q")
    state.grade = "CLASS_4"
    state.retrieval_rounds = 1
    state.iteration = 1
    shared = CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id="C4U06-LO02",
        name="LO",
        content="Identify fractions",
        grade="CLASS_4",
        subject="MATHEMATICS",
        topic="fractions",
        source_tool="search_curriculum",
    )
    state.evidence.append(shared)

    class _FakeTools:
        def llm_tool_specs(self):
            return [
                {"name": "get_topic", "description": "t", "parameters": {}},
                {"name": "search_curriculum", "description": "s", "parameters": {}},
            ]

        def execute(self, name, **kwargs):
            return ToolResult(
                success=True,
                data={
                    "evidence": [
                        {
                            "entity_type": "learning_outcome",
                            "entity_id": "C4U06-LO02",
                            "name": "LO",
                            "content": "Identify fractions",
                            "grade": "CLASS_4",
                            "subject": "MATHEMATICS",
                            "topic": "fractions",
                            "source_tool": name,
                        }
                    ]
                },
            )

    llm = MagicMock()
    llm.model = "stub"
    llm.set_active_node = MagicMock()
    llm.generate_with_tools.side_effect = [
        LLMResponse(
            content="",
            model="stub",
            tool_calls=[
                ToolCallRequest(
                    id="1",
                    name="get_topic",
                    arguments={"topic": "fractions", "grade": "CLASS_4"},
                )
            ],
        ),
        LLMResponse(content="done", model="stub", tool_calls=[]),
    ]
    node = RetrievalNode(llm=llm, tools=_FakeTools(), settings=settings)
    state = node.run(state)
    ids = [e.entity_id for e in state.evidence]
    assert ids.count("C4U06-LO02") == 1
    assert state.retrieval_state.duplicate_evidence_prevented >= 1


def test_d_targeted_verifier_retrieval():
    """Test D — missing C4U09 learning objectives → get_learning_objectives."""
    rs = RetrievalState()
    calls = targeted_tool_calls_from_missing(
        [
            MissingEvidenceRequest(
                type="learning_outcome",
                topic="C4U09",
                query="C4U09 learning objectives",
            )
        ],
        available_tools={
            "search_curriculum",
            "get_curriculum_structure",
            "get_topic",
            "get_learning_objectives",
        },
        grade="CLASS_4",
        subject="MATHEMATICS",
        topic="fractions",
        retrieval_state=rs,
    )
    assert calls
    assert calls[0].name == "get_learning_objectives"
    assert "C4U09" in str(calls[0].arguments)


def test_e_no_progress_when_all_candidates_duplicates(settings, tools):
    """Test E — all candidates duplicates → no new retrieval executed."""
    rs = RetrievalState()
    legacy_args = {
        "topic": "C4U09",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    }
    resolve_args = {
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
        "topic": "C4U09",
    }
    rs.remember_fingerprint(
        tool_fingerprint("get_learning_objectives", legacy_args), 2
    )
    rs.remember_fingerprint(
        tool_fingerprint("resolve_curriculum_context", resolve_args), 3
    )
    missing = [
        MissingEvidenceRequest(
            type="learning_outcome",
            topic="C4U09",
            query="C4U09 learning objectives",
        )
    ]
    assert not has_credible_retrieval_path(
        retrieval_state=rs,
        pending_missing=missing,
        available_tools={
            "resolve_curriculum_context",
            "get_learning_objectives",
            "search_curriculum",
            "get_topic",
            "get_curriculum_structure",
        },
        grade="CLASS_4",
        subject="MATHEMATICS",
        topic="fractions",
    )

    llm = MagicMock()
    llm.model = "stub"
    llm.set_active_node = MagicMock()
    # Follow-up must not call the LLM planner.
    node = RetrievalNode(llm=llm, tools=tools, settings=settings)
    state = CurriculumQAState.initial(question="q")
    state.grade = "CLASS_4"
    state.subject = "MATHEMATICS"
    state.retrieval_rounds = 2
    state.iteration = 2
    state.pending_missing_evidence = missing
    state.retrieval_state = rs
    before = state.tool_calls
    state = node.run(state)
    assert state.tool_calls == before
    assert state.retrieval_state.no_progress is True
    llm.generate_with_tools.assert_not_called()


def test_f_new_evidence_available_runs_targeted(settings, tools):
    """Test F — unused targeted tool executes on retrieve_more."""
    node = RetrievalNode(
        llm=StubLLMProvider(), tools=tools, settings=settings
    )
    state = CurriculumQAState.initial(
        question="What are the learning objectives for fractions in Primary 4?"
    )
    state.grade = "CLASS_4"
    state.subject = "MATHEMATICS"
    state.topic = "fractions"
    state.retrieval_rounds = 2
    state.iteration = 2
    state.pending_missing_evidence = [
        MissingEvidenceRequest(
            type="learning_outcome",
            topic="fractions",
            query="learning objectives for fractions",
        )
    ]
    state = node.run(state)
    assert any(
        r.tool
        in {"resolve_curriculum_context", "get_learning_objectives"}
        for r in state.retrieval_history
    )
    assert state.retrieval_state.targeted_retrievals >= 1


def test_g_existing_successful_path_still_works(settings, tools):
    """Test G — Run-4-style first-pass LLM retrieval still succeeds."""
    from app.agent.context import ConversationStore
    from app.agent.orchestrator import CurriculumQAAgent
    from app.enums import AgentStatus

    agent = CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=tools,
        conversations=ConversationStore(),
    )
    state = agent.ask(
        "What topics are taught in Primary 4 Mathematics?"
    )
    assert state.status == AgentStatus.COMPLETED
    assert state.tool_calls >= 1
    assert state.retrieval_rounds == 1


def test_fingerprint_normalizes_subject_casing():
    a = tool_fingerprint(
        "get_curriculum_structure",
        {"grade": "CLASS_4", "subject": "Mathematics"},
    )
    b = tool_fingerprint(
        "get_curriculum_structure",
        {"grade": "CLASS_4", "subject": "MATHEMATICS"},
    )
    assert a == b


def test_objective_from_missing_evidence():
    obj = build_retrieval_objective(
        pending_missing=[
            MissingEvidenceRequest(
                type="learning_outcome",
                topic="C4U09",
                query="C4U09 learning objectives",
            )
        ],
        grade="CLASS_4",
        subject="MATHEMATICS",
        topic="fractions",
    )
    assert "C4U09" in obj
    assert obj.startswith("Find:")


def test_regression_trace_duplicate_fingerprints_would_skip():
    """Reproduce failed-run retrieval state: duplicate structure/topic skipped."""
    fixtures = {
        "run-ed19de4bcebd4a2f": DIAG / "run_1_trace.json",
        "run-5df89dbf77e74152": DIAG / "run_4_trace.json",
        "run-68f2aa1ce8734ae2": DIAG / "run_5_trace.json",
    }
    for run_id, path in fixtures.items():
        if not path.exists():
            pytest.skip(f"missing fixture {path}")
        data = json.loads(path.read_text())
        rs = RetrievalState()
        would_skip = []
        for event in data.get("events", []):
            if event.get("event") != "agent.tool.start":
                continue
            if event.get("skipped"):
                continue
            name = event.get("tool_name")
            args = event.get("arguments") or {}
            fp = tool_fingerprint(name, args)
            if rs.has_fingerprint(fp):
                would_skip.append((name, args, event.get("iteration")))
            else:
                rs.remember_fingerprint(fp, event.get("tool_call_number") or 0)
                if name == "get_curriculum_structure":
                    rs.note_structure_call(args)
        # Failed runs should have at least one near-end duplicate that our
        # state would now skip; success path may have fewer.
        if run_id == "run-ed19de4bcebd4a2f":
            assert would_skip, f"expected duplicates in {run_id}"
            assert any(t[0] == "get_topic" for t in would_skip) or any(
                t[0] == "get_curriculum_structure" for t in would_skip
            )
        # Targeting: if missing LO for a unit appears, prefer get_learning_objectives.
        missing_events = [
            e
            for e in data.get("events", [])
            if e.get("event") == "agent.verification.end"
            and e.get("recommendation") == "retrieve_more"
        ]
        if missing_events:
            me = missing_events[0].get("missing_evidence") or []
            if me:
                normalized = []
                for m in me:
                    if isinstance(m, dict):
                        normalized.append(MissingEvidenceRequest.model_validate(m))
                    else:
                        normalized.append(m)
                calls = targeted_tool_calls_from_missing(
                    normalized,
                    available_tools={
                        "search_curriculum",
                        "get_curriculum_structure",
                        "get_topic",
                        "get_learning_objectives",
                    },
                    grade="CLASS_4",
                    subject="MATHEMATICS",
                    topic="fractions",
                    retrieval_state=RetrievalState(),
                )
                blob = json.dumps(me).lower()
                if "learning" in blob or "objective" in blob:
                    assert any(
                        c.name == "get_learning_objectives" for c in calls
                    ) or any(c.name == "get_topic" for c in calls)
