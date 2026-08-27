"""Sprint 5 — LangGraph node, routing, and integration tests."""

from __future__ import annotations

import httpx
import pytest

from app.agent.context import ConversationStore
from app.agent.graph import build_curriculum_qa_graph, graph_ascii, graph_mermaid
from app.agent.graph_nodes import GraphNodes
from app.agent.graph_routing import (
    route_after_prepare,
    route_after_verification,
    validate_route,
)
from app.agent.graph_state import initial_graph_input
from app.agent.orchestrator import CurriculumQAAgent
from app.agent.state import CurriculumQAState
from app.agent.verify import VerificationNode
from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.enums import AgentStatus
from app.llm.provider import StubLLMProvider
from app.schemas.verification import (
    VerificationRecommendation,
    VerificationResult,
    VerificationStatus,
)
from app.tools.registry import build_default_registry
from tests.tools.test_curriculum_tools import _router


@pytest.fixture
def settings() -> Settings:
    return Settings(
        curriculum_api_base_url="http://curriculum.test",
        agent_checkpointing_enabled=True,
        agent_checkpoint_backend="memory",
        agent_max_iterations=3,
        agent_max_retrieval_rounds=3,
        agent_max_tool_calls=10,
    )


@pytest.fixture
def agent(settings: Settings) -> CurriculumQAAgent:
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    return CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
    )


def test_validate_route_rejects_arbitrary_destinations():
    with pytest.raises(ValueError):
        validate_route("hack_the_graph")


def test_route_after_verification_accept(settings):
    qa = CurriculumQAState.initial(question="q")
    qa.verification = VerificationResult(
        passed=True,
        score=1.0,
        recommendation=VerificationRecommendation.ACCEPT,
    )
    assert route_after_verification({"qa": qa}, settings=settings) == "finish"


def test_route_after_verification_retrieve_more(settings):
    qa = CurriculumQAState.initial(question="q")
    qa.verification = VerificationResult(
        passed=False,
        score=0.2,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
    )
    assert (
        route_after_verification({"qa": qa}, settings=settings) == "retrieve_more"
    )


def test_route_after_verification_retrieve_more_hits_limit(settings):
    qa = CurriculumQAState.initial(question="q")
    qa.retrieval_rounds = settings.agent_max_retrieval_rounds
    qa.verification = VerificationResult(
        passed=False,
        score=0.2,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
    )
    assert route_after_verification({"qa": qa}, settings=settings) == "fallback"


def test_route_after_prepare_limits(settings):
    qa = CurriculumQAState.initial(question="q")
    assert route_after_prepare({"qa": qa, "max_iterations_hit": False}) == "retrieve"
    assert route_after_prepare({"qa": qa, "max_iterations_hit": True}) == "fallback"


def test_understand_node_inherits_prior_filters(agent):
    qa = CurriculumQAState.initial(question="Tell me more about fractions.")
    out = agent.nodes.understand(
        {
            "qa": qa,
            "prior_filters": {
                "grade": "CLASS_4",
                "subject": "MATHEMATICS",
                "level": None,
                "topic": None,
            },
            "visited_nodes": [],
        }
    )
    assert out["qa"].grade == "CLASS_4"
    assert out["qa"].subject == "MATHEMATICS"
    assert out["visited_nodes"] == ["understand"]


def test_clarify_and_fallback_nodes(agent):
    qa = CurriculumQAState.initial(question="What is taught?")
    qa.verification = VerificationResult(
        passed=False,
        score=0.0,
        recommendation=VerificationRecommendation.CLARIFY,
        clarification="Which grade?",
    )
    clarified = agent.nodes.clarify({"qa": qa, "visited_nodes": ["verify_answer"]})
    assert clarified["qa"].status == AgentStatus.NEEDS_CLARIFICATION
    assert clarified["route"] == "clarify"

    fb = agent.nodes.fallback(
        {
            "qa": CurriculumQAState.initial(question="x"),
            "visited_nodes": ["verify_answer"],
            "fallback_reason": "verification_fallback",
        }
    )
    assert fb["qa"].status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert fb["route"] == "fallback"


def test_graph_successful_path(agent):
    state = agent.ask("What topics are taught in Primary 4 Mathematics?")
    assert state.status == AgentStatus.COMPLETED
    path = state.metadata.get("visited_nodes") or []
    assert path[:4] == [
        "understand",
        "prepare_cycle",
        "retrieve",
        "generate_answer",
    ]
    assert "verify_answer" in path
    assert path[-1] == "finish"
    assert state.final_answer
    assert state.metadata.get("graph_run_id")


def test_graph_follow_up_reuses_thread(agent):
    first = agent.ask("What topics are in Primary 4 Mathematics?")
    second = agent.ask(
        "Tell me more about fractions.",
        conversation_id=first.conversation_id,
    )
    assert second.conversation_id == first.conversation_id
    assert second.grade == "CLASS_4"
    assert second.metadata.get("graph_run_id") != first.metadata.get("graph_run_id")


def test_graph_inspection(agent):
    ascii_view = graph_ascii(agent.graph)
    assert "understand" in ascii_view
    assert "verify_answer" in ascii_view
    mermaid = graph_mermaid(agent.graph)
    assert "understand" in mermaid or mermaid.startswith("#")


class _AlwaysRetrieveMoreVerifier(VerificationNode):
    def run(self, state, *, request_id=None):
        state.verification_attempts += 1
        state.verification = VerificationResult(
            passed=False,
            score=0.1,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            issues=["need more evidence"],
            missing_evidence=["topic placement"],
        )
        state.verification_history.append(state.verification)
        state.verification_status = VerificationStatus.FAILED
        state.pending_missing_evidence = list(state.verification.missing_evidence)
        return state


def test_graph_max_iterations_goes_to_fallback(settings):
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    agent = CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
        verification_node=_AlwaysRetrieveMoreVerifier(
            llm=StubLLMProvider(), settings=settings
        ),
    )
    state = agent.ask("Is topic X taught in Primary 4 Mathematics?")
    assert state.status == AgentStatus.INSUFFICIENT_EVIDENCE
    path = state.metadata.get("visited_nodes") or []
    assert "fallback" in path
    assert path.count("retrieve") >= 1
    assert state.verification_status == VerificationStatus.MAX_ITERATIONS


class _ClarifyVerifier(VerificationNode):
    def run(self, state, *, request_id=None):
        state.verification_attempts += 1
        state.verification = VerificationResult(
            passed=False,
            score=0.0,
            recommendation=VerificationRecommendation.CLARIFY,
            clarification="Which grade should I check?",
        )
        state.verification_history.append(state.verification)
        state.verification_status = VerificationStatus.NEEDS_CLARIFICATION
        return state


def test_graph_clarification_path(settings):
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    agent = CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
        verification_node=_ClarifyVerifier(llm=StubLLMProvider(), settings=settings),
    )
    state = agent.ask("What topics are taught?")
    assert state.status == AgentStatus.NEEDS_CLARIFICATION
    assert state.clarification
    assert (state.metadata.get("visited_nodes") or [])[-1] == "clarify"


class _RetrieveThenPassVerifier(VerificationNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._calls = 0

    def run(self, state, *, request_id=None):
        self._calls += 1
        state.verification_attempts += 1
        if self._calls == 1:
            state.verification = VerificationResult(
                passed=False,
                score=0.2,
                recommendation=VerificationRecommendation.RETRIEVE_MORE,
                issues=["incomplete"],
                missing_evidence=["learning outcomes"],
            )
            state.verification_status = VerificationStatus.FAILED
            state.pending_missing_evidence = ["learning outcomes"]
        else:
            state.verification = VerificationResult(
                passed=True,
                score=0.95,
                recommendation=VerificationRecommendation.ACCEPT,
            )
            state.verification_status = VerificationStatus.PASSED
        state.verification_history.append(state.verification)
        return state


def test_graph_retrieve_more_loop_then_finish(settings):
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    agent = CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
        verification_node=_RetrieveThenPassVerifier(
            llm=StubLLMProvider(), settings=settings
        ),
    )
    state = agent.ask("What should Primary 4 pupils learn about fractions?")
    assert state.status == AgentStatus.COMPLETED
    path = state.metadata.get("visited_nodes") or []
    # Expected: understand → prepare → retrieve → generate → verify →
    #           prepare → retrieve → generate → verify → finish
    assert path.count("retrieve") == 2
    assert path.count("generate_answer") == 2
    assert path.count("verify_answer") == 2
    assert path[-1] == "finish"
    # Demonstrate the sprint-required loop shape in the recorded path.
    joined = " → ".join(path)
    assert "verify_answer → prepare_cycle → retrieve" in joined


def test_build_graph_without_checkpoint(settings):
    from app.agent.answer import AnswerGenerationNode
    from app.agent.retrieve import RetrievalNode

    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    tools = build_default_registry(settings=settings, client=client)
    nodes = GraphNodes(
        settings=settings,
        retrieval=RetrievalNode(
            llm=StubLLMProvider(), tools=tools, settings=settings
        ),
        answer_node=AnswerGenerationNode(llm=StubLLMProvider(), settings=settings),
        verification_node=VerificationNode(
            llm=StubLLMProvider(), settings=settings
        ),
    )
    graph = build_curriculum_qa_graph(
        nodes=nodes, settings=settings, checkpointer=None
    )
    qa = CurriculumQAState.initial(
        question="What topics are taught in Primary 4 Mathematics?"
    )
    result = graph.invoke(initial_graph_input(qa=qa))
    assert result["qa"].status in (
        AgentStatus.COMPLETED,
        AgentStatus.INSUFFICIENT_EVIDENCE,
        AgentStatus.NEEDS_CLARIFICATION,
    )


def test_agent_with_checkpointing_disabled(settings):
    settings = settings.model_copy(update={"agent_checkpointing_enabled": False})
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    agent = CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
        checkpointer=None,
    )
    assert agent.checkpointer is None
    state = agent.ask("What topics are taught in Primary 4 Mathematics?")
    assert state.status == AgentStatus.COMPLETED
    assert "finish" in (state.metadata.get("visited_nodes") or [])


def test_sqlite_checkpointer_persists_prior_filters(settings, tmp_path):
    from app.agent.memory import build_checkpointer

    db_path = tmp_path / "checkpoints.sqlite"
    settings = settings.model_copy(
        update={
            "agent_checkpoint_backend": "sqlite",
            "agent_checkpoint_sqlite_path": str(db_path),
            "agent_checkpointing_enabled": True,
        }
    )
    checkpointer = build_checkpointer(settings)
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )
    store = ConversationStore()
    agent = CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        conversations=store,
        checkpointer=checkpointer,
    )
    first = agent.ask("What topics are in Primary 4 Mathematics?")
    assert first.grade == "CLASS_4"
    assert db_path.exists()

    # New agent process simulation: same sqlite file, fresh ConversationStore.
    agent2 = CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
        checkpointer=build_checkpointer(settings),
    )
    second = agent2.ask(
        "Tell me more about fractions.",
        conversation_id=first.conversation_id,
    )
    assert second.conversation_id == first.conversation_id
    assert second.grade == "CLASS_4"
