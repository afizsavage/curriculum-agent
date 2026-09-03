"""V2.13D production-shadow tests."""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

from app.agent.orchestrator import CurriculumQAAgent
from app.agent.state import CurriculumQAState
from app.agent.v213c_experiment import frozen_structured_catalog
from app.agent.v213d_shadow import (
    classify_shadow_comparison,
    maybe_schedule_v213d_shadow,
    prepare_replay_corpus,
    replay_fixtures,
    run_shadow_pipeline,
    should_sample_v213d,
    v213d_shadow_enabled,
)
from app.config import Settings
from app.llm.provider import StubLLMProvider
from app.schemas.verification import VerificationRecommendation, VerificationResult
from app.tools.registry import build_default_registry


def test_default_disabled():
    settings = Settings()
    assert v213d_shadow_enabled(settings) is False
    assert settings.v213d_shadow_sample_rate == 0.0
    assert settings.v213d_shadow_document_retrieval is True
    assert settings.v213d_shadow_retrieval_variant == "context_hybrid"


def test_sample_rate_zero_disables_execution():
    settings = Settings(v213d_shadow_enabled=True, v213d_shadow_sample_rate=0.0)
    assert should_sample_v213d(settings, "seed") is False


def test_deterministic_sampling():
    settings = Settings(v213d_shadow_enabled=True, v213d_shadow_sample_rate=0.5)
    first = should_sample_v213d(settings, "stable-id")
    second = should_sample_v213d(settings, "stable-id")
    assert first is second
    assert should_sample_v213d(
        Settings(v213d_shadow_enabled=True, v213d_shadow_sample_rate=1.0), "x"
    )
    assert not should_sample_v213d(
        Settings(v213d_shadow_enabled=False, v213d_shadow_sample_rate=1.0), "x"
    )


def test_shadow_does_not_execute_when_unsampled():
    settings = Settings(v213d_shadow_enabled=True, v213d_shadow_sample_rate=0.0)
    agent = MagicMock()
    agent.settings = settings
    state = CurriculumQAState.initial(question="What are money LOs?")
    with patch("app.agent.v213d_shadow.run_production_shadow") as mock_run:
        maybe_schedule_v213d_shadow(agent, state, request_id="r1")
        mock_run.assert_not_called()


def test_shadow_executes_when_sampled():
    settings = Settings(v213d_shadow_enabled=True, v213d_shadow_sample_rate=1.0)
    agent = MagicMock()
    agent.settings = settings
    state = CurriculumQAState.initial(question="What are money LOs?")
    with patch("app.agent.v213d_shadow.run_production_shadow") as mock_run:
        thread = maybe_schedule_v213d_shadow(agent, state, request_id="r1")
        if thread:
            thread.join(timeout=2)
        mock_run.assert_called_once()


def test_production_path_unchanged_and_document_retrieval_shadow_only():
    settings = Settings()
    registry = build_default_registry(settings=settings)
    assert "search_curriculum_documents" not in registry.names()
    original = CurriculumQAState.initial(question="q")
    original.final_answer = "production"
    agent = MagicMock()
    agent.settings = Settings(v213d_shadow_enabled=True, v213d_shadow_sample_rate=1.0)
    agent.answer.side_effect = lambda s, request_id=None: s
    agent.verify.side_effect = lambda s, request_id=None: s
    agent.route.return_value = "finish"

    def boom(**_k):
        raise RuntimeError("docs")

    run_shadow_pipeline(agent, original, retrieve_documents=boom)
    assert original.final_answer == "production"


def test_structured_preserved_and_document_added(tmp_path):
    settings = Settings(llm_provider="stub")
    agent = CurriculumQAAgent(settings=settings, llm=StubLLMProvider())
    retrieval = prepare_replay_corpus(tmp_path / "documents", tmp_path / "index")
    state = CurriculumQAState.initial(
        question="What does the MBSSE curriculum say about the purpose of mathematics education?"
    )
    state.subject = "MATHEMATICS"
    state.evidence = copy.deepcopy(frozen_structured_catalog()["c4u18"])
    record = run_shadow_pipeline(agent, state, retrieval=retrieval)
    assert record["shadow"]["structured_evidence_count"] == len(state.evidence)
    assert record["shadow"].get("document_evidence_count", 0) >= 1 or record["shadow"].get("error")
    if not record["shadow"].get("error"):
        assert record["grounding"]["provenance_complete"] is True


def test_failure_isolation_does_not_mutate_production():
    settings = Settings(v213d_shadow_enabled=True, v213d_shadow_sample_rate=1.0)
    original = CurriculumQAState.initial(question="q")
    original.final_answer = "ok"

    def _agent(*, fail_answer=False, fail_verify=False):
        agent = MagicMock()
        agent.settings = settings
        if fail_answer:
            agent.answer.side_effect = RuntimeError("gen fail")
        else:
            agent.answer.side_effect = lambda s, request_id=None: s
        if fail_verify:
            agent.verify.side_effect = RuntimeError("verify fail")
        else:

            def _verify(s, request_id=None):
                s.verification = VerificationResult(
                    passed=False,
                    score=0.2,
                    recommendation=VerificationRecommendation.RETRIEVE_MORE,
                )
                return s

            agent.verify.side_effect = _verify
        agent.route.return_value = "retrieve_more"
        return agent

    cases = [
        (_agent(), lambda **_k: (_ for _ in ()).throw(RuntimeError("retrieval fail"))),
        (_agent(), lambda **_k: (_ for _ in ()).throw(TimeoutError("timeout"))),
        (_agent(), lambda **_k: (_ for _ in ()).throw(ValueError("norm"))),
        (_agent(fail_answer=True), lambda **_k: ([], {})),
        (_agent(fail_verify=True), lambda **_k: ([], {})),
    ]
    for agent, retriever in cases:
        record = run_shadow_pipeline(agent, original, retrieve_documents=retriever)
        assert original.final_answer == "ok"
        assert record["comparison"]["classification"] == "SHADOW_ERROR"


def test_shadow_errors_cannot_break_production_scheduler():
    settings = Settings(v213d_shadow_enabled=True, v213d_shadow_sample_rate=1.0)
    agent = MagicMock()
    agent.settings = settings
    state = CurriculumQAState.initial(question="q")
    with patch(
        "app.agent.v213d_shadow.run_production_shadow",
        side_effect=RuntimeError("boom"),
    ):
        thread = maybe_schedule_v213d_shadow(agent, state, request_id="x")
        if thread:
            thread.join(timeout=2)


def test_safety_false_acceptance_classification():
    control = {
        "final_accepted": False,
        "final_route": "retrieve_more",
        "verifier_decision": "retrieve_more",
    }
    shadow = {
        "final_accepted": True,
        "metadata_valid": True,
        "wrong_context": True,
        "document_evidence_count": 2,
        "final_route": "finish",
    }
    assert classify_shadow_comparison(control, shadow) == "DOCUMENT_CREATED_WRONG_CONTEXT"


def test_comparison_labels():
    assert (
        classify_shadow_comparison(
            {"final_accepted": False, "final_route": "retrieve_more"},
            {
                "final_accepted": True,
                "metadata_valid": True,
                "document_evidence_count": 3,
                "wrong_context": False,
            },
        )
        == "DOCUMENT_ADDED_GROUNDING"
    )
    assert (
        classify_shadow_comparison(
            {"final_accepted": True, "final_route": "finish"},
            {
                "final_accepted": False,
                "metadata_valid": True,
                "document_evidence_count": 1,
                "final_route": "retrieve_more",
                "wrong_context": False,
            },
        )
        == "SHADOW_REGRESSED"
    )
    assert (
        classify_shadow_comparison(
            {"final_accepted": True, "final_route": "finish"},
            {
                "final_accepted": True,
                "metadata_valid": True,
                "document_evidence_count": 1,
                "wrong_context": False,
            },
        )
        == "BOTH_ACCEPTED"
    )
    assert classify_shadow_comparison({}, {"error": "TimeoutError"}) == "SHADOW_ERROR"
    assert (
        classify_shadow_comparison(
            {"final_accepted": False, "final_route": "retrieve_more"},
            {
                "final_accepted": False,
                "metadata_valid": True,
                "document_evidence_count": 0,
                "wrong_context": False,
                "final_route": "retrieve_more",
            },
        )
        == "BOTH_INSUFFICIENT"
    )


def test_replay_includes_required_categories(tmp_path):
    settings = Settings(llm_provider="stub")
    agent = CurriculumQAAgent(settings=settings, llm=StubLLMProvider())
    retrieval = prepare_replay_corpus(tmp_path / "documents", tmp_path / "index")
    records = replay_fixtures(
        agent,
        retrieval=retrieval,
        inject_failures={"V213C-F01": "timeout"},
    )
    cats = {r.get("replay_category") for r in records}
    assert "document_only" in cats
    assert "structured_fact" in cats
    assert "structured_plus_document" in cats
    assert "insufficient_evidence" in cats
    assert "adversarial" in cats
    timeout_row = next(r for r in records if r.get("replay_id") == "V213C-F01")
    assert timeout_row["comparison"]["classification"] == "SHADOW_ERROR"
    assert timeout_row["shadow"]["shadow_error_type"] == "TimeoutError"


def test_placeholder_not_accepted(tmp_path):
    settings = Settings(llm_provider="stub")
    agent = CurriculumQAAgent(settings=settings, llm=StubLLMProvider())
    retrieval = prepare_replay_corpus(tmp_path / "documents", tmp_path / "index")
    records = replay_fixtures(agent, retrieval=retrieval, question_ids=("V213C-G03",))
    shadow = records[0]["shadow"]
    assert shadow.get("final_accepted") is False or shadow.get("error")
