"""V2.12B production-shadow evaluation tests."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.agent.evidence_snapshot import evidence_snapshot_hash
from app.agent.orchestrator import CurriculumQAAgent
from app.agent.state import CurriculumQAState
from app.agent.v212_contract import EquivalenceClassification
from app.agent.v212_langchain import (
    Implementation,
    default_implementation,
    run_post_retrieval_implementation,
    run_post_retrieval_pair,
)
from app.agent.v212b_shadow import (
    REAL_QUESTIONS,
    aggregate_metrics,
    anonymize_request_id,
    build_evidence_snapshot,
    is_c4u18_path,
    maybe_schedule_production_shadow,
    persist_trace,
    replay_evaluation,
    run_production_shadow,
    run_shadow_evaluation,
    should_sample_shadow,
    v212b_shadow_enabled,
)
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence
from app.schemas.verification import VerificationRecommendation, VerificationResult


def _lo() -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id="lo-1",
        name="C4U18-LO01",
        content="Order operations using BODMAS.",
        grade="CLASS_4",
        subject="MATHEMATICS",
        metadata={"code": "C4U18-LO01", "parent_content_code": "C4-U18"},
    )


def _unit() -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="unit",
        entity_id="unit-1",
        name="Everyday Arithmetic Money",
        content="Everyday Arithmetic Money",
        grade="CLASS_4",
        subject="MATHEMATICS",
        metadata={"code": "C4-U18"},
    )


class _FixedVerifier:
    def __init__(self, *, score: float = 1.0, recommendation=VerificationRecommendation.ACCEPT):
        self.score = score
        self.recommendation = recommendation

    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=self.recommendation == VerificationRecommendation.ACCEPT,
            score=self.score,
            recommendation=self.recommendation,
        )


def test_shadow_does_not_alter_production_response():
    settings = Settings(v212b_shadow_enabled=True, v212b_shadow_sample_rate=1.0)
    agent = CurriculumQAAgent(settings=settings)
    state_before = CurriculumQAState.initial(question="test")
    state_before.final_answer = "production answer"
    state_before.evidence = [_lo()]

    with patch("app.agent.v212b_shadow.run_production_shadow") as mock_shadow:
        maybe_schedule_production_shadow(agent, state_before, request_id="req-1")
        time.sleep(0.05)
    assert state_before.final_answer == "production answer"
    mock_shadow.assert_called_once()


def test_langgraph_remains_production_default():
    assert default_implementation(Settings()) == Implementation.LANGGRAPH
    assert default_implementation(Settings(v212_langchain_experiment=False)) == Implementation.LANGGRAPH


def test_langchain_remains_shadow_only_by_default():
    settings = Settings()
    assert not v212b_shadow_enabled(settings)
    assert not settings.v212_langchain_experiment


def test_both_implementations_receive_identical_evidence_snapshots():
    evidence = [_unit(), _lo()]
    verifier = _FixedVerifier()
    pair = run_post_retrieval_pair(
        question="What are the learning objectives for C4-U18?",
        raw_evidence=evidence,
        generated_answer="Pupils learn money skills.",
        verifier=verifier,
        category="c4u18",
        evaluation_id="test_snap",
        evidence_source="c4u18",
    )
    assert pair["langgraph"]["raw_evidence_hash"] == pair["langchain"]["raw_evidence_hash"]
    assert pair["comparison"]["classification"] != EquivalenceClassification.RETRIEVAL_VARIANCE.value


def test_evidence_hash_is_stable():
    evidence = [_unit(), _lo()]
    h1 = evidence_snapshot_hash(evidence)
    h2 = evidence_snapshot_hash(copy.deepcopy(evidence))
    assert h1 == h2


def test_normalization_is_identical_across_implementations():
    evidence = [_unit(), _lo()]
    verifier = _FixedVerifier()
    pair = run_post_retrieval_pair(
        question="fractions",
        raw_evidence=evidence,
        generated_answer="answer",
        verifier=verifier,
    )
    assert (
        pair["langgraph"]["normalized_evidence_hash"]
        == pair["langchain"]["normalized_evidence_hash"]
    )


def test_metadata_validation_is_identical():
    evidence = [_unit(), _lo()]
    verifier = _FixedVerifier()
    pair = run_post_retrieval_pair(
        question="money",
        raw_evidence=evidence,
        generated_answer="answer",
        verifier=verifier,
        evidence_source="c4u18",
    )
    assert (
        pair["langgraph"]["metadata_integrity_valid"]
        == pair["langchain"]["metadata_integrity_valid"]
    )


def test_mapper_behavior_is_identical_with_same_verifier():
    evidence = [_unit(), _lo()]
    verifier = _FixedVerifier()
    pair = run_post_retrieval_pair(
        question="money",
        raw_evidence=evidence,
        generated_answer="grounded answer about money.",
        verifier=verifier,
        evidence_source="c4u18",
    )
    assert pair["langgraph"]["mapped_recommendation"] == pair["langchain"]["mapped_recommendation"]


def test_routing_comparison_works():
    evidence = [_unit(), _lo()]
    verifier = _FixedVerifier()
    pair = run_post_retrieval_pair(
        question="money",
        raw_evidence=evidence,
        generated_answer="answer",
        verifier=verifier,
        evidence_source="c4u18",
    )
    assert pair["langgraph"]["final_route"]
    assert pair["langchain"]["final_route"]


def test_safety_divergence_is_detected():
    evidence = [_unit(), _lo()]

    class _SplitVerifier:
        def __init__(self):
            self.calls = 0

        def verify(self, state, request_id=None):
            self.calls += 1
            if self.calls == 1:
                return VerificationResult(
                    passed=False,
                    score=0.1,
                    recommendation=VerificationRecommendation.FALLBACK,
                    unsupported_claims=["bad"],
                )
            return VerificationResult(
                passed=True,
                score=1.0,
                recommendation=VerificationRecommendation.ACCEPT,
            )

    verifier = _SplitVerifier()
    lg = run_post_retrieval_implementation(
        implementation=Implementation.LANGGRAPH,
        question="q",
        raw_evidence=evidence,
        generated_answer="unsafe",
        verifier=verifier,
    )
    lc = run_post_retrieval_implementation(
        implementation=Implementation.LANGCHAIN,
        question="q",
        raw_evidence=evidence,
        generated_answer="unsafe",
        verifier=verifier,
    )
    assert lg.final_accepted != lc.final_accepted or lg.verifier_score != lc.verifier_score


def test_placeholder_divergence_detection_path():
    evidence = [
        CurriculumEvidence(
            entity_type="learning_outcome",
            entity_id="ph-1",
            name="LO",
            content="",
            metadata={"code": "X"},
        )
    ]
    verifier = _FixedVerifier(score=0.9, recommendation=VerificationRecommendation.RETRIEVE_MORE)
    pair = run_post_retrieval_pair(
        question="placeholder",
        raw_evidence=evidence,
        generated_answer="answer",
        verifier=verifier,
    )
    assert pair["langgraph"]["mapped_recommendation"] == pair["langchain"]["mapped_recommendation"]


def test_metadata_divergence_same_pipeline_stages():
    evidence = [_lo()]
    verifier = _FixedVerifier()
    pair = run_post_retrieval_pair(
        question="meta",
        raw_evidence=evidence,
        generated_answer="a",
        verifier=verifier,
    )
    assert pair["langgraph"]["metadata_violations"] == pair["langchain"]["metadata_violations"]


def test_replay_reproduces_same_evidence(tmp_path: Path):
    evidence = [_unit(), _lo()]
    verifier = _FixedVerifier()
    trace = {
        "evaluation_id": "replay_test_01",
        "question": "money?",
        "category": "c4u18",
        "generated_answer": "answer",
        "c4u18_path": True,
        "evidence_snapshot": build_evidence_snapshot(
            raw_evidence=evidence,
            retrieval_metadata={"grade": "CLASS_4"},
        ),
        "langgraph": {},
        "langchain": {},
        "comparison": {},
    }
    trace_dir = tmp_path
    (trace_dir / "replay_test_01.json").write_text(json.dumps(trace))
    replay = replay_evaluation(
        evaluation_id="replay_test_01",
        trace_dir=trace_dir,
        verifier=verifier,
    )
    assert replay["evidence_hash_match"] is True


def test_shadow_errors_cannot_break_production():
    settings = Settings(v212b_shadow_enabled=True, v212b_shadow_sample_rate=1.0)
    agent = CurriculumQAAgent(settings=settings)
    state = CurriculumQAState.initial(question="q")
    state.evidence = [_lo()]
    state.final_answer = "ok"

    with patch(
        "app.agent.v212b_shadow.run_production_shadow",
        side_effect=RuntimeError("shadow boom"),
    ):
        maybe_schedule_production_shadow(agent, state, request_id="x")
        time.sleep(0.05)
    assert state.final_answer == "ok"


def test_shadow_timeout_cannot_break_production():
    settings = Settings(
        v212b_shadow_enabled=True,
        v212b_shadow_sample_rate=1.0,
        v212b_shadow_timeout_seconds=0.001,
    )
    agent = CurriculumQAAgent(settings=settings)
    state = CurriculumQAState.initial(question="q")
    state.final_answer = "stable"

    def slow_shadow(*_a, **_k):
        time.sleep(0.2)
        return {}

    with patch("app.agent.v212b_shadow.run_production_shadow", side_effect=slow_shadow):
        maybe_schedule_production_shadow(agent, state, request_id="y")
    assert state.final_answer == "stable"


def test_shadow_llm_failure_isolated_in_background_thread():
    settings = Settings(v212b_shadow_enabled=True, v212b_shadow_sample_rate=1.0)
    agent = CurriculumQAAgent(settings=settings)
    state = CurriculumQAState.initial(question="q")
    state.final_answer = "stable"

    with patch(
        "app.agent.v212b_shadow.run_production_shadow",
        side_effect=Exception("LLM down"),
    ):
        maybe_schedule_production_shadow(agent, state, request_id="z")
        time.sleep(0.05)
    assert state.final_answer == "stable"


def test_sampling_rate_is_respected():
    settings = Settings(v212b_shadow_enabled=True, v212b_shadow_sample_rate=0.0)
    assert should_sample_shadow(settings, "seed") is False
    settings_high = Settings(v212b_shadow_enabled=True, v212b_shadow_sample_rate=1.0)
    assert should_sample_shadow(settings_high, "seed") is True


def test_sensitive_request_identifiers_are_not_persisted_unnecessarily(tmp_path: Path):
    trace = {
        "evaluation_id": "anon_test",
        "question": "q",
        "request_trace_id": anonymize_request_id("user-12345-secret"),
        "evidence_snapshot": build_evidence_snapshot(
            raw_evidence=[_lo()],
            retrieval_metadata={},
        ),
        "langgraph": {"final_accepted": False},
        "langchain": {"final_accepted": False},
        "comparison": {},
    }
    assert "user-12345-secret" not in json.dumps(trace)
    path = persist_trace(trace, trace_dir=tmp_path)
    stored = json.loads(path.read_text())
    assert stored["request_trace_id"] != "user-12345-secret"
    assert len(stored["request_trace_id"]) == 16


def test_c4u18_path_detection():
    assert is_c4u18_path("What is C4-U18?", [])
    assert is_c4u18_path("money", [_unit()])


def test_aggregate_metrics_structure():
    row = {
        "question": "q",
        "retrieved_evidence_count": 2,
        "evidence_snapshot": build_evidence_snapshot(
            raw_evidence=[_lo()],
            retrieval_metadata={},
        ),
        "langgraph": {
            "final_accepted": True,
            "verifier_score": 1.0,
            "verifier_recommendation": "accept",
            "mapped_recommendation": "accept",
            "final_route": "finish",
            "metadata_integrity_valid": True,
            "timings": {"total_ms": 10.0},
        },
        "langchain": {
            "final_accepted": True,
            "verifier_score": 1.0,
            "verifier_recommendation": "accept",
            "mapped_recommendation": "accept",
            "final_route": "finish",
            "metadata_integrity_valid": True,
            "timings": {"total_ms": 12.0},
        },
        "comparison": {"classification": "EXACT_EQUIVALENCE"},
    }
    metrics = aggregate_metrics([row])
    assert metrics["n_evaluations"] == 1
    assert "retrieval_statistics" in metrics


def test_real_questions_set_not_empty():
    assert len(REAL_QUESTIONS) >= 20


def test_run_shadow_evaluation_with_mocked_retrieval():
    agent = MagicMock()
    agent.settings = Settings()
    verifier = _FixedVerifier()
    agent.verification_node = MagicMock()
    agent.verification_node.verifier = verifier
    state = CurriculumQAState.initial(question="money?")
    state.evidence = [_unit(), _lo()]
    state.final_answer = "answer"
    state.grade = "CLASS_4"
    state.subject = "MATHEMATICS"
    agent.understand.return_value = state
    agent.retrieve.return_value = state
    agent.answer.return_value = state
    agent.route.return_value = "finish"

    with patch(
        "app.agent.v212b_shadow.collect_retrieval_snapshot",
        return_value=(state, [_unit(), _lo()], "answer", {"grade": "CLASS_4"}),
    ):
        trace = run_shadow_evaluation(
            agent,
            question_spec=REAL_QUESTIONS[0],
            request_id="mock",
        )
    assert trace["langgraph"]["raw_evidence_hash"] == trace["langchain"]["raw_evidence_hash"]
