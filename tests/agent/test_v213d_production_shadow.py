"""V2.13D Phase 1 production-shadow tests."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.agent.orchestrator import CurriculumQAAgent
from app.agent.state import CurriculumQAState
from app.agent.v213c_experiment import frozen_structured_catalog
from app.agent.v213d_shadow import (
    aggregate_records,
    classify_shadow_comparison,
    classify_shadow_outcome,
    configured_sample_rate,
    format_v213d_startup_banner,
    maybe_schedule_v213d_shadow,
    prepare_replay_corpus,
    record_traffic_event,
    replay_fixtures,
    run_shadow_pipeline,
    should_sample_v213d,
    v213d_runtime_config,
    v213d_shadow_enabled,
)
from app.config import Settings
from app.llm.provider import StubLLMProvider
from app.schemas.verification import VerificationRecommendation, VerificationResult
from app.tools.registry import build_default_registry


def test_default_disabled_without_env_file():
    settings = Settings(_env_file=None)
    assert v213d_shadow_enabled(settings) is False
    assert settings.v213d_shadow_sample_rate == 0.0
    assert settings.v213d_shadow_document_retrieval is True
    assert settings.v213d_shadow_retrieval_variant == "context_hybrid"


def test_sample_rate_zero_disables_execution():
    settings = Settings(
        _env_file=None, v213d_shadow_enabled=True, v213d_shadow_sample_rate=0.0
    )
    assert should_sample_v213d(settings, "seed") is False


def test_one_percent_sampling_configuration():
    settings = Settings(
        _env_file=None, v213d_shadow_enabled=True, v213d_shadow_sample_rate=0.01
    )
    assert configured_sample_rate(settings) == 0.01
    sampled = sum(1 for i in range(10000) if should_sample_v213d(settings, str(i)))
    assert 50 <= sampled <= 150


def test_sampling_cannot_silently_exceed_configured_rate():
    settings = Settings(
        _env_file=None, v213d_shadow_enabled=True, v213d_shadow_sample_rate=0.01
    )
    # Hash threshold is int(rate * 10000); rate cannot exceed configured value.
    assert configured_sample_rate(settings) == 0.01
    for seed in ("a", "b", "c", "request-1"):
        if should_sample_v213d(settings, seed):
            digest_bucket = __import__("hashlib").sha256(
                f"v213d:{seed}".encode()
            ).hexdigest()
            bucket = int(digest_bucket[:8], 16) % 10000
            assert bucket < 100


def test_deterministic_sampling():
    settings = Settings(
        _env_file=None, v213d_shadow_enabled=True, v213d_shadow_sample_rate=0.5
    )
    first = should_sample_v213d(settings, "stable-id")
    second = should_sample_v213d(settings, "stable-id")
    assert first is second
    assert should_sample_v213d(
        Settings(_env_file=None, v213d_shadow_enabled=True, v213d_shadow_sample_rate=1.0),
        "x",
    )
    assert not should_sample_v213d(
        Settings(_env_file=None, v213d_shadow_enabled=False, v213d_shadow_sample_rate=1.0),
        "x",
    )


def test_shadow_does_not_execute_when_unsampled(tmp_path):
    settings = Settings(
        _env_file=None, v213d_shadow_enabled=True, v213d_shadow_sample_rate=0.0
    )
    agent = MagicMock()
    agent.settings = settings
    state = CurriculumQAState.initial(question="What are money LOs?")
    with patch("app.agent.v213d_shadow.run_production_shadow") as mock_run:
        with (
            patch("app.agent.v213d_shadow._TRAFFIC", tmp_path / "traffic.json"),
            patch("app.agent.v213d_shadow._FUNNEL", tmp_path / "funnel.json"),
        ):
            maybe_schedule_v213d_shadow(agent, state, request_id="r1")
        mock_run.assert_not_called()


def test_shadow_executes_when_sampled_asynchronously(tmp_path):
    settings = Settings(
        _env_file=None, v213d_shadow_enabled=True, v213d_shadow_sample_rate=1.0
    )
    agent = MagicMock()
    agent.settings = settings
    state = CurriculumQAState.initial(question="What are money LOs?")

    def slow_run(*_a, **_k):
        time.sleep(0.3)
        return {}

    with patch("app.agent.v213d_shadow.run_production_shadow", side_effect=slow_run):
        with (
            patch("app.agent.v213d_shadow._TRAFFIC", tmp_path / "traffic.json"),
            patch("app.agent.v213d_shadow._FUNNEL", tmp_path / "funnel.json"),
        ):
            started = time.perf_counter()
            thread = maybe_schedule_v213d_shadow(agent, state, request_id="r1")
            elapsed = time.perf_counter() - started
        assert elapsed < 0.15
        assert thread is not None
        thread.join(timeout=2)


def test_production_path_unchanged_and_document_retrieval_shadow_only():
    settings = Settings(_env_file=None)
    registry = build_default_registry(settings=settings)
    assert "search_curriculum_documents" not in registry.names()
    original = CurriculumQAState.initial(question="q")
    original.final_answer = "production"
    original.evidence = copy.deepcopy(frozen_structured_catalog()["c4u18"])
    frozen_evidence = [e.model_dump() for e in original.evidence]
    agent = MagicMock()
    agent.settings = Settings(
        _env_file=None, v213d_shadow_enabled=True, v213d_shadow_sample_rate=1.0
    )
    agent.answer.side_effect = lambda s, request_id=None: s
    agent.verify.side_effect = lambda s, request_id=None: s
    agent.route.return_value = "finish"

    def boom(**_k):
        raise RuntimeError("docs")

    run_shadow_pipeline(agent, original, retrieve_documents=boom)
    assert original.final_answer == "production"
    assert [e.model_dump() for e in original.evidence] == frozen_evidence


def test_structured_preserved_and_document_added(tmp_path):
    settings = Settings(_env_file=None, llm_provider="stub")
    agent = CurriculumQAAgent(settings=settings, llm=StubLLMProvider())
    retrieval = prepare_replay_corpus(tmp_path / "documents", tmp_path / "index")
    state = CurriculumQAState.initial(
        question="What does the MBSSE curriculum say about the purpose of mathematics education?"
    )
    state.subject = "MATHEMATICS"
    state.evidence = copy.deepcopy(frozen_structured_catalog()["c4u18"])
    record = run_shadow_pipeline(agent, state, retrieval=retrieval)
    assert record["shadow"]["structured_evidence_count"] == len(state.evidence)
    assert record["shadow"].get("document_evidence_count", 0) >= 1 or record["shadow"].get(
        "error"
    )
    if not record["shadow"].get("error"):
        assert record["grounding"]["provenance_complete"] is True
        assert isinstance(record["shadow"].get("document_passages"), list)


def test_failure_isolation_does_not_mutate_production():
    settings = Settings(
        _env_file=None, v213d_shadow_enabled=True, v213d_shadow_sample_rate=1.0
    )
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
    expected = {
        "DOCUMENT_RETRIEVAL_FAILURE",
        "GENERATION_FAILURE",
        "VERIFIER_FAILURE",
        "DOCUMENT_NOISE",
    }
    for agent, retriever in cases:
        record = run_shadow_pipeline(agent, original, retrieve_documents=retriever)
        assert original.final_answer == "ok"
        assert record["comparison"]["classification"] in expected
        assert record["shadow"].get("shadow_error_type")


def test_shadow_errors_cannot_break_production_scheduler(tmp_path):
    settings = Settings(
        _env_file=None, v213d_shadow_enabled=True, v213d_shadow_sample_rate=1.0
    )
    agent = MagicMock()
    agent.settings = settings
    state = CurriculumQAState.initial(question="q")
    with patch(
        "app.agent.v213d_shadow.run_production_shadow",
        side_effect=RuntimeError("boom"),
    ):
        with (
            patch("app.agent.v213d_shadow._TRAFFIC", tmp_path / "traffic.json"),
            patch("app.agent.v213d_shadow._FUNNEL", tmp_path / "funnel.json"),
        ):
            thread = maybe_schedule_v213d_shadow(agent, state, request_id="x")
        if thread:
            thread.join(timeout=2)


def test_wrong_context_classification_and_gate():
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
    assert classify_shadow_comparison(control, shadow) == "WRONG_CONTEXT"


def test_comparison_labels_phase1():
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
        == "DOCUMENT_ADDED_MISSING_CONTEXT"
    )
    assert classify_shadow_outcome(
        {"final_accepted": False, "final_route": "retrieve_more"},
        {
            "final_accepted": True,
            "metadata_valid": True,
            "document_evidence_count": 3,
            "wrong_context": False,
        },
    )["newly_recoverable"]
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
        == "DOCUMENT_NOISE"
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
        == "STRUCTURED_DATA_ALREADY_SUFFICIENT"
    )
    assert (
        classify_shadow_comparison({}, {"error": "TimeoutError", "shadow_stage": "document_retrieval"})
        == "DOCUMENT_RETRIEVAL_FAILURE"
    )
    assert (
        classify_shadow_comparison(
            {"final_accepted": False, "final_route": "retrieve_more"},
            {
                "final_accepted": False,
                "metadata_valid": True,
                "document_evidence_count": 2,
                "wrong_context": False,
                "final_route": "retrieve_more",
            },
        )
        == "DOCUMENT_DID_NOT_HELP"
    )


def test_aggregation_improvement_and_regression_metrics():
    records = [
        {
            "shadow": {"document_evidence_count": 2, "final_accepted": True},
            "grounding": {
                "metadata_valid": True,
                "provenance_complete": True,
                "wrong_context": False,
                "placeholder_evidence": False,
                "unsupported_claims": [],
            },
            "comparison": {
                "classification": "DOCUMENT_ADDED_MISSING_CONTEXT",
                "improved": True,
                "regressed": False,
                "newly_recoverable": True,
                "control_correct_shadow_worse": False,
            },
            "latency_ms": 40,
        },
        {
            "shadow": {
                "document_evidence_count": 1,
                "final_accepted": False,
                "document_retrieval_latency_ms": 12,
            },
            "grounding": {
                "metadata_valid": True,
                "provenance_complete": True,
                "wrong_context": False,
                "placeholder_evidence": False,
                "unsupported_claims": [],
            },
            "comparison": {
                "classification": "DOCUMENT_NOISE",
                "improved": False,
                "regressed": True,
                "newly_recoverable": False,
                "control_correct_shadow_worse": True,
            },
            "latency_ms": 50,
        },
    ]
    metrics = aggregate_records(records, traffic={"total_production_requests": 200, "sampled_requests": 2})
    assert metrics["newly_recoverable_count"] == 1
    assert metrics["control_correct_shadow_worse"] == 1
    assert metrics["regressions"] == 1
    assert metrics["phase1_status"] == "INSUFFICIENT_SAMPLE"
    assert metrics["phase1_recommendation"] == "CONTINUE SHADOW"


def test_safety_blocked_status():
    records = [
        {
            "shadow": {"document_evidence_count": 1, "final_accepted": True},
            "grounding": {
                "metadata_valid": True,
                "provenance_complete": True,
                "wrong_context": True,
                "placeholder_evidence": False,
                "unsupported_claims": [],
            },
            "comparison": {
                "classification": "WRONG_CONTEXT",
                "improved": False,
                "regressed": False,
                "newly_recoverable": False,
                "control_correct_shadow_worse": True,
            },
            "latency_ms": 10,
        }
    ]
    metrics = aggregate_records(records)
    assert metrics["safety_metrics"]["wrong_context_false_acceptance"] == 1
    assert metrics["phase1_status"] == "SAFETY_BLOCKED"
    assert metrics["phase1_recommendation"] == "STOP — SAFETY ISSUE"


def test_runtime_config_banner():
    settings = Settings(
        _env_file=None,
        v213d_shadow_enabled=True,
        v213d_shadow_sample_rate=0.01,
        v213d_shadow_document_retrieval=True,
        v213d_shadow_retrieval_variant="context_hybrid",
        v213d_shadow_timeout_seconds=30,
    )
    cfg = v213d_runtime_config(settings)
    assert cfg["sample_rate"] == 0.01
    banner = format_v213d_startup_banner(settings)
    assert "V2.13D shadow enabled: true" in banner
    assert "V2.13D sample rate: 0.01" in banner


def test_traffic_counter(tmp_path: Path):
    path = tmp_path / "traffic.json"
    first = record_traffic_event(sampled=False, path=path)
    second = record_traffic_event(sampled=True, path=path)
    assert first["total_production_requests"] == 1
    assert second["total_production_requests"] == 2
    assert second["sampled_requests"] == 1


def test_pipeline_funnel_stages(tmp_path: Path):
    settings = Settings(
        _env_file=None, v213d_shadow_enabled=True, v213d_shadow_sample_rate=1.0
    )
    agent = MagicMock()
    agent.settings = settings
    state = CurriculumQAState.initial(question="q")
    state.final_answer = "ok"

    def fake_run(agent, snapshot, *, request_id=None, retrieval=None, jsonl_path=None):
        from app.agent.v213d_shadow import persist_record, record_pipeline_stage

        record_pipeline_stage("shadow_started", request_id=request_id)
        record_pipeline_stage("shadow_completed", request_id=request_id)
        record = {
            "experiment": "v2.13d",
            "request_id": "abc",
            "shadow": {"error": None},
            "comparison": {"classification": "STRUCTURED_DATA_ALREADY_SUFFICIENT"},
        }
        persist_record(record, jsonl_path or (tmp_path / "out.jsonl"))
        return record

    with (
        patch("app.agent.v213d_shadow._FUNNEL", tmp_path / "funnel.json"),
        patch("app.agent.v213d_shadow._TRAFFIC", tmp_path / "traffic.json"),
        patch("app.agent.v213d_shadow.run_production_shadow", side_effect=fake_run),
    ):
        thread = maybe_schedule_v213d_shadow(
            agent, state, request_id="funnel-1", jsonl_path=tmp_path / "out.jsonl"
        )
        assert thread is not None
        thread.join(timeout=2)
    funnel = json.loads((tmp_path / "funnel.json").read_text())
    stages = funnel["stages"]
    assert stages["request_seen"] >= 1
    assert stages["shadow_eligible"] >= 1
    assert stages["shadow_sampled"] >= 1
    assert stages["shadow_started"] >= 1
    assert stages["shadow_completed"] >= 1
    assert stages["shadow_persisted"] >= 1
    assert state.final_answer == "ok"


def test_persist_error_is_counted(tmp_path: Path):
    from app.agent.v213d_shadow import persist_record, record_pipeline_stage

    funnel = tmp_path / "funnel.json"
    with patch("app.agent.v213d_shadow._FUNNEL", funnel):
        bad = tmp_path / "missing_dir_as_file"
        bad.write_text("not-a-dir")
        try:
            persist_record({"request_id": "x", "shadow": {}}, bad / "out.jsonl")
            assert False, "expected persist failure"
        except Exception:
            pass
        stages = json.loads(funnel.read_text())["stages"]
        assert stages["persist_error"] >= 1


def test_jsonl_runtime_path_absolute():
    from app.agent.v213d_shadow import jsonl_runtime_path

    path = jsonl_runtime_path()
    assert path.is_absolute()
    assert path.name == "v213d_shadow.jsonl"

def test_replay_includes_required_categories(tmp_path):
    settings = Settings(_env_file=None, llm_provider="stub")
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
    assert timeout_row["comparison"]["classification"] == "DOCUMENT_RETRIEVAL_FAILURE"
    assert timeout_row["shadow"]["shadow_error_type"] == "TimeoutError"


def test_placeholder_not_accepted(tmp_path):
    settings = Settings(_env_file=None, llm_provider="stub")
    agent = CurriculumQAAgent(settings=settings, llm=StubLLMProvider())
    retrieval = prepare_replay_corpus(tmp_path / "documents", tmp_path / "index")
    records = replay_fixtures(agent, retrieval=retrieval, question_ids=("V213C-G03",))
    shadow = records[0]["shadow"]
    assert shadow.get("final_accepted") is False or shadow.get("error")


def test_metadata_guard_blocks_acceptance(tmp_path):
    settings = Settings(_env_file=None, llm_provider="stub")
    agent = CurriculumQAAgent(settings=settings, llm=StubLLMProvider())
    retrieval = prepare_replay_corpus(tmp_path / "documents", tmp_path / "index")
    records = replay_fixtures(agent, retrieval=retrieval, question_ids=("V213C-G03",))
    shadow = records[0]["shadow"]
    if not shadow.get("error"):
        assert shadow.get("final_accepted") is False
