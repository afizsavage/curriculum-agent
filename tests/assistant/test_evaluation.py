"""Deterministic assistant behaviour evaluation. No live provider calls."""

from __future__ import annotations

import json

import pytest

from app.assistant.evaluation import (
    LiveEvaluationRefused,
    assert_live_allowed,
    dataset_path,
    format_summary,
    load_dataset,
    run_deterministic,
)
from app.assistant.identity import current_identity_gate
from app.config import Settings


@pytest.fixture(scope="module")
def report():
    return run_deterministic()


def test_dataset_covers_the_behaviour_groups():
    payload = load_dataset()
    groups = {case["group"] for case in payload["cases"]}
    assert groups == {
        "curriculum",
        "general",
        "mixed",
        "ambiguous",
        "failure",
        "isolation",
        "follow_up",
    }
    assert all(case["gold_curriculum_facts"] == [] for case in payload["cases"])
    assert payload["version"] == "assistant-behaviour-v1"
    assert dataset_path().is_file()


def test_dataset_rejects_encoded_curriculum_facts(tmp_path):
    payload = load_dataset()
    payload["cases"][0]["gold_curriculum_facts"] = ["Primary 4 teaches fractions."]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


def test_required_routing_matches_by_route(report):
    routing = report["metrics"]["routing_accuracy"]
    assert routing["denominator"] > 0
    assert routing["correct"] == routing["denominator"]
    assert sum(bucket["denominator"] for bucket in routing["by_route"].values()) == (
        routing["denominator"]
    )
    for case in report["cases"]:
        if case["acceptance"] == "required" and case["expected_route"] is not None:
            assert case["route_match"] is True
            assert case["safety_pass"] is True


def test_metrics_keep_separate_denominators(report):
    metrics = report["metrics"]
    intent = metrics["intent_agreement"]
    assert intent["correct"] == intent["denominator"]
    evidence = metrics["insufficient_or_incomplete_evidence"]
    assert evidence["passed"] == evidence["denominator"] == 3
    partial = metrics["partial_evidence"]
    assert partial["blocked"] == partial["denominator"] == 1
    authority = metrics["authority_separation"]
    assert authority["passed"] == authority["denominator"] > 0
    gold = metrics["complete_list_gold_facts"]
    assert gold["encoded"] == 0
    assert gold["denominator"] >= 6
    factual = metrics["curriculum_factual_accuracy"]
    assert factual["measured"] is False
    assert factual["correct"] is None
    assert factual["denominator"] is None
    assert metrics["conversation_isolation"] == {"passed": 1, "denominator": 1}
    follow = metrics["follow_up"]
    assert follow["prior_context_used"] == follow["denominator"] >= 3
    assert metrics["failure_handling"]["passed"] == metrics["failure_handling"]["denominator"]
    latency = metrics["latency_ms"]
    assert latency["scope"] == "harness"
    assert latency["n"] == len(report["cases"]) >= 20
    assert latency["median"] is not None
    assert latency["p95"] is not None
    assert metrics["provider_cost"]["estimated_usd"] is None


def test_report_omits_answer_text_and_secrets(report):
    encoded = json.dumps(report)
    assert "Fixture curriculum sentence" not in encoded
    assert "Fixture general teaching note" not in encoded
    assert "Bearer " not in encoded
    assert "api_key" not in encoded
    assert report["include_answers"] is False
    assert report["production_authentication"] is False
    assert report["assistant_enabled_default"] is False
    summary = format_summary(report)
    assert "Fixture curriculum sentence" not in summary
    assert "Routing accuracy (required cases):" in summary
    assert "Unmeasured" in summary
    by_id = {case["id"]: case for case in report["cases"]}
    assert by_id["obs-teach-fractions-no-marker"]["actual_route"] == "mixed"
    assert by_id["obs-partial-allows-advice"]["general_called"] is False
    assert by_id["obs-partial-allows-advice"]["general_suppressed_reason"] == "partial_evidence"
    assert by_id["follow-tell-me-more"]["prior_context_used"] is True
    assert by_id["follow-after-general"]["actual_route"] == "general"
    assert by_id["follow-after-general"]["qa_called"] is False
    assert by_id["follow-missing-context"]["actual_route"] == "clarify"
    assert by_id["follow-missing-context"]["prior_context_used"] is False
    assert by_id["follow-after-insufficient"]["general_called"] is False
    assert by_id["gen-noisy-class"]["actual_route"] == "general"
    assert by_id["gen-place-value"]["actual_route"] == "general"
    assert by_id["gen-mixed-ability"]["actual_route"] == "general"
    assert by_id["curr-class3-learn-multiplication"]["actual_route"] == "curriculum"
    assert all(not case.get("gap") for case in report["cases"])


def test_eval_gate_is_cleared_after_the_run(report):
    assert report["identity_mode"] == "eval-test-gate"
    assert current_identity_gate() is None


def test_live_evaluation_is_refused_without_an_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("ASSISTANT_EVAL_LIVE", raising=False)
    settings = Settings(
        _env_file=None,
        environment="development",
        assistant_enabled=False,
        assistant_local_stub_identity=False,
        llm_provider="openai",
        llm_api_key="present",
    )
    with pytest.raises(LiveEvaluationRefused):
        assert_live_allowed(settings)


def test_live_evaluation_is_refused_in_production(monkeypatch):
    monkeypatch.setenv("ASSISTANT_EVAL_LIVE", "1")
    settings = Settings(
        _env_file=None,
        environment="production",
        assistant_enabled=False,
        assistant_local_stub_identity=False,
        llm_provider="openai",
        llm_api_key="present",
    )
    with pytest.raises(LiveEvaluationRefused, match="production"):
        assert_live_allowed(settings)


def test_live_evaluation_does_not_use_the_local_stub(monkeypatch):
    monkeypatch.setenv("ASSISTANT_EVAL_LIVE", "1")
    settings = Settings(
        _env_file=None,
        environment="development",
        assistant_enabled=False,
        assistant_local_stub_identity=True,
        llm_provider="openai",
        llm_api_key="present",
    )
    with pytest.raises(LiveEvaluationRefused, match="stub"):
        assert_live_allowed(settings)
