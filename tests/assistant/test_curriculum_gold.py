"""Gold-set scoring. These tests do not call the curriculum API or a model."""

from __future__ import annotations

import json

import pytest

from app.assistant.curriculum_gold import (
    SCORING_VERSION,
    GoldEvaluationRefused,
    assert_live_allowed,
    format_summary,
    load_dataset,
    run_audit,
    score_observation,
)
from app.config import Settings


def _case(case_id: str) -> dict:
    dataset = load_dataset()
    return next(case for case in dataset["cases"] if case["id"] == case_id)


def _answer_all(case: dict) -> str:
    parts = [case["grade_labels"][0], case["subject_labels"][0]]
    for fact in case["required_facts"]:
        parts.append(fact["match_any"][0])
    return " ".join(parts)


def test_dataset_facts_come_from_enumerated_records():
    payload = load_dataset()
    assert payload["version"] == "curriculum-qa-gold-v1"
    assert payload["scoring_version"] == SCORING_VERSION
    assert len(payload["cases"]) >= 10
    subjects = {case["subject_code"] for case in payload["cases"]}
    grades = {case["grade_code"] for case in payload["cases"]}
    assert {"MATHEMATICS", "SCIENCE", "ENGLISH_LANGUAGE", "BIOLOGY"} <= subjects
    assert {"CLASS_3", "CLASS_4", "JSS_1", "SSS_1"} <= grades
    for case in payload["cases"]:
        for fact in case["required_facts"]:
            assert fact["source_ids"]
            assert fact["text"]
        if case["must_be_complete"]:
            record_ids = {record["id"] for record in case["source_records"]}
            for fact in case["required_facts"]:
                assert set(fact["source_ids"]) <= record_ids


def test_complete_science_list_is_the_full_published_set():
    case = _case("gold-c4-science-content-complete")
    assert len(case["required_facts"]) == 20
    codes = {fact["source_code"] for fact in case["required_facts"]}
    assert "C4-U01" in codes
    assert "C4-T14" in codes
    assert len(codes) == 20


def test_mathematics_complete_list_keeps_colliding_codes_and_skips_absent_unit():
    case = _case("gold-c4-math-units-complete")
    codes = [fact["source_code"] for fact in case["required_facts"]]
    assert codes[0] == "C4-U01"
    assert "C4-U12" not in codes
    assert codes[-1] == "C4-U28"
    assert len(codes) == 27
    names = [fact["record_name"] for fact in case["required_facts"]]
    assert names.count("Number and Numeration") > 1


def test_exact_gold_facts_score_as_supported():
    case = _case("gold-c4-math-u01-outcomes-complete")
    metrics = score_observation(
        case,
        {
            "answer": _answer_all(case),
            "status": "completed",
            "evidence_status": "found",
            "coverage_decision": "accept",
            "verification_passed": True,
        },
    )
    assert metrics["factual_recall"] == 1
    assert metrics["factual_precision"] == 1
    assert metrics["unsupported_claim_rate"] == 0
    assert metrics["completeness"] is True
    assert metrics["subject_grade_accuracy"] is True
    assert metrics["deterministic_pass"] is True
    assert metrics["failure_categories"] == []


def test_a_plausible_partial_list_is_not_complete():
    case = _case("gold-c4-science-content-complete")
    one = case["required_facts"][0]["match_any"][1]
    metrics = score_observation(
        case,
        {
            "answer": f"Class 4 Science includes {one}. That is the complete list.",
            "status": "completed",
            "evidence_status": "found",
            "coverage_decision": "accept",
            "verification_passed": True,
            "evidence": [
                {
                    "entity_id": case["required_facts"][0]["source_ids"][0],
                    "grade": "CLASS_4",
                    "subject": "SCIENCE",
                }
            ],
        },
    )
    assert metrics["factual_recall"] < 1
    assert metrics["completeness"] is False
    assert metrics["deterministic_pass"] is False
    assert "coverage_validation_missed_incomplete_answer" in metrics["failure_categories"]
    assert "verification_incorrectly_accepted" in metrics["failure_categories"]


def test_paraphrase_without_the_gold_wording_needs_review():
    case = _case("gold-c4-math-u01-outcomes-complete")
    metrics = score_observation(
        case,
        {
            "answer": (
                "Class 4 Mathematics helps pupils understand large numbers "
                "and place value in a practical way."
            ),
            "status": "completed",
            "evidence_status": "found",
        },
    )
    assert metrics["factual_recall"] == 0
    assert metrics["semantic_review_required"] is True
    assert "semantic_review_required" in metrics["failure_categories"]
    assert metrics["deterministic_pass"] is False


def test_wrong_subject_unit_is_an_unsupported_claim():
    case = _case("gold-c4-science-not-integrated")
    metrics = score_observation(
        case,
        {
            "answer": (
                "Class 4 Science begins with Meaning, branches and Importance of Science."
            ),
            "status": "completed",
            "evidence_status": "found",
            "evidence": [
                {
                    "entity_id": "83b85707-2bfc-429e-bbff-9c2be76dd28a",
                    "grade": "JSS_1",
                    "subject": "INTEGRATED_SCIENCE",
                }
            ],
        },
    )
    assert metrics["unsupported_claims"] == 1
    assert metrics["factual_precision"] == 0
    assert metrics["subject_grade_accuracy"] is True
    assert "context_preparation_wrong_scope" in metrics["failure_categories"]
    assert "generation_misstated_evidence" in metrics["failure_categories"]


def test_insufficient_french_abstains_and_does_not_borrow_appreciation():
    case = _case("gold-c4-french-subject-absent")
    abstained = score_observation(
        case,
        {
            "answer": "Class 4 French is not published in the curriculum records I checked.",
            "status": "insufficient_evidence",
            "evidence_status": "not_found",
        },
    )
    assert abstained["evidence_abstention"] is True
    assert abstained["factual_recall"] is None
    assert abstained["deterministic_pass"] is True

    borrowed = score_observation(
        case,
        {
            "answer": "Class 4 French covers Ordinal Numbers (premier, deuxième etc.).",
            "status": "completed",
            "evidence_status": "found",
        },
    )
    assert borrowed["evidence_abstention"] is False
    assert borrowed["unsupported_claims"] == 1
    assert "insufficient_evidence_behaviour_failed" in borrowed["failure_categories"]


def test_period_count_cannot_be_invented_for_a_null_field():
    case = _case("gold-c4-math-u01-periods-partial")
    invented = score_observation(
        case,
        {
            "answer": _answer_all(case) + " The unit assigns 12 periods.",
            "status": "completed",
            "evidence_status": "found",
        },
    )
    assert invented["factual_recall"] == 1
    assert invented["unsupported_claims"] == 1
    assert invented["completeness"] is False
    assert invented["evidence_abstention"] is False

    limited = score_observation(
        case,
        {
            "answer": _answer_all(case) + " No recommended period count is recorded.",
            "status": "completed",
            "evidence_status": "partial",
        },
    )
    assert limited["deterministic_pass"] is True
    assert limited["evidence_abstention"] is True


def test_general_advice_must_stay_out_of_the_curriculum_answer():
    case = _case("gold-jss1-integrated-science-first-unit")
    advice = "Use a class discussion before the definition."
    merged = score_observation(
        case,
        {
            "answer": _answer_all(case) + " " + advice,
            "status": "completed",
            "evidence_status": "found",
            "general_text": advice,
            "general_label": "General AI assistance",
        },
    )
    assert merged["channel_integrity"] is False

    separate = score_observation(
        case,
        {
            "answer": _answer_all(case),
            "status": "completed",
            "evidence_status": "found",
            "general_text": advice,
            "general_label": "General AI assistance",
        },
    )
    assert separate["channel_integrity"] is True
    assert separate["deterministic_pass"] is True


def test_retrieved_evidence_that_is_left_out_of_the_answer_is_generation_omission():
    case = _case("gold-c4-science-solar-outcomes-complete")
    first = case["required_facts"][0]
    metrics = score_observation(
        case,
        {
            "answer": f"Class 4 Science includes {first['match_any'][0]}",
            "status": "completed",
            "evidence_status": "found",
            "evidence": [
                {"entity_id": fact["source_ids"][0], "grade": "CLASS_4", "subject": "SCIENCE"}
                for fact in case["required_facts"]
            ],
        },
    )
    assert "generation_omitted_evidence" in metrics["failure_categories"]


def test_technical_failure_is_excluded_from_factual_denominators():
    from app.assistant.curriculum_gold import _technical_failure_row, aggregate

    case = _case("gold-c4-science-not-integrated")
    row = _technical_failure_row(
        case,
        TimeoutError("provider timed out"),
        latency_ms=1200,
        request_id="gold-c4-science-not-integrated",
    )
    assert row["scored"] is False
    assert row["technical_failure"] is True
    assert row["factual_recall"] is None
    assert row["failure_categories"] == ["provider_or_infrastructure_failure"]
    scored = score_observation(
        case,
        {"answer": _answer_all(case), "status": "completed", "evidence_status": "found"},
    )
    metrics = aggregate([row, scored])
    assert metrics["factual_recall"]["applicable"] == 1
    assert metrics["factual_recall"]["value"] == 1


def test_audit_does_not_score_model_answers_and_omits_fixture_text():
    report = run_audit()
    summary = format_summary(report)
    assert report["live_executed"] is False
    assert report["case_count"] == 12
    assert report["metrics"]["factual_recall"]["measured"] is False
    assert "Count up to 5000 using a number line." not in summary
    assert report["dataset_sha256"]
    encoded = json.dumps(report)
    assert "api_key" not in encoded
    assert "Bearer " not in encoded


def test_live_gold_evaluation_is_refused_without_an_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("CURRICULUM_GOLD_EVAL_LIVE", raising=False)
    settings = Settings(
        _env_file=None,
        environment="development",
        assistant_enabled=False,
        llm_provider="openai",
        llm_api_key="present",
    )
    with pytest.raises(GoldEvaluationRefused):
        assert_live_allowed(settings)


def test_loader_rejects_a_fact_without_a_source_record(tmp_path):
    payload = load_dataset()
    payload["cases"][0]["required_facts"][0]["source_ids"] = []
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)
