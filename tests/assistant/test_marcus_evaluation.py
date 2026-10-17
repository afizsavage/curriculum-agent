"""Marcus dataset integrity. These tests do not call a model."""

from __future__ import annotations

import json

import pytest

from app.assistant.curriculum_gold import SCORING_VERSION, file_sha256, load_dataset
from app.assistant.evaluation import load_dataset as load_behaviour
from app.assistant.marcus_evaluation import (
    MarcusNotScoreable,
    MarcusReportRefused,
    assert_safe_output,
    build_baseline_report,
    construct_dataset,
    load_dataset as load_marcus,
    score_marcus_case,
    write_dataset,
)
from app.config import Settings


def test_marcus_dataset_has_the_24_cases_and_stays_on_scoring_v1(tmp_path):
    payload = construct_dataset()
    path = tmp_path / "marcus_evaluation_v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_marcus(path)
    assert loaded["version"] == "marcus-evaluation-v1"
    assert loaded["official_scoring_version"] == SCORING_VERSION
    assert len(loaded["cases"]) == 24
    assert Settings(_env_file=None).assistant_enabled is False
    ids = [case["id"] for case in loaded["cases"]]
    assert len(set(ids)) == 24


def test_exact_questions_match_gold_and_approximate_questions_are_documented():
    gold = {case["id"]: case for case in load_dataset()["cases"]}
    marcus = {case["id"]: case for case in construct_dataset()["cases"]}
    exact = [case for case in marcus.values() if case["question_match"] == "exact"]
    approximate = [case for case in marcus.values() if case["question_match"] == "approximate"]
    assert len(exact) == 10
    assert len(approximate) == 2
    for case in exact:
        historical = gold[case["historical_case_id"]]
        assert case["question"] == historical["question"]
        assert [fact["fact_id"] for fact in case["required_facts"]] == [
            fact["fact_id"] for fact in historical["required_facts"]
        ]
    for case in approximate:
        assert case["question"] != case["historical_question"]
        assert case["question_difference"]
        assert case["requires_live_run"] is True
        assert case["saved_answer_available"] is True


def test_unenumerated_cases_cannot_pass_factual_scoring():
    marcus = {case["id"]: case for case in construct_dataset()["cases"]}
    blocked = marcus["marcus-p4-astronomy-subject"]
    assert blocked["required_facts"] is None
    assert blocked["requires_fact_enumeration"] is True
    with pytest.raises(MarcusNotScoreable):
        score_marcus_case(blocked, {"answer": "Primary 4 Astronomy covers the planets."})
    fractions = marcus["marcus-teach-fractions"]
    with pytest.raises(MarcusNotScoreable):
        score_marcus_case(
            fractions,
            {"answer": "Class 4 fractions are fully listed here.", "status": "completed"},
        )


def test_enumerated_abstention_still_rejects_a_borrowed_subject():
    marcus = {case["id"]: case for case in construct_dataset()["cases"]}
    french = marcus["marcus-c4-french"]
    borrowed = score_marcus_case(
        french,
        {
            "answer": "Class 4 French covers Ordinal Numbers (premier, deuxième etc.).",
            "status": "completed",
            "evidence_status": "found",
        },
    )
    assert borrowed["evidence_abstention"] is False
    assert borrowed["deterministic_pass"] is False
    abstained = score_marcus_case(
        french,
        {
            "answer": "Class 4 French is not published in the curriculum records I checked.",
            "status": "insufficient_evidence",
            "evidence_status": "not_found",
            "verification_passed": False,
            "termination_reason": "subject_mismatch",
        },
    )
    assert abstained["evidence_abstention"] is True
    assert "verification_incorrectly_rejected" not in abstained["failure_categories"]


def test_gold_and_behaviour_loaders_keep_their_contracts(tmp_path):
    gold = load_dataset()
    assert gold["scoring_version"] == SCORING_VERSION
    assert file_sha256()
    behaviour = load_behaviour()
    assert all(case["gold_curriculum_facts"] == [] for case in behaviour["cases"])
    broken = json.loads(json.dumps(behaviour))
    broken["cases"][0]["gold_curriculum_facts"] = ["invented fact"]
    path = tmp_path / "assistant_behaviour_v1.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError):
        load_behaviour(path)


def test_writer_refuses_historical_artifacts(tmp_path):
    with pytest.raises(MarcusReportRefused):
        assert_safe_output(tmp_path / "curriculum_qa_gold_v1_batch17.json")
    with pytest.raises(MarcusReportRefused):
        assert_safe_output(tmp_path / "curriculum_qa_gold_v1.json")
    allowed = assert_safe_output(tmp_path / "marcus_evaluation_v1.json")
    assert allowed.name == "marcus_evaluation_v1.json"
    written = write_dataset(tmp_path / "marcus_evaluation_v1.json")
    assert len(json.loads(written.read_text(encoding="utf-8"))["cases"]) == 24
    stage = assert_safe_output(tmp_path / "marcus_evaluation_v1_stage2_absence.json")
    stage.write_text("{}", encoding="utf-8")
    with pytest.raises(MarcusReportRefused):
        assert_safe_output(stage)


def test_saved_answer_baseline_separates_exact_and_approximate_matches():
    report = build_baseline_report()
    assert report["live_executed"] is False
    assert report["assistant_enabled"] is False
    assert report["official_scoring_version"] == SCORING_VERSION
    assert report["exact_saved_answer_overlaps"] == 10
    assert report["approximate_saved_answer_overlaps"] == 2
    assert report["selector_difference_count"] == 0
    assert all(row["result"] == "no selector difference" for row in report["selector_audit"])
    assert all(row["counted_as_improvement"] is False for row in report["selector_audit"])
    by_id = {row["id"]: row for row in report["cases"]}
    assert by_id["marcus-c4-u01-outcomes"]["execution"] == "approximate_rescore"
    assert by_id["marcus-c4-u01-outcomes"]["official_v1"]["scoring_version"] == SCORING_VERSION
    assert "curriculum-qa-gold-scoring-v2" in by_id["marcus-c4-science-all"]["secondary"]
    assert by_id["marcus-teach-c4-u01"]["execution"] == "not_executed"
    assert by_id["marcus-p4-astronomy-subject"]["official_v1"] is None
    assert by_id["marcus-c4-science-first"]["trace_question_matches"] is True


def _file_cases():
    return {case["id"]: case for case in load_marcus()["cases"]}


def test_blocked_cases_keep_an_explicit_contract_and_cannot_pass():
    cases = _file_cases()
    fractions = cases["marcus-teach-fractions"]
    assert fractions["ready_to_score"] is False
    assert fractions["required_facts"] is None
    assert fractions["source_limitation"]
    assert [row["source_code"] for row in fractions["related_records"]] == [
        "C4-U04",
        "C4-U05",
        "C4-U06",
    ]
    with pytest.raises(MarcusNotScoreable):
        score_marcus_case(fractions, {"answer": "Teach equivalent fractions first.", "status": "completed"})
    appreciation = cases["marcus-c4-french-appreciation"]
    assert appreciation["subject_code"] == "FRENCH_APPRECIATION"
    assert appreciation["required_facts"] is None
    assert appreciation["source_records"][0]["source_code"] == "FRENCH_APPRECIATION"
    assert "French subject" not in appreciation["source_limitation"] or "must not be substituted" in appreciation["source_limitation"]
    with pytest.raises(MarcusNotScoreable):
        score_marcus_case(
            appreciation,
            {"answer": "Class 4 French Appreciation covers Ordinal Numbers.", "status": "completed"},
        )


def test_catalogue_abstentions_require_a_stated_absence():
    cases = _file_cases()
    samples = {
        "marcus-c4-u12-absent": "Class 4 Mathematics does not publish unit C4-U12.",
        "marcus-c4-english-subject": "Class 4 does not publish a subject named English.",
        "marcus-c4-integrated-science": "Class 4 does not publish Integrated Science.",
        "marcus-p4-astronomy-subject": "Primary 4 does not publish an Astronomy subject.",
    }
    for case_id, answer in samples.items():
        case = cases[case_id]
        assert case["evaluation_type"] == "catalogue_abstention"
        assert case["required_facts"] == []
        assert case["source_records"]
        blank = score_marcus_case(case, {"answer": "", "status": "insufficient_evidence"})
        assert blank["deterministic_pass"] is False
        stated = score_marcus_case(
            case,
            {"answer": answer, "status": "insufficient_evidence", "evidence_status": "not_found"},
        )
        assert stated["evidence_abstention"] is True
        assert stated["deterministic_pass"] is True
        borrowed = case["forbidden_substitutions"][0]["match_any"][0]
        wrong = score_marcus_case(
            case,
            {"answer": f"{answer} {borrowed}", "status": "insufficient_evidence"},
        )
        assert wrong["deterministic_pass"] is False


def test_identity_lookup_is_not_scored_as_the_full_unit_list():
    case = _file_cases()["marcus-c4-math-u01-identity"]
    assert case["evaluation_type"] == "identity_lookup"
    codes = [fact["source_code"] for fact in case["required_facts"] if fact["text"].startswith("C4-")]
    assert codes == ["C4-U01", "C4-U07", "C4-U08", "C4-U09", "C4-U10"]
    assert "C4-U02" not in codes
    only_first = score_marcus_case(
        case,
        {
            "answer": "Class 4 Mathematics unit Number and Numeration is C4-U01.",
            "status": "completed",
            "evidence_status": "found",
        },
    )
    assert only_first["completeness"] is False
    assert only_first["deterministic_pass"] is False
    full_identity = (
        "Class 4 Mathematics units named Number and Numeration are "
        "C4-U01, C4-U07, C4-U08, C4-U09, and C4-U10."
    )
    matched = score_marcus_case(
        case,
        {"answer": full_identity, "status": "completed", "evidence_status": "found"},
    )
    assert matched["factual_recall"] == 1.0
    assert matched["deterministic_pass"] is True
    assert matched["evaluation_type"] == "identity_lookup"


def test_lesson_plans_are_not_scored_with_the_fact_list_rubric():
    cases = _file_cases()
    for case_id in ("marcus-teach-c4-u01", "marcus-teach-solar", "marcus-teach-bio-topics"):
        case = cases[case_id]
        assert case["evaluation_type"] == "lesson_plan"
        assert case["ready_to_score"] is False
        assert case["required_facts"] is None
        assert case["curriculum_scope_facts"]
        assert case["lesson_rubric"]["must_not_score_as"]
        sentences = " ".join(fact["text"] for fact in case["curriculum_scope_facts"])
        with pytest.raises(MarcusNotScoreable):
            score_marcus_case(case, {"answer": sentences, "status": "completed"})
