"""Evaluation labels describe disagreements without calling the verifier wrong.

These tests do not call a model or rewrite gold, traces, or saved reports.
"""

from __future__ import annotations

from app.assistant.curriculum_gold import (
    SCORING_VERSION,
    SCORING_VERSION_V2,
    SCORING_VERSION_V3,
    diagnose,
    format_summary,
    load_dataset,
    rescore_saved_report,
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


def _answer_replacing(case: dict, replacements: dict[str, str]) -> str:
    parts = [case["grade_labels"][0], case["subject_labels"][0]]
    for fact in case["required_facts"]:
        parts.append(replacements.get(fact["source_code"], fact["match_any"][0]))
    return " ".join(parts)


def test_default_scorer_remains_v1_and_the_assistant_stays_disabled():
    assert SCORING_VERSION == "curriculum-qa-gold-scoring-v1"
    case = _case("gold-c4-math-u01-outcomes-complete")
    metrics = score_observation(case, {"answer": _answer_all(case), "status": "completed"})
    assert metrics["scoring_version"] == SCORING_VERSION
    assert Settings(_env_file=None).assistant_enabled is False


def test_french_subject_mismatch_abstention_is_not_an_incorrect_rejection():
    case = _case("gold-c4-french-subject-absent")
    observation = {
        "answer": "Class 4 French is not published in the curriculum records I checked.",
        "status": "insufficient_evidence",
        "evidence_status": "not_found",
        "verification_passed": False,
        "termination_reason": "subject_mismatch",
    }
    categories = diagnose(
        case,
        observation,
        {
            "deterministic_pass": True,
            "semantic_review_required": False,
            "evidence_abstention": True,
        },
    )
    assert categories == ["factual_pass_verifier_disagreement"]
    assert "verification_incorrectly_rejected" not in categories

    metrics = score_observation(case, observation)
    assert metrics["evidence_abstention"] is True
    assert metrics["deterministic_pass"] is True
    assert metrics["failure_categories"] == ["factual_pass_verifier_disagreement"]
    assert metrics["termination_reason"] == "subject_mismatch"
    assert any("Not an incorrect rejection." in note for note in metrics["diagnostic_notes"])
    assert any("Subject-mismatch" in note for note in metrics["diagnostic_notes"])


def test_an_incomplete_list_accepted_by_the_verifier_stays_an_incorrect_acceptance():
    case = _case("gold-c4-science-content-complete")
    one = case["required_facts"][0]["match_any"][1]
    observation = {
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
    }
    metrics = score_observation(case, observation)
    assert metrics["completeness"] is False
    categories = diagnose(case, observation, metrics)
    assert categories == metrics["failure_categories"]
    assert "verification_incorrectly_accepted" in categories
    assert "coverage_validation_missed_incomplete_answer" in categories
    assert "scorer_dependent_wording_mismatch" not in categories


def test_null_period_count_abstention_is_not_an_incorrect_rejection():
    case = _case("gold-c4-math-u01-periods-partial")
    invented = score_observation(
        case,
        {
            "answer": _answer_all(case) + " The unit assigns 12 periods.",
            "status": "completed",
            "evidence_status": "found",
            "verification_passed": True,
        },
    )
    assert invented["scoring_version"] == SCORING_VERSION
    assert invented["unsupported_claims"] == 1
    assert invented["evidence_abstention"] is False
    assert "verification_incorrectly_accepted" in invented["failure_categories"]

    limited = _answer_replacing(
        case,
        {
            "C4U01-LO03": "Compare numbers up to 10,000 using a number line.",
            "C4U01-LO05": "Use the place value system up to 10,000,000.",
        },
    ) + " The recommended period count cannot be determined."
    observation = {
        "answer": limited,
        "status": "insufficient_evidence",
        "evidence_status": "partial",
        "verification_passed": False,
        "termination_reason": "no_retrieval_progress",
    }
    for version in (SCORING_VERSION_V2, SCORING_VERSION_V3):
        metrics = score_observation(case, observation, scoring_version=version)
        assert metrics["factual_recall"] == 1
        assert metrics["evidence_abstention"] is True
        assert metrics["deterministic_pass"] is True
        assert metrics["failure_categories"] == ["factual_pass_verifier_disagreement"]
        assert "verification_incorrectly_rejected" not in metrics["failure_categories"]
        assert any("Not an incorrect rejection." in note for note in metrics["diagnostic_notes"])
        assert metrics["termination_reason"] == "no_retrieval_progress"

    v1 = score_observation(case, observation)
    assert v1["scoring_version"] == SCORING_VERSION
    assert v1["factual_recall"] < 1
    assert "verification_incorrectly_rejected" not in v1["failure_categories"]
    assert "factual_pass_verifier_disagreement" not in v1["failure_categories"]


def test_biology_topic_list_disagreement_does_not_claim_a_rejected_sentence():
    case = _case("gold-sss1-bio-intro-topics-complete")
    observation = {
        "answer": _answer_all(case),
        "status": "insufficient_evidence",
        "evidence_status": "found",
        "verification_passed": False,
        "termination_reason": "no_retrieval_progress",
        "verifier_requested_records": [
            {"type": "learning_outcome", "query": "S1-T1-01-LO04"}
        ],
    }
    metrics = score_observation(case, observation)
    assert metrics["factual_recall"] == 1
    assert metrics["deterministic_pass"] is True
    assert metrics["failure_categories"] == ["factual_pass_verifier_disagreement"]
    notes = " ".join(metrics["diagnostic_notes"])
    assert "S1-T1-01-LO04" in notes
    assert "does not require" in notes
    assert "does not establish that those topic sentences were rejected" in notes
    assert "or that the requested outcome was retrieved" in notes
    assert "rejected the supported topic" not in notes
    assert "LO04 was retrieved" not in notes


def test_v1_wording_mismatch_does_not_make_the_verifier_wrong():
    math_case = _case("gold-c4-math-u01-outcomes-complete")
    math_answer = _answer_replacing(
        math_case,
        {
            "C4U01-LO03": "Compare numbers up to 10,000 using a number line.",
            "C4U01-LO05": "Use the place value system up to 10,000,000.",
        },
    )
    observation = {
        "answer": math_answer,
        "status": "completed",
        "evidence_status": "found",
        "verification_passed": True,
        "coverage_decision": "accept",
    }
    v1 = score_observation(math_case, observation)
    assert v1["scoring_version"] == SCORING_VERSION
    assert v1["factual_recall"] < 1
    assert "verification_incorrectly_accepted" not in v1["failure_categories"]
    assert "scorer_dependent_wording_mismatch" in v1["failure_categories"]
    assert "generation_omitted_evidence" not in v1["failure_categories"]

    for version in (SCORING_VERSION_V2, SCORING_VERSION_V3):
        later = score_observation(math_case, observation, scoring_version=version)
        assert later["factual_recall"] == 1
        assert later["deterministic_pass"] is True
        assert later["failure_categories"] == []

    class3 = _case("gold-c3-math-u15-outcomes-complete")
    class3_answer = _answer_replacing(
        class3,
        {
            "C3U15-LO04": "Write the multiplication table of 3 and 6.",
            "C3U15-LO05": "Write the division table of 3 and 6.",
        },
    )
    class3_observation = {
        "answer": class3_answer,
        "status": "completed",
        "verification_passed": True,
        "coverage_decision": "accept",
    }
    class3_v1 = score_observation(class3, class3_observation)
    class3_v2 = score_observation(
        class3, class3_observation, scoring_version=SCORING_VERSION_V2
    )
    class3_v3 = score_observation(
        class3, class3_observation, scoring_version=SCORING_VERSION_V3
    )
    assert class3_v1["factual_recall"] < 1
    assert class3_v2["factual_recall"] < 1
    assert "verification_incorrectly_accepted" not in class3_v1["failure_categories"]
    assert "verification_incorrectly_accepted" not in class3_v2["failure_categories"]
    assert "scorer_dependent_wording_mismatch" in class3_v1["failure_categories"]
    assert "scorer_dependent_wording_mismatch" in class3_v2["failure_categories"]
    assert class3_v3["factual_recall"] == 1
    assert class3_v3["failure_categories"] == []

    science = _case("gold-c4-science-not-integrated")
    title = {
        "answer": "The first published unit of Class 4 Science is Living and Non-living Things.",
        "status": "completed",
        "verification_passed": True,
    }
    science_v1 = score_observation(science, title)
    science_v3 = score_observation(science, title, scoring_version=SCORING_VERSION_V3)
    assert science_v1["factual_recall"] == 0
    assert "verification_incorrectly_accepted" not in science_v1["failure_categories"]
    assert "scorer_dependent_wording_mismatch" in science_v1["failure_categories"]
    assert science_v3["factual_recall"] == 1
    assert "verification_incorrectly_accepted" not in science_v3["failure_categories"]


def test_astronomy_acceptance_still_fails_the_required_abstention():
    case = _case("gold-c4-math-astronomy-absent")
    metrics = score_observation(
        case,
        {
            "answer": "Class 4 Mathematics covers astronomy in the solar system.",
            "status": "completed",
            "evidence_status": "found",
            "verification_passed": True,
            "termination_reason": "verification_passed",
        },
    )
    assert metrics["evidence_abstention"] is False
    assert "verification_incorrectly_accepted" in metrics["failure_categories"]
    assert "insufficient_evidence_behaviour_failed" in metrics["failure_categories"]
    assert "factual_pass_verifier_disagreement" not in metrics["failure_categories"]


def test_summary_shows_the_disagreement_and_its_termination_context():
    saved = {
        "cases": [
            {
                "case_id": "gold-c4-french-subject-absent",
                "answer": "Class 4 French is not published in the curriculum records I checked.",
                "status": "insufficient_evidence",
                "evidence_status": "not_found",
                "verification_passed": False,
                "termination_reason": "subject_mismatch",
                "subject_code": "FRENCH",
                "grade_code": "CLASS_4",
                "evidence": [],
            }
        ]
    }
    rescored = rescore_saved_report(saved, scoring_version=SCORING_VERSION)
    assert rescored["scoring_version"] == SCORING_VERSION
    assert rescored["live_executed"] is False
    failure = next(item for item in rescored["failures"] if item["id"] == saved["cases"][0]["case_id"])
    assert failure["failure_categories"] == ["factual_pass_verifier_disagreement"]
    assert failure["termination_reason"] == "subject_mismatch"
    summary = format_summary(rescored)
    assert "factual_pass_verifier_disagreement" in summary
    assert "termination_reason=subject_mismatch" in summary
    assert "Not an incorrect rejection." in summary
    assert "verification_incorrectly_rejected" not in summary
