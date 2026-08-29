"""V2.7 verifier decision-boundary harness tests."""

from __future__ import annotations

from app.agent.v27_experiment import (
    ANSWER_FIXTURES,
    apply_experimental_decision_boundary,
    is_false_retrieval,
    prepare_evidence_for_fixture,
    replay_verifier_control,
    summarize_threshold_sweep,
    threshold_sweep,
    v27_experiment_enabled,
)
from app.agent.v25_experiment import is_imperfect_learning_outcome
from app.agent.state import CurriculumQAState
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence
from app.schemas.verification import VerificationRecommendation, VerificationResult


def _imperfect_lo() -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id="lo-garbled",
        name="C4U06-LO02",
        content=(
            "Multiply like fractions with denominators up to multiply like fractions "
            "with denominators up to multiply related fractions with denominators up to"
        ),
        metadata={"code": "C4U06-LO02"},
    )


def _clean_lo() -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id="lo-clean",
        name="C4U05-LO01",
        content="Add Equivalent fractions.",
        metadata={"code": "C4U05-LO01"},
    )


def _baseline() -> list[CurriculumEvidence]:
    return [_clean_lo(), _imperfect_lo()]


class _HighScoreRetrieveMore:
    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=False,
            score=0.9,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            unsupported_claims=[],
        )


class _UnsupportedVerifier:
    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=False,
            score=0.0,
            recommendation=VerificationRecommendation.FALLBACK,
            unsupported_claims=["C4U99-LO01"],
        )


def test_fixture_evidence_modes():
    baseline = _baseline()
    clean = prepare_evidence_for_fixture(baseline, "FAITHFUL_COMPLETE")
    imperfect = prepare_evidence_for_fixture(baseline, "FAITHFUL_IMPERFECT")
    missing = prepare_evidence_for_fixture(baseline, "MISSING_EVIDENCE")
    assert any(is_imperfect_learning_outcome(e) for e in imperfect)
    assert not any(is_imperfect_learning_outcome(e) for e in clean)
    assert not any((e.metadata or {}).get("code") == "C4U06-LO02" for e in missing)


def test_false_retrieval_detected_for_faithful_imperfect():
    row = {
        "retrieve_more_requested": True,
        "evidence_present": True,
        "unsupported_claims": [],
        "truncation_reconstruction": False,
        "verifier_score": 0.9,
    }
    assert is_false_retrieval(fixture_class="FAITHFUL_IMPERFECT", row=row)


def test_decision_boundary_accepts_faithful_imperfect_at_threshold():
    control = replay_verifier_control(
        question="q",
        answer=ANSWER_FIXTURES["FAITHFUL_IMPERFECT"]["answer"],
        baseline_evidence=_baseline(),
        fixture_class="FAITHFUL_IMPERFECT",
        verifier=_HighScoreRetrieveMore(),
    )
    experimental = apply_experimental_decision_boundary(
        control, fixture_class="FAITHFUL_IMPERFECT", threshold=0.85
    )
    assert experimental["policy_applied"]
    assert experimental["experimental_accepted"]
    assert experimental["experimental_decision"] == "accept"


def test_decision_boundary_rejects_unsupported():
    control = replay_verifier_control(
        question="q",
        answer=ANSWER_FIXTURES["UNSUPPORTED"]["answer"],
        baseline_evidence=_baseline(),
        fixture_class="UNSUPPORTED",
        verifier=_UnsupportedVerifier(),
    )
    experimental = apply_experimental_decision_boundary(
        control, fixture_class="UNSUPPORTED", threshold=0.70
    )
    assert not experimental["experimental_accepted"]


def test_decision_boundary_rejects_reconstruction():
    control = {
        "fixture_class": "RECONSTRUCTED_IMPERFECT",
        "verifier_score": 0.95,
        "verifier_accepted": False,
        "verifier_decision": "retrieve_more",
        "retrieve_more_requested": True,
        "evidence_present": True,
        "unsupported_claims": [],
        "truncation_reconstruction": True,
        "speculative_claims": False,
    }
    experimental = apply_experimental_decision_boundary(
        control, fixture_class="RECONSTRUCTED_IMPERFECT", threshold=0.70
    )
    assert not experimental["experimental_accepted"]


def test_threshold_sweep_structure():
    control = replay_verifier_control(
        question="q",
        answer=ANSWER_FIXTURES["FAITHFUL_IMPERFECT"]["answer"],
        baseline_evidence=_baseline(),
        fixture_class="FAITHFUL_IMPERFECT",
        verifier=_HighScoreRetrieveMore(),
    )
    sweep = summarize_threshold_sweep([control], threshold=0.85)
    assert sweep["threshold"] == 0.85
    assert "faithful_imperfect_acceptance" in sweep


def test_production_flag_default_off():
    assert Settings().v27_verifier_decision_boundary_experiment is False
    assert not v27_experiment_enabled(Settings(), CurriculumQAState.initial(question="q"))


def test_threshold_sweep_values():
    assert threshold_sweep() == (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def test_control_arm_does_not_apply_policy():
    control = replay_verifier_control(
        question="q",
        answer="x",
        baseline_evidence=_baseline(),
        fixture_class="FAITHFUL_IMPERFECT",
        verifier=_HighScoreRetrieveMore(),
    )
    assert control["arm"] == "A"
    assert not control.get("experimental_accepted")


def test_no_database_mutation():
    baseline = _baseline()
    original = baseline[1].content
    prepare_evidence_for_fixture(baseline, "FAITHFUL_COMPLETE")
    assert baseline[1].content == original
