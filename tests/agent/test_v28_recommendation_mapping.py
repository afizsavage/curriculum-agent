"""V2.8 recommendation-mapping harness tests."""

from __future__ import annotations

import copy

from app.agent.state import CurriculumQAState
from app.agent.v28_recommendation_mapping import (
    FIXTURES,
    MappedRecommendation,
    detect_placeholder,
    map_recommendation,
    remap_row_for_threshold,
    replay_fixture,
    v28_experiment_enabled,
)
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence
from app.schemas.verification import VerificationRecommendation, VerificationResult


def _evidence() -> list[CurriculumEvidence]:
    return [
        CurriculumEvidence(
            entity_type="learning_outcome",
            entity_id="lo1",
            name="C4U18-LO01",
            content="Order operations using BODMAS.",
            metadata={"code": "C4U18-LO01"},
        )
    ]


class _HighRetrieve:
    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=False,
            score=0.9,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
        )


class _LowRetrieve:
    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=False,
            score=0.65,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
        )


class _Unsupported:
    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=False,
            score=0.99,
            recommendation=VerificationRecommendation.FALLBACK,
            unsupported_claims=["C4U18-LO99"],
        )


def test_faithful_imperfect_maps_to_accept_at_threshold():
    result = VerificationResult(
        passed=False,
        score=0.9,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
    )
    mapped = map_recommendation(
        result,
        fixture_class="FAITHFUL_IMPERFECT",
        evidence=_evidence(),
        answer="faithful text",
        threshold=0.85,
    )
    assert mapped.mapped_recommendation == MappedRecommendation.ACCEPT
    assert mapped.policy_applied


def test_low_score_faithful_imperfect_stays_retrieve_more():
    result = VerificationResult(
        passed=False,
        score=0.65,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
    )
    mapped = map_recommendation(
        result,
        fixture_class="FAITHFUL_IMPERFECT",
        evidence=_evidence(),
        answer="faithful text",
        threshold=0.85,
    )
    assert mapped.mapped_recommendation == MappedRecommendation.RETRIEVE_MORE


def test_unsupported_claim_rejects_even_high_score():
    result = VerificationResult(
        passed=False,
        score=0.99,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        unsupported_claims=["C4U18-LO99"],
    )
    mapped = map_recommendation(
        result,
        fixture_class="UNSUPPORTED_CLAIM",
        evidence=_evidence(),
        answer="unsupported",
        threshold=0.70,
    )
    assert mapped.mapped_recommendation == MappedRecommendation.REJECT


def test_placeholder_never_accepted_via_score():
    result = VerificationResult(
        passed=False,
        score=0.95,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        unsupported_claims=["[CLEAN_EVIDENCE_PLACEHOLDER]"],
    )
    evidence = [
        CurriculumEvidence(
            entity_type="learning_outcome",
            entity_id="lo-p",
            name="C4U04-LO04",
            content="C4U04-LO04 — [CLEAN_EVIDENCE_PLACEHOLDER]",
            metadata={"code": "C4U04-LO04"},
        )
    ]
    detected, cls = detect_placeholder(evidence=evidence, answer="placeholder answer", verifier_result=result)
    mapped = map_recommendation(
        result,
        fixture_class="CLEAN_PLACEHOLDER",
        evidence=evidence,
        answer="placeholder answer",
        threshold=0.70,
    )
    assert detected
    assert mapped.mapped_recommendation == MappedRecommendation.REJECT


def test_missing_evidence_insufficient_or_retrieve():
    result = VerificationResult(
        passed=False,
        score=0.4,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
    )
    mapped = map_recommendation(
        result,
        fixture_class="MISSING_EVIDENCE",
        evidence=_evidence(),
        answer="missing lo claim",
        threshold=0.85,
    )
    assert mapped.mapped_recommendation in {
        MappedRecommendation.RETRIEVE_MORE,
        MappedRecommendation.INSUFFICIENT_EVIDENCE,
    }


def test_threshold_boundary_deterministic():
    row = {
        "fixture_class": "FAITHFUL_IMPERFECT",
        "verifier_score": 0.85,
        "verifier_accepted": False,
        "verifier_decision": "retrieve_more",
        "retrieve_more_requested": True,
        "placeholder_detected": False,
        "unsupported_claims": [],
    }
    at = remap_row_for_threshold(row, 0.85)["mapped_accepted"]
    below = remap_row_for_threshold(row, 0.86)["mapped_accepted"]
    assert at is True
    assert below is False


def test_mapper_does_not_mutate_verifier_result():
    result = VerificationResult(
        passed=False,
        score=0.9,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
    )
    before = result.model_dump()
    map_recommendation(
        result,
        fixture_class="FAITHFUL_IMPERFECT",
        evidence=_evidence(),
        answer="x",
        threshold=0.85,
    )
    assert result.model_dump() == before


def test_replay_snapshot_immutable():
    baseline = _evidence()
    row = replay_fixture(
        fixture_class="FAITHFUL_COMPLETE",
        c4u18_baseline=baseline,
        fractions_baseline=baseline,
        verifier=_HighRetrieve(),
    )
    snap = copy.deepcopy(row["verifier_result_snapshot"])
    row["verifier_result_snapshot"]["score"] = 0.1
    assert snap["score"] == 0.9


def test_safety_fixture_reconstruction_rejects():
    mapped = map_recommendation(
        VerificationResult(passed=False, score=0.95, recommendation=VerificationRecommendation.RETRIEVE_MORE),
        fixture_class="RECONSTRUCTION",
        evidence=_evidence(),
        answer="reconstructed",
        threshold=0.70,
    )
    assert mapped.mapped_recommendation == MappedRecommendation.REJECT


def test_speculative_and_absence_reject():
    for fixture in ("SPECULATIVE", "UNSUPPORTED_ABSENCE"):
        mapped = map_recommendation(
            VerificationResult(passed=False, score=0.95, recommendation=VerificationRecommendation.RETRIEVE_MORE),
            fixture_class=fixture,  # type: ignore[arg-type]
            evidence=_evidence(),
            answer="bad",
            threshold=0.70,
        )
        assert mapped.mapped_recommendation == MappedRecommendation.REJECT


def test_production_flag_default_off():
    assert Settings().v28_recommendation_mapping_experiment is False
    assert not v28_experiment_enabled(Settings(), CurriculumQAState.initial(question="q"))


def test_all_fixtures_defined():
    assert len(FIXTURES) == 8
