"""V2.10 integrated experiment harness tests."""

from __future__ import annotations

import copy

from app.agent.state import CurriculumQAState
from app.agent.v25_experiment import _CLEAN_PLACEHOLDER
from app.agent.v210_integrated_experiment import (
    PIPELINES,
    Pipeline,
    _prepare_integrated_evidence,
    compute_attribution,
    run_pipeline,
    v210_experiment_enabled,
)
from app.agent.v29_evidence_normalization import normalize_evidence, NormalizationVariant
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence
from app.schemas.verification import VerificationRecommendation, VerificationResult


def _c4u18_unit() -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="unit",
        entity_id="unit-1",
        name="Everyday Arithmetic Money",
        content="Everyday Arithmetic Money",
        metadata={"code": "C4-U18"},
    )


def _c4u18_lo(topic: str = "unit-1") -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id="lo1",
        name="C4U18-LO01",
        content="Order operations using BODMAS.",
        topic=topic,
        grade="CLASS_4",
        subject="MATHEMATICS",
        metadata={
            "code": "C4U18-LO01",
            "parent_content_name": "Everyday Arithmetic Money",
            "parent_content_code": "C4-U18",
        },
    )


class _AcceptAll:
    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=True,
            score=1.0,
            recommendation=VerificationRecommendation.ACCEPT,
        )


class _HighRetrieve:
    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=False,
            score=0.9,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
        )


class _HighScoreUnsupported:
    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=False,
            score=0.99,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            unsupported_claims=["C4U18-LO99"],
        )


class _RetrieveMissing:
    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=False,
            score=0.2,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            missing_evidence=[{"type": "learning_outcome"}],
        )


def test_raw_pipeline_baseline():
    row = run_pipeline(
        fixture_class="FAITHFUL_COMPLETE",
        pipeline=Pipeline.A_RAW_VERIFIER,
        c4u18_baseline=[_c4u18_unit(), _c4u18_lo()],
        fractions_baseline=[],
        verifier=_AcceptAll(),
    )
    assert row["pipeline"] == Pipeline.A_RAW_VERIFIER.value
    assert row["final_accepted"] is True
    assert row["mapping_applied"] is False


def test_normalization_improves_topic_resolution():
    evidence = [_c4u18_unit(), _c4u18_lo(topic="unit-1")]
    normalized = normalize_evidence(evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION)
    lo = next(e for e in normalized.evidence if e.entity_type == "learning_outcome")
    assert lo.topic == "Everyday Arithmetic Money"


def test_mapper_improves_fi_decision():
    row = run_pipeline(
        fixture_class="FAITHFUL_IMPERFECT",
        pipeline=Pipeline.D_NORMALIZED_VERIFIER_MAPPER,
        c4u18_baseline=[_c4u18_unit()],
        fractions_baseline=[_c4u18_lo()],
        verifier=_HighRetrieve(),
        raw_baseline_row={"verifier_accepted": False},
        normalized_baseline_row={"verifier_accepted": False},
    )
    assert row["verifier_accepted"] is False
    assert row["final_accepted"] is True
    assert row["mapped_recommendation"] == "accept"


def test_normalization_before_verification_order():
    calls: list[str] = []

    class _Spy:
        def verify(self, state, request_id=None):
            calls.append("verify")
            topic = state.evidence[1].topic if len(state.evidence) > 1 else None
            assert topic == "Everyday Arithmetic Money"
            return VerificationResult(
                passed=True,
                score=1.0,
                recommendation=VerificationRecommendation.ACCEPT,
            )

    run_pipeline(
        fixture_class="FAITHFUL_COMPLETE",
        pipeline=Pipeline.B_NORMALIZED_VERIFIER,
        c4u18_baseline=[_c4u18_unit(), _c4u18_lo(topic="unit-1")],
        fractions_baseline=[],
        verifier=_Spy(),
    )
    assert calls == ["verify"]


def test_mapper_after_verification():
    row = run_pipeline(
        fixture_class="FAITHFUL_IMPERFECT",
        pipeline=Pipeline.C_RAW_VERIFIER_MAPPER,
        c4u18_baseline=[_c4u18_unit()],
        fractions_baseline=[_c4u18_lo()],
        verifier=_HighRetrieve(),
        raw_baseline_row={"verifier_accepted": False},
    )
    assert row["verifier_decision"] == "retrieve_more"
    assert row["mapped_recommendation"] == "accept"


def test_high_score_unsupported_rejected():
    row = run_pipeline(
        fixture_class="HIGH_SCORE_UNSUPPORTED",
        pipeline=Pipeline.D_NORMALIZED_VERIFIER_MAPPER,
        c4u18_baseline=[_c4u18_unit(), _c4u18_lo()],
        fractions_baseline=[],
        verifier=_HighScoreUnsupported(),
        raw_baseline_row={"verifier_accepted": False},
        normalized_baseline_row={"verifier_accepted": False},
    )
    assert row["final_accepted"] is False
    assert row["final_recommendation"] == "reject"


def test_placeholder_high_score_rejected():
    row = run_pipeline(
        fixture_class="PLACEHOLDER_PLUS_HIGH_SCORE",
        pipeline=Pipeline.D_NORMALIZED_VERIFIER_MAPPER,
        c4u18_baseline=[_c4u18_unit()],
        fractions_baseline=[
            CurriculumEvidence(
                entity_type="learning_outcome",
                entity_id="ph",
                name="C4U04-LO04",
                content=f"C4U04-LO04 — {_CLEAN_PLACEHOLDER}",
                metadata={"code": "C4U04-LO04"},
            )
        ],
        verifier=_HighRetrieve(),
        raw_baseline_row={"verifier_accepted": False},
        normalized_baseline_row={"verifier_accepted": False},
    )
    assert row["final_accepted"] is False


def test_missing_evidence_not_accepted():
    row = run_pipeline(
        fixture_class="MISSING_EVIDENCE",
        pipeline=Pipeline.D_NORMALIZED_VERIFIER_MAPPER,
        c4u18_baseline=[_c4u18_unit(), _c4u18_lo()],
        fractions_baseline=[],
        verifier=_RetrieveMissing(),
        raw_baseline_row={"verifier_accepted": False},
        normalized_baseline_row={"verifier_accepted": False},
    )
    assert row["final_accepted"] is False


def test_unresolvable_topic_not_invented():
    evidence, _ = _prepare_integrated_evidence(
        fixture_class="NORMALIZATION_MUST_NOT_INVENT",
        c4u18_baseline=[_c4u18_unit(), _c4u18_lo(topic="unit-1")],
        fractions_baseline=[],
    )
    lo = next(e for e in evidence if e.entity_type == "learning_outcome")
    normalized = normalize_evidence(evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION)
    lo_norm = next(e for e in normalized.evidence if e.entity_type == "learning_outcome")
    assert lo.topic != lo_norm.topic or lo_norm.topic == "00000000-0000-0000-0000-000000000099"
    assert "parent_content_name" not in (lo.metadata or {})


def test_wrong_subject_not_relabeled():
    evidence, _ = _prepare_integrated_evidence(
        fixture_class="ADV_WRONG_SUBJECT",
        c4u18_baseline=[_c4u18_unit(), _c4u18_lo()],
        fractions_baseline=[],
    )
    lo = next(e for e in evidence if e.entity_type == "learning_outcome")
    assert lo.subject == "ENGLISH"


def test_wrong_grade_not_relabeled():
    evidence, _ = _prepare_integrated_evidence(
        fixture_class="ADV_WRONG_GRADE",
        c4u18_baseline=[_c4u18_unit(), _c4u18_lo()],
        fractions_baseline=[],
    )
    lo = next(e for e in evidence if e.entity_type == "learning_outcome")
    assert lo.grade == "CLASS_5"


def test_normalization_answer_independent():
    evidence = [_c4u18_unit(), _c4u18_lo()]
    a = normalize_evidence(evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION)
    b = normalize_evidence(evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION)
    assert a.diagnostics.evidence_hash_out == b.diagnostics.evidence_hash_out


def test_normalization_idempotent():
    evidence = [_c4u18_unit(), _c4u18_lo(topic="unit-1")]
    once = normalize_evidence(evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION)
    twice = normalize_evidence(once.evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION)
    assert once.diagnostics.evidence_hash_out == twice.diagnostics.evidence_hash_out


def test_raw_evidence_not_mutated():
    evidence = [_c4u18_unit(), _c4u18_lo(topic="unit-1")]
    original = copy.deepcopy(evidence)
    _prepare_integrated_evidence(
        fixture_class="NORMALIZATION_MUST_NOT_INVENT",
        c4u18_baseline=evidence,
        fractions_baseline=[],
    )
    assert evidence[1].topic == original[1].topic


def test_mapper_does_not_mutate_verifier_snapshot():
    row = run_pipeline(
        fixture_class="FAITHFUL_IMPERFECT",
        pipeline=Pipeline.C_RAW_VERIFIER_MAPPER,
        c4u18_baseline=[_c4u18_unit()],
        fractions_baseline=[_c4u18_lo()],
        verifier=_HighRetrieve(),
        raw_baseline_row={"verifier_accepted": False},
    )
    snap = row["verifier_result_snapshot"]
    assert snap["score"] == 0.9
    assert snap["recommendation"] == "retrieve_more"


def test_threshold_mapping_deterministic():
    row = run_pipeline(
        fixture_class="FAITHFUL_IMPERFECT",
        pipeline=Pipeline.D_NORMALIZED_VERIFIER_MAPPER,
        c4u18_baseline=[_c4u18_unit()],
        fractions_baseline=[_c4u18_lo()],
        verifier=_HighRetrieve(),
        threshold=0.85,
        raw_baseline_row={"verifier_accepted": False},
        normalized_baseline_row={"verifier_accepted": False},
    )
    row2 = run_pipeline(
        fixture_class="FAITHFUL_IMPERFECT",
        pipeline=Pipeline.D_NORMALIZED_VERIFIER_MAPPER,
        c4u18_baseline=[_c4u18_unit()],
        fractions_baseline=[_c4u18_lo()],
        verifier=_HighRetrieve(),
        threshold=0.85,
        raw_baseline_row={"verifier_accepted": False},
        normalized_baseline_row={"verifier_accepted": False},
    )
    assert row["mapped_recommendation"] == row2["mapped_recommendation"]


def test_integrated_result_deterministic():
    kwargs = dict(
        fixture_class="FAITHFUL_COMPLETE",
        pipeline=Pipeline.B_NORMALIZED_VERIFIER,
        c4u18_baseline=[_c4u18_unit(), _c4u18_lo(topic="unit-1")],
        fractions_baseline=[],
        verifier=_AcceptAll(),
    )
    assert run_pipeline(**kwargs)["final_recommendation"] == run_pipeline(**kwargs)[
        "final_recommendation"
    ]


def test_attribution_mapper_changed():
    attr = compute_attribution(
        pipeline=Pipeline.C_RAW_VERIFIER_MAPPER,
        fixture_class="FAITHFUL_IMPERFECT",
        raw_verifier_accepted=False,
        normalized_verifier_accepted=False,
        final_accepted=True,
        mapping_applied=True,
        unsupported=False,
        placeholder=False,
    )
    assert attr == "MAPPER_CHANGED_DECISION"


def test_v210_flag_default_off():
    settings = Settings()
    qa = CurriculumQAState.initial(question="q")
    assert v210_experiment_enabled(settings, qa) is False


def test_four_pipelines_defined():
    assert len(PIPELINES) == 4
