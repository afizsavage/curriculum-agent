"""V2.11 metadata-integrity experiment tests."""

from __future__ import annotations

import copy

from app.agent.state import CurriculumQAState
from app.agent.v25_experiment import _CLEAN_PLACEHOLDER
from app.agent.v211_metadata_integrity import (
    ALL_FIXTURES,
    PipelineVariant,
    apply_metadata_policy,
    run_pipeline,
    validate_metadata_integrity,
    v211_experiment_enabled,
)
from app.agent.v29_evidence_normalization import NormalizationVariant, normalize_evidence
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.schemas.verification import VerificationRecommendation, VerificationResult


def _c4u18_unit() -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="unit",
        entity_id="unit-1",
        name="Everyday Arithmetic Money",
        content="Everyday Arithmetic Money",
        subject="MATHEMATICS",
        grade="CLASS_4",
        metadata={"code": "C4-U18"},
    )


def _c4u18_lo(*, topic: str = "unit-1", subject: str = "MATHEMATICS", grade: str = "CLASS_4") -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id="lo1",
        name="C4U18-LO01",
        content="Order operations using BODMAS.",
        topic=topic,
        grade=grade,
        subject=subject,
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


class _RetrieveMissing:
    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=False,
            score=0.2,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            missing_evidence=[{"type": "learning_outcome"}],
        )


def _normalize(evidence: list[CurriculumEvidence]) -> list[CurriculumEvidence]:
    return normalize_evidence(evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION).evidence


def test_valid_hierarchy_passes():
    result = validate_metadata_integrity(_normalize([_c4u18_unit(), _c4u18_lo()]))
    assert result.valid is True
    assert not result.violations


def test_c4u18_topic_uuid_resolves():
    result = validate_metadata_integrity(_normalize([_c4u18_unit(), _c4u18_lo(topic="unit-1")]))
    assert result.valid is True
    rel = next(r for r in result.resolved_relationships if r.lo_code == "C4U18-LO01")
    assert rel.topic_resolved == "Everyday Arithmetic Money"


def test_unresolved_uuid_fails():
    evidence = _normalize([_c4u18_lo(topic="00000000-0000-0000-0000-000000000099")])
    result = validate_metadata_integrity(evidence)
    assert result.valid is False
    assert any(v.violation_type == "unresolvable_topic_uuid" for v in result.violations)


def test_conflicting_parent_fails():
    lo = _c4u18_lo()
    lo.metadata = dict(lo.metadata or {})
    lo.metadata["parent_content_name"] = "Conflicting Unit A"
    result = validate_metadata_integrity(_normalize([_c4u18_unit(), lo]))
    assert result.valid is False
    assert any(v.violation_type == "conflicting_parent" for v in result.violations)


def test_conflicting_subject_fails():
    lo_a = _c4u18_lo()
    lo_b = copy.deepcopy(lo_a)
    lo_b.entity_id = "lo-conflict"
    lo_b.subject = "ENGLISH"
    result = validate_metadata_integrity(_normalize([_c4u18_unit(), lo_a, lo_b]))
    assert result.valid is False
    assert any(v.violation_type == "conflicting_subject" for v in result.violations)


def test_conflicting_grade_fails():
    lo_a = _c4u18_lo()
    lo_b = copy.deepcopy(lo_a)
    lo_b.entity_id = "lo-conflict"
    lo_b.grade = "CLASS_5"
    result = validate_metadata_integrity(_normalize([_c4u18_unit(), lo_a, lo_b]))
    assert result.valid is False
    assert any(v.violation_type == "conflicting_grade" for v in result.violations)


def test_placeholder_parent_fails():
    unit = _c4u18_unit()
    unit.content = _CLEAN_PLACEHOLDER
    unit.name = _CLEAN_PLACEHOLDER
    result = validate_metadata_integrity(_normalize([unit, _c4u18_lo()]))
    assert result.valid is False
    assert any(v.violation_type == "placeholder_parent" for v in result.violations)


def test_placeholder_topic_fails():
    lo = _c4u18_lo()
    lo.topic = _CLEAN_PLACEHOLDER
    result = validate_metadata_integrity(_normalize([_c4u18_unit(), lo]))
    assert result.valid is False
    assert any(v.violation_type == "placeholder_topic" for v in result.violations)


def test_parent_child_mismatch_fails():
    lo = _c4u18_lo()
    lo.metadata = dict(lo.metadata or {})
    lo.metadata["parent_content_name"] = "Wrong Parent Unit"
    lo.metadata["parent_content_code"] = "C4-U99"
    result = validate_metadata_integrity(_normalize([_c4u18_unit(), lo]))
    assert result.valid is False
    assert any(v.violation_type == "parent_child_mismatch" for v in result.violations)


def test_wrong_subject_fails():
    lo = _c4u18_lo(subject="ENGLISH")
    result = validate_metadata_integrity(_normalize([_c4u18_unit(), lo]))
    assert result.valid is False
    assert any(v.violation_type == "subject_topic_mismatch" for v in result.violations)


def test_wrong_grade_fails():
    lo = _c4u18_lo(grade="CLASS_5")
    result = validate_metadata_integrity(_normalize([_c4u18_unit(), lo]))
    assert result.valid is False
    assert any(v.violation_type == "grade_topic_mismatch" for v in result.violations)


def test_validator_is_answer_independent():
    evidence = _normalize([_c4u18_unit(), _c4u18_lo(subject="ENGLISH")])
    r1 = validate_metadata_integrity(evidence)
    r2 = validate_metadata_integrity(evidence)
    assert r1.valid == r2.valid
    assert [v.violation_type for v in r1.violations] == [v.violation_type for v in r2.violations]


def test_validator_is_deterministic():
    evidence = _normalize([_c4u18_unit(), _c4u18_lo()])
    assert validate_metadata_integrity(evidence).to_dict() == validate_metadata_integrity(evidence).to_dict()


def test_valid_unrelated_evidence_survives_suppression():
    unit = _c4u18_unit()
    lo_valid = _c4u18_lo()
    lo_valid.entity_id = "lo-valid"
    lo_valid.metadata = dict(lo_valid.metadata or {})
    lo_valid.metadata["code"] = "C4U18-LO02"
    lo_valid.name = "C4U18-LO02"
    lo_bad = _c4u18_lo(subject="ENGLISH")
    lo_bad.entity_id = "lo-bad"
    evidence = _normalize([unit, lo_valid, lo_bad])
    integrity = validate_metadata_integrity(evidence)
    suppressed, blocked, _ = apply_metadata_policy(
        evidence, integrity, variant=PipelineVariant.C_METADATA_SUPPRESS
    )
    assert blocked is True
    assert any(e.entity_id == "lo-valid" for e in suppressed)
    assert not any(e.entity_id == "lo-bad" for e in suppressed)


def test_raw_evidence_not_mutated_by_validator():
    raw = [_c4u18_unit(), _c4u18_lo(subject="ENGLISH")]
    before = copy.deepcopy(raw)
    validate_metadata_integrity(_normalize(raw))
    assert raw == before


def test_c4u18_fc_pipeline_remains_valid():
    row = run_pipeline(
        fixture_class="FAITHFUL_COMPLETE",
        variant=PipelineVariant.C_METADATA_SUPPRESS,
        c4u18_baseline=[_c4u18_unit(), _c4u18_lo(topic="unit-1")],
        fractions_baseline=[],
        verifier=_AcceptAll(),
    )
    assert row["metadata_integrity"]["valid"] is True
    assert row["final_accepted"] is True


def test_high_score_invalid_metadata_blocked():
    row = run_pipeline(
        fixture_class="ADV_WRONG_SUBJECT",
        variant=PipelineVariant.C_METADATA_SUPPRESS,
        c4u18_baseline=[_c4u18_unit(), _c4u18_lo()],
        fractions_baseline=[],
        verifier=_AcceptAll(),
    )
    assert row["metadata_integrity"]["valid"] is False
    assert row["final_accepted"] is False


def test_mapper_cannot_bypass_metadata_failure():
    row = run_pipeline(
        fixture_class="ADV_CONFLICTING_PARENT",
        variant=PipelineVariant.B_METADATA_VALIDATE,
        c4u18_baseline=[_c4u18_unit(), _c4u18_lo()],
        fractions_baseline=[],
        verifier=_AcceptAll(),
    )
    assert row["mapped_accepted"] is True
    assert row["metadata_blocked"] is True
    assert row["final_accepted"] is False


def test_missing_evidence_remains_insufficient():
    row = run_pipeline(
        fixture_class="MISSING_EVIDENCE",
        variant=PipelineVariant.C_METADATA_SUPPRESS,
        c4u18_baseline=[],
        fractions_baseline=[],
        verifier=_RetrieveMissing(),
    )
    assert row["final_accepted"] is False
    assert row["final_recommendation"] in {"insufficient_evidence", "retrieve_more", "reject"}


def test_experiment_flag_default_off():
    settings = Settings()
    qa = CurriculumQAState.initial(question="q")
    assert v211_experiment_enabled(settings, qa) is False


def test_fi_pipeline_with_valid_metadata():
    row = run_pipeline(
        fixture_class="FAITHFUL_IMPERFECT",
        variant=PipelineVariant.C_METADATA_SUPPRESS,
        c4u18_baseline=[_c4u18_unit()],
        fractions_baseline=[_c4u18_lo()],
        verifier=_HighRetrieve(),
    )
    assert row["metadata_integrity"]["valid"] is True
    assert row["final_accepted"] is True


def test_topic_uuid_collision_fails():
    lo_a = _c4u18_lo()
    lo_b = copy.deepcopy(lo_a)
    lo_b.entity_id = "collision-lo-b"
    lo_b.metadata = dict(lo_b.metadata or {})
    lo_b.metadata["parent_content_name"] = "Colliding Unit Label"
    lo_b.metadata["parent_content_code"] = "C4-U99"
    result = validate_metadata_integrity(_normalize([_c4u18_unit(), lo_a, lo_b]))
    assert result.valid is False
    assert any(v.violation_type == "topic_uuid_collision" for v in result.violations)


def test_adversarial_fixtures_have_modes():
    from app.agent.v211_metadata_integrity import ADVERSARIAL_FIXTURE_CLASSES

    for fixture in ADVERSARIAL_FIXTURE_CLASSES:
        assert fixture in ALL_FIXTURES
