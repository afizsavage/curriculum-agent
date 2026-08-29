"""V2.9 evidence normalization harness tests."""

from __future__ import annotations

import copy

from app.agent.state import CurriculumQAState
from app.agent.v25_experiment import _CLEAN_PLACEHOLDER
from app.agent.v29_evidence_normalization import (
    NormalizationVariant,
    classify_substance,
    detect_placeholder_content,
    normalize_evidence,
    replay_fixture,
    v29_experiment_enabled,
)
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence
from app.schemas.verification import VerificationRecommendation, VerificationResult


def _substantive_lo() -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id="lo1",
        name="C4U18-LO01",
        content="Order operations using BODMAS.",
        metadata={"code": "C4U18-LO01", "parent_content_name": "Everyday Arithmetic Money"},
    )


def _placeholder_lo() -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id="lo2",
        name="C4U04-LO04",
        content=f"C4U04-LO04 — {_CLEAN_PLACEHOLDER}",
        metadata={"code": "C4U04-LO04"},
    )


def _unit() -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="unit",
        entity_id="unit-1",
        name="Everyday Arithmetic Money",
        content="Everyday Arithmetic Money",
        metadata={"code": "C4-U18"},
    )


class _RejectUnsupported:
    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=False,
            score=0.0,
            recommendation=VerificationRecommendation.FALLBACK,
            unsupported_claims=["bad claim"],
        )


class _RetrieveMissing:
    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=False,
            score=0.2,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            missing_evidence=[{"type": "learning_outcome"}],
        )


def test_placeholder_sentinel_detected():
    detected, classification = detect_placeholder_content(_CLEAN_PLACEHOLDER)
    assert detected is True
    assert classification == "sentinel"


def test_placeholder_does_not_become_substantive():
    result = normalize_evidence([_placeholder_lo()], NormalizationVariant.PLACEHOLDER_FILTER)
    assert result.evidence == []
    assert result.diagnostics.placeholder_filtered == 1


def test_empty_evidence_remains_empty():
    result = normalize_evidence([], NormalizationVariant.SEMANTIC_EVIDENCE_EXTRACTION)
    assert result.evidence == []
    assert result.diagnostics.records_out == 0


def test_substantive_evidence_survives_normalization():
    evidence = [_unit(), _substantive_lo()]
    result = normalize_evidence(evidence, NormalizationVariant.SEMANTIC_EVIDENCE_EXTRACTION)
    assert len(result.evidence) == 2
    assert any(_substantive_lo().content == item.content for item in result.evidence)


def test_normalization_preserves_substantive_text():
    raw = [_substantive_lo()]
    result = normalize_evidence(raw, NormalizationVariant.STRUCTURAL_NORMALIZATION)
    assert result.evidence[0].content == "Order operations using BODMAS."


def test_normalization_is_deterministic():
    evidence = [_unit(), _substantive_lo(), _placeholder_lo()]
    first = normalize_evidence(evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION)
    second = normalize_evidence(evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION)
    assert first.diagnostics.evidence_hash_out == second.diagnostics.evidence_hash_out


def test_normalization_is_idempotent():
    evidence = [_unit(), _substantive_lo(), _placeholder_lo()]
    once = normalize_evidence(evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION)
    twice = normalize_evidence(once.evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION)
    assert once.diagnostics.evidence_hash_out == twice.diagnostics.evidence_hash_out


def test_normalization_does_not_depend_on_answer():
    evidence = [_unit(), _substantive_lo()]
    result_a = normalize_evidence(evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION)
    result_b = normalize_evidence(evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION)
    assert result_a.diagnostics.evidence_hash_out == result_b.diagnostics.evidence_hash_out


def test_unsupported_claim_fixture_still_rejected_by_verifier_stub():
    row = replay_fixture(
        fixture_class="UNSUPPORTED_CLAIM",
        variant=NormalizationVariant.RAW,
        c4u18_baseline=[_unit(), _substantive_lo()],
        fractions_baseline=[],
        verifier=_RejectUnsupported(),
    )
    assert row["verifier_accepted"] is False


def test_unsupported_absence_fixture_still_rejected():
    row = replay_fixture(
        fixture_class="UNSUPPORTED_ABSENCE",
        variant=NormalizationVariant.RAW,
        c4u18_baseline=[_unit(), _substantive_lo()],
        fractions_baseline=[],
        verifier=_RejectUnsupported(),
    )
    assert row["verifier_accepted"] is False


def test_speculation_fixture_still_rejected():
    row = replay_fixture(
        fixture_class="SPECULATIVE",
        variant=NormalizationVariant.RAW,
        c4u18_baseline=[_unit(), _substantive_lo()],
        fractions_baseline=[],
        verifier=_RejectUnsupported(),
    )
    assert row["verifier_accepted"] is False


def test_reconstruction_fixture_still_rejected():
    row = replay_fixture(
        fixture_class="RECONSTRUCTION",
        variant=NormalizationVariant.RAW,
        c4u18_baseline=[_unit(), _substantive_lo()],
        fractions_baseline=[_placeholder_lo()],
        verifier=_RejectUnsupported(),
    )
    assert row["verifier_accepted"] is False


def test_missing_evidence_remains_insufficient_or_retrieve():
    row = replay_fixture(
        fixture_class="MISSING_EVIDENCE",
        variant=NormalizationVariant.RAW,
        c4u18_baseline=[_unit(), _substantive_lo()],
        fractions_baseline=[],
        verifier=_RetrieveMissing(),
    )
    assert row["verifier_accepted"] is False
    assert row["retrieve_more_requested"] or row["insufficient_evidence"]


def test_placeholder_cannot_be_accepted_through_normalization():
    evidence = [_placeholder_lo()]
    for variant in NormalizationVariant:
        normalized = normalize_evidence(evidence, variant)
        if variant in {
            NormalizationVariant.PLACEHOLDER_FILTER,
            NormalizationVariant.SEMANTIC_EVIDENCE_EXTRACTION,
        }:
            assert normalized.evidence == []
        elif variant == NormalizationVariant.RAW:
            assert len(normalized.evidence) == 1
            assert classify_substance(normalized.evidence[0]) == "NON_SUBSTANTIVE"
        else:
            assert all(
                classify_substance(item) != "SUBSTANTIVE" for item in normalized.evidence
            )


def test_raw_evidence_not_mutated():
    evidence = [_substantive_lo(), _placeholder_lo()]
    original = copy.deepcopy(evidence)
    normalize_evidence(evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION)
    assert evidence[0].content == original[0].content
    assert evidence[1].content == original[1].content


def test_normalization_diagnostics_are_deterministic():
    evidence = [_unit(), _substantive_lo()]
    first = normalize_evidence(evidence, NormalizationVariant.SEMANTIC_EVIDENCE_EXTRACTION)
    second = normalize_evidence(evidence, NormalizationVariant.SEMANTIC_EVIDENCE_EXTRACTION)
    assert first.diagnostics.to_dict() == second.diagnostics.to_dict()


def test_v29_experiment_flag_default_off():
    settings = Settings()
    qa = CurriculumQAState.initial(question="q")
    assert v29_experiment_enabled(settings, qa) is False


def test_structural_normalization_resolves_topic_uuid():
    unit = _unit()
    lo = _substantive_lo()
    lo.topic = unit.entity_id
    result = normalize_evidence([unit, lo], NormalizationVariant.STRUCTURAL_NORMALIZATION)
    lo_out = next(item for item in result.evidence if item.entity_type == "learning_outcome")
    assert lo_out.topic == "Everyday Arithmetic Money"
