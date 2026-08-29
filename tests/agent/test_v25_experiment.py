"""V2.5 verifier evidence-quality experiment harness tests."""

from __future__ import annotations

from app.agent.state import CurriculumQAState
from app.agent.v25_experiment import (
    _CLEAN_PLACEHOLDER,
    apply_v25_evidence_transform,
    build_evidence_inventory,
    configure_v25_experiment,
    evidence_condition_for_arm,
    finalize_v25_diagnostics,
    transform_evidence_for_condition,
)
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence


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


def test_arm_conditions():
    assert evidence_condition_for_arm("A") == "clean"
    assert evidence_condition_for_arm("B") == "original_imperfect"
    assert evidence_condition_for_arm("C") == "clean_annotated"
    assert evidence_condition_for_arm("D") == "original_annotated"


def test_clean_transform_only_in_harness():
    evidence = [_clean_lo(), _imperfect_lo()]
    original_content = evidence[1].content
    clean = transform_evidence_for_condition(evidence, condition="clean")
    assert _CLEAN_PLACEHOLDER in clean[1].content
    assert evidence[1].content == original_content
    assert _CLEAN_PLACEHOLDER not in evidence[1].content


def test_annotation_preserves_source_wording():
    evidence = [_imperfect_lo()]
    annotated = transform_evidence_for_condition(
        evidence, condition="original_annotated"
    )
    assert annotated[0].content == evidence[0].content
    eq = annotated[0].metadata.get("evidence_quality")
    assert eq["status"] == "truncated"
    assert eq["original_text_present"] is True
    assert eq["source_record_id"] == "C4U06-LO02"


def test_apply_transform_records_original_hash():
    settings = Settings()
    state = CurriculumQAState.initial(question="q")
    configure_v25_experiment(state, settings=settings, arm="A")
    state.evidence = [_clean_lo(), _imperfect_lo()]
    apply_v25_evidence_transform(state)
    assert state.metadata["v25_baseline_evidence_hash"]
    assert state.metadata["v25_transformed_evidence_hash"]
    assert state.metadata["v25_baseline_evidence_hash"] != state.metadata[
        "v25_transformed_evidence_hash"
    ]
    assert _CLEAN_PLACEHOLDER in state.evidence[1].content


def test_original_arm_b_keeps_byte_equivalent_content():
    evidence = [_clean_lo(), _imperfect_lo()]
    original = transform_evidence_for_condition(
        evidence, condition="original_imperfect"
    )
    assert original[1].content == evidence[1].content


def test_inventory_detects_imperfect_records():
    inventory = build_evidence_inventory([_clean_lo(), _imperfect_lo()])
    assert len(inventory) == 1
    assert inventory[0]["lo_code"] == "C4U06-LO02"
    assert inventory[0]["quality_status"] == "GARBLED"


def test_finalize_records_evidence_condition():
    settings = Settings()
    state = CurriculumQAState.initial(question="q")
    configure_v25_experiment(state, settings=settings, arm="D")
    state.evidence = [_imperfect_lo()]
    apply_v25_evidence_transform(state)
    diag = finalize_v25_diagnostics(state)
    assert diag["evidence_condition"] == "original_annotated"
    assert diag["imperfect_evidence_count"] == 1
    assert state.metadata["v25_diagnostics"]["arm"] == "D"


def test_counterfactual_replay_uses_same_answer():
    from app.agent.v25_experiment import build_counterfactual_pair
    from app.schemas.verification import VerificationRecommendation, VerificationResult

    class FakeVerifier:
        def verify(self, state, request_id=None):
            content = " ".join(
                (item.content or "") for item in state.evidence if item.content
            )
            passed = _CLEAN_PLACEHOLDER in content
            return VerificationResult(
                passed=passed,
                score=0.95 if passed else 0.7,
                recommendation=VerificationRecommendation.ACCEPT
                if passed
                else VerificationRecommendation.RETRIEVE_MORE,
            )

    state = CurriculumQAState.initial(question="q")
    configure_v25_experiment(state, settings=Settings(), arm="B")
    state.evidence = [_clean_lo(), _imperfect_lo()]
    apply_v25_evidence_transform(state)
    answer = "C4U06-LO02 lists garbled wording."
    state.final_answer = answer
    pair = build_counterfactual_pair(state, verifier=FakeVerifier(), request_id="cf-test")
    assert not pair.get("skipped")
    assert pair["clean_evidence_accepted"] is True
    assert pair["original_evidence_accepted"] is False
    assert pair["acceptance_delta"] == 1
