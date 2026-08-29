"""V2.6 verifier evidence-state isolation harness tests."""

from __future__ import annotations

from app.agent.state import CurriculumQAState
from app.agent.v26_experiment import (
    V26_EVIDENCE_STATE_INSTRUCTION,
    answer_hash,
    get_v26_system_prompt_suffix,
    prepare_evidence_for_arm,
    replay_verifier_for_arm,
)
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


def test_evidence_state_classifications_per_arm():
    b = prepare_evidence_for_arm(_baseline(), "B")
    imperfect = next(e for e in b if (e.metadata or {}).get("code") == "C4U06-LO02")
    assert imperfect.metadata["evidence_state"] == "EVIDENCE_PRESENT_IMPERFECT"

    c = prepare_evidence_for_arm(_baseline(), "C")
    assert all(
        (e.metadata or {}).get("evidence_state") == "EVIDENCE_PRESENT_COMPLETE"
        for e in c
        if e.entity_type == "learning_outcome"
    )

    d = prepare_evidence_for_arm(_baseline(), "D")
    assert not any((e.metadata or {}).get("code") == "C4U06-LO02" for e in d)


def test_arm_a_preserves_original_content():
    a = prepare_evidence_for_arm(_baseline(), "A")
    b = _baseline()
    assert a[1].content == b[1].content
    assert "evidence_state" not in (a[1].metadata or {})


def test_experimental_prompt_isolated_behind_flag():
    off = CurriculumQAState.initial(question="q")
    assert get_v26_system_prompt_suffix(off) == ""

    on = CurriculumQAState.initial(question="q")
    on.metadata["v26_verifier_replay"] = True
    on.metadata["v26_experiment_arm"] = "B"
    assert V26_EVIDENCE_STATE_INSTRUCTION.split()[0] in get_v26_system_prompt_suffix(on)

    arm_a = CurriculumQAState.initial(question="q")
    arm_a.metadata["v26_verifier_replay"] = True
    arm_a.metadata["v26_experiment_arm"] = "A"
    assert get_v26_system_prompt_suffix(arm_a) == ""


def test_imperfect_not_auto_accepted():
    class StrictVerifier:
        def verify(self, state, request_id=None):
            return VerificationResult(
                passed=False,
                score=0.5,
                recommendation=VerificationRecommendation.RETRIEVE_MORE,
                unsupported_claims=["garbled wording"],
            )

    row = replay_verifier_for_arm(
        question="q",
        answer="C4U06-LO02 — garbled text repeated.",
        baseline_evidence=_baseline(),
        arm="B",
        verifier=StrictVerifier(),
    )
    assert not row["verifier_accepted"]


def test_unsupported_claim_rejected_arm_b():
    class RejectVerifier:
        def verify(self, state, request_id=None):
            return VerificationResult(
                passed=False,
                score=0.2,
                recommendation=VerificationRecommendation.FALLBACK,
                unsupported_claims=["C4U99-LO01"],
            )

    row = replay_verifier_for_arm(
        question="q",
        answer="C4U99-LO01 — Divide mixed fractions.",
        baseline_evidence=_baseline(),
        arm="B",
        verifier=RejectVerifier(),
    )
    assert not row["verifier_accepted"]


def test_truncation_reconstruction_classified():
    from app.agent.v26_experiment import classify_v26_claim
    from app.schemas.verification import VerificationResult

    result = VerificationResult(passed=False, unsupported_claims=["reconstructed"])
    cls = classify_v26_claim(
        "C4U06-LO02 — denominators up to 12",
        answer="C4U06-LO02 — Multiply like fractions with denominators up to 12.",
        result=result,
        evidence_state="EVIDENCE_PRESENT_IMPERFECT",
    )
    assert cls == "TRUNCATION_RECONSTRUCTION"


def test_faithful_truncation_classified():
    from app.agent.v26_experiment import classify_v26_claim
    from app.schemas.verification import VerificationResult

    answer = (
        "C4U06-LO02 — garbled text. Note: reported as supplied rather than reconstructed."
    )
    result = VerificationResult(passed=True, score=0.9)
    cls = classify_v26_claim(
        "C4U06-LO02 — garbled text",
        answer=answer,
        result=result,
        evidence_state="EVIDENCE_PRESENT_IMPERFECT",
    )
    assert cls == "TRUNCATION_FAITHFUL"


def test_missing_arm_removes_imperfect_lo():
    d = prepare_evidence_for_arm(_baseline(), "D")
    assert len(d) == 1
    assert d[0].metadata.get("code") == "C4U05-LO01"


def test_answer_hash_stable():
    assert answer_hash("same") == answer_hash("same")
    assert answer_hash("a") != answer_hash("b")


def test_production_settings_flag_default_off():
    assert Settings().v26_verifier_evidence_state_experiment is False


def test_no_database_mutation_is_harness_only():
    original = _baseline()
    prepare_evidence_for_arm(original, "C")
    assert original[1].content == _imperfect_lo().content
