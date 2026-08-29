"""V2.4 routing / verifier isolation instrumentation tests."""

from __future__ import annotations

from app.agent.context_boundary import ContextBoundarySnapshot
from app.agent.evidence_snapshot import evidence_snapshot_hash
from app.agent.retrieval_state import RetrievalState
from app.agent.state import CurriculumQAState
from app.agent.v24_diagnostics import (
    classify_retrieval_delta,
    configure_v24_experiment,
    evidence_record_from_state,
    frozen_retrieval_enabled,
    record_retrieval_delta,
    single_pass_enabled,
)
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence
from app.schemas.verification import (
    VerificationRecommendation,
    VerificationResult,
)


def _evidence() -> list[CurriculumEvidence]:
    return [
        CurriculumEvidence(
            entity_type="learning_outcome",
            entity_id="lo-1",
            name="C4U04-LO01",
            content="Simplify like fraction with common denominators.",
            metadata={"code": "C4U04-LO01"},
        ),
        CurriculumEvidence(
            entity_type="learning_outcome",
            entity_id="lo-garbled",
            name="C4U06-LO02",
            content="Multiply like fractions with denominators up to multiply like fractions",
            metadata={"code": "C4U06-LO02"},
        ),
    ]


def test_arm_a_and_b_share_frozen_retrieval_flag():
    settings = Settings()
    a = CurriculumQAState.initial(question="q")
    b = CurriculumQAState.initial(question="q")
    configure_v24_experiment(a, settings=settings, arm="A")
    configure_v24_experiment(b, settings=settings, arm="B")
    assert frozen_retrieval_enabled(settings, a)
    assert frozen_retrieval_enabled(settings, b)
    assert single_pass_enabled(a)
    assert not single_pass_enabled(b)


def test_arm_c_live_single_pass():
    settings = Settings()
    state = CurriculumQAState.initial(question="q")
    configure_v24_experiment(state, settings=settings, arm="C")
    assert not frozen_retrieval_enabled(settings, state)
    assert single_pass_enabled(state)


def test_frozen_evidence_hash_stable():
    ev = _evidence()
    assert evidence_snapshot_hash(ev) == evidence_snapshot_hash(list(reversed(ev)))


def test_frozen_arms_share_evidence_hash():
    """Arm A and B use identical frozen evidence snapshots."""
    settings = Settings()
    evidence = _evidence()
    a = CurriculumQAState.initial(question="q")
    b = CurriculumQAState.initial(question="q")
    configure_v24_experiment(a, settings=settings, arm="A")
    configure_v24_experiment(b, settings=settings, arm="B")
    a.evidence = list(evidence)
    b.evidence = list(evidence)
    assert evidence_snapshot_hash(a.evidence) == evidence_snapshot_hash(b.evidence)


def test_retrieve_more_records_evidence_already_present():
    settings = Settings()
    state = CurriculumQAState.initial(question="q")
    configure_v24_experiment(state, settings=settings, arm="B")
    state.evidence = _evidence()
    state.retrieval_state = RetrievalState(
        context_boundary=ContextBoundarySnapshot(
            context_resolved=True,
            learning_outcome_ids=["lo-1", "lo-garbled"],
            lo_codes=["C4U04-LO01", "C4U06-LO02"],
        )
    )
    result = VerificationResult(
        passed=False,
        score=0.7,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        unsupported_claims=["C4U06-LO02 — garbled source wording repeated"],
    )
    state.verification = result
    from app.agent.v24_diagnostics import classify_evidence_presence

    assert classify_evidence_presence(state, result=result) == (
        "EVIDENCE_PRESENT_BUT_IMPERFECT"
    )


def test_retrieval_delta_classifies_no_new_evidence():
    assert (
        classify_retrieval_delta(
            evidence_before_count=13,
            evidence_after_count=13,
            new_evidence_count=0,
            duplicate_evidence_count=2,
            new_evidence_ids=[],
        )
        == "DUPLICATE_ONLY"
    )


def test_retrieval_delta_recorded_on_state():
    state = CurriculumQAState.initial(question="q")
    state.evidence = _evidence()
    before_hash = evidence_snapshot_hash(state.evidence)
    delta = record_retrieval_delta(
        state,
        evidence_before_count=2,
        evidence_hash_before=before_hash,
        new_evidence_count=0,
        duplicate_evidence_count=1,
        new_evidence_ids=[],
    )
    assert delta["evidence_changed"] is False
    assert delta["retrieval_result_class"] == "DUPLICATE_ONLY"
    assert state.metadata["v24_retrieval_deltas"]


def test_evidence_record_includes_lo_codes():
    state = CurriculumQAState.initial(question="q")
    state.evidence = _evidence()
    record = evidence_record_from_state(state)
    assert "C4U04-LO01" in record["learning_outcome_codes"]
    assert record["learning_outcome_count"] == 2


def test_routing_sequence_captures_terminal_failure():
    from app.agent.v24_diagnostics import build_routing_sequence

    state = CurriculumQAState.initial(question="q")
    state.evidence = _evidence()
    state.metadata["v24_experiment_arm"] = "B"
    state.metadata["termination_reason"] = "no_retrieval_progress"
    state.verification = VerificationResult(
        passed=False,
        score=0.7,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        issues=["Garbled C4U06-LO02 text"],
    )
    state.verification_history = [state.verification]
    seq = build_routing_sequence(state)
    assert seq["retrieve_more_requested"] is True
    assert seq["final_failure_reason"] == "no_retrieval_progress"
