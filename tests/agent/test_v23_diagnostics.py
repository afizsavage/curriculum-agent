"""V2.3 evidence snapshot and diagnostic helpers."""

from __future__ import annotations

from app.agent.evidence_snapshot import evidence_snapshot_hash
from app.agent.state import CurriculumQAState
from app.agent.v23_diagnostics import (
    classify_v23_failure,
    configure_v23_experiment,
    detect_speculative_answer,
)
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence
from app.schemas.verification import (
    MissingEvidenceRequest,
    VerificationRecommendation,
    VerificationResult,
)


def test_evidence_snapshot_hash_stable():
    evidence = [
        CurriculumEvidence(
            entity_type="learning_outcome",
            entity_id="lo-1",
            name="LO1",
            content="Identify fractions.",
            metadata={"code": "C4U04-LO01"},
        ),
        CurriculumEvidence(
            entity_type="unit",
            entity_id="u-1",
            name="Fractions",
            metadata={"code": "C4-U04"},
        ),
    ]
    assert evidence_snapshot_hash(evidence) == evidence_snapshot_hash(list(reversed(evidence)))


def test_configure_v23_sets_flags():
    state = CurriculumQAState.initial(question="q")
    configure_v23_experiment(
        state,
        settings=Settings(),
        generation_mode="constrained",
        enabled=True,
    )
    assert state.metadata["v23_diagnostic_experiment"] is True
    assert state.metadata["context_boundary_experiment"] is True
    assert state.metadata["generation_mode"] == "constrained"


def test_speculative_detection():
    assert detect_speculative_answer("The objective likely means addition.")
    assert not detect_speculative_answer("C4U04-LO01: Identify fractions.")


def test_classify_verifier_missing_when_evidence_present():
    from app.agent.context_boundary import ContextBoundarySnapshot
    from app.agent.retrieval_state import RetrievalState

    state = CurriculumQAState.initial(question="q")
    state.retrieval_state = RetrievalState(
        context_boundary=ContextBoundarySnapshot(
            context_resolved=True,
            learning_outcome_ids=["lo-1"],
            lo_codes=["C4U04-LO01"],
            unit_codes=["C4-U04"],
        )
    )
    state.evidence = [
        CurriculumEvidence(
            entity_type="learning_outcome",
            entity_id="lo-1",
            content="Identify fractions.",
            metadata={"code": "C4U04-LO01"},
        )
    ]
    result = VerificationResult(
        passed=False,
        score=0.4,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        missing_evidence=[
            MissingEvidenceRequest(query="Need LO C4U04-LO01 wording")
        ],
    )
    assert classify_v23_failure(state, result=result) == "VERIFIER_MISSING_EVIDENCE"
