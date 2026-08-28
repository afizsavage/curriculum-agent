"""V2.2 context boundary experiment helpers."""

from __future__ import annotations

from app.agent.context_boundary import (
    ContextBoundarySnapshot,
    capture_context_boundary,
    is_redundant_legacy_retrieval,
    missing_evidence_covered_by_boundary,
)
from app.curriculum.evidence import CurriculumEvidence
from app.schemas.verification import MissingEvidenceRequest


def _boundary() -> ContextBoundarySnapshot:
    return ContextBoundarySnapshot(
        context_resolved=True,
        curriculum_id="cur-1",
        curriculum_version="2020",
        curriculum_version_id="cur-1:2020",
        grade_id="g4",
        grade_code="CLASS_4",
        subject_id="math",
        subject_code="MATHEMATICS",
        unit_ids=["u4", "u5", "u6"],
        unit_codes=["C4-U04", "C4-U05", "C4-U06"],
        learning_outcome_ids=[f"lo-{i}" for i in range(10)],
        lo_codes=[f"C4U0{i}-LO01" for i in range(4, 7)],
        resolution_status="resolved",
    )


def test_capture_context_boundary_from_resolve_payload():
    snap = capture_context_boundary(
        {
            "resolution": {"status": "resolved"},
            "curriculum": {"id": "cur-1", "version": "2020"},
            "grade": {"id": "g4", "code": "CLASS_4"},
            "subject": {"id": "math", "code": "MATHEMATICS"},
            "units": [{"id": "u4", "code": "C4-U04", "name": "Fractions"}],
            "learning_outcomes": [
                {"id": "lo-1", "code": "C4U04-LO01", "description": "Identify fractions."}
            ],
        }
    )
    assert snap is not None
    assert snap.context_resolved
    assert snap.unit_codes == ["C4-U04"]
    assert snap.lo_codes == ["C4U04-LO01"]


def test_missing_evidence_covered_for_lo_gap():
    boundary = _boundary()
    pending = [
        MissingEvidenceRequest(
            type="learning_objective",
            query="Need clearer support for C4U06-LO02",
        )
    ]
    evidence = [
        CurriculumEvidence(
            entity_type="learning_outcome",
            entity_id="lo-6",
            name="C4U06-LO02",
            content="Represent fractions.",
            metadata={"code": "C4U06-LO02"},
        )
    ]
    assert missing_evidence_covered_by_boundary(
        pending, boundary=boundary, evidence=evidence
    )


def test_missing_evidence_not_covered_for_lesson_plan():
    boundary = _boundary()
    pending = [MissingEvidenceRequest(query="Need the lesson plan covering C4-U06")]
    assert not missing_evidence_covered_by_boundary(
        pending, boundary=boundary, evidence=[]
    )


def test_redundant_legacy_retrieval_after_boundary():
    boundary = _boundary()
    assert is_redundant_legacy_retrieval(
        "get_learning_objectives",
        {"grade": "CLASS_4", "subject": "MATHEMATICS", "topic": "C4-U06"},
        boundary=boundary,
        evidence=[],
    )
    assert is_redundant_legacy_retrieval(
        "search_curriculum",
        {"query": "fractions", "grade": "CLASS_4", "subject": "MATHEMATICS"},
        boundary=boundary,
        evidence=[],
    )
    assert not is_redundant_legacy_retrieval(
        "resolve_curriculum_context",
        {"grade": "CLASS_4", "subject": "MATHEMATICS"},
        boundary=boundary,
        evidence=[],
    )
