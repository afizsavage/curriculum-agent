"""A requested unit code can be absent only from a complete checked catalogue."""

from __future__ import annotations

from app.agent.graph_nodes import apply_fallback
from app.agent.graph_routing import route_after_verification
from app.agent.state import CurriculumQAState
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence
from app.curriculum.structure_scope import codes_absent_from_checked_catalogue
from app.enums import AgentStatus
from app.schemas.verification import (
    MissingEvidenceRequest,
    VerificationRecommendation,
    VerificationResult,
)

QUESTION = "Is C4-U12 a published Class 4 Mathematics unit?"
HEDGE = (
    "The supplied Class 4 Mathematics evidence does not include a unit with "
    "the code C4-U12. Because some records were omitted, this answer cannot "
    "confirm whether such a unit exists elsewhere."
)


def _settings() -> Settings:
    return Settings(_env_file=None, curriculum_api_base_url="http://curriculum.test")


def _unit(
    code: str,
    name: str,
    *,
    subject: str = "MATHEMATICS",
    exhaustive: bool = True,
    grade_curriculum_id: str = "gc-math",
) -> CurriculumEvidence:
    metadata = {
        "code": code,
        "curriculum_code": "MBSSE-BEC",
        "curriculum_version": "2020",
        "structure_scope": "grade_curriculum_content",
        "structure_exhaustive": exhaustive,
    }
    if grade_curriculum_id:
        metadata["grade_curriculum_id"] = grade_curriculum_id
    return CurriculumEvidence(
        entity_type="unit",
        entity_id=f"{subject}-{code}".lower(),
        name=name,
        grade="CLASS_4",
        subject=subject,
        source_reference="grade_curriculum.content",
        metadata=metadata,
    )


def _math_tree(*, exhaustive: bool = True) -> list[CurriculumEvidence]:
    return [
        _unit("C4-U01", "Number and Numeration", exhaustive=exhaustive),
        _unit("C4-U11", "Everyday Arithmetic", exhaustive=exhaustive),
        _unit("C4-U13", "Everyday Arithmetic Multiplication", exhaustive=exhaustive),
    ]


def _state(evidence: list[CurriculumEvidence], *, subject: str = "MATHEMATICS") -> CurriculumQAState:
    state = CurriculumQAState.initial(question=QUESTION)
    state.grade = "CLASS_4"
    state.subject = subject
    state.metadata["curriculum_version"] = "2020"
    state.evidence = evidence
    state.draft_answer = HEDGE
    state.final_answer = HEDGE
    state.verification = VerificationResult(
        passed=True,
        score=0.9,
        recommendation=VerificationRecommendation.ACCEPT,
        issues=["The hedge is consistent with the evidence that was shown."],
    )
    return state


def test_a_complete_mathematics_catalogue_answers_that_c4_u12_is_absent():
    science = _unit(
        "C4-U12",
        "Feeding in Humans",
        subject="SCIENCE",
        grade_curriculum_id="gc-science",
    )
    state = _state(_math_tree() + [science])
    assert codes_absent_from_checked_catalogue(
        question=state.question,
        evidence=list(state.evidence),
        grade=state.grade,
        subject=state.subject,
        curriculum_version=state.metadata["curriculum_version"],
    ) == ["C4-U12"]
    assert route_after_verification({"qa": state}, settings=_settings()) == "fallback"
    published = apply_fallback(state, reason=state.metadata["fallback_reason"])
    answer = published.final_answer or ""
    assert published.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert published.metadata["fallback_reason"] == "catalogue_code_absent"
    assert "C4-U12" in answer
    assert "CLASS_4" in answer
    assert "MATHEMATICS" in answer
    assert "not present in the checked" in answer
    assert "another subject" in answer
    assert "Science" not in answer
    assert "Feeding" not in answer
    assert "unpublished" not in answer.casefold()
    assert "cannot confirm" not in answer
    assert HEDGE not in answer


def test_retrieve_more_cannot_override_a_complete_catalogue_absence():
    """A retrieval request does not outrank a complete checked catalogue."""
    science = _unit(
        "C4-U12",
        "Feeding in Humans",
        subject="SCIENCE",
        grade_curriculum_id="gc-science",
    )
    state = _state(_math_tree() + [science])
    missing = [
        MissingEvidenceRequest(
            type="unit",
            grade="CLASS_4",
            subject="MATHEMATICS",
            topic="C4-U12",
            query="Class 4 Mathematics unit code C4-U12",
            detail="The shown block skips C4-U12.",
        )
    ]
    state.pending_missing_evidence = missing
    state.verification = VerificationResult(
        passed=False,
        score=0.4,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        issues=["The shown catalogue block looks truncated around C4-U12."],
        unsupported_claims=["The listed titles are presented as the complete catalogue."],
        missing_evidence=missing,
    )
    assert state.verification.recommendation == VerificationRecommendation.RETRIEVE_MORE
    assert route_after_verification({"qa": state}, settings=_settings()) == "fallback"
    assert state.verification.recommendation == VerificationRecommendation.RETRIEVE_MORE
    published = apply_fallback(state, reason=state.metadata["fallback_reason"])
    answer = published.final_answer or ""
    assert published.metadata["fallback_reason"] == "catalogue_code_absent"
    assert published.metadata["absent_catalogue_codes"] == ["C4-U12"]
    assert published.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert "C4-U12" in answer
    assert "CLASS_4" in answer
    assert "MATHEMATICS" in answer
    assert "not present in the checked" in answer
    assert "another subject or curriculum" in answer
    assert "Science" not in answer
    assert "Feeding" not in answer
    assert "unpublished" not in answer.casefold()
    assert "cannot confirm" not in answer
    assert HEDGE not in answer


def test_an_incomplete_catalogue_does_not_claim_the_code_is_absent():
    state = _state(_math_tree(exhaustive=False))
    assert codes_absent_from_checked_catalogue(
        question=state.question,
        evidence=list(state.evidence),
        grade=state.grade,
        subject=state.subject,
        curriculum_version="2020",
    ) == []
    assert route_after_verification({"qa": state}, settings=_settings()) == "finish"
    assert "not present in the checked" not in (state.final_answer or "")


def test_another_subjects_tree_does_not_prove_mathematics_absence():
    science = _unit(
        "C4-U01",
        "Living and Non-living Things",
        subject="SCIENCE",
        grade_curriculum_id="gc-science",
    )
    state = _state([science])
    assert codes_absent_from_checked_catalogue(
        question=state.question,
        evidence=list(state.evidence),
        grade=state.grade,
        subject=state.subject,
        curriculum_version="2020",
    ) == []
    assert route_after_verification({"qa": state}, settings=_settings()) == "finish"


def test_a_code_that_is_in_the_checked_catalogue_is_not_reported_absent():
    state = _state(_math_tree() + [_unit("C4-U12", "Number and Numeration")])
    state.question = QUESTION
    assert codes_absent_from_checked_catalogue(
        question=state.question,
        evidence=list(state.evidence),
        grade=state.grade,
        subject=state.subject,
        curriculum_version="2020",
    ) == []


def test_the_same_rule_applies_to_another_catalogue_code():
    evidence = [
        _unit("AG-U01", "Soils", subject="AGRICULTURE", grade_curriculum_id="gc-ag"),
        _unit("AG-U02", "Crops", subject="AGRICULTURE", grade_curriculum_id="gc-ag"),
    ]
    for item in evidence:
        item.grade = "CLASS_2"
    assert codes_absent_from_checked_catalogue(
        question="Is AG-U09 a published Class 2 Agriculture unit?",
        evidence=evidence,
        grade="CLASS_2",
        subject="AGRICULTURE",
        curriculum_version="2020",
    ) == ["AG-U09"]
