"""A subject-identity contradiction is not published through the absence guard."""

from __future__ import annotations

from app.agent.graph_nodes import apply_fallback
from app.agent.graph_routing import route_after_verification
from app.agent.state import CurriculumQAState
from app.agent.verifier import AnswerVerifier
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.enums import AgentStatus
from app.llm.provider import StubLLMProvider
from app.schemas.verification import (
    MissingEvidenceRequest,
    VerificationRecommendation,
    VerificationResult,
    VerificationStatus,
)

GOLD_QUESTION = "What units does the Class 4 French subject cover?"
APPRECIATION_ID = "44c8b694-1111-4111-8111-111111111111"
DRAFT = (
    'The evidence lists French as a Primary 4 subject under the name "French '
    'Appreciation." No unit or topic records for French are included.'
)


def _settings() -> Settings:
    return Settings(_env_file=None, curriculum_api_base_url="http://curriculum.test")


def _catalogue(code: str, name: str, entity_id: str, *, grade: str = "CLASS_4") -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="subject",
        entity_id=entity_id,
        name=name,
        grade=grade,
        subject=code,
        content=name,
        source_reference="grade_subjects",
        metadata={
            "code": code,
            "grade_subject_id": f"gs-{entity_id}",
            "requested_subject": "FRENCH",
            "requested_subject_syllabus": "not_found",
        },
    )


def _french_contradiction() -> CurriculumQAState:
    state = CurriculumQAState.initial(question=GOLD_QUESTION)
    state.grade = "CLASS_4"
    state.subject = "FRENCH"
    state.level = "primary"
    state.evidence = [
        _catalogue("FRENCH_APPRECIATION", "French Appreciation", APPRECIATION_ID),
        _catalogue("SCIENCE", "Science", "science-row"),
    ]
    state.evidence_status = EvidenceStatus.FOUND
    state.draft_answer = DRAFT
    AnswerVerifier(StubLLMProvider()).build_messages(state)
    state.verification = VerificationResult(
        passed=False,
        score=0.4,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        incorrect_claims=["Subject mismatch for FRENCH."],
        issues=["Requested subject FRENCH is not reflected in retrieved evidence."],
    )
    return state


def _publish(state: CurriculumQAState) -> CurriculumQAState:
    route_after_verification({"qa": state}, settings=_settings())
    return apply_fallback(state, reason=state.metadata["fallback_reason"])


def test_absence_guard_does_not_publish_the_french_contradiction():
    state = _publish(_french_contradiction())
    assert state.metadata["fallback_reason"] == "subject_mismatch"
    assert DRAFT not in (state.final_answer or "")
    assert "under the name" not in (state.final_answer or "")
    assert "not certified" not in (state.final_answer or "")


def test_public_answer_separates_french_from_french_appreciation():
    answer = _publish(_french_contradiction()).final_answer or ""
    assert "FRENCH_APPRECIATION" in answer
    assert "French Appreciation" in answer
    assert "does not include a row coded FRENCH" in answer
    assert "cannot be stated" in answer
    assert "is the French syllabus" not in answer
    assert "Ordinal Numbers" not in answer
    assert "premier" not in answer


def test_retrieve_more_cannot_override_a_checked_subject_mismatch():
    """A retrieval request does not publish another subject's units."""
    state = _french_contradiction()
    state.evidence.append(
        CurriculumEvidence(
            entity_type="unit",
            entity_id="science-living",
            name="Living and Non-living Things",
            grade="CLASS_4",
            subject="SCIENCE",
            source_reference="grade_curriculum.content",
            metadata={"code": "C4-U01"},
        )
    )
    missing = [
        MissingEvidenceRequest(
            type="unit_listing",
            grade="CLASS_4",
            subject="FRENCH",
            query="Class 4 French units",
            detail="No unit records for the requested subject were retrieved.",
        )
    ]
    state.pending_missing_evidence = missing
    state.verification = VerificationResult(
        passed=False,
        score=0.1,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        incorrect_claims=["Subject mismatch for FRENCH."],
        issues=["Requested subject FRENCH is not reflected in retrieved evidence."],
        missing_evidence=missing,
    )
    assert state.verification.recommendation == VerificationRecommendation.RETRIEVE_MORE
    assert route_after_verification({"qa": state}, settings=_settings()) == "fallback"
    assert state.verification.recommendation == VerificationRecommendation.RETRIEVE_MORE
    published = apply_fallback(state, reason=state.metadata["fallback_reason"])
    answer = published.final_answer or ""
    assert published.metadata["fallback_reason"] == "subject_mismatch"
    assert published.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert DRAFT not in answer
    assert "does not include a row coded FRENCH" in answer
    assert "do not establish the units or learning outcomes" in answer
    assert "cannot be stated" in answer
    assert "Living and Non-living Things" not in answer
    assert "Ordinal Numbers" not in answer
    assert "C4-U01" not in answer


def test_positive_verifier_result_cannot_accept_the_contradiction():
    state = _french_contradiction()
    state.verification = VerificationResult(
        passed=True,
        score=1.0,
        recommendation=VerificationRecommendation.ACCEPT,
        incorrect_claims=["Subject mismatch for FRENCH."],
    )
    published = _publish(state)
    assert published.metadata["fallback_reason"] == "subject_mismatch"
    assert published.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert DRAFT not in (published.final_answer or "")


def test_ordinary_absence_without_a_mismatch_still_keeps_its_draft():
    state = CurriculumQAState.initial(
        question="What does the Class 4 Mathematics syllabus say about astronomy?"
    )
    state.grade = "CLASS_4"
    state.subject = "MATHEMATICS"
    unit = CurriculumEvidence(
        entity_type="unit",
        entity_id="c4-math-unit",
        name="Number and Numeration",
        grade="CLASS_4",
        subject="MATHEMATICS",
        metadata={"code": "C4-U01"},
    )
    state.evidence = [unit]
    state.evidence_status = EvidenceStatus.FOUND
    state.draft_answer = (
        "The evidence does not include a curriculum statement about astronomy."
    )
    state.metadata["verification_prompt_ids"] = ["c4-math-unit"]
    state.verification = VerificationResult(
        passed=True,
        score=1.0,
        recommendation=VerificationRecommendation.ACCEPT,
    )
    published = _publish(state)
    assert published.metadata["fallback_reason"] == "absence_records_unchecked"
    assert published.draft_answer.split("\n\nNote:", 1)[0] in (published.final_answer or "")


def test_matching_subjects_are_not_rejected_as_a_contradiction():
    appreciation = CurriculumQAState.initial(
        question="What units does Class 4 French Appreciation cover?"
    )
    appreciation.grade = "CLASS_4"
    appreciation.subject = "FRENCH_APPRECIATION"
    appreciation.evidence = [
        _catalogue("FRENCH_APPRECIATION", "French Appreciation", APPRECIATION_ID)
    ]
    appreciation.evidence_status = EvidenceStatus.FOUND
    appreciation.draft_answer = "No unit records for French Appreciation were retrieved."
    AnswerVerifier(StubLLMProvider()).build_messages(appreciation)
    appreciation.verification = VerificationResult(
        passed=False,
        score=0.4,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
    )
    assert _publish(appreciation).metadata["fallback_reason"] == "absence_records_unchecked"

    foreign = CurriculumQAState.initial(
        question="What topics are in SSS 1 French as a Foreign Language?"
    )
    foreign.grade = "SSS_1"
    foreign.subject = "FRENCH_FOREIGN_LANGUAGE"
    foreign.evidence = [
        CurriculumEvidence(
            entity_type="unit",
            entity_id="sss-french-unit",
            name="Greetings",
            grade="SSS_1",
            subject="FRENCH_FOREIGN_LANGUAGE",
            metadata={"code": "SSS-FR-U01"},
        )
    ]
    foreign.evidence_status = EvidenceStatus.FOUND
    foreign.draft_answer = "Greetings is a published unit."
    foreign.verification = VerificationResult(
        passed=True, score=1.0, recommendation=VerificationRecommendation.ACCEPT
    )
    assert route_after_verification({"qa": foreign}, settings=_settings()) == "finish"

    science = CurriculumQAState.initial(question="What units does Class 4 Science cover?")
    science.grade = "CLASS_4"
    science.subject = "SCIENCE"
    science.evidence = [
        CurriculumEvidence(
            entity_type="unit",
            entity_id="science-unit",
            name="The Earth",
            grade="CLASS_4",
            subject="SCIENCE",
            metadata={"code": "C4-U01"},
        )
    ]
    science.evidence_status = EvidenceStatus.FOUND
    science.draft_answer = "The Earth is a Class 4 Science unit."
    science.verification = VerificationResult(
        passed=True, score=1.0, recommendation=VerificationRecommendation.ACCEPT
    )
    assert route_after_verification({"qa": science}, settings=_settings()) == "finish"

    missing = CurriculumQAState.initial(question=GOLD_QUESTION)
    missing.grade = "CLASS_4"
    missing.subject = "FRENCH"
    missing.draft_answer = "No unit records for French were retrieved."
    missing.metadata["verification_prompt_ids"] = []
    missing.verification = VerificationResult(
        passed=True, score=1.0, recommendation=VerificationRecommendation.ACCEPT
    )
    assert _publish(missing).metadata["fallback_reason"] == "absence_records_unchecked"


def test_subject_mismatch_stays_an_insufficient_evidence_response():
    state = _publish(_french_contradiction())
    assert state.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert state.verification_status == VerificationStatus.INSUFFICIENT_EVIDENCE
    assert state.verification is not None
    assert state.verification.passed is False
    assert state.metadata["fallback_reason"] == "subject_mismatch"
    assert state.metadata["termination_reason"] == "subject_mismatch"
