"""An unresolved subject still uses the catalogue mismatch answer."""

from __future__ import annotations

from app.agent.absence_evidence import bind_unresolved_requested_subject
from app.agent.graph_nodes import apply_fallback
from app.agent.graph_routing import route_after_verification
from app.agent.state import CurriculumQAState
from app.agent.verification_checks import run_deterministic_checks
from app.agent.verifier import AnswerVerifier
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.enums import AgentStatus
from app.llm.provider import StubLLMProvider
from app.schemas.verification import VerificationRecommendation, VerificationResult

ASTRONOMY_QUESTION = "What topics are taught in Primary 4 Astronomy?"
HEADING = "## Subjects Listed for Primary 4"
ENGLISH_ID = "9da3f7f7-60c2-49c6-8950-fd20a1913661"
SCIENCE_ID = "d0f1291c-baa3-4651-a922-6d8590ace82e"


def _settings() -> Settings:
    return Settings(_env_file=None, curriculum_api_base_url="http://curriculum.test")


def _catalogue(code: str, name: str, entity_id: str, *, requested: str) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="subject",
        entity_id=entity_id,
        name=name,
        grade="CLASS_4",
        subject=code,
        content=name,
        source_reference="grade_subjects",
        metadata={
            "code": code,
            "requested_subject": requested,
            "requested_subject_syllabus": "not_found",
        },
    )


def _astronomy_state() -> CurriculumQAState:
    state = CurriculumQAState.initial(question=ASTRONOMY_QUESTION)
    state.grade = "CLASS_4"
    state.subject = None
    state.level = "primary"
    state.evidence = [
        _catalogue("ENGLISH_LANGUAGE", "English Language", ENGLISH_ID, requested="ASTRONOMY"),
        _catalogue("SCIENCE", "Science", SCIENCE_ID, requested="ASTRONOMY"),
    ]
    state.evidence_status = EvidenceStatus.FOUND
    state.retrieval_state.resolved_subject = "ENGLISH_LANGUAGE"
    state.draft_answer = HEADING
    state.final_answer = HEADING
    return state


def test_not_found_catalogue_preserves_astronomy_without_copying_the_first_row():
    state = _astronomy_state()
    assert bind_unresolved_requested_subject(state) == "ASTRONOMY"
    assert state.subject == "ASTRONOMY"
    assert state.metadata["requested_subject"] == "ASTRONOMY"
    assert state.retrieval_state.resolved_subject == "ENGLISH_LANGUAGE"


def test_unresolved_astronomy_uses_the_subject_mismatch_answer():
    state = _astronomy_state()
    AnswerVerifier(StubLLMProvider()).build_messages(state)
    checked = run_deterministic_checks(state)
    assert "Subject mismatch for ASTRONOMY." in checked.incorrect_claims
    state.verification = checked
    assert route_after_verification({"qa": state}, settings=_settings()) == "fallback"
    published = apply_fallback(state, reason=state.metadata["fallback_reason"])
    answer = published.final_answer or ""
    assert published.metadata["fallback_reason"] == "subject_mismatch"
    assert published.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert published.subject == "ASTRONOMY"
    assert "ASTRONOMY" in answer
    assert "does not include a row coded ASTRONOMY" in answer
    assert "syllabus was not found" in answer
    assert answer.strip() != HEADING
    assert HEADING not in answer


def test_empty_retrieval_does_not_declare_astronomy_absent():
    state = CurriculumQAState.initial(question=ASTRONOMY_QUESTION)
    state.grade = "CLASS_4"
    state.subject = None
    state.evidence = []
    state.evidence_status = EvidenceStatus.NOT_FOUND
    state.draft_answer = HEADING
    state.final_answer = HEADING
    state.metadata["verification_prompt_ids"] = []
    assert bind_unresolved_requested_subject(state) is None
    assert state.subject is None
    checked = run_deterministic_checks(state)
    assert "ASTRONOMY" not in " ".join(checked.incorrect_claims)
    state.verification = VerificationResult(
        passed=True,
        score=1.0,
        recommendation=VerificationRecommendation.ACCEPT,
    )
    assert route_after_verification({"qa": state}, settings=_settings()) == "finish"
    assert state.subject is None
    assert "ASTRONOMY" not in (state.final_answer or "")


def test_a_catalogue_without_the_not_found_stamp_does_not_bind_a_subject():
    state = CurriculumQAState.initial(question=ASTRONOMY_QUESTION)
    state.grade = "CLASS_4"
    state.subject = None
    state.evidence = [
        CurriculumEvidence(
            entity_type="subject",
            entity_id=ENGLISH_ID,
            name="English Language",
            grade="CLASS_4",
            subject="ENGLISH_LANGUAGE",
            source_reference="grade_subjects",
            metadata={"code": "ENGLISH_LANGUAGE"},
        )
    ]
    state.evidence_status = EvidenceStatus.FOUND
    state.draft_answer = HEADING
    state.final_answer = HEADING
    assert bind_unresolved_requested_subject(state) is None
    assert state.subject is None
    checked = run_deterministic_checks(state)
    assert checked.incorrect_claims == []
