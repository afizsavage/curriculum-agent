"""A missing subject syllabus keeps the grade catalogue visible to the verifier."""

from __future__ import annotations

from app.agent.absence_evidence import (
    select_records_for_absence_verification,
    serialized_entity_ids,
)
from app.agent.graph_routing import route_after_verification
from app.agent.state import CurriculumQAState
from app.agent.verification_checks import run_deterministic_checks
from app.agent.verifier import AnswerVerifier
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.llm.provider import StubLLMProvider
from app.schemas.verification import VerificationRecommendation, VerificationResult

GOLD_QUESTION = "What units does the Class 4 French subject cover?"
APPRECIATION_ID = "44c8b694-1111-4111-8111-111111111111"
SCIENCE_ID = "77777777-7777-4777-8777-777777777777"
FOREIGN_UNIT_ID = "sss-french-unit"
MATH_UNIT_ID = "c4-math-unit"

ABSENCE_ANSWER = (
    'The evidence lists French as a Primary 4 subject under the name "French '
    'Appreciation." No unit or topic records for French are included, so the '
    "units covered cannot be stated from this evidence."
)


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


def _french_state() -> CurriculumQAState:
    state = CurriculumQAState.initial(question=GOLD_QUESTION)
    state.grade = "CLASS_4"
    state.subject = "FRENCH"
    state.level = "primary"
    state.evidence = [
        _catalogue("FRENCH_APPRECIATION", "French Appreciation", APPRECIATION_ID),
        _catalogue("SCIENCE", "Science", SCIENCE_ID),
    ]
    state.evidence_status = EvidenceStatus.FOUND
    state.draft_answer = ABSENCE_ANSWER
    return state


def _settings() -> Settings:
    return Settings(_env_file=None, curriculum_api_base_url="http://curriculum.test")


def test_missing_french_syllabus_keeps_the_grade_catalogue_selectable():
    state = _french_state()
    selected = select_records_for_absence_verification(
        list(state.evidence), grade=state.grade, subject=state.subject
    )
    assert {item.entity_id for item in selected} == {APPRECIATION_ID, SCIENCE_ID}
    assert all(item.source_reference == "grade_subjects" for item in selected)


def test_catalogue_identifiers_and_codes_reach_the_verifier_prompt():
    state = _french_state()
    message = AnswerVerifier(StubLLMProvider()).build_messages(state)[1].content
    assert f"Entity ID: {APPRECIATION_ID}" in message
    assert "Subject code: FRENCH_APPRECIATION" in message
    assert "grade subject catalogue" in message
    assert "does not establish" in message
    assert APPRECIATION_ID in state.metadata["verification_evidence_ids"]
    assert serialized_entity_ids(message) == [APPRECIATION_ID, SCIENCE_ID]


def test_french_appreciation_catalogue_row_remains_a_subject_mismatch():
    state = _french_state()
    state.evidence = state.evidence[:1]
    result = run_deterministic_checks(state)
    assert result.passed is False
    assert "Subject mismatch for FRENCH." in result.incorrect_claims


def test_catalogue_does_not_certify_absence_or_override_the_mismatch():
    state = _french_state()
    AnswerVerifier(StubLLMProvider()).build_messages(state)
    state.verification = VerificationResult(
        passed=True,
        score=1.0,
        recommendation=VerificationRecommendation.ACCEPT,
        incorrect_claims=["Subject mismatch for FRENCH."],
    )
    route = route_after_verification({"qa": state}, settings=_settings())
    assert route == "fallback"
    assert state.metadata["fallback_reason"] == "subject_mismatch"

    accepted = VerificationResult(
        passed=True,
        score=1.0,
        recommendation=VerificationRecommendation.ACCEPT,
    )
    merged = AnswerVerifier(StubLLMProvider())._merge(
        run_deterministic_checks(state), accepted
    )
    assert merged.passed is False
    assert "Subject mismatch for FRENCH." in merged.incorrect_claims
    plain = _french_state()
    plain.draft_answer = "Class 4 publishes French Appreciation."
    plain.verification = merged
    assert route_after_verification({"qa": plain}, settings=_settings()) != "finish"

    empty = CurriculumQAState.initial(question=GOLD_QUESTION)
    empty.grade = "CLASS_4"
    empty.subject = "FRENCH"
    empty.draft_answer = "No unit records for French were retrieved."
    empty.metadata["verification_prompt_ids"] = []
    empty.verification = VerificationResult(
        passed=True, score=1.0, recommendation=VerificationRecommendation.ACCEPT
    )
    assert (
        select_records_for_absence_verification(
            [], grade=empty.grade, subject=empty.subject
        )
        == []
    )
    assert route_after_verification({"qa": empty}, settings=_settings()) == "fallback"
    assert empty.metadata["fallback_reason"] == "absence_records_unchecked"


def test_catalogue_fallback_stays_out_of_ordinary_subject_checks():
    math_unit = CurriculumEvidence(
        entity_type="unit",
        entity_id=MATH_UNIT_ID,
        name="Number and Numeration",
        grade="CLASS_4",
        subject="MATHEMATICS",
        metadata={"code": "C4-U01"},
    )
    appreciation = CurriculumEvidence(
        entity_type="subject",
        entity_id=APPRECIATION_ID,
        name="French Appreciation",
        grade="CLASS_4",
        subject="FRENCH_APPRECIATION",
        source_reference="subjects",
        metadata={"code": "FRENCH_APPRECIATION"},
    )
    selected = select_records_for_absence_verification(
        [math_unit, appreciation], grade="CLASS_4", subject="MATHEMATICS"
    )
    assert [item.entity_id for item in selected] == [MATH_UNIT_ID]

    appreciation_unit = CurriculumEvidence(
        entity_type="unit",
        entity_id="fr-app-unit",
        name="Ordinal Numbers",
        grade="CLASS_4",
        subject="FRENCH_APPRECIATION",
        metadata={"code": "C4-FA-U01"},
    )
    appreciation_selected = select_records_for_absence_verification(
        [appreciation_unit, _catalogue("SCIENCE", "Science", SCIENCE_ID)],
        grade="CLASS_4",
        subject="FRENCH_APPRECIATION",
    )
    assert [item.entity_id for item in appreciation_selected] == ["fr-app-unit"]

    foreign_unit = CurriculumEvidence(
        entity_type="unit",
        entity_id=FOREIGN_UNIT_ID,
        name="Greetings",
        grade="SSS_1",
        subject="FRENCH_FOREIGN_LANGUAGE",
        metadata={"code": "SSS-FR-U01"},
    )
    foreign_selected = select_records_for_absence_verification(
        [foreign_unit, _catalogue("FRENCH_APPRECIATION", "French Appreciation", APPRECIATION_ID)],
        grade="SSS_1",
        subject="FRENCH_FOREIGN_LANGUAGE",
    )
    assert [item.entity_id for item in foreign_selected] == [FOREIGN_UNIT_ID]
    wrong_grade = select_records_for_absence_verification(
        [_catalogue("FRENCH_APPRECIATION", "French Appreciation", APPRECIATION_ID)],
        grade="SSS_1",
        subject="FRENCH_FOREIGN_LANGUAGE",
    )
    assert wrong_grade == []
