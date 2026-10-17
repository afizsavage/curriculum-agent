"""Class 4 French is scoped to FRENCH, not to French Appreciation."""

from __future__ import annotations

from app.agent.absence_evidence import (
    format_absence_prompt,
    select_records_for_absence_verification,
    serialized_entity_ids,
)
from app.agent.graph_routing import route_after_verification
from app.agent.state import CurriculumQAState
from app.agent.verification_checks import run_deterministic_checks
from app.config import Settings
from app.curriculum.codes import extract_filters_from_question
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus

GOLD_QUESTION = "What units does the Class 4 French subject cover?"

SUBSTITUTED_ANSWER = (
    'The evidence lists French as a Primary 4 subject under the name "French '
    'Appreciation." It does not provide any unit or topic breakdown for this '
    "subject. No unit or topic records for French are included, so the specific "
    "units covered cannot be stated from this evidence."
)


def _class4_subject_rows() -> list[CurriculumEvidence]:
    return [
        CurriculumEvidence(
            entity_type="subject",
            entity_id=f"subject-{code.lower()}",
            name=name,
            grade="CLASS_4",
            subject=code,
        )
        for code, name in (
            ("MATHEMATICS", "Mathematics"),
            ("SCIENCE", "Science"),
            ("FRENCH_APPRECIATION", "French Appreciation"),
        )
    ]


def _state_from_gold_question() -> CurriculumQAState:
    filters = extract_filters_from_question(GOLD_QUESTION)
    state = CurriculumQAState.initial(question=GOLD_QUESTION)
    state.grade = filters["grade"]
    state.subject = filters["subject"]
    state.evidence = _class4_subject_rows()
    state.evidence_status = EvidenceStatus.FOUND
    state.draft_answer = SUBSTITUTED_ANSWER
    return state


def test_french_appreciation_rows_are_a_subject_mismatch_for_the_gold_question():
    result = run_deterministic_checks(_state_from_gold_question())
    assert result.passed is False
    assert "Subject mismatch for FRENCH." in result.incorrect_claims


def test_french_appreciation_rows_still_satisfy_a_french_appreciation_question():
    state = CurriculumQAState.initial(
        question="What units does Class 4 French Appreciation cover?"
    )
    filters = extract_filters_from_question(state.question)
    state.grade = filters["grade"]
    state.subject = filters["subject"]
    state.evidence = _class4_subject_rows()
    state.evidence_status = EvidenceStatus.FOUND
    state.draft_answer = "French Appreciation is a Class 4 subject."
    result = run_deterministic_checks(state)
    assert not any("Subject mismatch" in claim for claim in result.incorrect_claims)


def test_absence_guard_still_decides_the_route_when_the_subject_mismatches():
    qa = _state_from_gold_question()
    selected = select_records_for_absence_verification(
        list(qa.evidence), grade=qa.grade, subject=qa.subject
    )
    qa.metadata["verification_prompt_ids"] = serialized_entity_ids(
        format_absence_prompt(selected)
    )
    qa.verification = run_deterministic_checks(qa)
    assert "Subject mismatch for FRENCH." in qa.verification.incorrect_claims

    route = route_after_verification(
        {"qa": qa},
        settings=Settings(_env_file=None, curriculum_api_base_url="http://curriculum.test"),
    )
    assert route == "fallback"
    assert qa.metadata["fallback_reason"] == "absence_records_unchecked"
