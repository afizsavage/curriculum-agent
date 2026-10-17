"""Capacity boundaries for the 24-record evidence window."""

from __future__ import annotations

from app.agent.answer_generator import select_evidence_for_prompt
from app.agent.graph_routing import route_after_verification
from app.agent.retrieval_state import tool_fingerprint
from app.agent.state import CurriculumQAState
from app.agent.verifier import AnswerVerifier
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence
from app.llm.provider import StubLLMProvider
from app.schemas.verification import (
    MissingEvidenceRequest,
    VerificationRecommendation,
    VerificationResult,
)

_TOPIC_QUESTION = "What topics are listed under the Alpha theme?"
_OUTCOME_QUESTION = "What are the learning outcomes under the Alpha theme?"


def _settings() -> Settings:
    return Settings(_env_file=None, curriculum_api_base_url="http://curriculum.test")


def _parent(parent_id: str, name: str) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="theme",
        entity_id=parent_id,
        name=name,
        content=name,
    )


def _outcome(
    parent_id: str,
    number: int,
    content: str = "plain statement",
    prefix: str = "S1-T1-01",
) -> CurriculumEvidence:
    code = f"{prefix}-LO{number:02d}"
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id=f"{parent_id}-lo-{number:02d}",
        name=code,
        topic=parent_id,
        content=content,
        metadata={"code": code},
    )


def _filler(index: int) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="theme",
        entity_id=f"filler-{index:02d}",
        name=f"Other theme {index}",
        content=f"Other theme {index}",
    )


def _select(evidence: list[CurriculumEvidence], question: str = _TOPIC_QUESTION):
    return select_evidence_for_prompt(evidence, question=question, max_records=24)


def _normalize(state: CurriculumQAState, result: VerificationResult) -> VerificationResult:
    return AnswerVerifier(StubLLMProvider())._normalize_recommendation(state, result)


def _absence(code: str) -> MissingEvidenceRequest:
    return MissingEvidenceRequest(
        type="learning_outcome",
        query=code,
        detail=f"The retrieved evidence does not include {code}.",
    )


def test_window_default_stays_at_24_records():
    assert select_evidence_for_prompt.__kwdefaults__["max_records"] == 24


def test_sibling_group_that_fits_is_kept_at_23_and_24_records():
    for count in (22, 23):
        parent = _parent("parent-a", "Alpha")
        outcomes = [_outcome("parent-a", number) for number in range(1, count + 1)]
        evidence = [parent, *outcomes]
        first, first_ids = _select(evidence)
        second, second_ids = _select(evidence)
        assert len(first) == count + 1
        assert first_ids == second_ids
        assert set(first_ids) == {item.entity_id for item in evidence}


def test_sibling_group_beyond_24_is_capped_and_the_omitted_outcome_is_not_shown():
    parent = _parent("parent-a", "Alpha")
    outcomes = [_outcome("parent-a", number) for number in range(1, 25)]
    evidence = [parent, *outcomes]
    first, first_ids = _select(evidence)
    _, second_ids = _select(evidence)
    assert len(first) == 24
    assert first_ids == second_ids
    omitted = [item for item in outcomes if item.entity_id not in first_ids]
    assert [item.metadata["code"] for item in omitted] == ["S1-T1-01-LO24"]
    assert "parent-a" in first_ids

    state = CurriculumQAState.initial(question=_OUTCOME_QUESTION)
    state.evidence = evidence
    state.draft_answer = "The outcomes stop before the last one."
    state.metadata["verification_evidence_ids"] = list(first_ids)
    result = _normalize(
        state,
        VerificationResult(
            passed=False,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            missing_evidence=[_absence("S1-T1-01-LO24")],
        ),
    )
    assert result.missing_evidence
    assert "shown_outcome_absence_suppressed" not in result.metadata
    assert result.recommendation == VerificationRecommendation.RETRIEVE_MORE


def test_competing_parents_keep_their_own_context():
    parents = [_parent("parent-a", "Alpha"), _parent("parent-b", "Beta")]
    seeded = [
        _outcome("parent-a", 1, "alpha listed topic", prefix="S1-T1-01"),
        _outcome("parent-b", 1, "beta listed topic", prefix="S1-T1-02"),
    ]
    rest = [
        _outcome(parent_id, number, prefix=prefix)
        for parent_id, prefix in (("parent-a", "S1-T1-01"), ("parent-b", "S1-T1-02"))
        for number in range(2, 16)
    ]
    evidence = [*parents, *seeded, *rest, *[_filler(index) for index in range(20)]]
    question = "What topics are listed under the Alpha theme and the Beta theme?"
    first, first_ids = select_evidence_for_prompt(evidence, question=question, max_records=24)
    _, second_ids = select_evidence_for_prompt(evidence, question=question, max_records=24)
    assert len(first) == 24
    assert first_ids == second_ids
    assert "parent-a" in first_ids
    assert "parent-b" in first_ids
    selected_codes = {
        item.metadata["code"]
        for item in first
        if (item.entity_type or "") == "learning_outcome"
    }
    assert "S1-T1-01-LO01" in selected_codes
    assert "S1-T1-02-LO01" in selected_codes
    assert len(selected_codes) < 30


def test_topic_question_does_not_require_every_shown_outcome():
    parent = _parent("parent-a", "Alpha")
    shown = _outcome("parent-a", 4, "list the life processes shared by all living things.")
    state = CurriculumQAState.initial(question=_TOPIC_QUESTION)
    state.evidence = [parent, shown, _filler(1)]
    state.draft_answer = "The Alpha theme lists the living-things topic."
    state.metadata["verification_evidence_ids"] = ["parent-a", shown.entity_id]
    result = _normalize(
        state,
        VerificationResult(
            passed=False,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            missing_evidence=[_absence(shown.metadata["code"])],
            issues=[f"{shown.metadata['code']} is not present in the retrieved evidence."],
        ),
    )
    assert result.passed is True
    assert result.recommendation == VerificationRecommendation.ACCEPT
    assert result.missing_evidence == []


def test_outcome_question_still_flags_a_shown_outcome_omitted_from_the_answer():
    shown = _outcome("parent-a", 4, "list the life processes shared by all living things.")
    state = CurriculumQAState.initial(question=_OUTCOME_QUESTION)
    state.evidence = [_parent("parent-a", "Alpha"), shown]
    state.draft_answer = "Pupils learn one earlier outcome."
    state.metadata["verification_evidence_ids"] = [shown.entity_id]
    result = _normalize(
        state,
        VerificationResult(
            passed=False,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            missing_evidence=[_absence(shown.metadata["code"])],
        ),
    )
    assert result.passed is False
    assert result.missing_evidence == []
    assert any("omitted from the answer" in issue for issue in result.issues)


def test_outcome_in_the_bag_but_not_in_the_prompt_is_not_treated_as_shown():
    parent = _parent("parent-a", "Alpha")
    outcomes = [_outcome("parent-a", number) for number in range(1, 25)]
    evidence = [parent, *outcomes]
    _, ids = _select(evidence)
    omitted = next(item for item in outcomes if item.entity_id not in ids)
    state = CurriculumQAState.initial(question=_OUTCOME_QUESTION)
    state.evidence = evidence
    state.draft_answer = "A partial outcome list."
    state.metadata["verification_evidence_ids"] = list(ids)
    result = _normalize(
        state,
        VerificationResult(
            passed=False,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            missing_evidence=[_absence(omitted.metadata["code"])],
        ),
    )
    assert result.missing_evidence
    assert "shown_outcome_absence_suppressed" not in result.metadata

    unset = CurriculumQAState.initial(question=_OUTCOME_QUESTION)
    unset.evidence = evidence
    unset.draft_answer = "A partial outcome list."
    untouched = _normalize(
        unset,
        VerificationResult(
            passed=False,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            missing_evidence=[_absence(omitted.metadata["code"])],
        ),
    )
    assert untouched.missing_evidence
    assert "shown_outcome_absence_suppressed" not in untouched.metadata


def test_french_subject_mismatch_is_unchanged_by_the_window_guard():
    result = _normalize(
        CurriculumQAState.initial(question="What units does the Class 4 French subject cover?"),
        VerificationResult(
            passed=False,
            score=0.45,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            incorrect_claims=["Subject mismatch for FRENCH."],
            missing_evidence=[_absence("S1-T1-01-LO04")],
        ),
    )
    assert result.passed is False
    assert result.incorrect_claims == ["Subject mismatch for FRENCH."]
    assert result.missing_evidence


def test_retrieve_more_without_new_evidence_still_stops():
    state = CurriculumQAState.initial(
        question=(
            "What are the learning outcomes for the SSS 1 Biology theme "
            "Introduction to Biology?"
        )
    )
    state.grade = "SSS_1"
    state.subject = "BIOLOGY"
    state.draft_answer = "A partial outcome list."
    missing = [
        MissingEvidenceRequest(
            type="learning_outcome",
            grade="SSS_1",
            subject="BIOLOGY",
            topic="Introduction to Biology",
            query="S1-T1-01-LO24",
            detail="The retrieved evidence does not include S1-T1-01-LO24.",
        )
    ]
    state.pending_missing_evidence = missing
    state.verification = VerificationResult(
        passed=False,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        missing_evidence=missing,
    )
    args = {
        "topic": "Introduction to Biology",
        "grade": "SSS_1",
        "subject": "BIOLOGY",
    }
    state.retrieval_state.remember_fingerprint(
        tool_fingerprint("get_learning_objectives", args), 3
    )
    state.retrieval_state.remember_fingerprint(
        tool_fingerprint("resolve_curriculum_context", args), 4
    )
    assert route_after_verification({"qa": state}, settings=_settings()) == "fallback"
    assert state.metadata.get("fallback_reason") == "no_retrieval_progress"
