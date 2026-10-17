"""Shown outcomes are not reported as absent from the verifier evidence."""

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

_THEME = "theme-intro"
_LO04 = "3013755a-2b6d-44f2-9089-ff6a04174603"
_LO04_CODE = "S1-T1-01-LO04"
_LO04_TEXT = "list the life processes shared by all living things."
_TOPIC_QUESTION = (
    "What topics are listed under the SSS 1 Biology theme Introduction to Biology?"
)
_OUTCOME_QUESTION = (
    "What are the learning outcomes for the SSS 1 Biology theme Introduction to Biology?"
)


def _settings() -> Settings:
    return Settings(_env_file=None, curriculum_api_base_url="http://curriculum.test")


def _theme() -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="theme",
        entity_id=_THEME,
        name="Introduction to Biology",
        content="Introduction to Biology",
        metadata={"code": "S1-T1-01"},
    )


def _outcome(number: int, content: str, entity_id: str | None = None) -> CurriculumEvidence:
    code = f"S1-T1-01-LO{number:02d}"
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id=entity_id or code.lower(),
        name=code,
        topic=_THEME,
        content=content,
        metadata={"code": code},
    )


def _topic(name: str, entity_id: str) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="topic",
        entity_id=entity_id,
        name=name,
        content=f"{name} biology",
        metadata={"code": entity_id, "parent_id": _THEME},
    )


def _biology_evidence() -> list[CurriculumEvidence]:
    outcomes = [
        _outcome(1, "understand the meaning of biology.", "lo01"),
        _outcome(2, "state the various branches of biology.", "lo02"),
        _outcome(3, "state the reasons why biology should be studied in school.", "lo03"),
        _outcome(4, _LO04_TEXT, _LO04),
        _outcome(5, "understand what classification means.", "lo05"),
    ]
    topics = [
        _topic("Living and non-living things", "topic-living"),
        _topic("Diversity of living things", "topic-diversity"),
        _topic("Classification", "topic-class"),
    ]
    fillers = [
        CurriculumEvidence(
            entity_type="theme",
            entity_id=f"other-theme-{index}",
            name=f"Other theme {index}",
            content=f"Other theme {index}",
        )
        for index in range(18)
    ]
    return [_theme(), *outcomes, *topics, *fillers]


def _normalize(state: CurriculumQAState, result: VerificationResult) -> VerificationResult:
    return AnswerVerifier(StubLLMProvider())._normalize_recommendation(state, result)


def _absence_of_lo04() -> MissingEvidenceRequest:
    return MissingEvidenceRequest(
        type="learning_outcome",
        grade="SSS_1",
        subject="BIOLOGY",
        topic="Introduction to Biology",
        query=_LO04_CODE,
        detail=(
            "The retrieved evidence includes LO01, LO02, LO03, and LO05 but not LO04. "
            "The answer does not mention LO04."
        ),
    )


def test_prompt_window_keeps_a_sibling_outcome_the_lexical_rank_dropped():
    evidence = _biology_evidence()
    selected, ids = select_evidence_for_prompt(
        evidence, question=_TOPIC_QUESTION, max_records=24
    )
    assert len(selected) == 24
    assert _LO04 in ids
    assert "topic-living" in ids
    assert "topic-diversity" in ids
    assert "topic-class" in ids


def test_shown_outcome_is_not_reported_absent_for_a_topic_list():
    evidence = _biology_evidence()
    state = CurriculumQAState.initial(question=_TOPIC_QUESTION)
    state.evidence = evidence
    state.draft_answer = (
        "The topics are Living and non-living things, Diversity of living things, "
        "and Classification."
    )
    state.metadata["verification_evidence_ids"] = [
        item.entity_id for item in evidence if item.entity_id
    ]
    result = _normalize(
        state,
        VerificationResult(
            passed=False,
            score=0.9,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            missing_evidence=[_absence_of_lo04()],
            issues=[
                f"The answer omits {_LO04_CODE}, which is not present in the retrieved evidence."
            ],
        ),
    )
    assert result.passed is True
    assert result.recommendation == VerificationRecommendation.ACCEPT
    assert result.missing_evidence == []
    assert _LO04_CODE in result.metadata["shown_outcome_absence_suppressed"]
    assert not any("not present" in issue for issue in result.issues)
    state.verification = result
    state.pending_missing_evidence = list(result.missing_evidence)
    assert route_after_verification({"qa": state}, settings=_settings()) == "finish"


def test_shown_outcome_omitted_from_an_outcome_answer_stays_unaccepted():
    evidence = _biology_evidence()
    state = CurriculumQAState.initial(question=_OUTCOME_QUESTION)
    state.evidence = evidence
    state.draft_answer = "Pupils understand the meaning of biology."
    state.metadata["verification_evidence_ids"] = [_LO04, "lo01"]
    result = _normalize(
        state,
        VerificationResult(
            passed=False,
            score=0.9,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            missing_evidence=[_absence_of_lo04()],
        ),
    )
    assert result.passed is False
    assert result.missing_evidence == []
    assert any("omitted from the answer" in issue for issue in result.issues)
    assert not any("not present" in issue.casefold() for issue in result.issues)
    state.verification = result
    state.pending_missing_evidence = list(result.missing_evidence)
    assert route_after_verification({"qa": state}, settings=_settings()) != "retrieve_more"


def test_a_missing_outcome_can_still_request_retrieval():
    state = CurriculumQAState.initial(question=_OUTCOME_QUESTION)
    state.evidence = [_theme(), _outcome(1, "understand the meaning of biology.", "lo01")]
    state.draft_answer = "Pupils understand the meaning of biology."
    state.metadata["verification_evidence_ids"] = [_THEME, "lo01"]
    result = _normalize(
        state,
        VerificationResult(
            passed=False,
            score=0.4,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            missing_evidence=[_absence_of_lo04()],
        ),
    )
    assert result.missing_evidence
    assert result.recommendation == VerificationRecommendation.RETRIEVE_MORE
    state.verification = result
    state.pending_missing_evidence = list(result.missing_evidence)
    assert route_after_verification({"qa": state}, settings=_settings()) == "retrieve_more"


def test_retrieve_more_with_no_new_evidence_still_stops():
    state = CurriculumQAState.initial(question=_OUTCOME_QUESTION)
    state.grade = "SSS_1"
    state.subject = "BIOLOGY"
    state.draft_answer = "Pupils understand the meaning of biology."
    missing = [_absence_of_lo04()]
    state.pending_missing_evidence = missing
    state.verification = VerificationResult(
        passed=False,
        score=0.4,
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


def test_period_gap_and_subject_mismatch_are_not_cleared():
    period_state = CurriculumQAState.initial(
        question=(
            "What are the learning outcomes for Class 4 Mathematics unit C4-U01, "
            "and how many recommended periods does that unit assign?"
        )
    )
    period_state.draft_answer = (
        "The supplied evidence does not include a recommended number of periods."
    )
    period_gap = MissingEvidenceRequest(
        type="unit_periods",
        grade="CLASS_4",
        subject="MATHEMATICS",
        topic="Number and Numeration",
        query="C4-U01 recommended periods",
        detail="No evidence record contains a recommended number of periods for unit C4-U01.",
    )
    period_result = _normalize(
        period_state,
        VerificationResult(
            passed=False,
            score=0.9,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            missing_evidence=[period_gap],
        ),
    )
    assert period_result.passed is False
    assert period_result.recommendation == VerificationRecommendation.RETRIEVE_MORE
    assert period_result.missing_evidence

    mismatch = _normalize(
        CurriculumQAState.initial(
            question="What units does the Class 4 French subject cover?"
        ),
        VerificationResult(
            passed=False,
            score=0.45,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            incorrect_claims=["Subject mismatch for FRENCH."],
            missing_evidence=[_absence_of_lo04()],
        ),
    )
    assert mismatch.passed is False
    assert mismatch.incorrect_claims == ["Subject mismatch for FRENCH."]
    assert mismatch.missing_evidence
