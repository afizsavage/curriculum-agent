"""Requested curriculum codes survive redaction. Other internal ids do not."""

from __future__ import annotations

from app.agent.answer_generator import AnswerGenerator
from app.agent.generation_policy import (
    question_requests_identifiers,
    redact_internal_identifiers,
)
from app.agent.state import CurriculumQAState
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.llm.base import LLMMessage, LLMProvider, LLMResponse


UNIT_QUESTION = "Is C4-U12 a published Class 4 Mathematics unit?"
UUID = "f293e00a-bae7-44db-ad55-3c1263e20387"


def _evidence() -> list[CurriculumEvidence]:
    return [
        CurriculumEvidence(
            entity_type="unit",
            entity_id=UUID,
            name="Number and Numeration",
            grade="CLASS_4",
            subject="MATHEMATICS",
            metadata={"code": "C4-U01"},
        )
    ]


def test_a_named_unit_code_is_kept_and_other_identifiers_are_removed():
    answer = (
        "Class 4 Mathematics lists C4-U01 through C4-U28, but not C4-U12. "
        f"Record {UUID} and grade_curriculum_id are internal."
    )
    cleaned = redact_internal_identifiers(answer, _evidence(), question=UNIT_QUESTION)
    assert "C4-U12" in cleaned
    assert "C4-U01" not in cleaned
    assert "C4-U28" not in cleaned
    assert UUID not in cleaned
    assert "grade_curriculum_id" not in cleaned
    assert question_requests_identifiers(UNIT_QUESTION) is False


def test_another_grade_unit_code_in_the_question_is_preserved():
    question = "What does C3-U15 require?"
    cleaned = redact_internal_identifiers(
        "C3-U15 is published. C3-U16 and C3U15-LO02 are not requested.",
        question=question,
    )
    assert "C3-U15" in cleaned
    assert "C3-U16" not in cleaned
    assert "C3U15-LO02" not in cleaned


def test_an_ordinary_question_still_loses_unit_codes_and_internal_ids():
    cleaned = redact_internal_identifiers(
        f"See C4-U01 and {UUID}.",
        _evidence(),
        question="What is taught in Class 4 Mathematics?",
    )
    assert "C4-U01" not in cleaned
    assert UUID not in cleaned


def test_redaction_without_a_question_still_removes_every_unit_code():
    cleaned = redact_internal_identifiers("Units C4-U01 and C4-U12.", _evidence())
    assert "C4-U01" not in cleaned
    assert "C4-U12" not in cleaned


def test_an_explicit_identifier_request_still_skips_redaction():
    question = "Give the unit codes for Class 4 Mathematics."
    assert question_requests_identifiers(question) is True
    published = AnswerGenerator(_FixedAnswer(
        f"C4-U01 and {UUID}."
    )).generate(_math_state(question))
    assert "C4-U01" in (published.answer or "")
    assert UUID in (published.answer or "")


def test_unrequested_codes_do_not_leave_empty_parentheses():
    answer = (
        "The units include codes such as C4-U01 (Everyday Arithmetic Addition "
        "and Subtraction) and C4-U02 (Everyday Arithmetic Multiplication), "
        "but no C4-U12. Internal marker grade_curriculum_id and an empty () should disappear."
    )
    cleaned = redact_internal_identifiers(answer, _evidence(), question=UNIT_QUESTION)
    assert "C4-U12" in cleaned
    assert "C4-U01" not in cleaned
    assert "C4-U02" not in cleaned
    assert "()" not in cleaned
    assert "grade_curriculum_id" not in cleaned
    assert "Everyday Arithmetic Addition and Subtraction" in cleaned
    assert "(Everyday Arithmetic" not in cleaned


def test_generate_keeps_only_the_code_the_question_names():
    published = AnswerGenerator(_FixedAnswer(
        f"Not C4-U12. Also C4-U01 and {UUID}."
    )).generate(_math_state(UNIT_QUESTION))
    assert "C4-U12" in (published.answer or "")
    assert "C4-U01" not in (published.answer or "")
    assert UUID not in (published.answer or "")


class _FixedAnswer(LLMProvider):
    def __init__(self, answer: str) -> None:
        self._answer = answer

    @property
    def name(self) -> str:
        return "fixed"

    @property
    def model(self) -> str:
        return "fixed"

    def generate(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        return LLMResponse(content=self._answer)

    def generate_structured(self, messages: list[LLMMessage], **kwargs) -> dict:
        return {"answer": self._answer, "confidence": "low", "evidence": []}

    def generate_with_tools(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        return LLMResponse(content=self._answer)


def _math_state(question: str) -> CurriculumQAState:
    state = CurriculumQAState.initial(question=question)
    state.grade = "CLASS_4"
    state.subject = "MATHEMATICS"
    state.evidence = _evidence()
    state.evidence_status = EvidenceStatus.FOUND
    return state
