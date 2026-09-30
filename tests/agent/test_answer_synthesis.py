"""User-facing curriculum answer synthesis."""

from __future__ import annotations

import re

from app.agent.answer_generator import AnswerGenerator
from app.agent.state import CurriculumQAState
from app.agent.verifier import AnswerVerifier
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.llm.base import LLMProvider, LLMResponse
from app.llm.provider import StubLLMProvider

_GARBLED_MULTIPLY = (
    "Multiply like fractions with denominators up to multiply like fractions "
    "with denominators up to multiply related fractions"
)
_GARBLED_COMPARE = (
    "Compare fractions greater than compare fractions greater than"
)
_CODE_RE = re.compile(r"C4-U\d+|C4U\d+-LO\d+|grade_curriculum_id", re.I)
_BULLET_RE = re.compile(r"(?m)^\s*\*\s+")


def _state(question: str, evidence: list[CurriculumEvidence]) -> CurriculumQAState:
    state = CurriculumQAState.initial(question=question)
    state.grade = "CLASS_4"
    state.subject = "MATHEMATICS"
    state.evidence = evidence
    state.evidence_status = EvidenceStatus.FOUND
    return state


def _unit(entity_id: str, name: str, code: str) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="unit",
        entity_id=entity_id,
        name=name,
        grade="CLASS_4",
        subject="MATHEMATICS",
        metadata={"code": code, "grade_curriculum_id": "gc-class-4-math"},
    )


def _outcome(
    entity_id: str,
    code: str,
    topic: str,
    content: str,
    parent_code: str,
) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id=entity_id,
        name=code,
        grade="CLASS_4",
        subject="MATHEMATICS",
        topic=topic,
        content=content,
        metadata={
            "code": code,
            "parent_content_code": parent_code,
            "grade_curriculum_id": "gc-class-4-math",
        },
    )


def _primary4_fractions_evidence() -> list[CurriculumEvidence]:
    return [
        _unit("unit-fraction", "Fraction", "C4-U04"),
        _unit("unit-operations", "Operation on Fractions", "C4-U05"),
        _unit("unit-multiplication", "Fraction Multiplication", "C4-U06"),
        _outcome(
            "lo-simplify",
            "C4U04-LO01",
            "Fraction",
            "Simplify like fractions with common denominators.",
            "C4-U04",
        ),
        _outcome(
            "lo-equivalent",
            "C4U04-LO02",
            "Fraction",
            "Compare and order fractions and identify equivalent fractions.",
            "C4-U04",
        ),
        _outcome(
            "lo-add",
            "C4U05-LO01",
            "Operation on Fractions",
            "Add and subtract fractions.",
            "C4-U05",
        ),
        _outcome(
            "lo-problems",
            "C4U05-LO02",
            "Operation on Fractions",
            "Solve problems involving addition and subtraction of fractions.",
            "C4-U05",
        ),
        _outcome(
            "lo-multiply-garbled",
            "C4U06-LO02",
            "Fraction Multiplication",
            _GARBLED_MULTIPLY,
            "C4-U06",
        ),
    ]


def _generate(question: str, evidence: list[CurriculumEvidence]):
    return AnswerGenerator(StubLLMProvider()).generate(_state(question, evidence))


def _assert_no_internal_identifiers(answer: str, evidence: list[CurriculumEvidence]) -> None:
    assert _CODE_RE.search(answer) is None
    for item in evidence:
        if item.entity_id:
            assert item.entity_id not in answer
        code = (item.metadata or {}).get("code")
        if code:
            assert str(code) not in answer
        hidden = (item.metadata or {}).get("grade_curriculum_id")
        if hidden:
            assert str(hidden) not in answer


def test_primary4_fractions_is_synthesized():
    evidence = _primary4_fractions_evidence()
    result = _generate(
        "What should a Primary 4 pupil learn about fractions?",
        evidence,
    )
    answer = result.answer
    lowered = answer.lower()
    assert "fraction" in lowered
    assert "operation on fractions" in lowered
    assert "fraction multiplication" in lowered
    assert "simplify" in lowered
    assert "equivalent" in lowered
    assert "add" in lowered and "subtract" in lowered
    assert "multiply" in lowered
    bullets = _BULLET_RE.findall(answer)
    outcome_count = sum(
        1 for item in evidence if item.entity_type == "learning_outcome"
    )
    assert len(bullets) < outcome_count
    assert "denominators up to multiply" not in lowered
    assert "related fractions" not in lowered
    combined = lowered + " " + " ".join(result.limitations).lower()
    assert "unreliable" in combined
    _assert_no_internal_identifiers(answer, evidence)
    evidence_ids = {ref.entity_id for ref in result.evidence}
    assert {
        "unit-fraction",
        "unit-operations",
        "unit-multiplication",
        "lo-simplify",
        "lo-equivalent",
        "lo-add",
        "lo-problems",
        "lo-multiply-garbled",
    } <= evidence_ids


def test_multiple_units_form_one_explanation():
    evidence = [
        item
        for item in _primary4_fractions_evidence()
        if item.entity_id
        not in {"unit-multiplication", "lo-multiply-garbled"}
    ]
    result = _generate(
        "What should a Primary 4 pupil learn about fractions?",
        evidence,
    )
    answer = result.answer.lower()
    assert "fraction" in answer
    assert "operation on fractions" in answer
    assert "simplify" in answer
    assert "equivalent" in answer
    assert "add" in answer and "subtract" in answer
    bullets = _BULLET_RE.findall(result.answer)
    outcome_count = sum(
        1 for item in evidence if item.entity_type == "learning_outcome"
    )
    assert 1 < len(bullets) < outcome_count
    assert "**—**" not in result.answer
    _assert_no_internal_identifiers(result.answer, evidence)


def test_explicit_identifier_question_keeps_lo_code():
    evidence = [
        _unit("unit-fraction", "Fraction", "C4-U04"),
        _outcome(
            "lo-simplify",
            "C4U04-LO01",
            "Fraction",
            "Simplify like fractions with common denominators.",
            "C4-U04",
        ),
        _outcome(
            "lo-add",
            "C4U05-LO01",
            "Operation on Fractions",
            "Add and subtract fractions.",
            "C4-U05",
        ),
    ]
    result = _generate(
        "What is the learning-objective code for simplifying like fractions in Primary 4?",
        evidence,
    )
    assert "C4U04-LO01" in result.answer
    assert "C4U05-LO01" not in result.answer
    assert "lo-simplify" in {ref.entity_id for ref in result.evidence}


def test_malformed_evidence_is_not_repaired():
    evidence = [
        _unit("unit-compare", "Comparing fractions", "C4-U04"),
        _outcome(
            "lo-compare-garbled",
            "C4U04-LO04",
            "Comparing fractions",
            _GARBLED_COMPARE,
            "C4-U04",
        ),
    ]
    result = _generate(
        "What should a Primary 4 pupil learn about comparing fractions?",
        evidence,
    )
    lowered = result.answer.lower()
    assert "compare fractions" in lowered
    assert "one half" not in lowered
    assert "likely means" not in lowered
    assert "probably" not in lowered
    assert "the intended objective is" not in lowered
    assert "greater than compare" not in lowered
    combined = lowered + " " + " ".join(result.limitations).lower()
    assert "unreliable" in combined
    _assert_no_internal_identifiers(result.answer, evidence)


def test_synthesis_does_not_add_unsupported_concepts():
    evidence = [
        _unit("unit-fraction", "Fraction", "C4-U04"),
        _unit("unit-operations", "Operation on Fractions", "C4-U05"),
        _outcome(
            "lo-simplify",
            "C4U04-LO01",
            "Fraction",
            "Simplify like fractions with common denominators.",
            "C4-U04",
        ),
        _outcome(
            "lo-equivalent",
            "C4U04-LO02",
            "Fraction",
            "Compare and order fractions and identify equivalent fractions.",
            "C4-U04",
        ),
        _outcome(
            "lo-add",
            "C4U05-LO01",
            "Operation on Fractions",
            "Add and subtract fractions.",
            "C4-U05",
        ),
    ]
    result = _generate(
        "What should a Primary 4 pupil learn about fractions?",
        evidence,
    )
    lowered = result.answer.lower()
    assert "simplify" in lowered
    assert "equivalent" in lowered
    assert "subtract" in lowered
    for unsupported in ("division", "decimal", "percentage", "percent"):
        assert unsupported not in lowered


def test_ordinary_answer_hides_internal_identifiers():
    evidence = _primary4_fractions_evidence()
    result = _generate(
        "What should a Primary 4 pupil learn about fractions?",
        evidence,
    )
    _assert_no_internal_identifiers(result.answer, evidence)
    assert re.search(r"C4-U\d+", result.answer) is None
    assert re.search(r"C4U\d+-LO\d+", result.answer) is None
    assert "grade_curriculum_id" not in result.answer
    for ref in result.evidence:
        assert ref.entity_id


def test_model_code_dump_is_redacted_for_ordinary_questions():
    evidence = _primary4_fractions_evidence()
    payload = {
        "answer": (
            "- **C4U04-LO01** — Simplify like fractions with common denominators.\n"
            "- **C4-U06** unit-multiplication grade_curriculum_id"
        ),
        "confidence": "medium",
        "evidence": [
            {
                "entity_id": "lo-simplify",
                "entity_type": "learning_outcome",
                "claim": "Simplify like fractions.",
            }
        ],
        "limitations": [],
    }

    class DumpingLLM(LLMProvider):
        @property
        def name(self) -> str:
            return "openai"

        @property
        def model(self) -> str:
            return "test-model"

        def generate(self, messages, *, temperature=0.0, max_tokens=None) -> LLMResponse:
            return LLMResponse(content="unused")

        def generate_structured(self, messages, *, schema, temperature=0.0) -> dict:
            return payload

        def generate_with_tools(self, messages, *, tools, temperature=0.0) -> LLMResponse:
            return LLMResponse(content="unused")

    result = AnswerGenerator(DumpingLLM()).generate(
        _state("What should a Primary 4 pupil learn about fractions?", evidence)
    )
    _assert_no_internal_identifiers(result.answer, evidence)
    assert "simplify like fractions" in result.answer.lower()
    assert result.evidence[0].entity_id == "lo-simplify"


def test_verifier_reads_the_synthesized_prose():
    evidence = _primary4_fractions_evidence()
    state = _state(
        "What should a Primary 4 pupil learn about fractions?",
        evidence,
    )
    generated = AnswerGenerator(StubLLMProvider()).generate(state)
    state.draft_answer = generated.answer
    state.final_answer = generated.answer
    state.answer_evidence = list(generated.evidence)
    messages = AnswerVerifier(StubLLMProvider()).build_messages(state)
    user = messages[1].content or ""
    assert state.final_answer == state.draft_answer == generated.answer
    assert generated.answer in user
    assert "C4U04-LO01" not in generated.answer
    assert "C4U04-LO01" in user
