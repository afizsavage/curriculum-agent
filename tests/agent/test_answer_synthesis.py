"""User-facing curriculum answer synthesis."""

from __future__ import annotations

import re

from app.agent.answer import AnswerGenerationNode
from app.agent.answer_generator import AnswerGenerator, _render_stub_answer
from app.agent.verify import VerificationNode
from app.agent.verifier import AnswerVerifier
from app.config import Settings
from app.agent.state import CurriculumQAState
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
_CODE_RE = re.compile(
    r"C3-U\d+|C3U\d+-LO\d+|C4-U\d+|C4U\d+-LO\d+|grade_curriculum_id",
    re.I,
)
_BULLET_RE = re.compile(r"(?m)^\s*\*\s+")
_SECTION_RE = re.compile(r"(?m)^### \d+\. ")


def _state(
    question: str,
    evidence: list[CurriculumEvidence],
    *,
    grade: str = "CLASS_4",
) -> CurriculumQAState:
    state = CurriculumQAState.initial(question=question)
    state.grade = grade
    state.subject = "MATHEMATICS"
    state.evidence = evidence
    state.evidence_status = EvidenceStatus.FOUND
    return state


def _unit(
    entity_id: str,
    name: str,
    code: str,
    *,
    grade: str = "CLASS_4",
) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="unit",
        entity_id=entity_id,
        name=name,
        grade=grade,
        subject="MATHEMATICS",
        metadata={"code": code, "grade_curriculum_id": f"gc-{grade.lower()}-math"},
    )


def _outcome(
    entity_id: str,
    code: str,
    topic: str,
    content: str,
    parent_code: str,
    *,
    grade: str = "CLASS_4",
) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id=entity_id,
        name=code,
        grade=grade,
        subject="MATHEMATICS",
        topic=topic,
        content=content,
        metadata={
            "code": code,
            "parent_content_code": parent_code,
            "grade_curriculum_id": f"gc-{grade.lower()}-math",
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


def _primary3_fractions_evidence() -> list[CurriculumEvidence]:
    grade = "CLASS_3"
    return [
        _unit("unit-p3-unit-fractions", "Unit Fractions", "C3-U04", grade=grade),
        _unit(
            "unit-p3-non-unit",
            "Unit and Non-Unit Fractions",
            "C3-U05",
            grade=grade,
        ),
        _unit(
            "unit-p3-operations",
            "Equivalent Fractions and Fraction Operations",
            "C3-U06",
            grade=grade,
        ),
        _outcome(
            "lo-p3-den-1-5",
            "C3U04-LO01",
            "Unit Fractions",
            "Identify unit fractions with denominators 1-5 using pictorial representations.",
            "C3-U04",
            grade=grade,
        ),
        _outcome(
            "lo-p3-den-6-10",
            "C3U04-LO02",
            "Unit Fractions",
            "Identify unit fractions with denominators 6-10 using pictorial representations.",
            "C3-U04",
            grade=grade,
        ),
        _outcome(
            "lo-p3-number-line",
            "C3U04-LO03",
            "Unit Fractions",
            "Locate unit fractions on the number line.",
            "C3-U04",
            grade=grade,
        ),
        _outcome(
            "lo-p3-non-unit",
            "C3U05-LO01",
            "Unit and Non-Unit Fractions",
            "Identify unit and non-unit fractions with denominators 2-10.",
            "C3-U05",
            grade=grade,
        ),
        _outcome(
            "lo-p3-pictorial",
            "C3U05-LO02",
            "Unit and Non-Unit Fractions",
            "Represent these fractions pictorially.",
            "C3-U05",
            grade=grade,
        ),
        _outcome(
            "lo-p3-locate",
            "C3U05-LO03",
            "Unit and Non-Unit Fractions",
            "Locate and identify fractions on the number line.",
            "C3-U05",
            grade=grade,
        ),
        _outcome(
            "lo-p3-equivalent",
            "C3U06-LO01",
            "Equivalent Fractions and Fraction Operations",
            "Work with equivalent fractions.",
            "C3-U06",
            grade=grade,
        ),
        _outcome(
            "lo-p3-add",
            "C3U06-LO02",
            "Equivalent Fractions and Fraction Operations",
            "Add like fractions.",
            "C3-U06",
            grade=grade,
        ),
        _outcome(
            "lo-p3-subtract",
            "C3U06-LO03",
            "Equivalent Fractions and Fraction Operations",
            "Subtract like fractions.",
            "C3-U06",
            grade=grade,
        ),
        _outcome(
            "lo-p3-word-problems",
            "C3U06-LO04",
            "Equivalent Fractions and Fraction Operations",
            "Solve word problems involving addition and subtraction of like fractions.",
            "C3-U06",
            grade=grade,
        ),
        _outcome(
            "lo-p3-equivalent-incomplete",
            "C3U06-LO05",
            "Equivalent Fractions and Fraction Operations",
            "Identify equivalent fractions with denominators up to",
            "C3-U06",
            grade=grade,
        ),
    ]


def _generate(
    question: str,
    evidence: list[CurriculumEvidence],
    *,
    grade: str = "CLASS_4",
):
    return AnswerGenerator(StubLLMProvider()).generate(
        _state(question, evidence, grade=grade)
    )


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
    assert answer.startswith("# Primary 4 Mathematics — Fractions")
    sections = _SECTION_RE.findall(answer)
    outcome_count = sum(
        1 for item in evidence if item.entity_type == "learning_outcome"
    )
    assert 1 < len(sections) < outcome_count
    assert "### Curriculum Evidence Note" in answer
    assert "denominators up to multiply" not in lowered
    assert "related fractions" not in lowered
    combined = lowered + " " + " ".join(result.limitations).lower()
    assert "cannot be confirmed" in combined or "incomplete" in combined
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


def test_primary3_fractions_presentation():
    evidence = _primary3_fractions_evidence()
    result = _generate(
        "What should a Primary 3 pupil learn about fractions?",
        evidence,
        grade="CLASS_3",
    )
    answer = result.answer
    lowered = answer.lower()
    assert answer.startswith("# Primary 3 Mathematics — Fractions")
    assert "organised into 3 main areas" in lowered
    assert "### 1. Unit Fractions" in answer
    assert "### 2. Unit and Non-Unit Fractions" in answer
    assert "### 3. Equivalent Fractions and Fraction Operations" in answer
    assert "Pupils learn to:" in answer
    assert "denominators 1-5" in lowered
    assert "denominators 6-10" in lowered
    assert "denominators 2-10" in lowered
    assert "number line" in lowered
    assert "equivalent fractions" in lowered
    assert "add like fractions" in lowered
    assert "subtract like fractions" in lowered
    assert "word problems" in lowered
    assert "### Curriculum Evidence Note" in answer
    assert "cannot be confirmed" in lowered
    assert "denominator range" in lowered
    assert "up to 12" not in lowered
    assert "up to 8" not in lowered
    for invented in ("decimal", "mixed number", "multiplication"):
        assert invented not in lowered
    sections = _SECTION_RE.findall(answer)
    outcome_count = sum(
        1 for item in evidence if item.entity_type == "learning_outcome"
    )
    assert len(sections) < outcome_count
    assert len(_BULLET_RE.findall(answer)) >= 8
    assert "Identify equivalent fractions." not in answer
    assert "lo-p3-equivalent-incomplete" in {ref.entity_id for ref in result.evidence}
    _assert_no_internal_identifiers(answer, evidence)


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
    sections = _SECTION_RE.findall(result.answer)
    outcome_count = sum(
        1 for item in evidence if item.entity_type == "learning_outcome"
    )
    assert len(sections) == 2
    assert len(sections) < outcome_count
    assert _BULLET_RE.search(result.answer)
    assert "**—**" not in result.answer
    assert "### Curriculum Evidence Note" not in result.answer
    _assert_no_internal_identifiers(result.answer, evidence)


def test_simple_question_stays_concise():
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
            "lo-compare",
            "C4U04-LO02",
            "Fraction",
            "Compare and order like fractions.",
            "C4-U04",
        ),
    ]
    result = _generate(
        "What should a Primary 4 pupil learn about like fractions?",
        evidence,
    )
    answer = result.answer
    assert answer.startswith("# ")
    assert "### " not in answer
    assert "Curriculum Evidence Note" not in answer
    assert "organised into" not in answer.lower()
    assert len(answer) < 700
    assert "simplify like fractions" in answer.lower()
    assert "compare and order like fractions" in answer.lower()
    _assert_no_internal_identifiers(answer, evidence)


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
        "What is the learning-objective code for simplifying like fractions?",
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
    assert "### Curriculum Evidence Note" in result.answer
    combined = lowered + " " + " ".join(result.limitations).lower()
    assert "cannot be confirmed" in combined or "incomplete" in combined
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
    cases = [
        (
            "What should a Primary 4 pupil learn about fractions?",
            _primary4_fractions_evidence(),
            "CLASS_4",
        ),
        (
            "What should a Primary 3 pupil learn about fractions?",
            _primary3_fractions_evidence(),
            "CLASS_3",
        ),
    ]
    for question, evidence, grade in cases:
        result = _generate(question, evidence, grade=grade)
        _assert_no_internal_identifiers(result.answer, evidence)
        assert re.search(r"C3-U\d+", result.answer) is None
        assert re.search(r"C3U\d+-LO\d+", result.answer) is None
        assert re.search(r"C4-U\d+", result.answer) is None
        assert re.search(r"C4U\d+-LO\d+", result.answer) is None
        assert "grade_curriculum_id" not in result.answer
        for ref in result.evidence:
            assert ref.entity_id
            assert ref.entity_id not in result.answer


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


def test_later_evidence_is_referenced_when_it_supports_a_claim():
    evidence = [
        _outcome(
            f"lo-skill-{index}",
            f"C4U04-LO{index:02d}",
            "Fractions",
            f"Describe fraction skill {index} using diagrams.",
            "C4-U04",
        )
        for index in range(1, 10)
    ]
    evidence.append(
        CurriculumEvidence(
            entity_type="annotation",
            entity_id="unused-later-record",
            name="Administrative note",
            content="This record is not a learning expectation.",
            grade="CLASS_4",
            subject="MATHEMATICS",
            metadata={"grade_curriculum_id": "gc-class-4-math"},
        )
    )
    assert len(evidence) > 8
    result = _generate(
        "What should a Primary 4 pupil learn about fractions?",
        evidence,
    )
    ids = {ref.entity_id for ref in result.evidence}
    assert "lo-skill-9" in ids
    assert "Describe fraction skill 9 using diagrams." in result.answer
    assert "unused-later-record" not in ids
    assert "Administrative note" not in result.answer


def test_near_duplicate_expectations_are_not_repeated():
    evidence = [
        _unit("unit-equivalent", "Equivalent Fractions", "C4-U04"),
        _outcome(
            "lo-eq-work",
            "C4U04-LO01",
            "Equivalent Fractions",
            "Work with equivalent fractions.",
            "C4-U04",
        ),
        _outcome(
            "lo-eq-identify",
            "C4U04-LO02",
            "Equivalent Fractions",
            "Identify equivalent fractions.",
            "C4-U04",
        ),
        _outcome(
            "lo-add",
            "C4U04-LO03",
            "Equivalent Fractions",
            "Add like fractions.",
            "C4-U04",
        ),
        _outcome(
            "lo-den-1",
            "C4U04-LO04",
            "Equivalent Fractions",
            "Identify unit fractions with denominators 1-5 using pictorial representations.",
            "C4-U04",
        ),
        _outcome(
            "lo-den-2",
            "C4U04-LO05",
            "Equivalent Fractions",
            "Identify unit fractions with denominators 6-10 using pictorial representations.",
            "C4-U04",
        ),
    ]
    result = _generate(
        "What should a Primary 4 pupil learn about fractions?",
        evidence,
    )
    assert "Work with equivalent fractions." in result.answer
    assert "Identify equivalent fractions." not in result.answer
    assert "Add like fractions." in result.answer
    assert "denominators 1-5" in result.answer
    assert "denominators 6-10" in result.answer
    ids = {ref.entity_id for ref in result.evidence}
    assert {"lo-eq-work", "lo-eq-identify", "lo-add", "lo-den-1", "lo-den-2"} <= ids


def test_awkward_unit_name_becomes_a_descriptive_heading():
    evidence = [
        _unit("unit-awkward", "Number and Numeration FRACTION", "C4-U04"),
        _outcome(
            "lo-simplify",
            "C4U04-LO01",
            "",
            "Simplify like fractions with common denominators.",
            "C4-U04",
        ),
        _outcome(
            "lo-compare",
            "C4U04-LO02",
            "",
            "Compare and order like fractions.",
            "C4-U04",
        ),
        _outcome(
            "lo-add",
            "C4U04-LO03",
            "",
            "Add like fractions.",
            "C4-U04",
        ),
        _outcome(
            "lo-subtract",
            "C4U04-LO04",
            "",
            "Subtract like fractions.",
            "C4-U04",
        ),
    ]
    result = _generate(
        "What should a Primary 4 pupil learn about fractions?",
        evidence,
    )
    assert "### Fractions" in result.answer
    assert "Number and Numeration" not in result.answer
    assert "C4-U04" not in result.answer
    assert "Simplify like fractions with common denominators." in result.answer
    assert "unit-awkward" in {ref.entity_id for ref in result.evidence}
    assert "lo-subtract" in {ref.entity_id for ref in result.evidence}


def test_stub_synthesis_is_structured_before_redaction():
    evidence = _primary4_fractions_evidence()
    state = _state(
        "What should a Primary 4 pupil learn about fractions?",
        evidence,
    )
    text, _limitations, _used = _render_stub_answer(
        state,
        grade_label="Primary 4",
        subject_label="Mathematics",
    )
    assert text.startswith("# Primary 4 Mathematics — Fractions")
    assert "### 1. Fractions" in text
    assert "Pupils learn to:" in text
    assert "* Simplify like fractions with common denominators." in text
    assert "C4U04-LO01" not in text
    assert "C4-U04" not in text
    assert not re.search(r"\*\*C\d+U\d+-LO\d+\*\*", text)


def test_end_to_end_synthesis_integrity():
    evidence = _primary3_fractions_evidence()
    evidence.append(
        CurriculumEvidence(
            entity_type="annotation",
            entity_id="unused-later-record",
            name="Administrative note",
            content="Not a learning expectation.",
            grade="CLASS_3",
            subject="MATHEMATICS",
            metadata={"grade_curriculum_id": "gc-class-3-math", "evidence_state": "UNUSED"},
        )
    )
    state = _state(
        "What should a Primary 3 pupil learn about fractions?",
        evidence,
        grade="CLASS_3",
    )
    settings = Settings()
    llm = StubLLMProvider()
    state = AnswerGenerationNode(llm=llm, settings=settings).run(state)
    synthesized = state.final_answer or ""
    state = VerificationNode(llm=llm, settings=settings).run(state)
    messages = AnswerVerifier(llm, settings=settings).build_messages(state)
    user = messages[1].content or ""
    ids = {ref.entity_id for ref in state.answer_evidence}

    assert state.final_answer == state.draft_answer == synthesized
    assert synthesized in user
    _assert_no_internal_identifiers(synthesized, evidence)
    assert "denominators 1-5" in synthesized
    assert "lo-p3-den-1-5" in ids
    assert "lo-p3-equivalent-incomplete" in ids
    assert "unused-later-record" not in ids
    assert "up to 12" not in synthesized.lower()
    assert "denominators up to" not in synthesized.lower()
    for hidden in ("grade_curriculum_id", "evidence_state", "entity_id", "UNUSED"):
        assert hidden not in synthesized
