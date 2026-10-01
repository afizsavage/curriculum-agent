"""Generation-window unit dedupe and conditional evidence-note prompt."""

from app.agent.answer_generator import (
    AnswerGenerator,
    dedupe_units_for_generation,
    format_evidence_for_prompt,
    select_evidence_for_prompt,
)
from app.agent.state import CurriculumQAState
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.llm.provider import StubLLMProvider

# Primary 3 Mathematics grade-curriculum units, in tree order.
_P3_UNITS = [
    ("C3-U01", "Number and Numeration"),
    ("C3-U02", "Number and Numeration"),
    ("C3-U03", "Number and Numeration."),
    ("C3-U04", "Number and Numeration"),
    ("C3-U05", "Number and Numeration"),
    ("C3-U06", "Number and Numeration writing numbers"),
    ("C3-U07", "Fraction."),
    ("C3-U08", "Fraction"),
    ("C3-U09", "Fraction"),
    *[("C3-U%02d" % index, "Everyday Arithmetic") for index in range(10, 25)],
    ("C3-U25", "Measurement and estimation"),
    ("C3-U26", "Measurement and estimation"),
    ("C3-U27", "Measurement and estimation"),
    ("C3-U28", "Measurement and estimation"),
    ("C3-U29", "Measurement and estimation."),
    ("C3-U30", "Measurement and Estimation."),
    ("C3-U31", "Measurement and Estimation."),
    ("C3-U32", "Time"),
    ("C3-U33", "Time"),
    ("C3-U34", "Time"),
    ("C3-U35", "GEOMETRY (SHAPES)"),
    ("C3-U36", "GEOMETRY (SHAPES)"),
    ("C3-U37", "GEOMETRY (ANGLES)"),
    ("C3-U38", "Number patterns."),
    ("C3-U39", "NUMBER PATTERNS."),
]

_QUESTION = "What are the topics in Primary 3 Mathematics?"


def _unit(code: str, name: str) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="unit",
        entity_id=code.lower(),
        name=name,
        grade="CLASS_3",
        subject="MATHEMATICS",
        topic=name,
        content=name,
        metadata={"code": code},
        source_reference="grade_curriculum.content",
    )


def _p3_evidence() -> list[CurriculumEvidence]:
    return [_unit(code, name) for code, name in _P3_UNITS]


def _names(evidence: list[CurriculumEvidence]) -> list[str]:
    return [item.name or "" for item in evidence]


def test_primary3_topic_window_keeps_every_distinct_unit_name():
    evidence = _p3_evidence()
    assert len(evidence) == 39
    prepared = dedupe_units_for_generation(evidence)
    assert len(prepared) == 9
    assert _names(prepared) == [
        "Number and Numeration",
        "Number and Numeration writing numbers",
        "Fraction.",
        "Everyday Arithmetic",
        "Measurement and estimation",
        "Time",
        "GEOMETRY (SHAPES)",
        "GEOMETRY (ANGLES)",
        "Number patterns.",
    ]
    selected, _ids = select_evidence_for_prompt(evidence, question=_QUESTION)
    block = format_evidence_for_prompt(evidence, question=_QUESTION)
    assert len(selected) == 9
    for name in (
        "Measurement and estimation",
        "Time",
        "GEOMETRY (SHAPES)",
        "GEOMETRY (ANGLES)",
        "Number patterns.",
    ):
        assert name in block
    assert block.count("Name: Everyday Arithmetic\n") == 1


def test_distinct_unit_names_are_not_merged():
    evidence = _p3_evidence()
    evidence.append(_unit("C3-U40", "Fractions"))
    prepared = dedupe_units_for_generation(evidence)
    names = _names(prepared)
    assert "GEOMETRY (SHAPES)" in names
    assert "GEOMETRY (ANGLES)" in names
    assert "Number and Numeration" in names
    assert "Number and Numeration writing numbers" in names
    assert names.count("Fraction.") == 1
    assert "Fractions" not in names
    assert len(prepared) == 9


def test_full_audit_evidence_is_not_replaced():
    evidence = _p3_evidence()
    original_ids = [item.entity_id for item in evidence]
    selected, _ids = select_evidence_for_prompt(evidence, question=_QUESTION)
    assert len(evidence) == 39
    assert [item.entity_id for item in evidence] == original_ids
    assert selected is not evidence
    assert len(selected) == 9


def test_generation_cap_stays_at_24_distinct_records():
    assert select_evidence_for_prompt.__kwdefaults__["max_records"] == 24
    distinct = [
        _unit(f"C3-U{index:02d}", f"Distinct topic {index}")
        for index in range(1, 31)
    ]
    selected, _ids = select_evidence_for_prompt(distinct, question=_QUESTION)
    assert len(distinct) == 30
    assert len(selected) == 24
    p3 = _p3_evidence()
    window, _ids = select_evidence_for_prompt(p3, question=_QUESTION, max_records=24)
    assert len(window) == 9
    assert len(window) < 24


def test_clean_topic_list_omits_the_evidence_note_heading():
    evidence = _p3_evidence()
    state = CurriculumQAState.initial(question=_QUESTION)
    state.grade = "CLASS_3"
    state.subject = "MATHEMATICS"
    state.evidence = evidence
    state.evidence_status = EvidenceStatus.FOUND
    result = AnswerGenerator(StubLLMProvider()).generate(state)
    assert "### Curriculum Evidence Note" not in result.answer
    assert len(state.evidence) == 39
    messages = AnswerGenerator(StubLLMProvider()).build_messages(state)
    user = messages[1].content or ""
    assert "### Curriculum Evidence Note\nInclude the evidence note only when" not in user
    assert "Do not output that heading when no note is necessary" in user
    assert "Measurement and estimation" in user
    assert "GEOMETRY (ANGLES)" in user


def test_damaged_evidence_can_still_include_an_evidence_note():
    evidence = [
        CurriculumEvidence(
            entity_type="learning_outcome",
            entity_id="lo-garbled",
            name="C4U04-LO04",
            grade="CLASS_4",
            subject="MATHEMATICS",
            topic="Comparing fractions",
            content=(
                "Compare fractions with denominators up to compare fractions "
                "with denominators up to"
            ),
            metadata={"code": "C4U04-LO04"},
        )
    ]
    state = CurriculumQAState.initial(
        question="What should a Primary 4 pupil learn about comparing fractions?"
    )
    state.grade = "CLASS_4"
    state.subject = "MATHEMATICS"
    state.evidence = evidence
    state.evidence_status = EvidenceStatus.FOUND
    result = AnswerGenerator(StubLLMProvider()).generate(state)
    assert "### Curriculum Evidence Note" in result.answer
