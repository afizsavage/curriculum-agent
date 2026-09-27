import json
import re

import pytest

from app.agent.answer_generator import AnswerGenerator, format_evidence_for_prompt
from app.agent.state import CurriculumQAState
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.exceptions import LLMProviderError
from app.llm.base import LLMMessage, LLMProvider, LLMResponse
from app.llm.provider import StubLLMProvider
from app.schemas.answer import GROUNDED_ANSWER_JSON_SCHEMA, AnswerConfidence


def _state_with_evidence(**kwargs) -> CurriculumQAState:
    state = CurriculumQAState.initial(
        question=kwargs.pop("question", "What topics are in Primary 4 Mathematics?")
    )
    state.grade = kwargs.pop("grade", "CLASS_4")
    state.subject = kwargs.pop("subject", "MATHEMATICS")
    state.evidence = kwargs.pop("evidence", [])
    state.evidence_status = kwargs.pop("evidence_status", EvidenceStatus.FOUND)
    return state


def test_format_evidence_preserves_hierarchy():
    evidence = [
        CurriculumEvidence(
            entity_type="topic",
            entity_id="topic-1",
            name="Fractions",
            level="primary",
            grade="CLASS_4",
            subject="MATHEMATICS",
            content="Identify equivalent fractions.",
            source_reference="learning_outcomes",
        )
    ]
    block = format_evidence_for_prompt(evidence)
    assert "Entity ID: topic-1" in block
    assert "Fractions" in block
    assert "CLASS_4" in block or "Primary 4" in block
    assert "Identify equivalent fractions" in block


def test_empty_evidence_returns_insufficient_answer():
    state = _state_with_evidence(evidence=[], evidence_status=EvidenceStatus.NOT_FOUND)
    result = AnswerGenerator(StubLLMProvider()).generate(state)
    assert "couldn't find sufficient MBSSE curriculum evidence" in result.answer
    assert result.confidence == AnswerConfidence.LOW
    assert result.limitations


def test_stub_generates_answer_from_evidence():
    evidence = [
        CurriculumEvidence(
            entity_type="subject",
            entity_id="sub-1",
            name="Mathematics",
            grade="CLASS_4",
            subject="MATHEMATICS",
        ),
        CurriculumEvidence(
            entity_type="topic",
            entity_id="topic-1",
            name="Fractions",
            grade="CLASS_4",
            subject="MATHEMATICS",
            content="Fractions topic",
        ),
    ]
    state = _state_with_evidence(evidence=evidence)
    result = AnswerGenerator(StubLLMProvider()).generate(state)
    assert result.answer
    assert "Fractions" in result.answer or "Mathematics" in result.answer
    assert result.evidence
    assert all(ref.entity_id in {"sub-1", "topic-1"} for ref in result.evidence)
    assert result.confidence in {AnswerConfidence.HIGH, AnswerConfidence.MEDIUM}


def test_validate_evidence_refs_rejects_invented_ids():
    evidence = [
        CurriculumEvidence(
            entity_type="topic",
            entity_id="real-id",
            name="Fractions",
        )
    ]
    generator = AnswerGenerator(StubLLMProvider())
    refs = generator._validate_evidence_refs(
        [
            {"entity_id": "real-id", "entity_type": "topic", "claim": "Valid"},
            {"entity_id": "fake-id", "entity_type": "topic", "claim": "Invalid"},
        ],
        evidence,
    )
    assert len(refs) == 1
    assert refs[0].entity_id == "real-id"


def test_grade_mismatch_downgrades_confidence():
    evidence = [
        CurriculumEvidence(
            entity_type="topic",
            entity_id="topic-x",
            name="Advanced Algebra",
            grade="CLASS_5",
            subject="MATHEMATICS",
        )
    ]
    state = _state_with_evidence(
        question="Is Advanced Algebra taught in Primary 4?",
        grade="CLASS_4",
        evidence=evidence,
    )
    result = AnswerGenerator(StubLLMProvider()).generate(state)
    assert result.confidence == AnswerConfidence.LOW
    assert any("CLASS_5" in lim for lim in result.limitations)


class StructuredLLMStub(LLMProvider):
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return "test-model"

    def generate(self, messages, *, temperature=0.0, max_tokens=None) -> LLMResponse:
        return LLMResponse(content="unused")

    def generate_structured(self, messages, *, schema, temperature=0.0) -> dict:
        return self._payload

    def generate_with_tools(self, messages, *, tools, temperature=0.0) -> LLMResponse:
        return LLMResponse(content="unused")


def test_empty_structured_answer_falls_back_to_evidence_summary():
    evidence = [
        CurriculumEvidence(entity_type="topic", entity_id="t1", name="Fractions")
    ]
    state = _state_with_evidence(evidence=evidence)
    generator = AnswerGenerator(StructuredLLMStub({"answer": ""}))
    result = generator.generate(state)
    assert result.answer
    assert "Fractions" in result.answer
    assert any("empty answer" in note.lower() for note in result.limitations)


def test_empty_structured_answer_without_evidence_raises():
    state = _state_with_evidence(evidence=[], evidence_status=EvidenceStatus.NOT_FOUND)

    class EmptyThenEmpty(StructuredLLMStub):
        def generate_structured(self, messages, *, schema, temperature=0.0) -> dict:
            raise LLMProviderError("LLM returned empty answer")

    # No evidence path uses insufficient-evidence helper before LLM.
    result = AnswerGenerator(EmptyThenEmpty({"answer": ""})).generate(state)
    assert "couldn't find sufficient" in result.answer.lower()


def test_structured_output_parsed_and_sanitized():
    evidence = [
        CurriculumEvidence(
            entity_type="topic",
            entity_id="t1",
            name="Fractions",
            grade="CLASS_4",
            subject="MATHEMATICS",
        )
    ]
    payload = {
        "answer": "Fractions is included in Primary 4 Mathematics.",
        "summary": "Fractions in P4 Math",
        "evidence": [
            {
                "entity_id": "t1",
                "entity_type": "topic",
                "claim": "Fractions is a Primary 4 Mathematics topic.",
            },
            {
                "entity_id": "invented",
                "entity_type": "topic",
                "claim": "Should be dropped.",
            },
        ],
        "limitations": [],
        "confidence": "high",
    }
    state = _state_with_evidence(evidence=evidence)
    result = AnswerGenerator(StructuredLLMStub(payload)).generate(state)
    assert result.answer.startswith("Fractions")
    assert len(result.evidence) == 1
    assert result.evidence[0].entity_id == "t1"


def test_build_messages_includes_evidence_not_full_state():
    evidence = [
        CurriculumEvidence(
            entity_type="topic",
            entity_id="t1",
            name="Fractions",
            grade="CLASS_4",
        )
    ]
    state = _state_with_evidence(evidence=evidence)
    messages = AnswerGenerator(StubLLMProvider()).build_messages(state)
    assert messages[0].role == "system"
    user = messages[1].content or ""
    assert "CURRICULUM EVIDENCE" in user
    assert "Entity ID: t1" in user
    assert "retrieval_history" not in user


def test_grounded_answer_schema_has_required_fields():
    assert "answer" in GROUNDED_ANSWER_JSON_SCHEMA["required"]
    assert "confidence" in GROUNDED_ANSWER_JSON_SCHEMA["required"]


def _subject_evidence(codes_and_names: list[tuple[str, str]], *, grade: str):
    return [
        CurriculumEvidence(
            entity_type="subject",
            entity_id=f"id-{code}",
            name=name,
            grade=grade,
            subject=code,
            metadata={"code": code, "classification": "CORE"},
        )
        for code, name in codes_and_names
    ]


def test_subject_list_heading_from_resolved_context():
    from app.agent.answer_generator import subject_list_heading

    assert (
        subject_list_heading(grade_code="CLASS_3", classification="CORE")
        == "Core Subjects Listed for Primary 3"
    )
    assert (
        subject_list_heading(grade_code="CLASS_3", classification=None)
        == "Subjects Listed for Primary 3"
    )
    assert (
        subject_list_heading(grade_code="CLASS_3", classification="NON_CORE")
        == "Non-Core Subjects Listed for Primary 3"
    )
    assert (
        subject_list_heading(grade_code="CLASS_4", classification="CORE")
        == "Core Subjects Listed for Primary 4"
    )


def test_header_core_plus_primary3():
    evidence = _subject_evidence(
        [
            ("ENGLISH", "English"),
            ("MATHEMATICS", "Mathematics"),
            ("CIVIC_EDUCATION", "Civic Education"),
        ],
        grade="CLASS_3",
    )
    state = CurriculumQAState.initial(
        question="What are the core subjects in Primary 3?"
    )
    state.grade = "CLASS_3"
    state.classification = "CORE"
    state.evidence = evidence
    state.evidence_status = EvidenceStatus.FOUND
    result = AnswerGenerator(StubLLMProvider()).generate(state)
    heading = result.answer.splitlines()[0]
    assert "Core" in heading
    assert "Primary 3" in heading
    assert "Subjects" in heading
    # Must not imply the full unfiltered Primary 3 catalogue.
    assert not re.search(r"^#+\s*Subjects Listed for Primary 3\s*$", heading)
    assert "English" in result.answer
    assert "Mathematics" in result.answer


def test_header_grade_only_omits_core():
    evidence = _subject_evidence(
        [
            ("ENGLISH", "English"),
            ("MATHEMATICS", "Mathematics"),
            ("HOME_ECONOMICS", "Home Economics"),
        ],
        grade="CLASS_3",
    )
    state = CurriculumQAState.initial(
        question="What subjects are taught in Primary 3?"
    )
    state.grade = "CLASS_3"
    state.classification = None
    state.evidence = evidence
    state.evidence_status = EvidenceStatus.FOUND
    result = AnswerGenerator(StubLLMProvider()).generate(state)
    heading = result.answer.splitlines()[0]
    assert "Primary 3" in heading
    assert "Subjects" in heading
    assert "Core" not in heading
    assert "Non-Core" not in heading


def test_header_non_core_plus_primary3():
    evidence = [
        CurriculumEvidence(
            entity_type="subject",
            entity_id="id-home",
            name="Home Economics",
            grade="CLASS_3",
            subject="HOME_ECONOMICS",
            metadata={"classification": "AVAILABLE"},
        ),
        CurriculumEvidence(
            entity_type="subject",
            entity_id="id-ict",
            name="ICT Literacy",
            grade="CLASS_3",
            subject="ICT_LITERACY",
            metadata={"classification": "AVAILABLE"},
        ),
    ]
    state = CurriculumQAState.initial(
        question="What are the non-core subjects in Primary 3?"
    )
    state.grade = "CLASS_3"
    state.classification = "NON_CORE"
    state.evidence = evidence
    state.evidence_status = EvidenceStatus.FOUND
    result = AnswerGenerator(StubLLMProvider()).generate(state)
    heading = result.answer.splitlines()[0]
    assert heading == "## Non-Core Subjects Listed for Primary 3"
    assert not heading.startswith("## Core ")


def test_header_core_primary4_is_generic():
    evidence = _subject_evidence(
        [("MATHEMATICS", "Mathematics"), ("SCIENCE", "Science")],
        grade="CLASS_4",
    )
    state = CurriculumQAState.initial(
        question="What are the core subjects in Primary 4?"
    )
    state.grade = "CLASS_4"
    state.classification = "CORE"
    state.evidence = evidence
    state.evidence_status = EvidenceStatus.FOUND
    result = AnswerGenerator(StubLLMProvider()).generate(state)
    heading = result.answer.splitlines()[0]
    assert "Core" in heading
    assert "Primary 4" in heading
    assert "Primary 3" not in result.answer.splitlines()[0]


def test_header_rewrites_broader_llm_subject_heading():
    """Classification in STRUCTURED INTENT must survive into the presented header."""
    from app.agent.answer_generator import _ensure_subject_list_scope_heading

    evidence = _subject_evidence(
        [("ENGLISH", "English"), ("MATHEMATICS", "Mathematics")],
        grade="CLASS_3",
    )
    state = CurriculumQAState.initial(
        question="What are the core subjects in Primary 3?"
    )
    state.grade = "CLASS_3"
    state.classification = "CORE"
    state.evidence = evidence
    broader = (
        "## Primary 3 Subjects (from resolved curriculum evidence)\n\n"
        "- English\n- Mathematics\n"
    )
    fixed = _ensure_subject_list_scope_heading(state, broader)
    assert fixed.splitlines()[0] == "## Core Subjects Listed for Primary 3"
    assert "English" in fixed
    assert "Primary 3 Subjects (from resolved" not in fixed
