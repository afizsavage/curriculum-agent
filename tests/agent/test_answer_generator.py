import json

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
