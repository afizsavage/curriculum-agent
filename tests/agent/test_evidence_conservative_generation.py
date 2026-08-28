"""Regression tests for V2.3 evidence-conservative production generation."""

from __future__ import annotations

import json

from app.agent.answer_generator import (
    SYSTEM_PROMPT,
    AnswerGenerator,
    format_evidence_for_prompt,
)
from app.agent.generation_policy import (
    GENERATION_POLICY,
    analyze_answer_quality,
    detect_speculative_wording,
    detect_truncation_mishandling,
    detect_unsupported_absence_claim,
    source_wording_preserved,
)
from app.agent.state import CurriculumQAState
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.llm.base import LLMMessage, LLMProvider, LLMResponse
from app.llm.provider import StubLLMProvider
from app.schemas.answer import AnswerConfidence


def _state(**kwargs) -> CurriculumQAState:
    state = CurriculumQAState.initial(
        question=kwargs.pop(
            "question",
            "What are the learning objectives for fractions in Primary 4?",
        )
    )
    state.grade = kwargs.pop("grade", "CLASS_4")
    state.subject = kwargs.pop("subject", "MATHEMATICS")
    state.evidence = kwargs.pop("evidence", [])
    state.evidence_status = kwargs.pop("evidence_status", EvidenceStatus.FOUND)
    return state


def _fractions_evidence() -> list[CurriculumEvidence]:
    return [
        CurriculumEvidence(
            entity_type="unit",
            entity_id="unit-frac",
            name="Number and Numeration FRACTION",
            grade="CLASS_4",
            subject="MATHEMATICS",
            metadata={"code": "C4-U04"},
        ),
        CurriculumEvidence(
            entity_type="learning_outcome",
            entity_id="lo-01",
            name="C4U04-LO01",
            grade="CLASS_4",
            subject="MATHEMATICS",
            content="Simplify like fraction with common denominators.",
            metadata={"code": "C4U04-LO01"},
        ),
        CurriculumEvidence(
            entity_type="learning_outcome",
            entity_id="lo-garbled",
            name="C4U06-LO02",
            grade="CLASS_4",
            subject="MATHEMATICS",
            content=(
                "Multiply like fractions with denominators up to multiply like fractions "
                "with denominators up to multiply related fractions"
            ),
            metadata={"code": "C4U06-LO02"},
        ),
    ]


class RecordingLLM(LLMProvider):
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.last_messages: list[LLMMessage] = []

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return "test-model"

    def generate(self, messages, *, temperature=0.0, max_tokens=None) -> LLMResponse:
        return LLMResponse(content="unused")

    def generate_structured(self, messages, *, schema, temperature=0.0) -> dict:
        self.last_messages = list(messages)
        return self._payload

    def generate_with_tools(self, messages, *, tools, temperature=0.0) -> LLMResponse:
        return LLMResponse(content="unused")


# Test 1 — Unsupported absence claim
def test_unsupported_absence_claim_detected():
    bad = (
        "There are no learning outcomes for division of fractions in Primary 4."
    )
    assert detect_unsupported_absence_claim(bad)
    quality = analyze_answer_quality(bad)
    assert quality["unsupported_claim_count"] >= 1
    assert quality["absence_claim_count"] >= 1


def test_safe_absence_wording_not_flagged():
    safe = (
        "The resolved evidence does not include a learning outcome specifically "
        "mentioning division of fractions."
    )
    assert not detect_unsupported_absence_claim(safe)


def test_build_messages_prohibits_unsupported_absence_claims():
    state = _state(evidence=_fractions_evidence())
    messages = AnswerGenerator(RecordingLLM({})).build_messages(state)
    combined = (messages[0].content or "") + (messages[1].content or "")
    assert "not observed" in combined.lower() or "does not exist" in combined.lower()
    assert "absence" in combined.lower()


# Test 2 — Truncated LO
def test_truncated_lo_policy_in_prompt():
    state = _state(evidence=_fractions_evidence())
    messages = AnswerGenerator(RecordingLLM({})).build_messages(state)
    combined = (messages[0].content or "") + (messages[1].content or "")
    assert "truncated" in combined.lower() or "garbled" in combined.lower()
    assert "reconstruct" in combined.lower() or "repair" in combined.lower()


def test_stub_preserves_truncated_lo_wording():
    state = _state(evidence=_fractions_evidence())
    result = AnswerGenerator(StubLLMProvider()).generate(state)
    assert "C4U06-LO02" in result.answer
    assert "denominators up to multiply" in result.answer
    assert "Source limitations" in result.answer or "incomplete" in result.answer.lower()


# Test 3 — Speculative completion
def test_speculative_completion_detected():
    bad = "C4U04-LO04 likely means students compare fractions greater than one half."
    assert detect_speculative_wording(bad)
    assert analyze_answer_quality(bad)["speculative_claim_count"] >= 1


def test_truncation_mishandling_detected():
    bad = "The missing text probably means compare equivalent fractions."
    assert detect_truncation_mishandling(bad)


def test_build_messages_forbids_speculative_completion():
    state = _state(evidence=_fractions_evidence())
    messages = AnswerGenerator(RecordingLLM({})).build_messages(state)
    combined = (messages[0].content or "") + (messages[1].content or "")
    assert "likely" in combined.lower()
    assert "probably" in combined.lower()


# Test 4 — Source wording preservation
def test_source_wording_preservation_heuristic():
    source = "Simplify like fraction with common denominators."
    answer = (
        "## Learning objectives\n"
        "- **C4U04-LO01** — Simplify like fraction with common denominators."
    )
    assert source_wording_preserved(
        answer, lo_code="C4U04-LO01", source_wording=source
    )


def test_build_messages_requires_source_wording():
    state = _state(evidence=_fractions_evidence())
    messages = AnswerGenerator(RecordingLLM({})).build_messages(state)
    system = messages[0].content or ""
    assert "source wording" in system.lower() or "lo code" in system.lower()


# Test 5 — Evidence boundary
def test_system_prompt_evidence_boundary():
    assert GENERATION_POLICY == "evidence_conservative"
    assert "Evidence is the boundary" in SYSTEM_PROMPT or "EVIDENCE IS THE BOUNDARY" in SYSTEM_PROMPT.upper()
    assert "general model knowledge" in SYSTEM_PROMPT.lower()


def test_format_evidence_does_not_add_external_facts():
    evidence = _fractions_evidence()
    block = format_evidence_for_prompt(evidence)
    assert "C4U04-LO01" in block
    assert "division" not in block.lower()


# Test 6 — Normal successful answer
def test_normal_lo_answer_is_readable():
    state = _state(evidence=_fractions_evidence())
    payload = {
        "answer": (
            "## Curriculum context\n"
            "Grade: Primary 4\nSubject: Mathematics\n\n"
            "## Learning objectives/outcomes\n"
            "- **C4U04-LO01** — Simplify like fraction with common denominators.\n"
            "- **C4U06-LO02** — Multiply like fractions with denominators up to multiply "
            "like fractions with denominators up to multiply related fractions\n\n"
            "## Source limitations\n"
            "- C4U06-LO02 source record appears repetitive/incomplete."
        ),
        "confidence": "high",
        "evidence": [
            {"entity_id": "lo-01", "entity_type": "learning_outcome", "claim": "LO01"},
            {"entity_id": "lo-garbled", "entity_type": "learning_outcome", "claim": "LO02"},
        ],
        "limitations": ["C4U06-LO02 source text is incomplete."],
    }
    llm = RecordingLLM(payload)
    result = AnswerGenerator(llm).generate(state)
    assert "C4U04-LO01" in result.answer
    assert result.confidence == AnswerConfidence.HIGH
    assert state.metadata.get("generation_policy") == "evidence_conservative"
    assert state.metadata.get("truncation_warning_count", 0) >= 1
