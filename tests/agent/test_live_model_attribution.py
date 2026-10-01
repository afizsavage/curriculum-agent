"""Live-model evidence attribution. The model is a deterministic fake."""

from __future__ import annotations

from app.agent.answer import AnswerGenerationNode
from app.agent.answer_generator import AnswerGenerator
from app.agent.verify import VerificationNode
from app.agent.verifier import AnswerVerifier
from app.config import Settings
from app.agent.state import CurriculumQAState
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.llm.base import LLMProvider, LLMResponse
from app.llm.provider import StubLLMProvider
from app.schemas.answer import GROUNDED_ANSWER_JSON_SCHEMA, AnswerConfidence

from tests.agent.test_answer_synthesis import (
    _assert_no_internal_identifiers,
    _primary3_fractions_evidence,
    _primary4_fractions_evidence,
    _state,
)


class _FakeModel(LLMProvider):
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return "live-attribution-fake"

    def generate(self, messages, *, temperature=0.0, max_tokens=None) -> LLMResponse:
        return LLMResponse(content="unused")

    def generate_structured(self, messages, *, schema, temperature=0.0) -> dict:
        return self._payload

    def generate_with_tools(self, messages, *, tools, temperature=0.0) -> LLMResponse:
        return LLMResponse(content="unused")


def _outcome(entity_id: str, content: str, *, name: str | None = None) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id=entity_id,
        name=name or entity_id,
        grade="CLASS_3",
        subject="MATHEMATICS",
        topic="Fractions",
        content=content,
    )


def _ask(question: str, evidence: list[CurriculumEvidence], payload: dict):
    state = CurriculumQAState.initial(question=question)
    state.grade = "CLASS_3"
    state.subject = "MATHEMATICS"
    state.evidence = evidence
    state.evidence_status = EvidenceStatus.FOUND
    result = AnswerGenerator(_FakeModel(payload)).generate(state)
    return state, result


def _payload(answer: str, refs: list, *, evidence: list | None = None) -> dict:
    body = {
        "answer": answer,
        "summary": None,
        "refs": refs,
        "limitations": [],
        "confidence": "high",
    }
    if evidence is not None:
        body["evidence"] = evidence
    return body


def test_schema_accepts_explicit_refs_without_dropping_the_evidence_array():
    assert "refs" in GROUNDED_ANSWER_JSON_SCHEMA["properties"]
    assert "evidence" in GROUNDED_ANSWER_JSON_SCHEMA["properties"]
    assert "refs" not in GROUNDED_ANSWER_JSON_SCHEMA["required"]
    state = _state("What should pupils learn about fractions?", [_outcome("LO_A", "Add like fractions.")])
    user = AnswerGenerator(StubLLMProvider()).build_messages(state)[1].content or ""
    assert "refs" in user
    assert "at most 8 evidence refs" not in user


def test_case_a_one_record_one_claim():
    evidence = [
        _outcome(
            "LO_A",
            "Identify unit fractions with denominators 1-5 using pictorial representations.",
        )
    ]
    answer = (
        "Pupils learn to:\n\n"
        "* Identify unit fractions with denominators 1-5 using pictorial representations."
    )
    state, result = _ask(
        "What should pupils learn about unit fractions?",
        evidence,
        _payload(answer, ["LO_A"]),
    )
    report = state.metadata["live_attribution"]
    assert [ref.entity_id for ref in result.evidence] == ["LO_A"]
    assert result.evidence[0].entity_type == "learning_outcome"
    assert report["status"] == "valid"
    assert report["invalid_refs"] == []
    assert report["unused_evidence_ids"] == []
    assert report["unsupported_ref_ids"] == []
    assert report["attribution_completeness"] == 1
    assert "LO_A" not in result.answer


def test_case_b_one_bullet_keeps_both_source_records():
    evidence = [
        _outcome("LO_A", "Work with equivalent fractions."),
        _outcome("LO_B", "Identify equivalent fractions."),
    ]
    answer = "Pupils learn to:\n\n* Work with equivalent fractions."
    state, result = _ask(
        "What should pupils learn about equivalent fractions?",
        evidence,
        _payload(answer, ["LO_A", "LO_B"]),
    )
    assert [ref.entity_id for ref in result.evidence] == ["LO_A", "LO_B"]
    report = state.metadata["live_attribution"]
    assert report["status"] == "valid"
    assert report["attribution_completeness"] == 1
    assert report["unused_evidence_ids"] == []


def test_case_c_unused_retrieved_evidence_is_not_added():
    evidence = [
        _outcome("LO_A", "Add like fractions."),
        _outcome("LO_B", "Subtract like fractions."),
        _outcome("LO_C", "Describe the water cycle in the local environment."),
    ]
    answer = (
        "Pupils learn to:\n\n"
        "* Add like fractions.\n"
        "* Subtract like fractions."
    )
    state, result = _ask(
        "What should pupils learn about fractions?",
        evidence,
        _payload(answer, ["LO_A", "LO_B"]),
    )
    assert [ref.entity_id for ref in result.evidence] == ["LO_A", "LO_B"]
    report = state.metadata["live_attribution"]
    assert report["unused_evidence_ids"] == ["LO_C"]
    assert "LO_C" not in report["valid_refs"]
    assert report["attribution_completeness"] == 1


def test_case_d_malformed_record_remains_attributable():
    malformed = _outcome(
        "LO_BAD",
        "Identify equivalent fractions with denominators up to",
    )
    answer = (
        "Pupils learn to:\n\n"
        "* Identify equivalent fractions.\n\n"
        "### Curriculum Evidence Note\n\n"
        "One learning outcome concerning equivalent fractions is incomplete "
        "in the source evidence. Therefore, the exact denominator range for "
        "that particular outcome cannot be confirmed from the available evidence."
    )
    state, result = _ask(
        "What should pupils learn about equivalent fractions?",
        [malformed],
        _payload(answer, ["LO_BAD"]),
    )
    assert [ref.entity_id for ref in result.evidence] == ["LO_BAD"]
    assert "up to 12" not in result.answer
    assert "up to 8" not in result.answer
    assert "cannot be confirmed" in result.answer
    report = state.metadata["live_attribution"]
    assert report["status"] == "valid"
    assert "LO_BAD" in report["valid_refs"]
    assert report["invalid_refs"] == []


def test_case_e_fake_ref_is_rejected_and_not_replaced():
    evidence = [_outcome("VALID_REF", "Add like fractions.")]
    answer = "Pupils learn to:\n\n* Add like fractions."
    state, result = _ask(
        "What should pupils learn about fractions?",
        evidence,
        _payload(answer, ["VALID_REF", "FAKE_REF"]),
    )
    assert [ref.entity_id for ref in result.evidence] == ["VALID_REF"]
    assert all(ref.entity_id != "FAKE_REF" for ref in result.evidence)
    report = state.metadata["live_attribution"]
    assert report["invalid_refs"] == ["FAKE_REF"]
    assert report["valid_refs"] == ["VALID_REF"]
    assert "FAKE_REF" not in {item.entity_id for item in evidence}


def test_case_f_missing_refs_are_not_treated_as_full_grounding():
    evidence = [_outcome("LO_A", "Add like fractions.")]
    answer = "Pupils learn to:\n\n* Add like fractions."
    state, result = _ask(
        "What should pupils learn about fractions?",
        evidence,
        _payload(
            answer,
            [],
            evidence=[
                {
                    "entity_id": "LO_A",
                    "entity_type": "learning_outcome",
                    "claim": "Add like fractions.",
                }
            ],
        ),
    )
    assert result.evidence == []
    assert result.confidence == AnswerConfidence.LOW
    assert any("attribution is incomplete" in note.lower() for note in result.limitations)
    assert result.answer == answer
    report = state.metadata["live_attribution"]
    assert report["status"] == "missing"
    assert report["valid_refs"] == []
    assert report["unused_evidence_ids"] == ["LO_A"]
    assert report["attribution_completeness"] == 0


def test_duplicate_refs_keep_one_source_identity():
    evidence = [_outcome("LO_A", "Add like fractions.", name="Adding")]
    _state_out, result = _ask(
        "What should pupils learn about fractions?",
        evidence,
        _payload("Pupils learn to:\n\n* Add like fractions.", ["LO_A", "LO_A"]),
    )
    assert [ref.entity_id for ref in result.evidence] == ["LO_A"]
    assert result.evidence[0].name == "Adding"
    assert result.evidence[0].grade == "CLASS_3"
    assert _state_out.metadata["live_attribution"]["duplicate_ref_count"] == 1


def test_cited_unrelated_record_is_flagged_unsupported():
    evidence = [
        _outcome("LO_A", "Add like fractions."),
        _outcome("LO_C", "Describe the water cycle in the local environment."),
    ]
    answer = "Pupils learn to:\n\n* Add like fractions."
    state, result = _ask(
        "What should pupils learn about fractions?",
        evidence,
        _payload(answer, ["LO_A", "LO_C"]),
    )
    assert [ref.entity_id for ref in result.evidence] == ["LO_A", "LO_C"]
    report = state.metadata["live_attribution"]
    assert report["unsupported_ref_ids"] == ["LO_C"]
    assert report["status"] == "unsupported"
    assert "water cycle" not in result.answer


def test_live_path_preserves_prose_through_verification():
    evidence = [
        _outcome("LO_A", "Add like fractions."),
        _outcome("LO_B", "Subtract like fractions."),
    ]
    answer = (
        "# Primary 3 Mathematics — Fractions\n\n"
        "Pupils learn to:\n\n"
        "* Add like fractions.\n"
        "* Subtract like fractions."
    )
    state = CurriculumQAState.initial(question="What should pupils learn about fractions?")
    state.grade = "CLASS_3"
    state.subject = "MATHEMATICS"
    state.evidence = evidence
    state.evidence_status = EvidenceStatus.FOUND
    settings = Settings()
    model = _FakeModel(_payload(answer, ["LO_A", "LO_B"]))
    state = AnswerGenerationNode(llm=model, settings=settings).run(state)
    generated = state.final_answer
    assert generated == answer
    assert state.draft_answer == generated
    state = VerificationNode(llm=StubLLMProvider(), settings=settings).run(state)
    assert state.final_answer == generated
    verifier_input = AnswerVerifier(StubLLMProvider(), settings=settings).build_messages(state)
    assert generated in (verifier_input[1].content or "")
    assert "LO_A" not in (state.final_answer or "")
    assert "LO_B" not in (state.final_answer or "")
    assert [ref.entity_id for ref in state.answer_evidence] == ["LO_A", "LO_B"]


def test_primary3_live_attribution_uses_only_cited_records():
    evidence = _primary3_fractions_evidence()
    evidence.append(
        CurriculumEvidence(
            entity_type="annotation",
            entity_id="unused-later-record",
            name="Administrative note",
            content="Not a learning expectation.",
            grade="CLASS_3",
            subject="MATHEMATICS",
            metadata={"code": "C3U99-LO99"},
        )
    )
    question = "What should a Primary 3 pupil learn about fractions?"
    stub_state = _state(question, evidence, grade="CLASS_3")
    stub = AnswerGenerator(StubLLMProvider()).generate(stub_state)
    cited = [ref.entity_id for ref in stub.evidence]
    assert "unused-later-record" not in cited
    assert "lo-p3-equivalent-incomplete" in cited
    state, result = _ask(
        question,
        evidence,
        _payload(stub.answer, cited),
    )
    state.grade = "CLASS_3"
    assert result.answer == stub.answer
    assert [ref.entity_id for ref in result.evidence] == cited
    assert "unused-later-record" not in {ref.entity_id for ref in result.evidence}
    assert "Identify equivalent fractions." not in result.answer
    _assert_no_internal_identifiers(result.answer, evidence)
    report = state.metadata["live_attribution"]
    assert report["invalid_refs"] == []
    assert "unused-later-record" in report["unused_evidence_ids"]
    assert report["attribution_completeness"] == 1
    assert report["status"] == "valid"


def test_primary4_live_attribution_keeps_the_multiplication_record():
    evidence = _primary4_fractions_evidence()
    question = "What should a Primary 4 pupil learn about fractions?"
    stub_state = _state(question, evidence)
    stub = AnswerGenerator(StubLLMProvider()).generate(stub_state)
    note = stub.answer.split("### Curriculum Evidence Note", 1)[1].lower()
    assert "multiply" in note
    assert "concerning like fractions" not in note
    cited = [ref.entity_id for ref in stub.evidence]
    assert "lo-multiply-garbled" in cited
    state, result = _ask(question, evidence, _payload(stub.answer, cited))
    assert result.answer == stub.answer
    assert [ref.entity_id for ref in result.evidence] == cited
    _assert_no_internal_identifiers(result.answer, evidence)
    report = state.metadata["live_attribution"]
    assert "lo-multiply-garbled" in report["valid_refs"]
    assert report["invalid_refs"] == []
    assert report["unsupported_ref_ids"] == []
    assert report["attribution_completeness"] == 1
    assert "C4U06-LO02" not in result.answer
