"""Experimental claim → evidence mappings, compared with flat refs."""

from __future__ import annotations

import re

from app.agent.answer import AnswerGenerationNode
from app.agent.answer_generator import AnswerGenerator
from app.agent.verify import VerificationNode
from app.agent.verifier import AnswerVerifier
from app.config import Settings
from app.agent.state import CurriculumQAState
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.llm.base import LLMProvider, LLMResponse
from app.llm.provider import StubLLMProvider
from app.schemas.answer import EXPERIMENTAL_CLAIM_MAPPING_SCHEMA, AnswerConfidence

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
        return "claim-attribution-fake"

    def generate(self, messages, *, temperature=0.0, max_tokens=None) -> LLMResponse:
        return LLMResponse(content="unused")

    def generate_structured(self, messages, *, schema, temperature=0.0) -> dict:
        return self._payload

    def generate_with_tools(self, messages, *, tools, temperature=0.0) -> LLMResponse:
        return LLMResponse(content="unused")


def _outcome(entity_id: str, content: str) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id=entity_id,
        name=entity_id,
        grade="CLASS_3",
        subject="MATHEMATICS",
        topic="Fractions",
        content=content,
    )


def _ask(evidence: list[CurriculumEvidence], payload: dict, *, grade: str = "CLASS_3"):
    state = CurriculumQAState.initial(question="What should pupils learn about fractions?")
    state.grade = grade
    state.subject = "MATHEMATICS"
    state.evidence = evidence
    state.evidence_status = EvidenceStatus.FOUND
    state.metadata["claim_shadow"] = True
    result = AnswerGenerator(_FakeModel(payload)).generate(state)
    return state, result


def _body(answer: str, **extra) -> dict:
    payload = {
        "answer": answer,
        "summary": None,
        "limitations": [],
        "confidence": "high",
        "evidence": [],
    }
    payload.update(extra)
    return payload


def _bullets(answer: str) -> list[str]:
    return [
        re.sub(r"^[\*\-]\s+", "", line.strip())
        for line in answer.splitlines()
        if re.match(r"^\s*[\*\-]\s+", line)
    ]


def test_production_generate_ignores_claim_mappings():
    evidence = [_outcome("LO_A", "Add like fractions.")]
    answer = "Pupils learn to:\n\n* Add like fractions."
    state = CurriculumQAState.initial(question="What should pupils learn about fractions?")
    state.grade = "CLASS_3"
    state.subject = "MATHEMATICS"
    state.evidence = evidence
    state.evidence_status = EvidenceStatus.FOUND
    result = AnswerGenerator(
        _FakeModel(
            _body(
                answer,
                refs=["LO_A"],
                claims=[{"text": "Pupils learn decimal numbers.", "refs": ["FAKE_REF"]}],
            )
        )
    ).generate(state)
    assert result.answer == answer
    assert [ref.entity_id for ref in result.evidence] == ["LO_A"]
    assert "claim_attribution" not in state.metadata
    user = AnswerGenerator(StubLLMProvider()).build_messages(state)[1].content or ""
    assert "SHADOW CLAIM ATTRIBUTION" not in user


def test_claim_schema_is_experimental_and_not_required_in_production():
    assert EXPERIMENTAL_CLAIM_MAPPING_SCHEMA["required"] == ["answer", "claims"]
    item = EXPERIMENTAL_CLAIM_MAPPING_SCHEMA["properties"]["claims"]["items"]
    assert item["required"] == ["text", "refs"]
    from app.schemas.answer import GROUNDED_ANSWER_JSON_SCHEMA

    assert "claims" not in GROUNDED_ANSWER_JSON_SCHEMA["properties"]
    state = _state("What should pupils learn about fractions?", [_outcome("LO_A", "Add like fractions.")])
    user = AnswerGenerator(StubLLMProvider()).build_messages(state)[1].content or ""
    assert "refs" in user
    assert "EXPERIMENTAL CLAIM MAPPING" not in user


def test_one_claim_one_record():
    evidence = [_outcome("LO_A", "Add like fractions.")]
    answer = "Pupils learn to:\n\n* Add like fractions."
    state, result = _ask(
        evidence,
        _body(answer, claims=[{"text": "Add like fractions.", "refs": ["LO_A"]}]),
    )
    report = state.metadata["claim_attribution"]
    assert [ref.entity_id for ref in result.evidence] == ["LO_A"]
    assert result.evidence[0].claim == "Add like fractions."
    assert report["status"] == "valid"
    assert report["valid_claim_count"] == 1
    assert report["unattributed_claims"] == []
    assert "LO_A" not in result.answer


def test_one_claim_keeps_multiple_records():
    evidence = [
        _outcome("LO_A", "Work with equivalent fractions."),
        _outcome("LO_B", "Identify equivalent fractions."),
    ]
    text = "Work with equivalent fractions and identify equivalent fractions."
    answer = f"Pupils learn to:\n\n* {text}"
    state, result = _ask(
        evidence,
        _body(answer, claims=[{"text": text, "refs": ["LO_A", "LO_B"]}]),
    )
    report = state.metadata["claim_attribution"]
    assert [ref.entity_id for ref in result.evidence] == ["LO_A", "LO_B"]
    assert report["multi_record_claim_count"] == 1
    assert report["valid_claims"] == [{"text": text, "refs": ["LO_A", "LO_B"]}]
    assert report["status"] == "valid"


def test_multiple_claims_map_separately():
    evidence = [
        _outcome(
            "LO_A",
            "Identify unit fractions with denominators 1-5 using pictorial representations.",
        ),
        _outcome(
            "LO_B",
            "Identify unit fractions with denominators 6-10 using pictorial representations.",
        ),
        _outcome("LO_C", "Locate unit fractions on the number line."),
    ]
    bullets = [
        "Identify unit fractions with denominators 1-5 using pictorial representations.",
        "Identify unit fractions with denominators 6-10 using pictorial representations.",
        "Locate unit fractions on the number line.",
    ]
    answer = "# Primary 3 Mathematics — Fractions\n\n### Unit Fractions\n\nPupils learn to:\n\n" + "\n".join(
        f"* {bullet}" for bullet in bullets
    )
    state, result = _ask(
        evidence,
        _body(
            answer,
            claims=[
                {"text": bullets[0], "refs": ["LO_A"]},
                {"text": bullets[1], "refs": ["LO_B"]},
                {"text": bullets[2], "refs": ["LO_C"]},
            ],
        ),
    )
    report = state.metadata["claim_attribution"]
    assert report["valid_claim_count"] == 3
    assert [claim["refs"] for claim in report["valid_claims"]] == [["LO_A"], ["LO_B"], ["LO_C"]]
    assert report["unattributed_claims"] == []
    assert "Pupils learn to" not in " ".join(report["unattributed_claims"])
    assert [ref.entity_id for ref in result.evidence] == ["LO_A", "LO_B", "LO_C"]
    assert "# Primary 3 Mathematics — Fractions" in result.answer


def test_unused_retrieved_evidence_is_not_attached_to_a_claim():
    evidence = [
        _outcome("LO_A", "Add like fractions."),
        _outcome("LO_B", "Subtract like fractions."),
        _outcome("LO_C", "Describe the water cycle in the local environment."),
    ]
    answer = "Pupils learn to:\n\n* Add like fractions.\n* Subtract like fractions."
    state, result = _ask(
        evidence,
        _body(
            answer,
            refs=["LO_A", "LO_B", "LO_C"],
            claims=[
                {"text": "Add like fractions.", "refs": ["LO_A"]},
                {"text": "Subtract like fractions.", "refs": ["LO_B"]},
            ],
        ),
    )
    assert [ref.entity_id for ref in result.evidence] == ["LO_A", "LO_B"]
    assert state.metadata["claim_attribution"]["status"] == "valid"
    assert any(item.entity_id == "LO_C" for item in evidence)


def test_unknown_ref_is_rejected():
    evidence = [_outcome("LO_A", "Add like fractions.")]
    answer = "Pupils learn to:\n\n* Add like fractions."
    state, result = _ask(
        evidence,
        _body(answer, claims=[{"text": "Add like fractions.", "refs": ["FAKE_REF"]}]),
    )
    report = state.metadata["claim_attribution"]
    assert result.evidence == []
    assert report["invalid_refs"] == ["FAKE_REF"]
    assert report["unknown_ref_claims"][0]["text"] == "Add like fractions."
    assert report["valid_claim_count"] == 0
    assert "FAKE_REF" not in {item.entity_id for item in evidence}


def test_claim_text_absent_from_the_answer_is_rejected():
    evidence = [_outcome("VALID_REF", "Add like fractions.")]
    answer = "Pupils learn to:\n\n* Add like fractions."
    state, result = _ask(
        evidence,
        _body(
            answer,
            claims=[{"text": "Pupils learn decimal numbers.", "refs": ["VALID_REF"]}],
        ),
    )
    report = state.metadata["claim_attribution"]
    assert result.answer == answer
    assert "decimal" not in result.answer
    assert result.evidence == []
    assert report["absent_claims"][0]["text"] == "Pupils learn decimal numbers."
    assert report["valid_claim_count"] == 0
    assert "VALID_REF" not in report["valid_refs"]


def test_unrelated_real_ref_is_unsupported_and_kept_in_retrieval():
    evidence = [
        _outcome("LO_A", "Add like fractions."),
        _outcome("LO_C", "Describe the water cycle in the local environment."),
    ]
    answer = "Pupils learn to:\n\n* Add like fractions."
    state, result = _ask(
        evidence,
        _body(answer, claims=[{"text": "Add like fractions.", "refs": ["LO_C"]}]),
    )
    report = state.metadata["claim_attribution"]
    assert result.evidence == []
    assert report["status"] == "unsupported"
    assert report["unsupported_claims"] == [{"text": "Add like fractions.", "refs": ["LO_C"]}]
    assert report["valid_claim_count"] == 0
    assert any(item.entity_id == "LO_C" for item in state.evidence)


def test_missing_claim_mapping_stays_incomplete():
    evidence = [
        _outcome("LO_A", "Add like fractions."),
        _outcome("LO_B", "Subtract like fractions."),
        _outcome("LO_C", "Locate unit fractions on the number line."),
    ]
    answer = (
        "Pupils learn to:\n\n"
        "* Add like fractions.\n"
        "* Subtract like fractions.\n"
        "* Locate unit fractions on the number line."
    )
    state, result = _ask(
        evidence,
        _body(
            answer,
            refs=["LO_A", "LO_B", "LO_C"],
            claims=[
                {"text": "Add like fractions.", "refs": ["LO_A"]},
                {"text": "Subtract like fractions.", "refs": ["LO_B"]},
            ],
        ),
    )
    report = state.metadata["claim_attribution"]
    assert [ref.entity_id for ref in result.evidence] == ["LO_A", "LO_B"]
    assert "LO_C" not in report["valid_refs"]
    assert report["status"] == "incomplete"
    assert any("number line" in claim for claim in report["unattributed_claims"])
    assert any("incomplete" in note.lower() for note in result.limitations)
    assert result.confidence == AnswerConfidence.MEDIUM
    assert "Locate unit fractions on the number line." in result.answer


def test_no_claim_mappings_are_not_filled_from_flat_refs():
    evidence = [
        _outcome("LO_A", "Add like fractions."),
        _outcome("LO_B", "Subtract like fractions."),
        _outcome("LO_C", "Locate unit fractions on the number line."),
    ]
    answer = (
        "Pupils learn to:\n\n"
        "* Add like fractions.\n"
        "* Subtract like fractions.\n"
        "* Locate unit fractions on the number line."
    )
    state, result = _ask(
        evidence,
        _body(answer, refs=["LO_A", "LO_B", "LO_C"], claims=[]),
    )
    assert result.evidence == []
    assert result.confidence == AnswerConfidence.LOW
    report = state.metadata["claim_attribution"]
    assert report["status"] == "missing"
    assert report["valid_claim_count"] == 0
    assert len(report["unattributed_claims"]) == 3
    assert state.metadata["live_attribution"]["valid_refs"] == ["LO_A", "LO_B", "LO_C"]


def test_slight_wording_change_is_rejected():
    evidence = [_outcome("LO_A", "Add like fractions.")]
    answer = "Pupils learn to:\n\n* Add like fractions."
    state, result = _ask(
        evidence,
        _body(answer, claims=[{"text": "Add like fraction", "refs": ["LO_A"]}]),
    )
    assert result.evidence == []
    assert state.metadata["claim_attribution"]["absent_claims"][0]["text"] == "Add like fraction"
    assert state.metadata["claim_attribution"]["valid_claim_count"] == 0


def test_malformed_primary3_note_maps_to_the_incomplete_record():
    evidence = _primary3_fractions_evidence()
    question = "What should a Primary 3 pupil learn about fractions?"
    stub = AnswerGenerator(StubLLMProvider()).generate(_state(question, evidence, grade="CLASS_3"))
    note = stub.answer.split("### Curriculum Evidence Note", 1)[1].strip()
    bullets = {
        "Identify unit fractions with denominators 1-5 using pictorial representations.": "lo-p3-den-1-5",
        "Identify unit fractions with denominators 6-10 using pictorial representations.": "lo-p3-den-6-10",
        "Locate unit fractions on the number line.": "lo-p3-number-line",
        "Identify unit and non-unit fractions with denominators 2-10.": "lo-p3-non-unit",
        "Represent these fractions pictorially.": "lo-p3-pictorial",
        "Locate and identify fractions on the number line.": "lo-p3-locate",
        "Work with equivalent fractions.": "lo-p3-equivalent",
        "Add like fractions.": "lo-p3-add",
        "Subtract like fractions.": "lo-p3-subtract",
        "Solve word problems involving addition and subtraction of like fractions.": "lo-p3-word-problems",
    }
    assert _bullets(stub.answer) == list(bullets)
    claims = [{"text": text, "refs": [entity_id]} for text, entity_id in bullets.items()]
    claims.append({"text": note, "refs": ["lo-p3-equivalent-incomplete"]})
    state, result = _ask(evidence, _body(stub.answer, claims=claims), grade="CLASS_3")
    report = state.metadata["claim_attribution"]
    assert result.answer == stub.answer
    assert report["status"] == "valid"
    assert report["unattributed_claims"] == []
    note_claims = [claim for claim in report["valid_claims"] if "incomplete" in claim["text"]]
    assert note_claims == [{"text": note, "refs": ["lo-p3-equivalent-incomplete"]}]
    assert "lo-p3-equivalent-incomplete" in report["valid_refs"]
    assert "up to 12" not in result.answer
    _assert_no_internal_identifiers(result.answer, evidence)


def test_malformed_primary4_note_maps_to_the_multiplication_record():
    evidence = _primary4_fractions_evidence()
    question = "What should a Primary 4 pupil learn about fractions?"
    stub = AnswerGenerator(StubLLMProvider()).generate(_state(question, evidence))
    note = stub.answer.split("### Curriculum Evidence Note", 1)[1].strip()
    assert "multiply like fractions" in note.lower()
    bullets = {
        "Simplify like fractions with common denominators.": "lo-simplify",
        "Compare and order fractions and identify equivalent fractions.": "lo-equivalent",
        "Add and subtract fractions.": "lo-add",
        "Solve problems involving addition and subtraction of fractions.": "lo-problems",
        "Multiply like fractions.": "lo-multiply-garbled",
    }
    assert _bullets(stub.answer) == list(bullets)
    claims = [{"text": text, "refs": [entity_id]} for text, entity_id in bullets.items()]
    claims.append({"text": note, "refs": ["lo-multiply-garbled"]})
    state, result = _ask(evidence, _body(stub.answer, claims=claims))
    report = state.metadata["claim_attribution"]
    assert result.answer == stub.answer
    assert report["status"] == "valid"
    assert report["unattributed_claims"] == []
    assert {"text": "Multiply like fractions.", "refs": ["lo-multiply-garbled"]} in report["valid_claims"]
    assert {"text": note, "refs": ["lo-multiply-garbled"]} in report["valid_claims"]
    assert report["valid_refs"].count("lo-multiply-garbled") == 1
    assert "denominators up to multiply" not in result.answer.lower()
    _assert_no_internal_identifiers(result.answer, evidence)


def test_claim_path_passes_the_same_prose_to_the_verifier():
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
    state.metadata["claim_shadow"] = True
    settings = Settings()
    model = _FakeModel(
        _body(
            answer,
            claims=[
                {"text": "Add like fractions.", "refs": ["LO_A"]},
                {"text": "Subtract like fractions.", "refs": ["LO_B"]},
            ],
        )
    )
    state = AnswerGenerationNode(llm=model, settings=settings).run(state)
    generated = state.final_answer
    assert generated == answer
    state = VerificationNode(llm=StubLLMProvider(), settings=settings).run(state)
    assert state.final_answer == generated
    verifier_input = AnswerVerifier(StubLLMProvider(), settings=settings).build_messages(state)
    assert generated in (verifier_input[1].content or "")
    assert "LO_A" not in (state.final_answer or "")
    assert [ref.entity_id for ref in state.answer_evidence] == ["LO_A", "LO_B"]
