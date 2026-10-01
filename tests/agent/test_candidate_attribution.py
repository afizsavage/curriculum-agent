"""Constrained candidate attribution reads frozen claims and does not rewrite them."""

from app.agent.answer_generator import AnswerGenerator
from app.agent.candidate_attribution import (
    CANDIDATE_ATTRIBUTION_PROMPT,
    attribute_one_claim,
    extract_answer_claims,
    measure_candidate_case,
    select_candidates,
)
from app.agent.state import CurriculumQAState
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.llm.base import LLMProvider, LLMResponse
from app.llm.provider import StubLLMProvider
from app.schemas.answer import CANDIDATE_CLAIM_REF_SCHEMA, GROUNDED_ANSWER_JSON_SCHEMA
from tests.agent.test_answer_synthesis import (
    _primary3_fractions_evidence,
    _primary4_fractions_evidence,
)


class _RefModel(LLMProvider):
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return "candidate-fake"

    def generate(self, messages, *, temperature=0.0, max_tokens=None) -> LLMResponse:
        return LLMResponse(content="unused")

    def generate_structured(self, messages, *, schema, temperature=0.0) -> dict:
        self.schema = schema
        self.messages = messages
        return self._payload

    def generate_with_tools(self, messages, *, tools, temperature=0.0) -> LLMResponse:
        return LLMResponse(content="unused")


def _outcome(entity_id: str, content: str, *, name: str | None = None) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id=entity_id,
        name=name or entity_id,
        grade="CLASS_4",
        subject="MATHEMATICS",
        topic="Fractions",
        content=content,
    )


def test_extractor_keeps_bullets_and_notes_and_skips_presentation():
    answer = """# Primary 2 Mathematics — Topics

The Primary 2 Mathematics curriculum covers the following areas:

### 1. Everyday Arithmetic
Pupils learn everyday arithmetic, including number patterns.

### 2. Operations
Pupils learn to:
* Add like fractions.
*

### Curriculum Evidence Note
One learning outcome is incomplete in the source.
"""
    claims = extract_answer_claims(answer)
    texts = [item["text"] for item in claims]
    assert texts == [
        "Pupils learn everyday arithmetic, including number patterns.",
        "Add like fractions.",
        "One learning outcome is incomplete in the source.",
    ]
    assert [item["kind"] for item in claims] == [
        "section_statement",
        "bullet",
        "evidence_note",
    ]
    headings_only = """# Primary 4 Social Studies — Topics

Primary 4 Social Studies covers the following units:

### 1. Healthy Living
### 2. Our Natural Resources
"""
    assert extract_answer_claims(headings_only) == []


def test_candidates_prefer_the_garbled_outcome_over_the_empty_unit():
    evidence = _primary4_fractions_evidence()
    claim = {
        "text": "Multiply fractions.",
        "kind": "bullet",
        "heading": "Fraction Multiplication",
    }
    selected = select_candidates(claim, evidence, ["lo-multiply-garbled", "unit-multiplication"])
    ids = [item["entity_id"] for item in selected["candidates"]]
    assert "lo-multiply-garbled" in ids
    assert ids[0] == "lo-multiply-garbled"
    assert "unit-multiplication" in ids
    assert selected["candidate_count"] <= 8


def test_unrelated_records_and_bare_production_refs_stay_out():
    evidence = [
        _outcome("LO_ADD", "Add like fractions."),
        _outcome("LO_WATER", "Describe the water cycle in the local environment."),
        _outcome("LO_SUB", "Subtract like fractions."),
    ]
    evidence[1] = evidence[1].model_copy(update={"topic": "Water"})
    selected = select_candidates(
        {"text": "Add like fractions.", "kind": "bullet", "heading": "Operations"},
        evidence,
        ["LO_ADD", "LO_WATER"],
    )
    ids = [item["entity_id"] for item in selected["candidates"]]
    assert ids[0] == "LO_ADD"
    assert "LO_WATER" not in ids
    assert "LO_SUB" in ids


def test_candidate_truncation_is_recorded():
    evidence = [
        _outcome(f"LO_{index}", f"Add like fractions item {index}.")
        for index in range(9)
    ]
    selected = select_candidates(
        {"text": "Add like fractions.", "kind": "bullet", "heading": None},
        evidence,
        [],
    )
    assert selected["truncated"] is True
    assert selected["candidate_count"] == 8
    assert selected["candidate_count_before_cap"] == 9
    assert len(selected["omitted_ids"]) == 1


def test_model_selection_is_kept_when_the_lexical_check_rejects_it():
    evidence = [
        _outcome("LO_ADD", "Add like fractions."),
        _outcome("LO_NEAR", "Identify unit fractions with denominators 1-5."),
    ]
    claim = {"text": "Add like fractions.", "kind": "bullet", "heading": "Operations"}
    selected = select_candidates(claim, evidence, ["LO_ADD"])
    result = attribute_one_claim(
        _RefModel({"answer": "Add fractions.", "refs": ["LO_NEAR", "FAKE", "LO_ADD"]}),
        question="What should pupils learn?",
        claim_text=claim["text"],
        candidates=selected["candidates"],
        evidence=evidence,
    )
    assert result["model_refs"] == ["LO_NEAR", "FAKE", "LO_ADD"]
    assert result["model-selected-valid-lexically"] == ["LO_ADD"]
    assert "LO_NEAR" in result["model-selected-but-lexically-rejected"]
    assert result["model-selected-unknown"] == ["FAKE"]
    assert result["discarded_answer_field"] == "Add fractions."
    assert "answer" not in CANDIDATE_CLAIM_REF_SCHEMA["properties"]
    messages = build_messages_for(claim["text"], selected["candidates"])
    assert "Prefer the smallest sufficient set" in (messages[0].content or "")
    assert "Add like fractions." in (messages[1].content or "")
    assert "LO_WATER" not in (messages[1].content or "")
    assert "DO NOT rewrite" in CANDIDATE_ATTRIBUTION_PROMPT


def build_messages_for(claim_text, candidates):
    from app.agent.candidate_attribution import build_candidate_messages

    return build_candidate_messages(
        question="What should pupils learn?",
        claim_text=claim_text,
        candidates=candidates,
    )


def test_empty_selection_is_distinct_from_a_missing_candidate_set():
    evidence = [_outcome("LO_ADD", "Add like fractions.")]
    empty = attribute_one_claim(
        _RefModel({"refs": []}),
        question="What should pupils learn?",
        claim_text="Add like fractions.",
        candidates=select_candidates(
            {"text": "Add like fractions.", "kind": "bullet", "heading": None},
            evidence,
            [],
        )["candidates"],
        evidence=evidence,
    )
    assert empty["model-selected-empty"] is True
    omitted = attribute_one_claim(
        _RefModel({"refs": ["LO_ADD"]}),
        question="What should pupils learn?",
        claim_text="Describe the water cycle.",
        candidates=[],
        evidence=evidence,
    )
    assert omitted["no_candidates"] is True
    assert omitted["model_called"] is False
    metrics = measure_candidate_case(
        answer="* Add like fractions.",
        claim_rows=[
            {
                "text": "Add like fractions.",
                "kind": "bullet",
                "candidate_count": 1,
                "truncated": False,
                "selection": empty,
            },
            {
                "text": "Describe the water cycle.",
                "kind": "bullet",
                "candidate_count": 0,
                "truncated": False,
                "selection": omitted,
            },
        ],
    )
    assert metrics["empty_mappings"] == 1
    assert metrics["no_candidate_claims"] == 1
    assert metrics["covered_claims"] == 0


def test_primary3_note_candidate_includes_the_incomplete_record():
    evidence = _primary3_fractions_evidence()
    note = (
        "One learning outcome about identifying equivalent fractions is incomplete "
        'in the source: the wording stops after "with denominators up to" and does '
        "not state the denominator range."
    )
    selected = select_candidates(
        {"text": note, "kind": "evidence_note", "heading": "Curriculum Evidence Note"},
        evidence,
        ["lo-p3-equivalent-incomplete"],
    )
    ids = [item["entity_id"] for item in selected["candidates"]]
    assert "lo-p3-equivalent-incomplete" in ids


def test_production_prompt_is_unchanged():
    assert "refs" in CANDIDATE_CLAIM_REF_SCHEMA["properties"]
    assert "claims" not in GROUNDED_ANSWER_JSON_SCHEMA["properties"]
    state = CurriculumQAState.initial(question="What should pupils learn?")
    state.evidence = [_outcome("LO_ADD", "Add like fractions.")]
    state.evidence_status = EvidenceStatus.FOUND
    production = AnswerGenerator(StubLLMProvider()).build_messages(state)[1].content or ""
    assert "smallest sufficient set" not in production
    assert "CANDIDATE RECORDS" not in production


def test_duplicate_generic_units_do_not_crowd_out_a_specific_record():
    generic = [
        CurriculumEvidence(
            entity_type="unit",
            entity_id=f"unit-{index}",
            name="Everyday Arithmetic",
            content="Everyday Arithmetic",
            grade="CLASS_2",
            subject="MATHEMATICS",
        )
        for index in range(8)
    ]
    specific = CurriculumEvidence(
        entity_type="unit",
        entity_id="unit-patterns",
        name="Everyday Arithmetic NUMBER PARTERN",
        content="Everyday Arithmetic NUMBER PARTERN",
        grade="CLASS_2",
        subject="MATHEMATICS",
    )
    selected = select_candidates(
        {
            "text": "Pupils learn everyday arithmetic, including number patterns.",
            "kind": "section_statement",
            "heading": "Everyday Arithmetic",
        },
        [*generic, specific],
        [],
    )
    ids = [item["entity_id"] for item in selected["candidates"]]
    assert "unit-patterns" in ids
    assert selected["candidates_removed_as_duplicates"] == 7
    assert sum(1 for item in selected["candidates"] if item["name"] == "Everyday Arithmetic") == 1


def test_relaxed_checker_accepts_short_spans_and_rejects_a_different_operation():
    from app.agent.answer_generator import _record_supports_text
    from app.agent.candidate_attribution import relaxed_record_supports

    reading = _outcome(
        "LO_READ",
        "Identify words in sentences, read independently and answer questions on passages read.",
    )
    assert _record_supports_text(reading, "Identify words in sentences.") is False
    assert relaxed_record_supports(reading, "Identify words in sentences.") is True
    subject = CurriculumEvidence(
        entity_type="subject",
        entity_id="agricultural",
        name="Agricultural Science",
        content="Agricultural Science",
    )
    assert relaxed_record_supports(subject, "Agricultural") is True
    subtract = _outcome("LO_SUB", "Subtract like fractions.")
    assert relaxed_record_supports(subtract, "Add like fractions.") is False
    water = _outcome("LO_WATER", "Describe the water cycle in the local environment.")
    assert relaxed_record_supports(water, "Add like fractions.") is False


def test_two_outcome_claim_keeps_both_records_available():
    evidence = _primary3_fractions_evidence()
    selected = select_candidates(
        {
            "text": "Identify unit fractions with denominators 1-5 and denominators 6-10.",
            "kind": "bullet",
            "heading": "Unit Fractions",
        },
        evidence,
        ["lo-p3-den-1-5", "lo-p3-den-6-10"],
    )
    ids = [item["entity_id"] for item in selected["candidates"]]
    assert "lo-p3-den-1-5" in ids
    assert "lo-p3-den-6-10" in ids
