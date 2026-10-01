"""Post-generation attribution reads a frozen answer and does not rewrite it."""

from app.agent.answer_generator import AnswerGenerator
from app.agent.post_generation_attribution import (
    POST_GENERATION_ATTRIBUTION_PROMPT,
    attribute_existing_answer,
    build_post_generation_messages,
    measure_post_generation_attribution,
)
from app.agent.state import CurriculumQAState
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.llm.base import LLMProvider, LLMResponse
from app.llm.provider import StubLLMProvider
from app.schemas.answer import GROUNDED_ANSWER_JSON_SCHEMA, POST_GENERATION_CLAIM_SCHEMA


class _AttributionModel(LLMProvider):
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls = 0

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return "post-generation-fake"

    def generate(self, messages, *, temperature=0.0, max_tokens=None) -> LLMResponse:
        return LLMResponse(content="unused")

    def generate_structured(self, messages, *, schema, temperature=0.0) -> dict:
        self.calls += 1
        self.schema = schema
        self.messages = messages
        return self._payload

    def generate_with_tools(self, messages, *, tools, temperature=0.0) -> LLMResponse:
        return LLMResponse(content="unused")


def _outcome(entity_id: str, content: str) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id=entity_id,
        name=entity_id,
        grade="CLASS_4",
        subject="MATHEMATICS",
        topic="Fractions",
        content=content,
    )


def test_attribution_schema_does_not_ask_for_an_answer():
    assert "answer" not in POST_GENERATION_CLAIM_SCHEMA["properties"]
    assert "claims" not in GROUNDED_ANSWER_JSON_SCHEMA["properties"]
    state = CurriculumQAState.initial(question="What should pupils learn?")
    state.evidence = [_outcome("LO_A", "Add like fractions.")]
    state.evidence_status = EvidenceStatus.FOUND
    production = AnswerGenerator(StubLLMProvider()).build_messages(state)[1].content or ""
    assert "SHADOW CLAIM ATTRIBUTION" not in production
    assert "DO NOT rewrite" not in production


def test_frozen_answer_is_not_rewritten():
    evidence = [
        _outcome("LO_A", "Multiply like fractions."),
        _outcome("LO_BAD", "Multiply like fractions with denominators up to"),
    ]
    answer = (
        "Pupils learn to:\n\n"
        "* Multiply like fractions.\n\n"
        "### Curriculum Evidence Note\n\n"
        "One learning outcome concerning multiply like fractions is incomplete "
        "in the source evidence. Therefore, the exact denominator range cannot "
        "be confirmed from the available evidence."
    )
    original = answer
    model = _AttributionModel(
        {
            "answer": "Multiply fractions.",
            "claims": [
                {"text": "Multiply fractions.", "refs": ["LO_BAD"]},
                {"text": "Multiply like fractions.", "refs": ["LO_BAD"]},
                {
                    "text": (
                        "One learning outcome concerning multiply like fractions is incomplete "
                        "in the source evidence."
                    ),
                    "refs": ["LO_BAD"],
                },
            ],
        }
    )
    result = attribute_existing_answer(
        model,
        question="What should pupils learn about fractions?",
        answer=answer,
        evidence=evidence,
        production_refs=["LO_A", "LO_BAD"],
        malformed_ids=["LO_BAD"],
    )
    assert answer == original
    assert result["production_answer"] == original
    assert result["discarded_answer_field"] == "Multiply fractions."
    assert "answer" not in model.schema["properties"]
    prompt = model.messages[0].content or ""
    assert "DO NOT rewrite" in prompt
    assert POST_GENERATION_ATTRIBUTION_PROMPT.splitlines()[0] in prompt
    metrics = result["metrics"]
    assert metrics["extra_mappings"] == 1
    assert metrics["exact_matches"] == 2
    assert metrics["malformed_evidence_mappings"] == 1
    assert "Multiply fractions." not in result["production_answer"]
    assert "Multiply like fractions." in result["production_answer"]


def test_missing_mapping_is_distinct_from_an_invalid_mapping():
    evidence = [
        _outcome("LO_A", "Add like fractions."),
        _outcome("LO_B", "Subtract like fractions."),
        _outcome("LO_C", "Describe the water cycle in the local environment."),
    ]
    answer = "Pupils learn to:\n\n* Add like fractions.\n* Subtract like fractions."
    metrics = measure_post_generation_attribution(
        answer=answer,
        returned_claims=[
            {"text": "Add like fractions.", "refs": ["LO_A", "LO_A"]},
            {"text": "Subtract like fractions.", "refs": ["FAKE_REF"]},
            {"text": "Locate unit fractions on the number line.", "refs": ["LO_C"]},
        ],
        evidence=evidence,
        production_refs=["LO_A", "LO_UNUSED"],
    )
    assert metrics["covered_claims"] == 1
    assert metrics["missing_mappings"] == 0
    assert len(metrics["returned_invalid_claims"]) == 1
    assert "Subtract like fractions" in metrics["returned_invalid_claims"][0]
    assert metrics["invalid_refs"] == 1
    assert metrics["extra_mappings"] == 1
    assert "LO_A" in metrics["production_refs_covered"]
    assert "LO_UNUSED" in metrics["production_refs_unattributed"]
    assert metrics["record_buckets"]["1"] == 1

    omitted = measure_post_generation_attribution(
        answer=answer,
        returned_claims=[{"text": "Add like fractions.", "refs": ["LO_A"]}],
        evidence=evidence,
        production_refs=["LO_A"],
    )
    assert omitted["missing_mappings"] == 1
    assert omitted["returned_invalid_mappings"] == 0
    assert "Subtract like fractions" in omitted["omitted_claims"][0]


def test_prompt_contains_the_immutable_answer_and_not_a_rewrite_request():
    evidence = [_outcome("LO_A", "Add like fractions.")]
    messages = build_post_generation_messages(
        question="What should pupils learn?",
        answer="* Add like fractions.",
        evidence=evidence,
        production_refs=["LO_A"],
    )
    user = messages[1].content or ""
    assert "IMMUTABLE ANSWER" in user
    assert "* Add like fractions." in user
    assert "Entity ID: LO_A" in user
    assert "Do not return an answer field" in user
