"""Shadow claim-span diagnostics. Near misses are recorded and not accepted."""

from app.agent.answer_generator import AnswerGenerator, CLAIM_SHADOW_APPENDIX
from app.agent.claim_shadow import classify_answer_delta, classify_claim_text, measure_shadow_case
from app.agent.state import CurriculumQAState
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.llm.provider import StubLLMProvider
from app.schemas.answer import GROUNDED_ANSWER_JSON_SCHEMA, shadow_claim_generation_schema


def test_production_prompt_does_not_request_shadow_claims():
    state = CurriculumQAState.initial(question="What are the subjects in Primary 3?")
    state.evidence = [
        CurriculumEvidence(entity_type="subject", entity_id="math", name="Mathematics", grade="CLASS_3")
    ]
    state.evidence_status = EvidenceStatus.FOUND
    user = AnswerGenerator(StubLLMProvider()).build_messages(state)[1].content or ""
    assert "SHADOW CLAIM ATTRIBUTION" not in user
    assert "claims" not in GROUNDED_ANSWER_JSON_SCHEMA["properties"]


def test_shadow_prompt_requests_verbatim_spans():
    state = CurriculumQAState.initial(question="What are the subjects in Primary 3?")
    state.metadata["claim_shadow"] = True
    state.evidence = [
        CurriculumEvidence(entity_type="subject", entity_id="math", name="Mathematics", grade="CLASS_3")
    ]
    state.evidence_status = EvidenceStatus.FOUND
    user = AnswerGenerator(StubLLMProvider()).build_messages(state)[1].content or ""
    assert CLAIM_SHADOW_APPENDIX.strip() in user
    assert "copied verbatim" in user
    schema = shadow_claim_generation_schema()
    assert "claims" in schema["properties"]
    assert "claims" not in schema["required"]


def test_near_miss_is_not_an_exact_match():
    answer = "Pupils learn to:\n\n* Add like fractions."
    assert classify_claim_text("Add like fractions.", answer) == "exact"
    assert classify_claim_text("Add like fraction.", answer) == "near_miss"
    assert classify_claim_text("Pupils learn decimal numbers.", answer) == "extra"
    assert classify_answer_delta(answer, answer) == "identical"
    assert classify_answer_delta(answer, answer.replace("*", "-")) == "formatting-only"


def test_measure_shadow_case_counts_exact_spans_only():
    report = measure_shadow_case(
        production_answer="Pupils learn to:\n\n* Add like fractions.",
        shadow_answer="Pupils learn to:\n\n* Add like fractions.",
        claim_report={
            "returned_claims": [
                {"text": "Add like fractions.", "refs": ["LO_A"]},
                {"text": "Add like fraction.", "refs": ["LO_A"]},
            ],
            "valid_claims": [{"text": "Add like fractions.", "refs": ["LO_A"]}],
            "unattributed_claims": [],
            "invalid_refs": [],
            "unsupported_claims": [],
            "substantive_claim_count": 1,
            "multi_record_claim_count": 0,
            "status": "valid",
        },
    )
    assert report["exact_matches"] == 1
    assert report["near_misses"] == 1
    assert report["answer_delta"] == "identical"
    assert report["complete"] is True
