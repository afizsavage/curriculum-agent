"""V2.13F document evidence arbitration tests (shadow/replay only)."""

from __future__ import annotations

from app.agent.v213f_arbitration import (
    ArbitrationPolicy,
    DocumentRole,
    DocumentUse,
    StructuredSufficiency,
    arbitrate_documents,
    counterfactual_outcome,
    select_generation_evidence,
)
from app.curriculum.evidence import CurriculumEvidence
from app.config import Settings


def _doc(**kwargs) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="document_passage",
        entity_id=kwargs.get("entity_id", "p1"),
        name=kwargs.get("name", "passage"),
        content=kwargs.get("content", "money class 4 mathematics teaching"),
        grade=kwargs.get("grade"),
        subject=kwargs.get("subject"),
        topic=kwargs.get("topic"),
        source="document",
        metadata={
            "retrieval_score": kwargs.get("retrieval_score", 0.03),
            "retrieval_rank": kwargs.get("retrieval_rank", 1),
        },
    )


def test_structured_sufficient_redundant_document_not_used_for_generation():
    structured = [
        CurriculumEvidence(
            entity_type="topic",
            entity_id=f"t{i}",
            name="Money",
            content="Money topic",
            grade="CLASS_4",
            subject="MATHEMATICS",
            source="structured",
        )
        for i in range(10)
    ]
    docs = [_doc(grade="CLASS_4", subject="MATHEMATICS", topic="money")]
    decision = arbitrate_documents(
        docs,
        structured=structured,
        question="What is taught about money in Class 4 Mathematics?",
        grade="CLASS_4",
        subject="MATHEMATICS",
        topic="money",
        control_accepted=True,
        control_route="finish",
        control_verifier_decision="accept",
        policy=ArbitrationPolicy.ARBITRATED,
    )
    assert decision.structured_sufficiency == StructuredSufficiency.SUFFICIENT
    assert decision.document_role == DocumentRole.REDUNDANT
    assert decision.document_use == DocumentUse.PROVENANCE_ONLY
    gen, prov = select_generation_evidence(structured, docs, decision)
    assert len(gen) == len(structured)
    assert len(prov) == 1


def test_structured_insufficient_decisive_document_used():
    docs = [
        _doc(
            grade="CLASS_5",
            subject="SCIENCE",
            topic="plants",
            content="plants photosynthesis class 5 science guidance",
        )
    ]
    decision = arbitrate_documents(
        docs,
        structured=[],
        question="What does the document say about plants in Class 5 Science?",
        grade="CLASS_5",
        subject="SCIENCE",
        topic="plants",
        control_accepted=False,
        control_route="fallback",
        control_verifier_decision="retrieve_more",
        policy=ArbitrationPolicy.ARBITRATED,
    )
    assert decision.structured_sufficiency == StructuredSufficiency.INSUFFICIENT
    assert decision.document_role == DocumentRole.DECISIVE
    assert decision.document_use == DocumentUse.INCLUDE_IN_GENERATION


def test_irrelevant_document_not_used_when_structured_sufficient():
    structured = [
        CurriculumEvidence(
            entity_type="subject",
            entity_id="s1",
            name="Mathematics",
            content="Mathematics",
            grade="CLASS_4",
            subject="MATHEMATICS",
            source="structured",
        )
    ]
    docs = [_doc(grade="JSS_1", subject="SCIENCE", topic="ecosystems", content="ecosystems")]
    decision = arbitrate_documents(
        docs,
        structured=structured,
        question="List Class 4 Mathematics topics",
        grade="CLASS_4",
        subject="MATHEMATICS",
        control_accepted=True,
        policy=ArbitrationPolicy.ARBITRATED,
    )
    assert decision.document_role in {DocumentRole.IRRELEVANT, DocumentRole.CONFLICTING}
    assert decision.document_use in {DocumentUse.DO_NOT_USE, DocumentUse.REQUIRE_REVIEW, DocumentUse.PROVENANCE_ONLY}


def test_structured_first_preserves_recovery():
    docs = [_doc(grade="CLASS_5", subject="SCIENCE", content="science guidance plants")]
    decision = arbitrate_documents(
        docs,
        structured=[],
        grade="CLASS_5",
        subject="SCIENCE",
        control_accepted=False,
        control_route="fallback",
        policy=ArbitrationPolicy.STRUCTURED_FIRST,
    )
    assert decision.document_use == DocumentUse.INCLUDE_IN_GENERATION


def test_dd1c57_style_regression_prevented_by_arbitration():
    record = {
        "question": {"hash": "dd1c57ff32af875b", "grade": "CLASS_4", "subject": None},
        "control": {
            "final_accepted": True,
            "final_route": "finish",
            "verifier_decision": "accept",
            "verifier_score": 1.0,
            "mapper_recommendation": "accept",
            "unsupported_claims": [],
            "answer_hash": "aaa",
            "answer_length": 400,
            "evidence_count": 40,
        },
        "shadow": {
            "structured_evidence_count": 40,
            "document_evidence_count": 5,
            "final_accepted": False,
            "final_route": "fallback",
            "verifier_decision": "retrieve_more",
            "verifier_score": 0.6,
            "mapper_recommendation": "reject",
            "unsupported_claims": ["Money in Class 4 should help pupils"],
            "answer_hash": "bbb",
            "answer_length": 900,
            "document_passages": [
                {
                    "document_id": "doc-f05cba561646",
                    "subject": "MATHEMATICS",
                    "grade": None,
                    "topic": None,
                    "retrieval_score": 0.03,
                },
                {
                    "document_id": "doc-bb746307a337",
                    "subject": "MATHEMATICS",
                    "grade": "CLASS_4",
                    "topic": "money",
                    "retrieval_score": 0.03,
                },
            ],
        },
        "comparison": {"control_correct_shadow_worse": True, "classification": "DOCUMENT_NOISE"},
        "grounding": {"wrong_context": False, "placeholder_evidence": False},
    }
    from app.agent.v213f_arbitration import arbitrate_from_shadow_record

    for policy in (ArbitrationPolicy.STRUCTURED_FIRST, ArbitrationPolicy.ARBITRATED):
        decision = arbitrate_from_shadow_record(record, policy=policy)
        cf = counterfactual_outcome(record, decision)
        assert decision.document_use in {
            DocumentUse.PROVENANCE_ONLY,
            DocumentUse.DO_NOT_USE,
        }
        assert cf["final_accepted"] is True
        assert cf["control_correct_shadow_worse"] is False
        assert cf["generator_drift_prevented"] is True


def test_v213f_flag_defaults_false():
    settings = Settings(_env_file=None)
    assert settings.v213f_document_arbitration_experiment is False


def test_shadow_pipeline_arbitration_excludes_docs_when_enabled(tmp_path):
    from app.agent.orchestrator import CurriculumQAAgent
    from app.agent.state import CurriculumQAState, AgentStatus
    from app.agent.v213d_shadow import run_shadow_pipeline
    from app.llm.provider import StubLLMProvider
    from app.curriculum.evidence import EvidenceStatus

    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        v213d_shadow_enabled=True,
        v213f_document_arbitration_experiment=True,
        v213f_arbitration_policy="B_STRUCTURED_FIRST",
    )
    agent = CurriculumQAAgent(settings=settings, llm=StubLLMProvider())
    structured = [
        CurriculumEvidence(
            entity_type="topic",
            entity_id=f"t{i}",
            name="Money",
            content="Money",
            grade="CLASS_4",
            subject="MATHEMATICS",
            source="structured",
        )
        for i in range(12)
    ]
    state = CurriculumQAState(
        question="What is taught about money in Class 4?",
        grade="CLASS_4",
        subject="MATHEMATICS",
        topic="money",
        evidence=structured,
        evidence_status=EvidenceStatus.FOUND,
        status=AgentStatus.ANSWERING,
        final_answer="Structured money answer",
        draft_answer="Structured money answer",
    )
    # Pre-populate verification-like acceptance via stub after answer/verify in control snapshot
    state = agent.verify(agent.answer(state, request_id="ctrl"), request_id="ctrl")

    docs = [
        _doc(grade="CLASS_4", subject="MATHEMATICS", topic="money", entity_id=f"d{i}")
        for i in range(3)
    ]

    def fake_retrieve(**_kwargs):
        return docs, {
            "variant": "context_hybrid",
            "latency_ms": 1.0,
            "count": len(docs),
            "passages": [],
            "corpus_available": True,
        }

    record = run_shadow_pipeline(
        agent,
        state,
        request_id="v213f-test",
        retrieve_documents=fake_retrieve,
    )
    assert record["shadow"].get("v213f_arbitration")
    assert (
        record["shadow"]["v213f_arbitration"]["document_use"]
        == DocumentUse.PROVENANCE_ONLY.value
    )
    assert record["shadow"]["v213f_arbitration"]["generation_document_count"] == 0
