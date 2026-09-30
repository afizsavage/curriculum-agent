"""V2.13G.1 PROVENANCE_ONLY generation-boundary tests."""

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
from app.agent.v213g1_provenance_only import (
    PROVENANCE_ONLY_SEMANTICS_PRESERVE,
    PROVENANCE_ONLY_SEMANTICS_REGENERATE,
    build_preserved_control_arm,
    classify_divergence_root_cause,
    should_preserve_control_answer,
)
from app.agent.v213g_live_shadow import run_dual_arm_shadow_pipeline
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.agent.state import AgentStatus, CurriculumQAState
from app.agent.orchestrator import CurriculumQAAgent
from app.llm.provider import StubLLMProvider


def _doc(**kwargs) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="document_passage",
        entity_id=kwargs.get("entity_id", "p1"),
        name=kwargs.get("name", "passage"),
        content=kwargs.get("content", "money class 4 mathematics teaching"),
        grade=kwargs.get("grade", "CLASS_4"),
        subject=kwargs.get("subject", "MATHEMATICS"),
        topic=kwargs.get("topic", "money"),
        source="document",
        metadata={
            "retrieval_score": kwargs.get("retrieval_score", 0.03),
            "retrieval_rank": kwargs.get("retrieval_rank", 1),
            "document_id": kwargs.get("document_id", "doc-bb746307a337"),
            "source_id": "math-primary-guidance",
            "page_number": 1,
        },
    )


def _structured(n: int = 12) -> list[CurriculumEvidence]:
    return [
        CurriculumEvidence(
            entity_type="topic",
            entity_id=f"t{i}",
            name="Money",
            content="Money topic",
            grade="CLASS_4",
            subject="MATHEMATICS",
            source="structured",
        )
        for i in range(n)
    ]


def test_sufficient_redundant_maps_provenance_only():
    decision = arbitrate_documents(
        [_doc()],
        structured=_structured(),
        question="What is taught about money in Class 4?",
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


def test_provenance_only_excludes_documents_from_generation():
    structured = _structured()
    docs = [_doc(entity_id="d1"), _doc(entity_id="d2")]
    decision = arbitrate_documents(
        docs,
        structured=structured,
        question="money class 4",
        grade="CLASS_4",
        subject="MATHEMATICS",
        topic="money",
        control_accepted=True,
        control_route="finish",
        control_verifier_decision="accept",
    )
    gen, prov = select_generation_evidence(structured, docs, decision)
    assert decision.document_use == DocumentUse.PROVENANCE_ONLY
    assert all(e.entity_type != "document_passage" for e in gen)
    assert len(prov) == len(docs)
    assert len(gen) == len(structured)


def test_sufficient_irrelevant_withholds_documents_from_generation():
    # Cross-grade/subject passage: may classify IRRELEVANT or CONFLICTING depending
    # on scoring; either way docs must not enter generation when control is sufficient.
    docs = [
        _doc(
            entity_id="irr",
            grade="JSS_3",
            subject="SCIENCE",
            topic="photosynthesis",
            content="photosynthesis chlorophyll plants",
        )
    ]
    structured = _structured()
    decision = arbitrate_documents(
        docs,
        structured=structured,
        question="What is taught about money in Class 4?",
        grade="CLASS_4",
        subject="MATHEMATICS",
        topic="money",
        control_accepted=True,
        control_route="finish",
        control_verifier_decision="accept",
    )
    assert decision.structured_sufficiency == StructuredSufficiency.SUFFICIENT
    assert decision.document_use != DocumentUse.INCLUDE_IN_GENERATION
    gen, prov = select_generation_evidence(structured, docs, decision)
    assert all(e.entity_type != "document_passage" for e in gen)
    if decision.document_use == DocumentUse.PROVENANCE_ONLY:
        assert len(prov) == len(docs)
    if decision.document_use == DocumentUse.DO_NOT_USE:
        assert prov == []


def test_insufficient_decisive_includes_documents():
    decision = arbitrate_documents(
        [_doc()],
        structured=[],
        question="What does the document say about money teaching?",
        grade="CLASS_4",
        subject="MATHEMATICS",
        topic="money",
        control_accepted=False,
        control_route="retrieve_more",
        control_verifier_decision="retrieve_more",
    )
    assert decision.structured_sufficiency == StructuredSufficiency.INSUFFICIENT
    assert decision.document_role == DocumentRole.DECISIVE
    assert decision.document_use == DocumentUse.INCLUDE_IN_GENERATION
    gen, prov = select_generation_evidence([], [_doc()], decision)
    assert any(e.entity_type == "document_passage" for e in gen)
    assert prov == []


def test_preserve_semantics_gate():
    assert should_preserve_control_answer(
        document_use="PROVENANCE_ONLY",
        control_accepted=True,
        semantics=PROVENANCE_ONLY_SEMANTICS_PRESERVE,
    )
    assert not should_preserve_control_answer(
        document_use="PROVENANCE_ONLY",
        control_accepted=True,
        semantics=PROVENANCE_ONLY_SEMANTICS_REGENERATE,
    )
    assert not should_preserve_control_answer(
        document_use="INCLUDE_IN_GENERATION",
        control_accepted=True,
        semantics=PROVENANCE_ONLY_SEMANTICS_PRESERVE,
    )


def test_preserved_control_arm_matches_control_hash():
    control = {
        "final_accepted": True,
        "final_route": "finish",
        "verifier_decision": "accept",
        "verifier_accepted": True,
        "verifier_score": 1.0,
        "mapper_recommendation": "accept",
        "mapped_accepted": True,
        "unsupported_claims": [],
        "answer_hash": "deadbeefcafebabe",
        "answer_length": 12,
        "answer_present": True,
        "evidence_snapshot": "snap1",
    }
    arm = build_preserved_control_arm(
        control,
        structured=_structured(3),
        documents=[_doc()],
        generation_evidence=_structured(3),
        provenance_evidence=[_doc()],
        retrieval_meta={"variant": "context_hybrid", "corpus_available": True},
        arbitration_info={"document_use": "PROVENANCE_ONLY"},
    )
    assert arm["outcome_source"] == "control"
    assert arm["answer_hash"] == "deadbeefcafebabe"
    assert arm["final_accepted"] is True
    assert arm["generation_document_count"] == 0
    assert arm["provenance_evidence_count"] == 1
    assert arm["generation_attempted"] is False
    assert arm["generation_evidence_fingerprint"]


def test_dual_arm_preserve_mode_reuses_control(monkeypatch):
    monkeypatch.setenv("V213G_PROVENANCE_ONLY_SEMANTICS", "preserve_control_answer")
    monkeypatch.setenv("V213G_LIVE_ARBITRATION_SHADOW", "true")
    monkeypatch.setenv("V213F_DOCUMENT_ARBITRATION_EXPERIMENT", "false")
    settings = Settings(_env_file=None)
    assert settings.v213g_provenance_only_semantics == "preserve_control_answer"
    agent = CurriculumQAAgent(settings=settings, llm=StubLLMProvider())
    structured = _structured(12)
    state = CurriculumQAState(
        question="What is taught about money in Class 4?",
        grade="CLASS_4",
        subject="MATHEMATICS",
        topic="money",
        evidence=structured,
        evidence_status=EvidenceStatus.FOUND,
        status=AgentStatus.ANSWERING,
        final_answer="Structured money answer preserved",
        draft_answer="Structured money answer preserved",
    )
    state = agent.verify(agent.answer(state, request_id="ctrl"), request_id="ctrl")
    ctrl_hash = __import__("app.agent.v26_experiment", fromlist=["answer_hash"]).answer_hash(
        state.final_answer or ""
    )
    docs = [_doc(entity_id=f"d{i}") for i in range(3)]

    def fake_retrieve(**_kwargs):
        return docs, {
            "variant": "context_hybrid",
            "latency_ms": 1.0,
            "count": len(docs),
            "passages": [],
            "corpus_available": True,
        }

    record = run_dual_arm_shadow_pipeline(
        agent,
        state,
        request_id="v213g1-preserve",
        retrieve_documents=fake_retrieve,
    )
    assert record["arbitration"]["document_use"] == DocumentUse.PROVENANCE_ONLY.value
    assert record["arbitration"]["generation_document_count"] == 0
    assert record["arbitrated_shadow"]["outcome_source"] == "control"
    assert record["arbitrated_shadow"]["generation_attempted"] is False
    assert record["arbitrated_shadow"]["answer_hash"] == (
        record["control"]["answer_hash"] or ctrl_hash
    )
    assert record["arbitrated_shadow"]["final_accepted"] is True


def test_dual_arm_preserve_does_not_block_decisive_include(monkeypatch):
    monkeypatch.setenv("V213G_PROVENANCE_ONLY_SEMANTICS", "preserve_control_answer")
    monkeypatch.setenv("V213G_LIVE_ARBITRATION_SHADOW", "true")
    monkeypatch.setenv("V213F_DOCUMENT_ARBITRATION_EXPERIMENT", "false")
    settings = Settings(_env_file=None)
    agent = CurriculumQAAgent(settings=settings, llm=StubLLMProvider())
    # Insufficient structured control
    state = CurriculumQAState(
        question="What does the curriculum document say about money teaching methods?",
        grade="CLASS_4",
        subject="MATHEMATICS",
        topic="money",
        evidence=[],
        evidence_status=EvidenceStatus.NOT_FOUND,
        status=AgentStatus.ANSWERING,
        final_answer="",
        draft_answer="",
    )
    docs = [
        _doc(entity_id=f"d{i}", content="money teaching methods class 4 mathematics guidance")
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

    record = run_dual_arm_shadow_pipeline(
        agent,
        state,
        request_id="v213g2-h2",
        retrieve_documents=fake_retrieve,
    )
    assert record["arbitration"]["document_use"] == DocumentUse.INCLUDE_IN_GENERATION.value
    assert record["arbitration"]["generation_document_count"] > 0
    assert record["arbitrated_shadow"].get("generation_attempted") is True
    assert record["arbitrated_shadow"].get("outcome_source") != "control"


def test_dd1c57_counterfactual_still_preserves_control():
    record = {
        "control": {
            "final_accepted": True,
            "final_route": "finish",
            "verifier_decision": "accept",
            "verifier_score": 1.0,
            "mapper_recommendation": "accept",
            "unsupported_claims": [],
            "answer_hash": "16088fa24c2d597f",
            "evidence_count": 40,
        },
        "shadow": {
            "final_accepted": False,
            "final_route": "retrieve_more",
            "verifier_decision": "retrieve_more",
            "unsupported_claims": ["x"],
            "answer_hash": "shadowhash",
            "structured_evidence_count": 40,
            "document_passages": [{"passage_id": "p1"}],
        },
        "question": {"hash": "dd1c57ff32af875b", "grade": "CLASS_4"},
        "comparison": {"control_correct_shadow_worse": True},
    }
    decision = arbitrate_documents(
        [_doc()],
        structured=_structured(40),
        question="What are the learning objectives for money in Primary 4?",
        grade="CLASS_4",
        control_accepted=True,
        control_route="finish",
        control_verifier_decision="accept",
    )
    assert decision.document_use == DocumentUse.PROVENANCE_ONLY
    cf = counterfactual_outcome(record, decision)
    assert cf["outcome_source"] == "control"
    assert cf["answer_hash"] == "16088fa24c2d597f"
    assert cf["final_accepted"] is True


def test_classify_control_regeneration_when_evidence_matches():
    case = {
        "arbitration": {"generation_document_count": 0},
        "control": {"evidence_snapshot": "abc"},
        "arbitrated_shadow": {"evidence_snapshot": "abc", "answer_hash": "x"},
    }
    assert classify_divergence_root_cause(case) == "CONTROL_REGENERATION"


def test_classify_document_leak():
    case = {
        "arbitration": {"generation_document_count": 2},
        "control": {"evidence_snapshot": "abc"},
        "arbitrated_shadow": {"evidence_snapshot": "abc"},
    }
    assert classify_divergence_root_cause(case) == "DOCUMENT_LEAK"
