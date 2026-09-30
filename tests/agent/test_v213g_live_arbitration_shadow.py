"""V2.13G live dual-arm document arbitration shadow tests."""

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
from app.agent.v213g_live_shadow import (
    decide_v213g_status,
    run_dual_arm_shadow_pipeline,
)
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence


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
            "document_id": kwargs.get("document_id", "doc-bb746307a337"),
            "source_id": "math-primary-guidance",
            "page_number": 1,
        },
    )


def test_v213g_flag_defaults_false(monkeypatch):
    for key in (
        "V213G_LIVE_ARBITRATION_SHADOW",
        "V213F_DOCUMENT_ARBITRATION_EXPERIMENT",
        "V213D_SHADOW_ENABLED",
        "V213D_SHADOW_SAMPLE_RATE",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.v213g_live_arbitration_shadow is False
    assert settings.v213f_document_arbitration_experiment is False


def test_structured_sufficient_redundant_provenance_only():
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
    assert all(e.entity_type != "document_passage" for e in gen)


def test_structured_insufficient_decisive_include():
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


def test_irrelevant_document_do_not_use():
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
    assert decision.document_use in {
        DocumentUse.DO_NOT_USE,
        DocumentUse.REQUIRE_REVIEW,
        DocumentUse.PROVENANCE_ONLY,
    }


def test_conflicting_documents_require_review():
    docs = [
        _doc(grade="JSS_3", subject="SCIENCE", content="wrong grade science"),
        _doc(
            grade="JSS_2",
            subject="LANGUAGE_ARTS",
            content="wrong subject",
            entity_id="p2",
        ),
    ]
    decision = arbitrate_documents(
        docs,
        structured=[
            CurriculumEvidence(
                entity_type="topic",
                entity_id="t1",
                name="Money",
                content="Money",
                grade="CLASS_4",
                subject="MATHEMATICS",
                source="structured",
            )
        ],
        grade="CLASS_4",
        subject="MATHEMATICS",
        control_accepted=True,
        policy=ArbitrationPolicy.ARBITRATED,
    )
    assert decision.document_role == DocumentRole.CONFLICTING
    assert decision.document_use == DocumentUse.REQUIRE_REVIEW


def test_dd1c57_regression_fixture_preserved():
    """Permanent V2.13F/G fixture: sufficient+redundant → provenance-only preserves control."""
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

    decision = arbitrate_from_shadow_record(record, policy=ArbitrationPolicy.ARBITRATED)
    cf = counterfactual_outcome(record, decision)
    assert decision.structured_sufficiency == StructuredSufficiency.SUFFICIENT
    assert decision.document_role == DocumentRole.REDUNDANT
    assert decision.document_use == DocumentUse.PROVENANCE_ONLY
    assert cf["final_accepted"] is True
    assert cf["control_correct_shadow_worse"] is False
    assert cf["generator_drift_prevented"] is True


def test_dual_arm_uses_same_retrieval_and_gates_docs(tmp_path):
    from app.agent.orchestrator import CurriculumQAAgent
    from app.agent.state import CurriculumQAState, AgentStatus
    from app.llm.provider import StubLLMProvider
    from app.curriculum.evidence import EvidenceStatus

    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        v213d_shadow_enabled=True,
        v213g_live_arbitration_shadow=True,
        v213f_document_arbitration_experiment=False,
        v213f_arbitration_policy="C_ARBITRATED",
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
    state = agent.verify(agent.answer(state, request_id="ctrl"), request_id="ctrl")

    docs = [
        _doc(grade="CLASS_4", subject="MATHEMATICS", topic="money", entity_id=f"d{i}")
        for i in range(3)
    ]
    retrieve_calls = {"n": 0}

    def fake_retrieve(**_kwargs):
        retrieve_calls["n"] += 1
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
        request_id="v213g-test",
        retrieve_documents=fake_retrieve,
    )
    assert retrieve_calls["n"] == 1
    assert record["experiment"] == "v2.13g"
    assert record["arbitration"]["structured_sufficiency"] == "SUFFICIENT"
    assert record["arbitration"]["document_use"] == DocumentUse.PROVENANCE_ONLY.value
    # Baseline merges docs into generation evidence; arbitrated keeps provenance-only.
    assert record["baseline_shadow"]["generation_evidence_count"] >= len(structured) + 1
    assert record["arbitrated_shadow"]["arbitration"]["generation_document_count"] == 0
    assert record["sampling"]["v213f_document_arbitration_experiment"] is False


def test_decide_status_insufficient_before_milestone():
    out = decide_v213g_status(
        n=12,
        baseline_reg=2,
        arb_reg=0,
        baseline_rec=3,
        arb_rec=3,
        safety_blocked=False,
        class_diag={},
    )
    assert out["status"] == "INSUFFICIENT_SAMPLE"


def test_decide_status_confirmed_at_milestone():
    out = decide_v213g_status(
        n=50,
        baseline_reg=5,
        arb_reg=0,
        baseline_rec=15,
        arb_rec=15,
        safety_blocked=False,
        class_diag={},
        structured_sufficient_with_docs=10,
    )
    assert out["status"] == "ARBITRATION_CONFIRMED"


def test_decide_status_untested_when_sufficient_cohort_missing():
    out = decide_v213g_status(
        n=52,
        baseline_reg=0,
        arb_reg=0,
        baseline_rec=30,
        arb_rec=36,
        safety_blocked=False,
        class_diag={},
        structured_sufficient_with_docs=0,
    )
    assert out["status"] == "INVESTIGATE_BEFORE_PROMOTION"
    assert "PRIMARY_REGRESSION_HYPOTHESIS_UNTESTED" in out["recommendation_text"]


def test_decide_status_stage1_sufficient_target_before_n50():
    out = decide_v213g_status(
        n=28,
        baseline_reg=0,
        arb_reg=0,
        baseline_rec=8,
        arb_rec=10,
        safety_blocked=False,
        class_diag={},
        structured_sufficient_with_docs=20,
    )
    assert out["status"] == "INSUFFICIENT_SAMPLE"
    assert "Stage-1 sufficient_with_docs target met" in out["recommendation_text"]


def test_classify_row_validity_api_failure_vs_genuine():
    from app.agent.v213g_live_shadow import classify_row_validity

    bad = classify_row_validity(
        structured_api_available=False, structured_count=0
    )
    assert bad["validity"] == "INFRASTRUCTURE_INVALID"
    assert bad["structured_count_zero_due_to_api_failure"] is True
    assert bad["structured_count_zero_due_to_genuine_no_evidence"] is False

    genuine = classify_row_validity(
        structured_api_available=True, structured_count=0
    )
    assert genuine["validity"] == "VALID"
    assert genuine["structured_count_zero_due_to_api_failure"] is False
    assert genuine["structured_count_zero_due_to_genuine_no_evidence"] is True

    ok = classify_row_validity(
        structured_api_available=True, structured_count=3
    )
    assert ok["validity"] == "VALID"
    assert ok["zero_structured_reason"] is None


def test_decide_status_safety_blocked():
    out = decide_v213g_status(
        n=50,
        baseline_reg=0,
        arb_reg=0,
        baseline_rec=1,
        arb_rec=1,
        safety_blocked=True,
        class_diag={},
    )
    assert out["status"] == "SAFETY_BLOCKED"
