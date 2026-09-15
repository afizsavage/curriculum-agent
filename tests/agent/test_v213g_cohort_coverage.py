"""V2.13G cohort-coverage investigation tests."""

from __future__ import annotations

from app.agent.v213f_arbitration import (
    ArbitrationPolicy,
    DocumentRole,
    DocumentUse,
    StructuredSufficiency,
    arbitrate_documents,
    arbitrate_from_shadow_record,
    counterfactual_outcome,
)
from app.agent.v213g_cohort_coverage import (
    analyze_live_g_rows,
    determine_root_cause,
    hypothesis_results,
    replay_v213f_through_classifier,
)
from app.curriculum.evidence import CurriculumEvidence


def _doc(**kwargs) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="document_passage",
        entity_id=kwargs.get("entity_id", "p1"),
        name="passage",
        content=kwargs.get("content", "money class 4 mathematics"),
        grade=kwargs.get("grade"),
        subject=kwargs.get("subject"),
        topic=kwargs.get("topic"),
        source="document",
        metadata={
            "retrieval_score": 0.03,
            "retrieval_rank": 1,
            "document_id": kwargs.get("document_id", "doc-bb746307a337"),
            "source_id": "math-primary",
            "page_number": 1,
        },
    )


def test_missing_structured_evidence_is_insufficient_decisive():
    decision = arbitrate_documents(
        [_doc(grade="CLASS_4", subject="MATHEMATICS")],
        structured=[],
        grade="CLASS_4",
        subject="MATHEMATICS",
        control_accepted=False,
        control_route="fallback",
        control_verifier_decision="retrieve_more",
        policy=ArbitrationPolicy.ARBITRATED,
    )
    assert decision.structured_sufficiency == StructuredSufficiency.INSUFFICIENT
    assert decision.document_role == DocumentRole.DECISIVE
    assert decision.document_use == DocumentUse.INCLUDE_IN_GENERATION


def test_structured_sufficient_irrelevant_do_not_use():
    structured = [
        CurriculumEvidence(
            entity_type="topic",
            entity_id="t1",
            name="Money",
            content="Money",
            grade="CLASS_4",
            subject="MATHEMATICS",
            source="structured",
        )
        for _ in range(8)
    ]
    docs = [_doc(grade="JSS_1", subject="SCIENCE", content="ecosystems")]
    decision = arbitrate_documents(
        docs,
        structured=structured,
        grade="CLASS_4",
        subject="MATHEMATICS",
        control_accepted=True,
        policy=ArbitrationPolicy.ARBITRATED,
    )
    assert decision.structured_sufficiency == StructuredSufficiency.SUFFICIENT
    assert decision.document_role in {DocumentRole.IRRELEVANT, DocumentRole.CONFLICTING}
    assert decision.document_use in {
        DocumentUse.DO_NOT_USE,
        DocumentUse.REQUIRE_REVIEW,
        DocumentUse.PROVENANCE_ONLY,
    }


def test_structured_insufficient_irrelevant_still_recovery_path_when_docs_present():
    # When structured failed, docs remain recovery candidates even without hierarchy match.
    decision = arbitrate_documents(
        [_doc(grade="JSS_1", subject="SCIENCE", content="ecosystems")],
        structured=[],
        grade="CLASS_4",
        subject="MATHEMATICS",
        control_accepted=False,
        control_route="fallback",
        policy=ArbitrationPolicy.ARBITRATED,
    )
    assert decision.structured_sufficiency == StructuredSufficiency.INSUFFICIENT
    assert decision.document_use == DocumentUse.INCLUDE_IN_GENERATION


def test_live_all_zero_structured_is_sampling_bias_not_bug():
    rows = [
        {
            "control": {"final_accepted": False, "final_route": "fallback", "evidence_count": 0},
            "baseline_shadow": {"structured_evidence_count": 0, "document_evidence_count": 5},
            "arbitration": {
                "structured_sufficiency": "INSUFFICIENT",
                "document_role": "DECISIVE",
                "document_use": "INCLUDE_IN_GENERATION",
                "structured_count": 0,
                "document_count": 5,
            },
            "question": {"category": "insufficient_evidence", "grade": None, "subject": None},
        }
        for _ in range(5)
    ]
    live = analyze_live_g_rows(rows)
    root = determine_root_cause(live, curriculum_api_reachable=False)
    assert live["all_insufficient_decisive_include"] is True
    assert root["determination"] == "SAMPLING_BIAS"
    assert root["classification_validity"] == "CLASSIFICATION_CORRECT"
    assert root["primary_regression_hypothesis"] == "UNTESTED"


def test_dd1c57_targeted_replay_fixture():
    record = {
        "question": {"hash": "dd1c57ff32af875b", "grade": "CLASS_4"},
        "v213f_group": "D_regression",
        "request_id": "63bbf28e5f0db6e1",
        "control": {
            "final_accepted": True,
            "final_route": "finish",
            "verifier_decision": "accept",
            "evidence_count": 40,
            "unsupported_claims": [],
            "answer_hash": "aaa",
        },
        "shadow": {
            "structured_evidence_count": 40,
            "document_evidence_count": 5,
            "final_accepted": False,
            "final_route": "fallback",
            "verifier_decision": "retrieve_more",
            "unsupported_claims": ["drift"],
            "answer_hash": "bbb",
            "document_passages": [
                {
                    "document_id": "doc-bb746307a337",
                    "subject": "MATHEMATICS",
                    "grade": "CLASS_4",
                    "topic": "money",
                    "retrieval_score": 0.03,
                }
            ],
        },
        "comparison": {"control_correct_shadow_worse": True},
    }
    out = replay_v213f_through_classifier([record])
    assert out["dd1c57"]["pass"] is True
    assert out["cohort_counts"]["structured_sufficient_redundant_docs"] == 1


def test_hypothesis_untested_when_live_sufficient_cohort_zero():
    hyp = hypothesis_results(
        live_sufficient_with_docs=0,
        live_baseline_reg=0,
        live_arb_reg=0,
        live_baseline_rec=30,
        live_arb_rec=36,
        live_baseline_unsup=30,
        live_arb_unsup=17,
        targeted_sufficient_with_docs=10,
        targeted_dd1c57_pass=True,
        safety_blocked=False,
    )
    assert hyp["H1_regression_prevention"]["result"] == "UNTESTED"
    assert hyp["H2_recovery_preservation"]["result"] == "CONFIRMED"
    assert hyp["primary_regression_hypothesis_live"] == "PRIMARY_REGRESSION_HYPOTHESIS_UNTESTED"
    assert hyp["recommendation"] == "PROCEED_TO_TARGETED_LIVE_VALIDATION"
    assert hyp["safety"] == "PASS"


def test_conflicting_metadata_require_review():
    docs = [
        _doc(grade="JSS_3", subject="SCIENCE", content="a"),
        _doc(grade="JSS_2", subject="LANGUAGE_ARTS", content="b", entity_id="p2"),
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
