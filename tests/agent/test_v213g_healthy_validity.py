"""V2.13G healthy-row validity, cohort accounting, and milestone tests.

Infrastructure failure must not be counted as valid structured insufficiency.
"""

from __future__ import annotations

from pathlib import Path

from app.agent.v213f_arbitration import (
    ArbitrationPolicy,
    DocumentRole,
    DocumentUse,
    StructuredSufficiency,
    arbitrate_documents,
)
from app.agent.v213g_live_shadow import (
    assess_healthy_live_milestone,
    classify_row_validity,
    count_sufficient_with_docs,
    gate_arbitration_labels,
    load_v213g_records,
)
from app.curriculum.evidence import CurriculumEvidence, ToolCallRecord


def _doc() -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="document_passage",
        entity_id="p1",
        name="passage",
        content="class 4 mathematics money",
        grade="CLASS_4",
        subject="MATHEMATICS",
        source="document",
        metadata={"retrieval_score": 0.4, "retrieval_rank": 1, "document_id": "doc-1"},
    )


def _decisive_decision() -> dict:
    decision = arbitrate_documents(
        [_doc()],
        structured=[],
        grade="CLASS_4",
        subject="MATHEMATICS",
        control_accepted=False,
        control_route="fallback",
        control_verifier_decision="retrieve_more",
        policy=ArbitrationPolicy.ARBITRATED,
    )
    return decision.to_dict()


def test_api_unavailable_is_infrastructure_invalid_not_decisive():
    validity = classify_row_validity(
        structured_api_available=False,
        structured_count=0,
        evidence_status="error",
        structured_failure="api_unavailable",
    )
    assert validity["validity"] == "INFRASTRUCTURE_INVALID"
    assert validity["infrastructure_valid"] is False
    assert validity["diagnostic_reason"] == "api_unavailable"
    assert validity["structured_count_reliable"] is False

    gated = gate_arbitration_labels(_decisive_decision(), validity)
    assert gated["structured_sufficiency"] != StructuredSufficiency.INSUFFICIENT.value
    assert gated["document_role"] != DocumentRole.DECISIVE.value
    assert gated["document_use"] != DocumentUse.INCLUDE_IN_GENERATION.value
    assert gated["classification_withheld"] is True

    rows = [
        {
            "batch_id": "batch_2_api_degraded",
            "source": "LIVE_TRAFFIC",
            "infrastructure": {**validity, "structured_count": None},
            "arbitration": gated,
            "retrieval": {"provenance_complete": True, "document_evidence_count": 5},
            "baseline_shadow": {"metadata_valid": True},
            "arbitrated_shadow": {},
        }
    ]
    coverage = count_sufficient_with_docs(rows)
    assert coverage["infrastructure_invalid"] == 1
    assert coverage["healthy_valid"] == 0
    assert coverage["structured_insufficient_decisive_docs"] == 0


def test_api_timeout_is_infrastructure_invalid_and_excluded_from_hypotheses():
    validity = classify_row_validity(
        structured_api_available=True,
        structured_count=0,
        evidence_status="error",
        structured_failure="timeout",
    )
    assert validity["validity"] == "INFRASTRUCTURE_INVALID"
    assert validity["diagnostic_reason"] == "timeout"
    assert validity["structured_count_zero_due_to_genuine_no_evidence"] is False

    record = {
        "batch_id": "batch_2_healthy_structured_api",
        "source": "LIVE_TRAFFIC",
        "infrastructure": validity,
        "arbitration": gate_arbitration_labels(_decisive_decision(), validity),
        "hypothesis": {
            "control_correct_baseline_worse": True,
            "control_correct_arbitrated_worse": True,
            "control_insufficient_baseline_accepted": True,
            "control_insufficient_arbitrated_accepted": True,
        },
        "retrieval": {"provenance_complete": True},
        "baseline_shadow": {"metadata_valid": True},
        "arbitrated_shadow": {},
        "cohorts": {
            "structured_sufficient_with_docs": {"in_cohort": True},
            "structured_insufficient_decisive_docs": {"in_cohort": True},
        },
    }
    from app.agent.v213g_live_shadow import aggregate_v213g_records

    summary = aggregate_v213g_records([record])
    assert summary["n_successful"] == 0
    assert summary["regression"]["baseline_regression"] == 0
    assert summary["regression"]["arbitrated_regression"] == 0
    assert summary["recovery"]["baseline_recovery"] == 0
    assert summary["recovery"]["arbitrated_recovery"] == 0
    assert summary["healthy_batch_coverage"]["infrastructure_invalid"] == 1
    assert summary["healthy_batch_coverage"]["healthy_valid"] == 0


def test_malformed_response_is_not_a_measured_zero():
    validity = classify_row_validity(
        structured_api_available=True,
        structured_count=0,
        evidence_status="error",
        structured_failure="malformed_response",
    )
    assert validity["validity"] == "INFRASTRUCTURE_INVALID"
    assert validity["structured_count_reliable"] is False
    assert validity["structured_count_zero_due_to_malformed_response"] is True
    assert validity["structured_count_zero_due_to_genuine_no_evidence"] is False
    gated = gate_arbitration_labels(_decisive_decision(), validity)
    assert gated["structured_sufficiency"] != "INSUFFICIENT"
    assert gated["classification_withheld"] is True
    assert validity["diagnostic_reason"] == "malformed_response"


def test_valid_zero_structured_evidence_stays_insufficient():
    validity = classify_row_validity(
        structured_api_available=True,
        structured_count=0,
        evidence_status="not_found",
    )
    assert validity["validity"] == "VALID"
    assert validity["infrastructure_valid"] is True
    assert validity["structured_count_reliable"] is True
    assert validity["zero_structured_reason"] == "genuine_no_evidence"

    decision = _decisive_decision()
    gated = gate_arbitration_labels(decision, validity)
    assert gated["structured_sufficiency"] == StructuredSufficiency.INSUFFICIENT.value
    assert gated.get("classification_withheld") is not True
    assert decision["structured_count"] == 0


def test_timeout_record_from_tool_history():
    from app.agent.state import CurriculumQAState
    from app.agent.v213g_live_shadow import infer_structured_failure

    state = CurriculumQAState(question="core subjects in class 4?")
    state.retrieval_history.append(
        ToolCallRecord(
            tool="get_curriculum_structure",
            status="error",
            error="Curriculum API request timed out",
            curriculum_api_status=504,
        )
    )
    assert infer_structured_failure(state, structured_count=0) == "timeout"
    validity = classify_row_validity(
        structured_api_available=True,
        structured_count=0,
        evidence_status="error",
        structured_failure=infer_structured_failure(state, structured_count=0),
    )
    assert validity["diagnostic_reason"] == "timeout"
    assert validity["validity"] == "INFRASTRUCTURE_INVALID"


def _healthy_row(
    *,
    sufficiency: str,
    role: str,
    document_count: int = 5,
    effect: str = "neutral",
) -> dict:
    return {
        "batch_id": "batch_2_healthy_structured_api",
        "source": "LIVE_TRAFFIC",
        "corpus_epoch": "post_corpus",
        "infrastructure": {
            "validity": "VALID",
            "infrastructure_valid": True,
            "structured_api_available": True,
        },
        "arbitration": {
            "structured_sufficiency": sufficiency,
            "document_role": role,
            "document_count": document_count,
        },
        "retrieval": {
            "provenance_complete": True,
            "retrieval_failure_kind": None,
            "document_evidence_count": document_count,
        },
        "baseline_shadow": {"metadata_valid": True},
        "arbitrated_shadow": {},
        "comparisons": {"control_vs_arbitrated": {"document_effect": effect}},
    }


def test_cohort_accounting_ignores_infrastructure_invalid_row():
    rows = [
        _healthy_row(sufficiency="SUFFICIENT", role="REDUNDANT"),
        _healthy_row(sufficiency="SUFFICIENT", role="IRRELEVANT"),
        _healthy_row(sufficiency="INSUFFICIENT", role="DECISIVE", effect="helped"),
        {
            "batch_id": "batch_2_healthy_structured_api",
            "source": "LIVE_TRAFFIC",
            "infrastructure": {
                "validity": "INFRASTRUCTURE_INVALID",
                "infrastructure_valid": False,
                "diagnostic_reason": "api_unavailable",
            },
            "arbitration": {
                "structured_sufficiency": "INSUFFICIENT",
                "document_role": "DECISIVE",
                "document_use": "INCLUDE_IN_GENERATION",
                "document_count": 5,
            },
            "retrieval": {
                "provenance_complete": True,
                "document_evidence_count": 5,
            },
            "baseline_shadow": {"metadata_valid": True},
            "arbitrated_shadow": {},
            "comparisons": {"control_vs_arbitrated": {"document_effect": "helped"}},
        },
    ]
    coverage = count_sufficient_with_docs(rows)
    assert coverage["structured_sufficient_with_docs"] == 2
    assert coverage["structured_sufficient_redundant_docs"] == 1
    assert coverage["structured_sufficient_irrelevant_docs"] == 1
    assert coverage["structured_insufficient_decisive_docs"] == 1
    assert coverage["healthy_valid"] == 3
    assert coverage["valid_healthy"] == 3
    assert coverage["infrastructure_invalid"] == 1
    assert coverage["neutral_document_cases"] == 2


def test_milestone_stays_insufficient_until_healthy_n_50():
    below = assess_healthy_live_milestone(
        healthy_valid=49, structured_sufficient_with_docs=20
    )
    assert below["status"] == "INSUFFICIENT_SAMPLE"
    assert "n>=50" in below["recommendation_text"]
    assert below["milestones"]["reached_first"] is False
    assert below["stage1_sufficient_with_docs"]["met"] is True
    assert below["production_promotion"] is False
    assert below["v213g_live_only"] is True
    assert below["v213f_document_arbitration_experiment"] is False

    reached = assess_healthy_live_milestone(
        healthy_valid=50, structured_sufficient_with_docs=20
    )
    assert reached["milestones"]["reached_first"] is True
    assert reached["production_promotion"] is False
    assert reached["v213g_live_only"] is True
    assert reached["v213f_document_arbitration_experiment"] is False
    assert reached["milestones"]["stronger"] == 100
    assert reached["milestones"]["preferred"] == 200
    assert reached["sample_rate_unchanged"] == 0.01


def test_live_jsonl_healthy_baseline_unchanged():
    path = Path("data/diagnostics/v213g_arbitration_shadow.jsonl")
    rows = load_v213g_records(path)
    coverage = count_sufficient_with_docs(rows)
    assert coverage["healthy_valid"] == 32
    assert coverage["structured_sufficient_with_docs"] == 20
    assert coverage["structured_insufficient_decisive_docs"] == 12
    assert coverage["infrastructure_invalid"] == 0
    assert coverage["historical_batch_1"] == 52

    preserved = Path(
        "data/diagnostics/v213g_arbitration_summary_batch1_api_unavailable.json"
    )
    assert preserved.is_file()
