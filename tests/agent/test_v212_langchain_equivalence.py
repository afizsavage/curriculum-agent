"""V2.12A LangChain vs LangGraph behavioral equivalence tests."""

from __future__ import annotations

import copy
import inspect

from app.agent.graph import build_curriculum_qa_graph
from app.agent.orchestrator import CurriculumQAAgent
from app.agent.state import CurriculumQAState
from app.agent.v212_contract import (
    EquivalenceClassification,
    PipelineRunResult,
    compare_pipeline_results,
)
from app.agent.v212_langchain import (
    Implementation,
    build_langchain_harness_chain,
    build_langgraph_harness_graph,
    default_implementation,
    run_equivalence_pair,
    run_implementation,
    v212_experiment_enabled,
)
from app.agent.v29_evidence_normalization import NormalizationVariant, normalize_evidence
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence
from app.schemas.verification import VerificationRecommendation, VerificationResult


def _c4u18_unit() -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="unit",
        entity_id="unit-1",
        name="Everyday Arithmetic Money",
        content="Everyday Arithmetic Money",
        subject="MATHEMATICS",
        grade="CLASS_4",
        metadata={"code": "C4-U18"},
    )


def _c4u18_lo() -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id="lo1",
        name="C4U18-LO01",
        content="Order operations using BODMAS.",
        topic="unit-1",
        grade="CLASS_4",
        subject="MATHEMATICS",
        metadata={
            "code": "C4U18-LO01",
            "parent_content_name": "Everyday Arithmetic Money",
            "parent_content_code": "C4-U18",
        },
    )


class _AcceptAll:
    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=True,
            score=1.0,
            recommendation=VerificationRecommendation.ACCEPT,
        )


class _HighRetrieve:
    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=False,
            score=0.9,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
        )


class _RejectAll:
    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=False,
            score=0.1,
            recommendation=VerificationRecommendation.FALLBACK,
            unsupported_claims=["fabricated"],
        )


class _RetrieveMissing:
    def verify(self, state, request_id=None):
        return VerificationResult(
            passed=False,
            score=0.0,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            missing_evidence=[{"type": "learning_outcome"}],
        )


def _run(fixture: str, impl: Implementation, verifier) -> PipelineRunResult:
    return run_implementation(
        implementation=impl,
        fixture_class=fixture,
        c4u18_baseline=[_c4u18_unit(), _c4u18_lo()],
        fractions_baseline=[_c4u18_lo()],
        verifier=verifier,
    )


def test_langgraph_control_still_runs():
    agent = CurriculumQAAgent()
    assert agent.graph is not None
    assert build_curriculum_qa_graph(
        nodes=agent.nodes, settings=agent.settings, checkpointer=None
    )


def test_langchain_implementation_runs():
    result = _run("FAITHFUL_COMPLETE", Implementation.LANGCHAIN, _AcceptAll())
    assert result.implementation == "langchain"
    assert result.execution_path


def test_default_production_implementation_is_langgraph():
    assert default_implementation(Settings()) == Implementation.LANGGRAPH


def test_experiment_flag_defaults_off():
    settings = Settings()
    qa = CurriculumQAState.initial(question="q")
    assert v212_experiment_enabled(settings, qa) is False


def test_both_accept_valid_c4u18_fc_path():
    lg = _run("FAITHFUL_COMPLETE", Implementation.LANGGRAPH, _AcceptAll())
    lc = _run("FAITHFUL_COMPLETE", Implementation.LANGCHAIN, _AcceptAll())
    assert lg.final_accepted is True
    assert lc.final_accepted is True
    assert lg.metadata_integrity_valid is True
    assert lc.metadata_integrity_valid is True


def test_normalization_equivalent():
    lg = _run("FAITHFUL_COMPLETE", Implementation.LANGGRAPH, _AcceptAll())
    lc = _run("FAITHFUL_COMPLETE", Implementation.LANGCHAIN, _AcceptAll())
    assert lg.normalized_evidence_hash == lc.normalized_evidence_hash


def test_metadata_validation_equivalent():
    lg = _run("ADV_WRONG_SUBJECT", Implementation.LANGGRAPH, _AcceptAll())
    lc = _run("ADV_WRONG_SUBJECT", Implementation.LANGCHAIN, _AcceptAll())
    assert lg.metadata_integrity_valid is False
    assert lc.metadata_integrity_valid is False
    assert lg.metadata_blocked is True
    assert lc.metadata_blocked is True


def test_unsupported_claim_rejected_both():
    for impl in (Implementation.LANGGRAPH, Implementation.LANGCHAIN):
        result = _run("UNSUPPORTED_CLAIM", impl, _RejectAll())
        assert result.final_accepted is False


def test_unsupported_absence_rejected_both():
    for impl in (Implementation.LANGGRAPH, Implementation.LANGCHAIN):
        result = _run("UNSUPPORTED_ABSENCE", impl, _RejectAll())
        assert result.final_accepted is False


def test_speculative_rejected_both():
    for impl in (Implementation.LANGGRAPH, Implementation.LANGCHAIN):
        result = _run("SPECULATIVE", impl, _RejectAll())
        assert result.final_accepted is False


def test_reconstruction_rejected_both():
    for impl in (Implementation.LANGGRAPH, Implementation.LANGCHAIN):
        result = _run("RECONSTRUCTION", impl, _RejectAll())
        assert result.final_accepted is False


def test_missing_evidence_non_acceptable_both():
    for impl in (Implementation.LANGGRAPH, Implementation.LANGCHAIN):
        result = _run("MISSING_EVIDENCE", impl, _RetrieveMissing())
        assert result.final_accepted is False


def test_placeholder_non_acceptable_both():
    for impl in (Implementation.LANGGRAPH, Implementation.LANGCHAIN):
        result = _run("CLEAN_PLACEHOLDER", impl, _AcceptAll())
        assert result.final_accepted is False


def test_conflicting_parent_blocked_both():
    for impl in (Implementation.LANGGRAPH, Implementation.LANGCHAIN):
        result = _run("ADV_CONFLICTING_PARENT", impl, _AcceptAll())
        assert result.final_accepted is False


def test_placeholder_parent_blocked_both():
    for impl in (Implementation.LANGGRAPH, Implementation.LANGCHAIN):
        result = _run("ADV_PLACEHOLDER_PARENT", impl, _AcceptAll())
        assert result.final_accepted is False


def test_wrong_subject_blocked_both():
    for impl in (Implementation.LANGGRAPH, Implementation.LANGCHAIN):
        result = _run("ADV_WRONG_SUBJECT", impl, _AcceptAll())
        assert result.final_accepted is False


def test_wrong_grade_blocked_both():
    for impl in (Implementation.LANGGRAPH, Implementation.LANGCHAIN):
        result = _run("ADV_WRONG_GRADE", impl, _AcceptAll())
        assert result.final_accepted is False


def test_mapper_safety_dominance_intact():
    result = _run("UNSUPPORTED_CLAIM", Implementation.LANGCHAIN, _AcceptAll())
    assert result.final_accepted is False


def test_routing_equivalent_on_fc():
    lg = _run("FAITHFUL_COMPLETE", Implementation.LANGGRAPH, _AcceptAll())
    lc = _run("FAITHFUL_COMPLETE", Implementation.LANGCHAIN, _AcceptAll())
    assert lg.final_route == lc.final_route


def test_langchain_cannot_bypass_metadata_integrity():
    result = _run("ADV_CONFLICTING_PARENT", Implementation.LANGCHAIN, _AcceptAll())
    assert result.metadata_blocked is True
    assert result.final_accepted is False


def test_langchain_cannot_bypass_recommendation_safety():
    result = _run("PLACEHOLDER_PLUS_HIGH_SCORE", Implementation.LANGCHAIN, _AcceptAll())
    assert result.final_accepted is False


def test_structured_result_contract_populated():
    result = _run("FAITHFUL_COMPLETE", Implementation.LANGCHAIN, _AcceptAll())
    payload = result.to_dict()
    for key in (
        "question",
        "raw_evidence_hash",
        "normalized_evidence_hash",
        "metadata_integrity_valid",
        "verifier_score",
        "mapped_recommendation",
        "final_route",
    ):
        assert key in payload


def test_comparator_classifications_work():
    control = PipelineRunResult(
        implementation="langgraph",
        fixture_class="FAITHFUL_COMPLETE",
        final_accepted=True,
        normalized_evidence_hash="abc",
        metadata_integrity_valid=True,
        metadata_blocked=False,
        verifier_score=1.0,
        verifier_recommendation="accept",
        final_route="finish",
        mapped_recommendation="accept",
    )
    experiment = copy.deepcopy(control)
    experiment.implementation = "langchain"
    comp = compare_pipeline_results(control, experiment, fixture_class="FAITHFUL_COMPLETE")
    assert comp.classification == EquivalenceClassification.EXACT_EQUIVALENCE


def test_unsafe_divergence_detected():
    control = PipelineRunResult(
        implementation="langgraph",
        fixture_class="ADV_WRONG_SUBJECT",
        final_accepted=False,
        metadata_integrity_valid=False,
        metadata_blocked=True,
    )
    experiment = PipelineRunResult(
        implementation="langchain",
        fixture_class="ADV_WRONG_SUBJECT",
        final_accepted=True,
        metadata_integrity_valid=False,
        metadata_blocked=True,
    )
    comp = compare_pipeline_results(control, experiment, fixture_class="ADV_WRONG_SUBJECT")
    assert comp.classification == EquivalenceClassification.UNSAFE_DIVERGENCE


def test_langgraph_production_implementation_untouched():
    source = inspect.getsource(build_curriculum_qa_graph)
    assert "understand" in source
    assert "verify_answer" in source


def test_fi_path_with_high_retrieve():
    result = _run("FAITHFUL_IMPERFECT", Implementation.LANGCHAIN, _HighRetrieve())
    assert result.metadata_integrity_valid is True
    assert result.final_accepted is True


def test_equivalence_pair_runs_both():
    row = run_equivalence_pair(
        fixture_class="FAITHFUL_COMPLETE",
        c4u18_baseline=[_c4u18_unit(), _c4u18_lo()],
        fractions_baseline=[_c4u18_lo()],
        verifier=_AcceptAll(),
    )
    assert "langgraph" in row
    assert "langchain" in row
    assert row["comparison"]["classification"] in {
        EquivalenceClassification.EXACT_EQUIVALENCE.value,
        EquivalenceClassification.BEHAVIORAL_EQUIVALENCE.value,
    }


def test_harness_graph_and_chain_build():
    assert build_langgraph_harness_graph() is not None
    assert build_langchain_harness_chain() is not None


def test_normalization_not_mutated_by_orchestration():
    raw = [_c4u18_unit(), _c4u18_lo()]
    before = copy.deepcopy(raw)
    _run("FAITHFUL_COMPLETE", Implementation.LANGCHAIN, _AcceptAll())
    normalized = normalize_evidence(raw, NormalizationVariant.STRUCTURAL_NORMALIZATION).evidence
    assert raw == before
    assert normalized
