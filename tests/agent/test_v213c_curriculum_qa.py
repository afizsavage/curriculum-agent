"""V2.13C controlled curriculum QA evaluation tests."""

from __future__ import annotations

import json

from app.agent.evidence_snapshot import evidence_snapshot_hash
from app.agent.v211_metadata_integrity import validate_metadata_integrity
from app.agent.v213c_dataset import DATASET_VERSION, build_v213c_dataset
from app.agent.v213c_experiment import (
    V213CEvaluationHarness,
    classify_difference,
    dataset_hash,
    frozen_structured_catalog,
    interpret_v213c,
    synthesize_answer,
    v213c_document_retrieval_enabled,
    v213c_experiment_enabled,
)
from app.agent.v28_recommendation_mapping import map_recommendation
from app.agent.v29_evidence_normalization import NormalizationVariant, normalize_evidence
from app.config import Settings
from app.schemas.verification import VerificationRecommendation, VerificationResult
from app.tools.registry import build_default_registry


def test_dataset_loading():
    items = build_v213c_dataset()
    assert len(items) >= 60
    categories = {q["category"] for q in items}
    assert {
        "document_only",
        "structured_plus_document",
        "source_grounding",
        "structured_fact",
        "ambiguous",
        "insufficient_evidence",
        "adversarial",
    } <= categories
    assert DATASET_VERSION
    assert dataset_hash(items) == dataset_hash(build_v213c_dataset())


def test_production_flags_remain_disabled():
    settings = Settings()
    assert v213c_experiment_enabled(settings) is False
    assert v213c_document_retrieval_enabled(settings) is False
    assert settings.v213c_retrieval_variant == "context_hybrid"
    registry = build_default_registry(settings=settings)
    assert "search_curriculum_documents" not in registry.names()


def test_control_and_experiment_execution(tmp_path):
    harness = V213CEvaluationHarness(
        store_root=tmp_path / "documents",
        index_root=tmp_path / "index",
    )
    harness.prepare_corpus()
    spec = next(q for q in build_v213c_dataset() if q["id"] == "V213C-A01")
    control = harness.run_arm(spec, arm="control")
    experiment = harness.run_arm(spec, arm="experiment")
    assert control["arm"] == "control"
    assert experiment["arm"] == "experiment"
    assert experiment["document_evidence_count"] >= 1
    assert control["structured_evidence_count"] == 0
    assert experiment["evidence_count"] >= control["evidence_count"]


def test_deterministic_pairing_and_frozen_snapshot(tmp_path):
    harness = V213CEvaluationHarness(
        store_root=tmp_path / "documents",
        index_root=tmp_path / "index",
    )
    harness.prepare_corpus()
    spec = next(q for q in build_v213c_dataset() if q["id"] == "V213C-D01")
    first = harness.run_arm(spec, arm="control")
    second = harness.run_arm(spec, arm="control")
    assert first["evidence_snapshot"] == second["evidence_snapshot"]
    structured = harness.structured_for(spec)
    assert first["evidence_snapshot"] == evidence_snapshot_hash(structured)


def test_document_evidence_inclusion_and_structured_preservation(tmp_path):
    harness = V213CEvaluationHarness(
        store_root=tmp_path / "documents",
        index_root=tmp_path / "index",
    )
    harness.prepare_corpus()
    spec = next(q for q in build_v213c_dataset() if q["id"] == "V213C-B01")
    experiment = harness.run_arm(spec, arm="experiment")
    assert experiment["structured_evidence_count"] >= 1
    assert experiment["document_evidence_count"] >= 1


def test_normalization_and_metadata_guard():
    catalog = frozen_structured_catalog()["c4u18"]
    normalized = normalize_evidence(catalog, NormalizationVariant.STRUCTURAL_NORMALIZATION)
    integrity = validate_metadata_integrity(normalized.evidence)
    assert normalized.diagnostics.evidence_hash_out
    assert isinstance(integrity.valid, bool)


def test_mapper_invocation():
    result = VerificationResult(
        passed=False,
        score=0.2,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        issues=["no evidence"],
    )
    mapping = map_recommendation(
        result,
        fixture_class="MISSING_EVIDENCE",
        evidence=[],
        answer="Insufficient curriculum evidence to answer this question.",
        threshold=0.85,
    )
    assert mapping.mapped_accepted is False


def test_provenance_preservation(tmp_path):
    harness = V213CEvaluationHarness(
        store_root=tmp_path / "documents",
        index_root=tmp_path / "index",
    )
    harness.prepare_corpus()
    spec = next(q for q in build_v213c_dataset() if q["id"] == "V213C-C01")
    experiment = harness.run_arm(spec, arm="experiment")
    assert experiment["provenance_complete"] is True
    assert experiment["document_evidence_count"] >= 1


def test_wrong_context_and_placeholder_rejection(tmp_path):
    harness = V213CEvaluationHarness(
        store_root=tmp_path / "documents",
        index_root=tmp_path / "index",
    )
    harness.prepare_corpus()
    placeholder = next(q for q in build_v213c_dataset() if q["id"] == "V213C-G03")
    wrong = next(q for q in build_v213c_dataset() if q["id"] == "V213C-G01")
    p_row = harness.run_arm(placeholder, arm="experiment")
    w_row = harness.run_arm(wrong, arm="experiment")
    assert p_row["final_accepted"] is False
    assert w_row["final_accepted"] is False


def test_insufficient_evidence_behavior(tmp_path):
    harness = V213CEvaluationHarness(
        store_root=tmp_path / "documents",
        index_root=tmp_path / "index",
    )
    harness.prepare_corpus()
    spec = next(q for q in build_v213c_dataset() if q["id"] == "V213C-F01")
    control = harness.run_arm(spec, arm="control")
    experiment = harness.run_arm(spec, arm="experiment")
    assert control["final_accepted"] is False
    assert experiment["final_accepted"] is False
    assert control["grounded_correct"] is True
    assert experiment["grounded_correct"] is True


def test_diagnostic_generation_and_interpret():
    conclusion, _, _ = interpret_v213c(
        newly_count=10,
        newly_doc=6,
        dataset_n=72,
        structured_delta=0.0,
        safety={
            "wrong_context_false_acceptance": 0,
            "placeholder_false_acceptance": 0,
            "metadata_integrity_false_acceptance": 0,
            "unsafe_adversarial_false_acceptance": 0,
        },
        document_only_gain=0.5,
    )
    assert conclusion == "SUPPORTED"
    unsafe, _, _ = interpret_v213c(
        newly_count=20,
        newly_doc=10,
        dataset_n=72,
        structured_delta=0.0,
        safety={
            "wrong_context_false_acceptance": 1,
            "placeholder_false_acceptance": 0,
            "metadata_integrity_false_acceptance": 0,
            "unsafe_adversarial_false_acceptance": 0,
        },
        document_only_gain=0.8,
    )
    assert unsafe == "NOT_SUPPORTED"


def test_synthesize_answer_and_classify_difference():
    evidence = frozen_structured_catalog()["c4u18"]
    answer = synthesize_answer("What are money LOs?", evidence)
    assert "C4U18" in answer or "BODMAS" in answer
    empty = synthesize_answer("none", [])
    assert "Insufficient" in empty
    label = classify_difference(
        {"grounded_correct": False, "evidence_count": 0},
        {"grounded_correct": True, "document_evidence_count": 3},
    )
    assert label == "DOCUMENT_ADDED_MISSING_CONTEXT"


def test_mini_eval_report(tmp_path):
    harness = V213CEvaluationHarness(
        store_root=tmp_path / "documents",
        index_root=tmp_path / "index",
    )
    items = [
        q
        for q in build_v213c_dataset()
        if q["id"] in {"V213C-A01", "V213C-D01", "V213C-F01", "V213C-G03"}
    ]
    report = harness.evaluate(items)
    assert "conclusion" in report
    assert report["dataset_size"] == 4
    assert "safety_metrics" in report
    json.dumps(report["safety_metrics"])
