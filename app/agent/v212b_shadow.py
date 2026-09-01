"""V2.12B production-shadow evaluation — real retrieval, observational only."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.agent.evidence_snapshot import evidence_snapshot_hash
from app.agent.orchestrator import CurriculumQAAgent
from app.agent.state import CurriculumQAState
from app.agent.v212_contract import (
    EquivalenceClassification,
    FailureOrigin,
    compare_pipeline_results,
)
from app.agent.v212_langchain import (
    Implementation,
    default_implementation,
    infer_mapper_fixture_class,
    run_post_retrieval_implementation,
    run_post_retrieval_pair,
    summarize_comparisons,
    summarize_implementation_rows,
)
from app.agent.v25_experiment import _deserialize_evidence, _serialize_evidence
from app.agent.v211_metadata_integrity import validate_metadata_integrity
from app.agent.v29_evidence_normalization import NormalizationVariant, normalize_evidence
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.schemas.verification import VerificationRecommendation

logger = logging.getLogger(__name__)

_EXPERIMENT_NAME = "v2.12b_production_shadow"
_ANALYTICAL_THRESHOLD = 0.85
_NORMALIZATION_VERSION = "v2.9"
_METADATA_GUARD_VERSION = "v2.11"
_DIAGNOSTICS_DIR = Path("data/diagnostics/v212b_production_shadow")
_AGGREGATE_JSON = Path("data/diagnostics/v212b_production_shadow.json")

REAL_QUESTIONS: list[dict[str, Any]] = [
    {
        "category": "learning_objectives",
        "question": "What are the learning objectives for money in Primary 4 Mathematics?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "c4u18",
        "question": "What should pupils learn in the Everyday Arithmetic Money unit (C4-U18)?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "learning_objectives",
        "question": "What are the learning outcomes for C4-U18?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "grade_subject",
        "question": "What does Class 4 Mathematics cover?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "topics",
        "question": "What topics are taught in Primary 4 Mathematics?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "learning_objectives",
        "question": "What should pupils learn about fractions in Class 4?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "units",
        "question": "What units are in Primary 4 Mathematics?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "relationships",
        "question": "What learning objectives belong to the fractions topic in Class 4?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "grade_subject",
        "question": "What does Class 5 Science cover?",
        "grade": "CLASS_5",
        "subject": "SCIENCE",
    },
    {
        "category": "learning_objectives",
        "question": "What are the learning objectives for Primary 4 English Language?",
        "grade": "CLASS_4",
        "subject": "ENGLISH",
    },
    {
        "category": "topics",
        "question": "What topics are in Primary 5 Mathematics?",
        "grade": "CLASS_5",
        "subject": "MATHEMATICS",
    },
    {
        "category": "units",
        "question": "List the units taught in Class 4 English.",
        "grade": "CLASS_4",
        "subject": "ENGLISH",
    },
    {
        "category": "relationships",
        "question": "Which subjects are available for Class 4?",
        "grade": "CLASS_4",
    },
    {
        "category": "learning_objectives",
        "question": "What should pupils learn about measurement in Primary 4?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "imperfect_source",
        "question": "What are the learning objectives for geometry in Class 4 Mathematics?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "c4u18",
        "question": "What does Class 4 teach about everyday arithmetic and money?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "topics",
        "question": "What fraction topics are covered in Primary 4?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "relationships",
        "question": "What learning objectives are under the number and operations strand in Class 4 Maths?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "grade_subject",
        "question": "What does Primary 3 Mathematics include?",
        "grade": "CLASS_3",
        "subject": "MATHEMATICS",
    },
    {
        "category": "imperfect_source",
        "question": "What should students know about data handling in Class 4?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "units",
        "question": "What is taught in the Primary 4 fractions unit?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "learning_objectives",
        "question": "What are pupils expected to learn about decimals in Class 4?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "relationships",
        "question": "How are Class 4 Mathematics topics organized?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "category": "imperfect_source",
        "question": "What learning objectives exist for problem solving in Primary 4 Maths?",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
]


def v212b_shadow_enabled(settings: Settings) -> bool:
    return bool(getattr(settings, "v212b_shadow_enabled", False))


def should_sample_shadow(settings: Settings, seed: str) -> bool:
    if not v212b_shadow_enabled(settings):
        return False
    rate = float(getattr(settings, "v212b_shadow_sample_rate", 0.0) or 0.0)
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    digest = hashlib.sha256(seed.encode()).hexdigest()
    bucket = int(digest[:8], 16) % 10000
    return bucket < int(rate * 10000)


def anonymize_request_id(request_id: str | None) -> str:
    if not request_id:
        return ""
    return hashlib.sha256(request_id.encode()).hexdigest()[:16]


def is_c4u18_path(
    question: str,
    evidence: list[CurriculumEvidence],
) -> bool:
    lowered = question.lower()
    if "c4-u18" in lowered or "everyday arithmetic money" in lowered:
        return True
    for item in evidence:
        meta = item.metadata or {}
        code = str(meta.get("code") or "").upper()
        name = (item.name or "").lower()
        if code == "C4-U18" or "everyday arithmetic money" in name:
            return True
    return False


def build_evidence_snapshot(
    *,
    raw_evidence: list[CurriculumEvidence],
    retrieval_metadata: dict[str, Any],
) -> dict[str, Any]:
    normalized = normalize_evidence(
        raw_evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION
    )
    integrity = validate_metadata_integrity(normalized.evidence)
    return {
        "evidence_hash": evidence_snapshot_hash(raw_evidence),
        "evidence_ids": [e.entity_id for e in raw_evidence if e.entity_id],
        "evidence_count": len(raw_evidence),
        "raw_evidence": _serialize_evidence(raw_evidence),
        "retrieval_metadata": retrieval_metadata,
        "normalization_version": _NORMALIZATION_VERSION,
        "metadata_guard_version": _METADATA_GUARD_VERSION,
        "normalized_evidence_hash": normalized.diagnostics.evidence_hash_out,
        "normalization_diagnostics": normalized.diagnostics.to_dict(),
        "metadata_integrity_valid": integrity.valid,
        "metadata_violations": [v.to_dict() for v in integrity.violations],
    }


def collect_retrieval_snapshot(
    agent: CurriculumQAAgent,
    *,
    question: str,
    request_id: str | None = None,
) -> tuple[CurriculumQAState, list[CurriculumEvidence], str, dict[str, Any]]:
    """Execute one shared retrieval + generation pass for shadow evaluation."""
    state = CurriculumQAState.initial(question=question)
    state = agent.understand(state)
    state = agent.retrieve(state, request_id=request_id)
    state = agent.answer(state, request_id=request_id)
    answer = state.final_answer or state.draft_answer or ""
    evidence = copy.deepcopy(state.evidence)
    retrieval_metadata = {
        "retrieval_rounds": state.retrieval_rounds,
        "evidence_status": state.evidence_status.value,
        "grade": state.grade,
        "subject": state.subject,
        "topic": state.topic,
        "tool_calls": state.tool_calls,
        "retrieval_query": state.metadata.get("retrieval_query"),
        "evidence_snapshot_hash": state.metadata.get("evidence_snapshot_hash"),
    }
    return state, evidence, answer, retrieval_metadata


def attribute_failure_origin(
    control: dict[str, Any],
    experiment: dict[str, Any],
    comparison: dict[str, Any],
) -> FailureOrigin:
    classification = comparison.get("classification", "")
    if classification == EquivalenceClassification.RETRIEVAL_VARIANCE.value:
        return FailureOrigin.RETRIEVAL
    if control.get("normalized_evidence_hash") != experiment.get("normalized_evidence_hash"):
        return FailureOrigin.NORMALIZATION
    if control.get("metadata_integrity_valid") != experiment.get("metadata_integrity_valid"):
        return FailureOrigin.METADATA_GUARD
    if control.get("generated_answer") != experiment.get("generated_answer"):
        return FailureOrigin.GENERATION
    if control.get("verifier_recommendation") != experiment.get("verifier_recommendation"):
        return FailureOrigin.VERIFIER
    if control.get("mapped_recommendation") != experiment.get("mapped_recommendation"):
        return FailureOrigin.MAPPER
    if control.get("final_route") != experiment.get("final_route"):
        return FailureOrigin.ROUTING
    impl = experiment.get("implementation", "")
    if impl == Implementation.LANGCHAIN.value and control.get("final_accepted") != experiment.get(
        "final_accepted"
    ):
        return FailureOrigin.LANGCHAIN_ORCHESTRATION
    if impl == Implementation.LANGGRAPH.value:
        return FailureOrigin.LANGGRAPH_ORCHESTRATION
    return FailureOrigin.UNKNOWN


def classify_real_unsafe_divergence(
    control: dict[str, Any],
    experiment: dict[str, Any],
) -> bool:
    """Detect unsafe LangChain acceptance on real questions."""
    if control.get("final_accepted"):
        return False
    if not experiment.get("final_accepted"):
        return False
    if experiment.get("metadata_blocked"):
        return False
    if not experiment.get("metadata_integrity_valid", True):
        return True
    if experiment.get("safety_status") == "unsafe_accept":
        return True
    unsupported = experiment.get("verifier_recommendation") == VerificationRecommendation.FALLBACK.value
    if unsupported and experiment.get("final_accepted"):
        return True
    return False


def run_shadow_evaluation(
    agent: CurriculumQAAgent,
    *,
    question_spec: dict[str, Any],
    run_index: int = 1,
    request_id: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Full shadow evaluation: shared retrieval then dual post-retrieval pipelines."""
    settings = settings or agent.settings
    question = question_spec["question"]
    category = question_spec.get("category", "REAL_SHADOW")
    evaluation_id = f"v212b_{uuid.uuid4().hex[:12]}"
    tag = request_id or evaluation_id

    prod_state, evidence, answer, retrieval_metadata = collect_retrieval_snapshot(
        agent,
        question=question,
        request_id=f"ret_{tag}",
    )
    evidence_source = "c4u18" if is_c4u18_path(question, evidence) else "real"
    snapshot = build_evidence_snapshot(
        raw_evidence=evidence,
        retrieval_metadata=retrieval_metadata,
    )

    pair = run_post_retrieval_pair(
        question=question,
        raw_evidence=evidence,
        generated_answer=answer,
        verifier=agent.verification_node.verifier,
        settings=settings,
        threshold=_ANALYTICAL_THRESHOLD,
        run_index=run_index,
        request_id=tag,
        category=category,
        evaluation_id=evaluation_id,
        retrieval_metadata=retrieval_metadata,
        evidence_source=evidence_source,
    )

    comparison = pair["comparison"]
    failure_origin = attribute_failure_origin(
        pair["langgraph"],
        pair["langchain"],
        comparison,
    )
    unsafe = classify_real_unsafe_divergence(pair["langgraph"], pair["langchain"])
    if unsafe and comparison.get("classification") != EquivalenceClassification.UNSAFE_DIVERGENCE.value:
        comparison = dict(comparison)
        comparison["classification"] = EquivalenceClassification.UNSAFE_DIVERGENCE.value
        comparison["notes"] = "LangChain accepted while control rejected on real retrieval."

    trace = {
        "experiment": _EXPERIMENT_NAME,
        "evaluation_id": evaluation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "category": category,
        "grade": question_spec.get("grade") or prod_state.grade,
        "subject": question_spec.get("subject") or prod_state.subject,
        "topic": question_spec.get("topic") or prod_state.topic,
        "request_trace_id": anonymize_request_id(request_id),
        "retrieval_query": retrieval_metadata.get("retrieval_query"),
        "retrieved_evidence_ids": snapshot["evidence_ids"],
        "retrieved_evidence_count": snapshot["evidence_count"],
        "evidence_snapshot": snapshot,
        "generated_answer": answer,
        "production_implementation": default_implementation(settings).value,
        "production_route": agent.route(prod_state),
        "production_accepted": bool(
            prod_state.verification and prod_state.verification.passed
        ),
        "c4u18_path": evidence_source == "c4u18",
        "langgraph": pair["langgraph"],
        "langchain": pair["langchain"],
        "comparison": comparison,
        "failure_origin": failure_origin.value,
        "unsafe_divergence": unsafe,
        "likely_cause": _likely_cause(comparison, failure_origin),
        "safety_impact": "high" if unsafe else "none",
    }
    return trace


def _likely_cause(comparison: dict[str, Any], origin: FailureOrigin) -> str:
    classification = comparison.get("classification", "")
    if classification == EquivalenceClassification.RETRIEVAL_VARIANCE.value:
        return "Evidence snapshot mismatch between implementations."
    if classification == EquivalenceClassification.EXPECTED_LLM_VARIANCE.value:
        return "Verifier LLM score variance within established tolerance."
    if origin == FailureOrigin.VERIFIER:
        return "Verifier recommendation or score differed."
    if origin == FailureOrigin.MAPPER:
        return "Recommendation mapper applied different policy outcome."
    if origin == FailureOrigin.LANGCHAIN_ORCHESTRATION:
        return "LangChain orchestration path differed from LangGraph control."
    return "No material divergence or undetermined."


def replay_evaluation(
    *,
    evaluation_id: str,
    trace_dir: Path | None = None,
    verifier: Any = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Replay a stored evaluation without performing new retrieval."""
    trace_dir = trace_dir or _DIAGNOSTICS_DIR
    path = trace_dir / f"{evaluation_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No trace found for evaluation_id={evaluation_id}")
    stored = json.loads(path.read_text())
    snapshot = stored["evidence_snapshot"]
    evidence = _deserialize_evidence(snapshot["raw_evidence"])
    question = stored["question"]
    answer = stored.get("generated_answer", "")
    category = stored.get("category", "REAL_SHADOW")
    retrieval_metadata = snapshot.get("retrieval_metadata", {})
    evidence_source = "c4u18" if stored.get("c4u18_path") else "real"

    if verifier is None:
        agent = CurriculumQAAgent(settings=settings or Settings())
        verifier = agent.verification_node.verifier
        settings = agent.settings

    replay = run_post_retrieval_pair(
        question=question,
        raw_evidence=evidence,
        generated_answer=answer,
        verifier=verifier,
        settings=settings,
        category=category,
        evaluation_id=evaluation_id,
        retrieval_metadata=retrieval_metadata,
        evidence_source=evidence_source,
    )
    replay["replay_of"] = evaluation_id
    replay["evidence_hash_match"] = (
        replay.get("evidence_hash") == snapshot.get("evidence_hash")
    )
    return replay


def persist_trace(trace: dict[str, Any], *, trace_dir: Path | None = None) -> Path:
    trace_dir = trace_dir or _DIAGNOSTICS_DIR
    trace_dir.mkdir(parents=True, exist_ok=True)
    evaluation_id = trace["evaluation_id"]
    path = trace_dir / f"{evaluation_id}.json"
    path.write_text(json.dumps(trace, indent=2, default=str))
    return path


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    langgraph_summary = summarize_implementation_rows(rows, key="langgraph")
    langchain_summary = summarize_implementation_rows(rows, key="langchain")
    comparison_summary = summarize_comparisons(rows)

    retrieval_stats = _aggregate_retrieval(rows)
    normalization_stats = _aggregate_normalization(rows)
    metadata_stats = _aggregate_metadata(rows)
    verifier_stats = _aggregate_verifier(rows, key="langgraph")
    verifier_stats_lc = _aggregate_verifier(rows, key="langchain")
    mapper_stats = _aggregate_mapper(rows, key="langgraph")
    routing_stats = _aggregate_routing(rows, key="langgraph")

    unsafe_count = sum(1 for r in rows if r.get("unsafe_divergence"))
    c4u18_rows = [r for r in rows if r.get("c4u18_path")]
    fi_rows = [
        r
        for r in rows
        if r.get("category") in {"imperfect_source", "learning_objectives"}
        and (r.get("langgraph", {}).get("verifier_recommendation") == "retrieve_more")
    ]

    latencies_lg = [
        r.get("langgraph", {}).get("timings", {}).get("total_ms", 0.0) for r in rows
    ]
    latencies_lc = [
        r.get("langchain", {}).get("timings", {}).get("total_ms", 0.0) for r in rows
    ]

    return {
        "experiment": _EXPERIMENT_NAME,
        "n_evaluations": len(rows),
        "langgraph_summary": langgraph_summary,
        "langchain_summary": langchain_summary,
        "comparison_summary": comparison_summary,
        "retrieval_statistics": retrieval_stats,
        "normalization_statistics": normalization_stats,
        "metadata_statistics": metadata_stats,
        "verifier_statistics": {
            "langgraph": verifier_stats,
            "langchain": verifier_stats_lc,
        },
        "mapper_statistics": mapper_stats,
        "routing_statistics": routing_stats,
        "safety": {
            "unsafe_divergence_count": unsafe_count,
            "metadata_false_acceptance": sum(
                1
                for r in rows
                if r.get("langchain", {}).get("final_accepted")
                and not r.get("langgraph", {}).get("metadata_integrity_valid", True)
            ),
            "placeholder_false_acceptance": sum(
                1
                for r in rows
                if r.get("langchain", {}).get("final_accepted")
                and r.get("langchain", {}).get("mapped_recommendation") == "reject"
                and "placeholder" in str(
                    r.get("langchain", {}).get("normalization_diagnostics", {})
                ).lower()
            ),
        },
        "c4u18": {
            "n": len(c4u18_rows),
            "langgraph_accept_rate": _accept_rate(c4u18_rows, "langgraph"),
            "langchain_accept_rate": _accept_rate(c4u18_rows, "langchain"),
        },
        "fi_monitoring": {
            "retrieve_more_rows": len(fi_rows),
            "langgraph_accept_after_retrieve_more": sum(
                1 for r in fi_rows if r.get("langgraph", {}).get("final_accepted")
            ),
            "langchain_accept_after_retrieve_more": sum(
                1 for r in fi_rows if r.get("langchain", {}).get("final_accepted")
            ),
        },
        "latency": {
            "langgraph_mean_ms": round(sum(latencies_lg) / n, 3),
            "langchain_mean_ms": round(sum(latencies_lc) / n, 3),
            "overhead_mean_ms": round(
                (sum(latencies_lc) - sum(latencies_lg)) / n, 3
            ),
        },
        "errors": sum(len(r.get("langgraph", {}).get("errors") or []) for r in rows),
        "shadow_errors": sum(1 for r in rows if r.get("shadow_error")),
    }


def _accept_rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(
        sum(1 for r in rows if r.get(key, {}).get("final_accepted")) / len(rows),
        3,
    )


def _aggregate_retrieval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [r.get("retrieved_evidence_count", 0) for r in rows]
    n = len(rows) or 1
    return {
        "mean_evidence_count": round(sum(counts) / n, 3),
        "no_evidence": sum(1 for c in counts if c == 0),
        "weak_evidence": sum(1 for c in counts if 0 < c < 3),
        "evidence_found": sum(1 for c in counts if c > 0),
    }


def _aggregate_normalization(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    successes = sum(
        1
        for r in rows
        if r.get("evidence_snapshot", {})
        .get("normalization_diagnostics", {})
        .get("records_normalized", 0)
        >= 0
    )
    uuid_resolution = sum(
        1
        for r in rows
        if r.get("evidence_snapshot", {})
        .get("normalization_diagnostics", {})
        .get("uuid_resolutions", 0)
    )
    return {
        "normalization_success_rate": round(successes / n, 3),
        "uuid_resolution_total": uuid_resolution,
    }


def _aggregate_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    valid = sum(
        1
        for r in rows
        if r.get("evidence_snapshot", {}).get("metadata_integrity_valid")
    )
    blocked = sum(1 for r in rows if r.get("langgraph", {}).get("metadata_blocked"))
    return {
        "valid_evidence_pct": round(valid / n, 3),
        "metadata_blocked_count": blocked,
    }


def _aggregate_verifier(rows: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    impl_rows = [r.get(key, {}) for r in rows]
    n = len(impl_rows) or 1
    scores = [r.get("verifier_score", 0.0) for r in impl_rows]
    recs = [r.get("verifier_recommendation", "") for r in impl_rows]
    return {
        "accept_pct": round(sum(1 for r in recs if r == "accept") / n, 3),
        "retrieve_more_pct": round(sum(1 for r in recs if r == "retrieve_more") / n, 3),
        "fallback_pct": round(sum(1 for r in recs if r == "fallback") / n, 3),
        "mean_score": round(sum(scores) / n, 3),
        "mapped_accept_pct": round(
            sum(1 for r in impl_rows if r.get("final_accepted")) / n, 3
        ),
    }


def _aggregate_mapper(rows: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    impl_rows = [r.get(key, {}) for r in rows]
    n = len(impl_rows) or 1
    return {
        "mapped_acceptance_pct": round(
            sum(1 for r in impl_rows if r.get("final_accepted")) / n, 3
        ),
        "safety_blocks": sum(
            1 for r in impl_rows if r.get("mapped_recommendation") == "reject"
        ),
    }


def _aggregate_routing(rows: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    impl_rows = [r.get(key, {}) for r in rows]
    n = len(impl_rows) or 1
    routes = [r.get("final_route", "") for r in impl_rows]
    return {
        "finish_pct": round(sum(1 for r in routes if r == "finish") / n, 3),
        "retrieve_more_pct": round(
            sum(1 for r in routes if r == "retrieve_more") / n, 3
        ),
        "fallback_pct": round(sum(1 for r in routes if r == "fallback") / n, 3),
    }


def interpret_v212b(metrics: dict[str, Any]) -> tuple[str, str, str]:
    """Return conclusion, note, and V2.13 recommendation."""
    unsafe = metrics.get("safety", {}).get("unsafe_divergence_count", 999)
    meta_false = metrics.get("safety", {}).get("metadata_false_acceptance", 999)
    placeholder_false = metrics.get("safety", {}).get("placeholder_false_acceptance", 999)
    comparison = metrics.get("comparison_summary", {})
    unsafe_div = comparison.get("unsafe_divergence_count", unsafe)

    if (
        unsafe_div == 0
        and meta_false == 0
        and placeholder_false == 0
        and metrics.get("shadow_errors", 0) == 0
    ):
        conclusion = "SUPPORTED"
        note = (
            "Real-retrieval shadow evaluation preserved safety invariants with zero "
            "unsafe LangChain divergences and reproducible evidence snapshots."
        )
        v213 = "V2.13 — Controlled LangChain Production Canary (LangGraph rollback retained)"
    elif unsafe_div == 0 and meta_false == 0:
        conclusion = "PARTIALLY_SUPPORTED"
        note = (
            "Architecture is safe under real retrieval but requires targeted improvements "
            "before migration."
        )
        v213 = "Targeted hardening experiments for identified FI or operational gaps"
    else:
        conclusion = "NOT_SUPPORTED"
        note = (
            "LangChain introduced unsafe behavior or metadata bypass under real retrieval."
        )
        v213 = "Retain LangGraph; document blocking architectural issues"

    return conclusion, note, v213


def run_production_shadow(
    agent: CurriculumQAAgent,
    state: CurriculumQAState,
    *,
    request_id: str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any] | None:
    """Observational shadow from a completed production request (does not affect response)."""
    settings = agent.settings
    if not should_sample_shadow(settings, request_id or state.question):
        return None

    timeout = timeout_seconds or float(
        getattr(settings, "v212b_shadow_timeout_seconds", 120.0)
    )
    question = state.question
    evidence = copy.deepcopy(state.evidence)
    answer = state.final_answer or state.draft_answer or ""
    retrieval_metadata = {
        "retrieval_rounds": state.retrieval_rounds,
        "evidence_status": state.evidence_status.value,
        "grade": state.grade,
        "subject": state.subject,
        "topic": state.topic,
        "retrieval_query": state.metadata.get("retrieval_query"),
        "production_shadow": True,
    }
    evaluation_id = f"prod_{uuid.uuid4().hex[:12]}"
    evidence_source = "c4u18" if is_c4u18_path(question, evidence) else "real"

    pair = run_post_retrieval_pair(
        question=question,
        raw_evidence=evidence,
        generated_answer=answer,
        verifier=agent.verification_node.verifier,
        settings=settings,
        evaluation_id=evaluation_id,
        retrieval_metadata=retrieval_metadata,
        evidence_source=evidence_source,
        category="PRODUCTION_SHADOW",
    )
    snapshot = build_evidence_snapshot(
        raw_evidence=evidence,
        retrieval_metadata=retrieval_metadata,
    )
    trace = {
        "experiment": _EXPERIMENT_NAME,
        "evaluation_id": evaluation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "request_trace_id": anonymize_request_id(request_id),
        "evidence_snapshot": snapshot,
        "generated_answer": answer,
        "production_shadow": True,
        "timeout_seconds": timeout,
        **pair,
    }
    persist_trace(trace)
    return trace


def maybe_schedule_production_shadow(
    agent: CurriculumQAAgent,
    state: CurriculumQAState,
    *,
    request_id: str | None = None,
) -> None:
    """Fire-and-forget production shadow; failures are isolated from the user response."""
    if not should_sample_shadow(agent.settings, request_id or state.question):
        return

    def _worker() -> None:
        try:
            run_production_shadow(agent, state, request_id=request_id)
        except Exception:
            logger.exception("v212b production shadow failed", extra={"request_id": request_id})

    thread = threading.Thread(target=_worker, daemon=True, name="v212b-shadow")
    thread.start()


__all__ = [
    "REAL_QUESTIONS",
    "aggregate_metrics",
    "anonymize_request_id",
    "attribute_failure_origin",
    "build_evidence_snapshot",
    "collect_retrieval_snapshot",
    "interpret_v212b",
    "is_c4u18_path",
    "maybe_schedule_production_shadow",
    "persist_trace",
    "replay_evaluation",
    "run_production_shadow",
    "run_shadow_evaluation",
    "should_sample_shadow",
    "v212b_shadow_enabled",
]
