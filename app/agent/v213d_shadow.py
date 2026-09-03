"""V2.13D production-shadow evaluation of context-hybrid document evidence.

Observational only: never replaces the production LangGraph response.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.agent.evidence_snapshot import evidence_snapshot_hash
from app.agent.state import CurriculumQAState
from app.agent.v211_metadata_integrity import (
    PipelineVariant,
    apply_metadata_policy,
    validate_metadata_integrity,
)
from app.agent.v212_langchain import infer_mapper_fixture_class
from app.agent.v213_experiment import (
    BENCHMARK_SOURCES,
    DocumentEvidencePipeline,
    benchmark_fixture_path,
    document_passage_to_evidence,
)
from app.agent.v213b_embeddings import FeatureHashEmbeddingProvider
from app.agent.v213b_retrieval_contract import RetrievalVariant
from app.agent.v213b_semantic_retrieval import HybridDocumentRetrievalService
from app.agent.v213b_vector_index import PassageVectorIndex
from app.agent.v213c_dataset import build_v213c_dataset
from app.agent.v213c_experiment import frozen_structured_catalog
from app.agent.v25_experiment import _CLEAN_PLACEHOLDER
from app.agent.v26_experiment import answer_hash
from app.agent.v28_recommendation_mapping import map_recommendation
from app.agent.v29_evidence_normalization import NormalizationVariant, normalize_evidence
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus, merge_evidence_bundles
from app.logging_utils import log_agent_event
from app.schemas.verification import VerificationRecommendation

logger = logging.getLogger(__name__)

_EXPERIMENT_NAME = "v2.13d"
_SCHEMA_VERSION = "v213d.1"
_ANALYTICAL_THRESHOLD = 0.85
_JSONL = Path("data/diagnostics/v213d_shadow.jsonl")
_WRITE_LOCK = threading.Lock()

REPLAY_QUESTION_IDS = (
    "V213C-A01",  # document-only
    "V213C-D01",  # structured
    "V213C-B01",  # mixed
    "V213C-F01",  # insufficient
    "V213C-G03",  # adversarial placeholder
)


def v213d_shadow_enabled(settings: Settings) -> bool:
    return bool(getattr(settings, "v213d_shadow_enabled", False))


def should_sample_v213d(settings: Settings, seed: str) -> bool:
    if not v213d_shadow_enabled(settings):
        return False
    rate = float(getattr(settings, "v213d_shadow_sample_rate", 0.0) or 0.0)
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    digest = hashlib.sha256(f"v213d:{seed}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 10000
    return bucket < int(rate * 10000)


def question_hash(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()[:16]


def infer_question_category(state: CurriculumQAState, document_count: int) -> str:
    q = (state.question or "").lower()
    if any(token in q for token in ("ignore previous", "system prompt", "blockchain")):
        return "adversarial_suspicious"
    if document_count and state.evidence:
        return "mixed"
    if any(
        token in q
        for token in ("purpose", "principle", "teaching", "assessment", "curriculum say")
    ):
        return "document_oriented"
    if state.grade and state.subject and (
        "learning objective" in q or "topics" in q or "unit" in q
    ):
        return "structured"
    if not state.grade and not state.subject:
        return "ambiguous"
    if not state.evidence:
        return "insufficient_evidence"
    return "structured"


def classify_shadow_comparison(control: dict[str, Any], shadow: dict[str, Any]) -> str:
    if shadow.get("error"):
        return "SHADOW_ERROR"
    if shadow.get("wrong_context") and shadow.get("final_accepted"):
        return "DOCUMENT_CREATED_WRONG_CONTEXT"
    c_route = control.get("final_route")
    s_route = shadow.get("final_route")
    c_accept = bool(control.get("final_accepted"))
    s_accept = bool(shadow.get("final_accepted")) and bool(shadow.get("metadata_valid"))
    docs = int(shadow.get("document_evidence_count") or 0)

    if (not c_accept) and s_accept and docs > 0:
        return "DOCUMENT_ADDED_GROUNDING"
    if c_accept and s_accept:
        return "BOTH_ACCEPTED"
    if (not c_accept) and (not s_accept):
        if c_route in {"retrieve_more", "fallback"} and s_route in {"retrieve_more", "fallback"}:
            return "BOTH_INSUFFICIENT"
        return "BOTH_REJECTED"
    if c_accept and not s_accept:
        if s_route == "retrieve_more" and c_route == "finish":
            return "SHADOW_REGRESSED"
        if docs:
            return "DOCUMENT_CREATED_NOISE"
        return "SHADOW_REGRESSED"
    if docs == 0:
        return "RETRIEVAL_FAILURE"
    if control.get("verifier_decision") != shadow.get("verifier_decision"):
        return "VERIFIER_DIFFERENCE"
    if control.get("answer_hash") != shadow.get("answer_hash"):
        return "GENERATION_DIFFERENCE"
    return "DOCUMENT_DID_NOT_HELP"


def _control_snapshot(state: CurriculumQAState, settings: Settings) -> dict[str, Any]:
    verification = state.verification
    mapping = None
    if verification is not None:
        mapping = map_recommendation(
            verification,
            fixture_class=infer_mapper_fixture_class(state.evidence, verification),  # type: ignore[arg-type]
            evidence=state.evidence,
            answer=state.final_answer or state.draft_answer or "",
            threshold=_ANALYTICAL_THRESHOLD,
        )
    from app.agent.graph_routing import route_after_verification

    route = "fallback"
    try:
        route = route_after_verification(
            {"qa": copy.deepcopy(state), "visited_nodes": []},
            settings=settings,
        )
    except Exception:
        if verification and (
            verification.passed or verification.recommendation == VerificationRecommendation.ACCEPT
        ):
            route = "finish"
    answer = state.final_answer or state.draft_answer or ""
    return {
        "evidence_count": len(state.evidence),
        "evidence_snapshot": evidence_snapshot_hash(state.evidence),
        "verifier_score": verification.score if verification else None,
        "verifier_decision": verification.recommendation.value if verification else None,
        "verifier_accepted": verification.passed if verification else False,
        "unsupported_claims": list(verification.unsupported_claims or []) if verification else [],
        "mapper_recommendation": mapping.mapped_recommendation.value if mapping else None,
        "mapped_accepted": mapping.mapped_accepted if mapping else False,
        "final_accepted": bool(mapping.mapped_accepted) if mapping else bool(verification and verification.passed),
        "final_route": route,
        "answer_hash": answer_hash(answer) if answer else "",
        "model": (state.metadata or {}).get("model"),
    }


def retrieve_document_evidence(
    *,
    question: str,
    grade: str | None,
    subject: str | None,
    topic: str | None,
    unit: str | None,
    settings: Settings,
    retrieval: HybridDocumentRetrievalService | None = None,
    timeout_seconds: float | None = None,
) -> tuple[list[CurriculumEvidence], dict[str, Any]]:
    if not bool(getattr(settings, "v213d_shadow_document_retrieval", True)):
        return [], {"skipped": True}
    service = retrieval or default_retrieval_service(settings)
    variant = getattr(settings, "v213d_shadow_retrieval_variant", "context_hybrid")
    started = time.perf_counter()
    result = service.search(
        query=question,
        variant=variant,
        grade=grade,
        subject=subject,
        topic=topic,
        unit=unit,
        limit=5,
    )
    latency = (time.perf_counter() - started) * 1000
    if timeout_seconds is not None and latency > timeout_seconds * 1000:
        raise TimeoutError("document retrieval exceeded shadow timeout")
    evidence = [document_passage_to_evidence(hit.passage) for hit in result.hits]
    return evidence, {
        "variant": variant,
        "latency_ms": latency,
        "count": len(evidence),
        "diagnostics": result.diagnostics.to_dict(),
    }


def default_retrieval_service(settings: Settings) -> HybridDocumentRetrievalService:
    from app.agent.v213_document_store import DocumentStore

    store = DocumentStore(root=Path("data/documents"))
    provider = FeatureHashEmbeddingProvider(
        dimension=int(getattr(settings, "v213b_embedding_dimension", 128)),
        model_name=getattr(settings, "v213b_embedding_model", "feature-hash-v1"),
    )
    index = PassageVectorIndex(
        root=Path("data/document_index"),
        store=store,
        provider=provider,
    )
    return HybridDocumentRetrievalService(
        store=store,
        provider=provider,
        index=index,
        settings=settings,
    )


def _provenance_complete(evidence: list[CurriculumEvidence]) -> bool:
    docs = [e for e in evidence if e.entity_type == "document_passage"]
    if not docs:
        return True
    return all(
        bool(e.metadata.get("document_id"))
        and bool(e.metadata.get("source_id") or (e.metadata.get("provenance") or {}).get("source_url"))
        and bool(e.metadata.get("page_number") or (e.metadata.get("provenance") or {}).get("page_number"))
        for e in docs
    )


def run_shadow_pipeline(
    agent: Any,
    production_state: CurriculumQAState,
    *,
    request_id: str | None = None,
    retrieval: HybridDocumentRetrievalService | None = None,
    retrieve_documents: Callable[..., tuple[list[CurriculumEvidence], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run document-enhanced generate→verify→map using frozen production structured evidence."""
    settings = agent.settings
    structured = copy.deepcopy(production_state.evidence)
    control = _control_snapshot(production_state, settings)
    started = time.perf_counter()
    stage = "start"
    try:
        stage = "document_retrieval"
        retriever = retrieve_documents or retrieve_document_evidence
        documents, retrieval_meta = retriever(
            question=production_state.question,
            grade=production_state.grade,
            subject=production_state.subject,
            topic=production_state.topic,
            unit=None,
            settings=settings,
            retrieval=retrieval,
            timeout_seconds=float(getattr(settings, "v213d_shadow_timeout_seconds", 30.0)),
        )
        stage = "merge"
        merged = merge_evidence_bundles(structured, documents)
        stage = "normalization"
        normalized = normalize_evidence(merged, NormalizationVariant.STRUCTURAL_NORMALIZATION)
        stage = "metadata_guard"
        integrity = validate_metadata_integrity(normalized.evidence)
        verify_evidence, metadata_blocked, metadata_policy = apply_metadata_policy(
            normalized.evidence,
            integrity,
            variant=PipelineVariant.B_METADATA_VALIDATE,
        )
        shadow_state = copy.deepcopy(production_state)
        shadow_state.evidence = verify_evidence
        shadow_state.evidence_status = (
            EvidenceStatus.FOUND if verify_evidence else EvidenceStatus.NOT_FOUND
        )
        shadow_state.final_answer = None
        shadow_state.draft_answer = None
        shadow_state.verification = None
        stage = "generation"
        shadow_state = agent.answer(shadow_state, request_id=f"{request_id or 'v213d'}-shadow")
        stage = "verification"
        shadow_state = agent.verify(shadow_state, request_id=f"{request_id or 'v213d'}-shadow")
        stage = "mapper"
        verifier_result = shadow_state.verification
        mapping = map_recommendation(
            verifier_result,
            fixture_class=infer_mapper_fixture_class(verify_evidence, verifier_result),  # type: ignore[arg-type]
            evidence=verify_evidence,
            answer=shadow_state.final_answer or shadow_state.draft_answer or "",
            threshold=_ANALYTICAL_THRESHOLD,
        )
        stage = "routing"
        final_route = agent.route(shadow_state)
        if metadata_blocked:
            final_accepted = False
            final_route = "fallback" if not verify_evidence else final_route
        else:
            final_accepted = mapping.mapped_accepted
        answer = shadow_state.final_answer or shadow_state.draft_answer or ""
        placeholder = any(
            _CLEAN_PLACEHOLDER.lower() in (e.content or "").lower() for e in verify_evidence
        )
        wrong_context = False
        if production_state.grade:
            wrong_context = any(
                e.grade and e.grade != production_state.grade for e in documents
            )
        if production_state.subject:
            wrong_context = wrong_context or any(
                e.subject and e.subject != production_state.subject for e in documents
            )
        shadow = {
            "structured_evidence_count": len(structured),
            "document_evidence_count": len(documents),
            "evidence_count": len(verify_evidence),
            "retrieval_variant": retrieval_meta.get("variant")
            or getattr(settings, "v213d_shadow_retrieval_variant", "context_hybrid"),
            "document_retrieval_latency_ms": retrieval_meta.get("latency_ms", 0),
            "normalization_status": "ok",
            "normalization_count": len(normalized.evidence),
            "metadata_valid": integrity.valid,
            "metadata_blocked": metadata_blocked,
            "metadata_policy": metadata_policy,
            "violations": [v.to_dict() for v in integrity.violations],
            "verifier_score": verifier_result.score if verifier_result else None,
            "verifier_decision": verifier_result.recommendation.value if verifier_result else None,
            "verifier_accepted": verifier_result.passed if verifier_result else False,
            "unsupported_claims": list(verifier_result.unsupported_claims or [])
            if verifier_result
            else [],
            "mapper_recommendation": mapping.mapped_recommendation.value,
            "mapped_accepted": mapping.mapped_accepted,
            "final_accepted": final_accepted and not metadata_blocked,
            "final_route": final_route,
            "answer_hash": answer_hash(answer) if answer else "",
            "provenance_complete": _provenance_complete(documents),
            "wrong_context": wrong_context,
            "placeholder_evidence": placeholder,
            "error": None,
        }
        classification = classify_shadow_comparison(control, shadow)
        record = {
            "experiment": _EXPERIMENT_NAME,
            "schema_version": _SCHEMA_VERSION,
            "request_id": hashlib.sha256((request_id or "").encode()).hexdigest()[:16]
            if request_id
            else "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sampling": {
                "enabled": v213d_shadow_enabled(settings),
                "rate": float(getattr(settings, "v213d_shadow_sample_rate", 0.0) or 0.0),
                "sampled": True,
            },
            "question": {
                "hash": question_hash(production_state.question),
                "category": infer_question_category(production_state, len(documents)),
                "grade": production_state.grade,
                "subject": production_state.subject,
                "topic": production_state.topic,
            },
            "control": control,
            "shadow": shadow,
            "grounding": {
                "metadata_valid": integrity.valid,
                "provenance_complete": shadow["provenance_complete"],
                "wrong_context": wrong_context,
                "placeholder_evidence": placeholder,
                "unsupported_claims": shadow["unsupported_claims"],
            },
            "comparison": {
                "classification": classification,
                "improved": classification in {"DOCUMENT_ADDED_GROUNDING", "SHADOW_IMPROVED"},
                "regressed": classification == "SHADOW_REGRESSED",
            },
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        log_agent_event(
            logger,
            "v213d.shadow.completed",
            request_id=request_id,
            extra_classification=classification,
        )
        if shadow["metadata_blocked"]:
            log_agent_event(logger, "v213d.shadow.metadata_blocked", request_id=request_id)
        if record["comparison"]["improved"]:
            log_agent_event(logger, "v213d.shadow.improved", request_id=request_id)
        if record["comparison"]["regressed"]:
            log_agent_event(logger, "v213d.shadow.regressed", request_id=request_id)
        return record
    except Exception as exc:
        log_agent_event(
            logger,
            "v213d.shadow.failed",
            request_id=request_id,
            error=type(exc).__name__,
        )
        return {
            "experiment": _EXPERIMENT_NAME,
            "schema_version": _SCHEMA_VERSION,
            "request_id": hashlib.sha256((request_id or "").encode()).hexdigest()[:16]
            if request_id
            else "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": {"hash": question_hash(production_state.question)},
            "control": control,
            "shadow": {
                "error": type(exc).__name__,
                "shadow_error_type": type(exc).__name__,
                "shadow_error_message_safe": str(exc)[:200],
                "shadow_stage": stage,
                "final_accepted": False,
            },
            "grounding": {"metadata_valid": False},
            "comparison": {
                "classification": "SHADOW_ERROR",
                "improved": False,
                "regressed": False,
            },
        }


def persist_record(record: dict[str, Any], path: Path | None = None) -> None:
    target = path or _JSONL
    target.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def run_production_shadow(
    agent: Any,
    state: CurriculumQAState,
    *,
    request_id: str | None = None,
    retrieval: HybridDocumentRetrievalService | None = None,
    jsonl_path: Path | None = None,
) -> dict[str, Any]:
    production_copy = copy.deepcopy(state)
    log_agent_event(logger, "v213d.shadow.started", request_id=request_id)
    record = run_shadow_pipeline(
        agent, production_copy, request_id=request_id, retrieval=retrieval
    )
    persist_record(record, jsonl_path)
    return record


def maybe_schedule_v213d_shadow(
    agent: Any,
    state: CurriculumQAState,
    *,
    request_id: str | None = None,
) -> None:
    """Fire-and-forget; exceptions never reach the production caller."""
    settings = agent.settings
    if not should_sample_v213d(settings, request_id or state.question):
        return
    snapshot = copy.deepcopy(state)
    timeout = float(getattr(settings, "v213d_shadow_timeout_seconds", 30.0))

    def _worker() -> None:
        try:
            started = time.perf_counter()
            run_production_shadow(agent, snapshot, request_id=request_id)
            elapsed = time.perf_counter() - started
            if elapsed > timeout:
                logger.warning("v213d shadow exceeded timeout after completion")
        except Exception:
            logger.exception("v213d production shadow failed")

    thread = threading.Thread(target=_worker, daemon=True, name="v213d-shadow")
    thread.start()
    return thread


def prepare_replay_corpus(store_root: Path, index_root: Path) -> HybridDocumentRetrievalService:
    from app.agent.v213_document_store import DocumentStore

    store = DocumentStore(root=store_root)
    pipeline = DocumentEvidencePipeline(store=store)
    for spec in BENCHMARK_SOURCES:
        pipeline.ingest_source(
            {
                "id": spec["id"],
                "name": spec["name"],
                "document_url": spec["document_url"],
                "version": spec["version"],
                "verification_status": spec["verification_status"],
                "authority": spec.get("authority"),
            },
            allow_local_path=str(benchmark_fixture_path(spec)),
            structure_hints=spec.get("structure_hints"),
        )
    provider = FeatureHashEmbeddingProvider()
    index = PassageVectorIndex(root=index_root, store=store, provider=provider)
    index.build_index(provider, force=True)
    return HybridDocumentRetrievalService(store=store, provider=provider, index=index)


def replay_fixtures(
    agent: Any,
    *,
    retrieval: HybridDocumentRetrievalService,
    question_ids: tuple[str, ...] = REPLAY_QUESTION_IDS,
    inject_failures: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    catalog = frozen_structured_catalog()
    dataset = {q["id"]: q for q in build_v213c_dataset()}
    records: list[dict[str, Any]] = []
    failures = inject_failures or {}
    for qid in question_ids:
        spec = dataset[qid]
        state = CurriculumQAState.initial(question=spec["question"])
        state.grade = spec.get("grade")
        state.subject = spec.get("subject")
        state.topic = spec.get("topic")
        key = spec.get("structured_key")
        state.evidence = copy.deepcopy(catalog.get(key or "", []))
        kind = spec.get("adversarial_kind")
        if spec.get("category") == "adversarial" and kind:
            from app.agent.v213c_experiment import _poison_structured

            state.evidence = _poison_structured(kind, state.evidence)
        state.evidence_status = EvidenceStatus.FOUND if state.evidence else EvidenceStatus.NOT_FOUND
        control_state = copy.deepcopy(state)
        if control_state.evidence or spec.get("category") != "insufficient_evidence":
            try:
                control_state = agent.answer(control_state, request_id=f"replay-{qid}-control")
                control_state = agent.verify(control_state, request_id=f"replay-{qid}-control")
            except Exception:
                pass
        retrieve_fn = retrieve_document_evidence
        if failures.get(qid) == "retrieval":
            def retrieve_fn(**_kwargs):  # type: ignore[misc]
                raise RuntimeError("forced retrieval failure")
        elif failures.get(qid) == "timeout":
            def retrieve_fn(**_kwargs):  # type: ignore[misc]
                raise TimeoutError("forced shadow timeout")
        elif failures.get(qid) == "normalization":
            def retrieve_fn(**kwargs):  # type: ignore[misc]
                raise ValueError("forced normalization failure")
        record = run_shadow_pipeline(
            agent,
            control_state,
            request_id=f"replay-{qid}",
            retrieval=retrieval,
            retrieve_documents=retrieve_fn,
        )
        record["replay_id"] = qid
        record["replay_category"] = spec["category"]
        records.append(record)
    return records


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records) or 1
    successful = [r for r in records if not (r.get("shadow") or {}).get("error")]
    errors = [r for r in records if (r.get("shadow") or {}).get("error")]
    improved = [r for r in records if r.get("comparison", {}).get("improved")]
    regressed = [r for r in records if r.get("comparison", {}).get("regressed")]
    newly = [
        r
        for r in records
        if r.get("comparison", {}).get("classification") == "DOCUMENT_ADDED_GROUNDING"
    ]
    safety = {
        "wrong_context_false_acceptance": sum(
            1
            for r in successful
            if r.get("grounding", {}).get("wrong_context")
            and (r.get("shadow") or {}).get("final_accepted")
        ),
        "placeholder_false_acceptance": sum(
            1
            for r in successful
            if r.get("grounding", {}).get("placeholder_evidence")
            and (r.get("shadow") or {}).get("final_accepted")
        ),
        "metadata_integrity_false_acceptance": sum(
            1
            for r in successful
            if (r.get("shadow") or {}).get("metadata_blocked")
            and (r.get("shadow") or {}).get("final_accepted")
        ),
        "unsafe_adversarial_false_acceptance": sum(
            1
            for r in successful
            if r.get("replay_category") == "adversarial"
            and (r.get("shadow") or {}).get("final_accepted")
        ),
        "shadow_errors_must_not_affect_production": True,
    }
    latencies = [float(r.get("latency_ms") or 0) for r in successful]
    retrieval_lat = [
        float((r.get("shadow") or {}).get("document_retrieval_latency_ms") or 0)
        for r in successful
    ]
    provenance = [
        r.get("grounding", {}).get("provenance_complete") for r in successful
    ]
    safety_fail = any(v > 0 for k, v in safety.items() if k != "shadow_errors_must_not_affect_production")
    error_rate = len(errors) / n
    if safety_fail:
        canary = "BLOCKED"
        note = "Safety gate failed; do not enable canary or production document retrieval."
    elif not successful:
        canary = "BLOCKED"
        note = "No successful shadow evaluations."
    elif newly and not regressed and not safety_fail and error_rate < 0.25:
        canary = "CANARY_NOT_READY"
        note = (
            "Local replay is operationally stable and shows document grounding gains, "
            "but real production traffic has not been collected. Enable shadow at a low "
            "sample rate; do not promote document retrieval."
        )
    else:
        canary = "CANARY_NOT_READY"
        note = "Shadow path works, but evidence is insufficient for canary of user-facing answers."
    return {
        "experiment": _EXPERIMENT_NAME,
        "schema_version": _SCHEMA_VERSION,
        "traffic_sampled": len(records),
        "successful_shadow_evaluations": len(successful),
        "shadow_errors": len(errors),
        "shadow_error_rate": error_rate,
        "retrieval_success": sum(
            1 for r in successful if int((r.get("shadow") or {}).get("document_evidence_count") or 0) > 0
        )
        / max(len(successful), 1),
        "provenance_complete_rate": sum(1 for p in provenance if p) / max(len(provenance), 1),
        "metadata_valid_rate": sum(
            1 for r in successful if r.get("grounding", {}).get("metadata_valid")
        )
        / max(len(successful), 1),
        "newly_recoverable_count": len(newly),
        "newly_recoverable_rate": len(newly) / n,
        "improvements": len(improved),
        "regressions": len(regressed),
        "unchanged": n - len(improved) - len(regressed) - len(errors),
        "classifications": _count_classifications(records),
        "safety_metrics": safety,
        "latency_metrics": {
            "shadow_mean_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "retrieval_mean_ms": sum(retrieval_lat) / len(retrieval_lat) if retrieval_lat else 0.0,
        },
        "canary_recommendation": canary,
        "canary_note": note,
        "v213c_comparison_note": (
            "V2.13C was a controlled harness (59.7%→90.3% grounded-correct). "
            "V2.13D replay is not statistically equivalent to that dataset."
        ),
    }


def _count_classifications(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        label = record.get("comparison", {}).get("classification") or "UNKNOWN"
        counts[label] = counts.get(label, 0) + 1
    return counts


def interpret_v213d(metrics: dict[str, Any]) -> str:
    return str(metrics.get("canary_recommendation") or "CANARY_NOT_READY")


__all__ = [
    "REPLAY_QUESTION_IDS",
    "aggregate_records",
    "classify_shadow_comparison",
    "interpret_v213d",
    "maybe_schedule_v213d_shadow",
    "persist_record",
    "prepare_replay_corpus",
    "replay_fixtures",
    "run_production_shadow",
    "run_shadow_pipeline",
    "should_sample_v213d",
    "v213d_shadow_enabled",
]
