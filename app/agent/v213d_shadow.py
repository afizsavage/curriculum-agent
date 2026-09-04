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
_SCHEMA_VERSION = "v213d.2"
_ANALYTICAL_THRESHOLD = 0.85
_JSONL = Path("data/diagnostics/v213d_shadow.jsonl")
_TRAFFIC = Path("data/diagnostics/v213d_traffic.json")
_FUNNEL = Path("data/diagnostics/v213d_pipeline_funnel.json")
_WRITE_LOCK = threading.Lock()
OBSERVATION_TARGET_MIN = 100
OBSERVATION_TARGET_MAX = 200
_FUNNEL_STAGES = (
    "request_seen",
    "shadow_eligible",
    "shadow_sampled",
    "shadow_not_sampled",
    "shadow_started",
    "shadow_completed",
    "shadow_failed",
    "shadow_persisted",
    "persist_error",
)

PHASE1_CATEGORIES = (
    "DOCUMENT_ADDED_MISSING_CONTEXT",
    "DOCUMENT_ADDED_EXPLANATION",
    "DOCUMENT_DISAMBIGUATED_CONTEXT",
    "DOCUMENT_PROVIDED_SOURCE",
    "STRUCTURED_DATA_ALREADY_SUFFICIENT",
    "DOCUMENT_DID_NOT_HELP",
    "DOCUMENT_CORPUS_UNAVAILABLE",
    "DOCUMENT_RETRIEVAL_FAILURE",
    "DOCUMENT_NO_MATCH",
    "DOCUMENT_NOISE",
    "WRONG_CONTEXT",
    "GENERATION_FAILURE",
    "VERIFIER_FAILURE",
)

REPLAY_QUESTION_IDS = (
    "V213C-A01",  # document-only
    "V213C-D01",  # structured
    "V213C-B01",  # mixed
    "V213C-F01",  # insufficient
    "V213C-G03",  # adversarial placeholder
)


def v213d_shadow_enabled(settings: Settings) -> bool:
    return bool(getattr(settings, "v213d_shadow_enabled", False))


def configured_sample_rate(settings: Settings) -> float:
    rate = float(getattr(settings, "v213d_shadow_sample_rate", 0.0) or 0.0)
    return min(max(rate, 0.0), 1.0)


def v213d_runtime_config(settings: Settings) -> dict[str, Any]:
    return {
        "shadow_enabled": v213d_shadow_enabled(settings),
        "sample_rate": configured_sample_rate(settings),
        "document_retrieval": bool(
            getattr(settings, "v213d_shadow_document_retrieval", True)
        ),
        "retrieval_variant": str(
            getattr(settings, "v213d_shadow_retrieval_variant", "context_hybrid")
        ),
        "timeout_seconds": float(
            getattr(settings, "v213d_shadow_timeout_seconds", 30.0) or 30.0
        ),
    }


def format_v213d_startup_banner(settings: Settings) -> str:
    cfg = v213d_runtime_config(settings)
    return "\n".join(
        [
            f"V2.13D shadow enabled: {str(cfg['shadow_enabled']).lower()}",
            f"V2.13D sample rate: {cfg['sample_rate']}",
            f"V2.13D retrieval variant: {cfg['retrieval_variant']}",
            f"V2.13D document retrieval: {str(cfg['document_retrieval']).lower()}",
            f"V2.13D timeout: {int(cfg['timeout_seconds'])}s",
        ]
    )


def log_v213d_startup(settings: Settings) -> None:
    banner = format_v213d_startup_banner(settings)
    for line in banner.splitlines():
        logger.info(line)
    log_agent_event(
        logger,
        "v213d.shadow.config",
        extra_config=v213d_runtime_config(settings),
    )


def should_sample_v213d(settings: Settings, seed: str) -> bool:
    if not v213d_shadow_enabled(settings):
        return False
    rate = configured_sample_rate(settings)
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


def _pack_classification(
    primary: str,
    *,
    secondaries: list[str] | None = None,
    improved: bool = False,
    regressed: bool = False,
    newly_recoverable: bool = False,
    control_correct_shadow_worse: bool = False,
) -> dict[str, Any]:
    return {
        "classification": primary,
        "primary_category": primary,
        "secondary_categories": secondaries or [],
        "improved": improved,
        "regressed": regressed,
        "newly_recoverable": newly_recoverable,
        "control_correct_shadow_worse": control_correct_shadow_worse,
    }


def classify_shadow_outcome(
    control: dict[str, Any],
    shadow: dict[str, Any],
    *,
    question_category: str | None = None,
) -> dict[str, Any]:
    """Deterministic Phase 1 comparison. Does not use verifier acceptance alone."""
    error = shadow.get("error") or shadow.get("shadow_error_type")
    stage = str(shadow.get("shadow_stage") or "")
    docs = int(shadow.get("document_evidence_count") or 0)
    c_accept = bool(control.get("final_accepted"))
    s_accept = bool(shadow.get("final_accepted")) and bool(
        shadow.get("metadata_valid", True)
    )
    skipped = bool(shadow.get("retrieval_skipped"))

    if error:
        err_l = str(error).lower()
        if "corpus" in err_l or shadow.get("corpus_available") is False:
            primary = "DOCUMENT_CORPUS_UNAVAILABLE"
        elif "timeout" in err_l or stage in {"document_retrieval", "start"}:
            primary = "DOCUMENT_RETRIEVAL_FAILURE"
        elif stage == "generation":
            primary = "GENERATION_FAILURE"
        elif stage in {"verification", "mapper", "routing"}:
            primary = "VERIFIER_FAILURE"
        elif stage in {"normalization", "metadata_guard", "merge"}:
            primary = "DOCUMENT_NOISE"
        else:
            primary = "DOCUMENT_RETRIEVAL_FAILURE"
        return _pack_classification(primary)

    if shadow.get("wrong_context"):
        return _pack_classification(
            "WRONG_CONTEXT",
            control_correct_shadow_worse=c_accept and s_accept,
        )

    if docs == 0 and not skipped:
        if shadow.get("corpus_available") is False or shadow.get(
            "retrieval_failure_kind"
        ) == "corpus_unavailable":
            return _pack_classification("DOCUMENT_CORPUS_UNAVAILABLE")
        if c_accept and s_accept:
            return _pack_classification("STRUCTURED_DATA_ALREADY_SUFFICIENT")
        if shadow.get("retrieval_failure_kind") == "no_match" or shadow.get(
            "corpus_available"
        ):
            return _pack_classification("DOCUMENT_NO_MATCH")
        return _pack_classification("DOCUMENT_RETRIEVAL_FAILURE")

    newly = (not c_accept) and s_accept and docs > 0
    worse = c_accept and (not s_accept)

    if newly:
        secondaries = ["DOCUMENT_PROVIDED_SOURCE"]
        if question_category == "ambiguous":
            secondaries.insert(0, "DOCUMENT_DISAMBIGUATED_CONTEXT")
        return _pack_classification(
            "DOCUMENT_ADDED_MISSING_CONTEXT",
            secondaries=secondaries,
            improved=True,
            newly_recoverable=True,
        )

    if c_accept and s_accept:
        if (
            docs > 0
            and control.get("answer_hash")
            and shadow.get("answer_hash")
            and control.get("answer_hash") != shadow.get("answer_hash")
        ):
            return _pack_classification(
                "DOCUMENT_ADDED_EXPLANATION",
                secondaries=["DOCUMENT_PROVIDED_SOURCE"],
                improved=True,
            )
        if question_category == "ambiguous" and docs > 0:
            return _pack_classification(
                "DOCUMENT_DISAMBIGUATED_CONTEXT",
                secondaries=["DOCUMENT_PROVIDED_SOURCE"],
            )
        secondaries = ["DOCUMENT_PROVIDED_SOURCE"] if docs > 0 else []
        return _pack_classification(
            "STRUCTURED_DATA_ALREADY_SUFFICIENT",
            secondaries=secondaries,
        )

    if worse:
        return _pack_classification(
            "DOCUMENT_NOISE",
            regressed=True,
            control_correct_shadow_worse=True,
        )

    if docs > 0:
        return _pack_classification("DOCUMENT_DID_NOT_HELP")
    return _pack_classification("DOCUMENT_RETRIEVAL_FAILURE")


def classify_shadow_comparison(
    control: dict[str, Any],
    shadow: dict[str, Any],
    *,
    question_category: str | None = None,
) -> str:
    return str(
        classify_shadow_outcome(
            control, shadow, question_category=question_category
        )["classification"]
    )


def _evidence_summary(evidence: list[CurriculumEvidence]) -> list[dict[str, Any]]:
    rows = []
    for item in evidence[:20]:
        rows.append(
            {
                "entity_type": item.entity_type,
                "entity_id": item.entity_id,
                "grade": item.grade,
                "subject": item.subject,
                "topic": item.topic,
            }
        )
    return rows


def _document_passage_summaries(
    evidence: list[CurriculumEvidence],
) -> list[dict[str, Any]]:
    rows = []
    for item in evidence:
        if item.entity_type != "document_passage":
            continue
        provenance = item.metadata.get("provenance") or {}
        rows.append(
            {
                "document_id": item.metadata.get("document_id"),
                "passage_id": item.entity_id,
                "source_url": provenance.get("source_url")
                or item.metadata.get("source_id"),
                "page_number": item.metadata.get("page_number")
                or provenance.get("page_number"),
                "document_hash": provenance.get("content_hash"),
                "retrieval_score": item.metadata.get("retrieval_score"),
                "retrieval_rank": item.metadata.get("retrieval_rank"),
                "grade": item.grade,
                "subject": item.subject,
                "topic": item.topic,
            }
        )
    return rows


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
        "status": state.status.value if getattr(state, "status", None) else None,
        "evidence_count": len(state.evidence),
        "evidence_snapshot": evidence_snapshot_hash(state.evidence),
        "evidence_summary": _evidence_summary(state.evidence),
        "verifier_score": verification.score if verification else None,
        "verifier_decision": verification.recommendation.value if verification else None,
        "verifier_accepted": verification.passed if verification else False,
        "unsupported_claims": list(verification.unsupported_claims or []) if verification else [],
        "mapper_recommendation": mapping.mapped_recommendation.value if mapping else None,
        "mapped_accepted": mapping.mapped_accepted if mapping else False,
        "final_accepted": bool(mapping.mapped_accepted) if mapping else bool(verification and verification.passed),
        "final_route": route,
        "answer_present": bool(answer),
        "answer_hash": answer_hash(answer) if answer else "",
        "model": (state.metadata or {}).get("model"),
        "generation_config": {
            "provider": getattr(settings, "llm_provider", None),
            "model": getattr(settings, "llm_model", None),
        },
    }


def production_corpus_status(
    store_root: Path | None = None,
    index_root: Path | None = None,
) -> dict[str, Any]:
    """Inspect live production document store + index availability."""
    store_path = store_root or Path("data/documents")
    index_path = index_root or Path("data/document_index")
    document_dirs = (
        sorted(p for p in store_path.glob("doc-*") if p.is_dir())
        if store_path.is_dir()
        else []
    )
    document_count = 0
    for doc_dir in document_dirs:
        if (doc_dir / "metadata.json").is_file() and (doc_dir / "passages.json").is_file():
            document_count += 1
    index_entries = 0
    index_file = index_path / "feature-hash-v1" / "index.json"
    if index_file.is_file():
        try:
            rows = json.loads(index_file.read_text() or "[]")
            index_entries = len(rows) if isinstance(rows, list) else 0
        except json.JSONDecodeError:
            index_entries = 0
    available = document_count > 0 and index_entries > 0
    return {
        "available": available,
        "document_count": document_count,
        "index_entry_count": index_entries,
        "store_root": str(store_path.resolve()) if store_path.exists() else str(store_path),
        "index_root": str(index_path.resolve()) if index_path.exists() else str(index_path),
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
        return [], {
            "skipped": True,
            "variant": getattr(settings, "v213d_shadow_retrieval_variant", "context_hybrid"),
            "corpus_available": production_corpus_status()["available"],
        }
    corpus = production_corpus_status()
    if retrieval is None and not corpus["available"]:
        return [], {
            "variant": getattr(settings, "v213d_shadow_retrieval_variant", "context_hybrid"),
            "latency_ms": 0.0,
            "count": 0,
            "passages": [],
            "corpus_available": False,
            "retrieval_failure_kind": "corpus_unavailable",
            "diagnostics": {
                "document_count": corpus["document_count"],
                "index_entry_count": corpus["index_entry_count"],
            },
        }
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
    evidence: list[CurriculumEvidence] = []
    for hit in result.hits:
        item = document_passage_to_evidence(hit.passage)
        item.metadata["retrieval_score"] = hit.retrieval_score
        item.metadata["retrieval_rank"] = hit.retrieval_rank
        evidence.append(item)
    corpus_available = True if retrieval is not None else corpus["available"]
    failure_kind = None
    if not evidence:
        failure_kind = "no_match" if corpus_available else "corpus_unavailable"
    return evidence, {
        "variant": variant,
        "latency_ms": latency,
        "count": len(evidence),
        "diagnostics": result.diagnostics.to_dict(),
        "passages": _document_passage_summaries(evidence),
        "corpus_available": corpus_available,
        "retrieval_failure_kind": failure_kind,
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
        # Shadow-only safety gates: never treat blocked evidence as a successful answer.
        if metadata_blocked or wrong_context or placeholder:
            final_accepted = False
        shadow = {
            "structured_evidence_count": len(structured),
            "document_evidence_count": len(documents),
            "evidence_count": len(verify_evidence),
            "evidence_summary": _evidence_summary(verify_evidence),
            "document_passages": retrieval_meta.get("passages")
            or _document_passage_summaries(documents),
            "retrieval_variant": retrieval_meta.get("variant")
            or getattr(settings, "v213d_shadow_retrieval_variant", "context_hybrid"),
            "document_retrieval_latency_ms": retrieval_meta.get("latency_ms", 0),
            "retrieval_skipped": bool(retrieval_meta.get("skipped")),
            "corpus_available": retrieval_meta.get("corpus_available"),
            "retrieval_failure_kind": retrieval_meta.get("retrieval_failure_kind"),
            "normalization_status": "ok",
            "normalization_count": len(normalized.evidence),
            "normalization_failures": 0,
            "metadata_valid": integrity.valid,
            "metadata_blocked": metadata_blocked,
            "metadata_policy": metadata_policy,
            "metadata_violations": [v.to_dict() for v in integrity.violations],
            "blocked_evidence": len(normalized.evidence) - len(verify_evidence),
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
            "answer_present": bool(answer),
            "answer_hash": answer_hash(answer) if answer else "",
            "provenance_complete": _provenance_complete(documents),
            "wrong_context": wrong_context,
            "placeholder_evidence": placeholder,
            "error": None,
            "generation_config": {
                "provider": getattr(settings, "llm_provider", None),
                "model": getattr(settings, "llm_model", None),
            },
        }
        category = infer_question_category(production_state, len(documents))
        comparison = classify_shadow_outcome(
            control, shadow, question_category=category
        )
        record = {
            "experiment": _EXPERIMENT_NAME,
            "schema_version": _SCHEMA_VERSION,
            "phase": "phase1",
            "request_id": hashlib.sha256((request_id or "").encode()).hexdigest()[:16]
            if request_id
            else "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sampling": {
                "enabled": v213d_shadow_enabled(settings),
                "rate": configured_sample_rate(settings),
                "sampled": True,
                "decision": "sampled",
            },
            "question": {
                "hash": question_hash(production_state.question),
                "category": category,
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
            "comparison": comparison,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        log_agent_event(
            logger,
            "v213d.shadow.completed",
            request_id=request_id,
            extra_classification=comparison["classification"],
        )
        if shadow["metadata_blocked"]:
            log_agent_event(logger, "v213d.shadow.metadata_blocked", request_id=request_id)
        if comparison.get("improved"):
            log_agent_event(logger, "v213d.shadow.improved", request_id=request_id)
        if comparison.get("regressed"):
            log_agent_event(logger, "v213d.shadow.regressed", request_id=request_id)
        if int(shadow.get("document_evidence_count") or 0) > 0:
            log_agent_event(logger, "v213d.shadow.retrieval_success", request_id=request_id)
        return record
    except Exception as exc:
        log_agent_event(
            logger,
            "v213d.shadow.failed",
            request_id=request_id,
            error=type(exc).__name__,
        )
        shadow_err = {
            "error": type(exc).__name__,
            "shadow_error_type": type(exc).__name__,
            "shadow_error_message_safe": str(exc)[:200],
            "shadow_stage": stage,
            "final_accepted": False,
            "metadata_valid": False,
            "document_evidence_count": 0,
        }
        comparison = classify_shadow_outcome(control, shadow_err)
        return {
            "experiment": _EXPERIMENT_NAME,
            "schema_version": _SCHEMA_VERSION,
            "phase": "phase1",
            "request_id": hashlib.sha256((request_id or "").encode()).hexdigest()[:16]
            if request_id
            else "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sampling": {
                "enabled": v213d_shadow_enabled(settings),
                "rate": configured_sample_rate(settings),
                "sampled": True,
                "decision": "sampled",
            },
            "question": {"hash": question_hash(production_state.question)},
            "control": control,
            "shadow": shadow_err,
            "grounding": {"metadata_valid": False},
            "comparison": comparison,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }


def jsonl_runtime_path() -> Path:
    return _JSONL.resolve()


def record_pipeline_stage(
    stage: str,
    *,
    path: Path | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Lightweight, privacy-conscious funnel counter. Never stores question text."""
    target = path or _FUNNEL
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK:
            current: dict[str, Any] = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "stages": {name: 0 for name in _FUNNEL_STAGES},
                "last_request_id_hash": "",
                "jsonl_path": str(jsonl_runtime_path()),
            }
            if target.exists():
                try:
                    loaded = json.loads(target.read_text() or "{}")
                    if isinstance(loaded, dict):
                        current.update(loaded)
                        stages = current.get("stages") or {}
                        for name in _FUNNEL_STAGES:
                            stages.setdefault(name, 0)
                        current["stages"] = stages
                except json.JSONDecodeError:
                    pass
            stages = current.setdefault("stages", {})
            stages[stage] = int(stages.get(stage) or 0) + 1
            current["updated_at"] = datetime.now(timezone.utc).isoformat()
            if request_id:
                current["last_request_id_hash"] = hashlib.sha256(
                    request_id.encode()
                ).hexdigest()[:16]
            current["jsonl_path"] = str(jsonl_runtime_path())
            target.write_text(json.dumps(current))
            return current
    except Exception:
        logger.debug("v213d pipeline funnel skipped", exc_info=True)
        return {}


def load_pipeline_funnel(path: Path | None = None) -> dict[str, Any]:
    target = path or _FUNNEL
    if not target.exists():
        return {
            "stages": {name: 0 for name in _FUNNEL_STAGES},
            "jsonl_path": str(jsonl_runtime_path()),
        }
    try:
        data = json.loads(target.read_text() or "{}")
    except json.JSONDecodeError:
        data = {}
    stages = data.get("stages") or {}
    for name in _FUNNEL_STAGES:
        stages.setdefault(name, 0)
    data["stages"] = stages
    data.setdefault("jsonl_path", str(jsonl_runtime_path()))
    return data


def persist_record(record: dict[str, Any], path: Path | None = None) -> None:
    target = path or _JSONL
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK:
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
        record_pipeline_stage("shadow_persisted", request_id=record.get("request_id"))
    except Exception as exc:
        record_pipeline_stage("persist_error", request_id=record.get("request_id"))
        log_agent_event(
            logger,
            "v213d.shadow.persist_failed",
            error=type(exc).__name__,
        )
        raise


def record_traffic_event(*, sampled: bool, path: Path | None = None) -> dict[str, int]:
    """Process-local counter (single uvicorn worker). Multi-worker deployments need aggregation."""
    target = path or _TRAFFIC
    target.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        current = {"total_production_requests": 0, "sampled_requests": 0}
        if target.exists():
            try:
                current.update(json.loads(target.read_text() or "{}"))
            except json.JSONDecodeError:
                pass
        current["total_production_requests"] = int(
            current.get("total_production_requests") or 0
        ) + 1
        if sampled:
            current["sampled_requests"] = int(current.get("sampled_requests") or 0) + 1
        target.write_text(json.dumps(current))
        return {
            "total_production_requests": int(current["total_production_requests"]),
            "sampled_requests": int(current.get("sampled_requests") or 0),
        }


def load_traffic_counters(path: Path | None = None) -> dict[str, int]:
    target = path or _TRAFFIC
    if not target.exists():
        return {"total_production_requests": 0, "sampled_requests": 0}
    try:
        data = json.loads(target.read_text() or "{}")
    except json.JSONDecodeError:
        return {"total_production_requests": 0, "sampled_requests": 0}
    return {
        "total_production_requests": int(data.get("total_production_requests") or 0),
        "sampled_requests": int(data.get("sampled_requests") or 0),
    }


def run_production_shadow(
    agent: Any,
    state: CurriculumQAState,
    *,
    request_id: str | None = None,
    retrieval: HybridDocumentRetrievalService | None = None,
    jsonl_path: Path | None = None,
) -> dict[str, Any]:
    production_copy = copy.deepcopy(state)
    record_pipeline_stage("shadow_started", request_id=request_id)
    log_agent_event(logger, "v213d.shadow.started", request_id=request_id)
    try:
        record = run_shadow_pipeline(
            agent, production_copy, request_id=request_id, retrieval=retrieval
        )
        if (record.get("shadow") or {}).get("error"):
            record_pipeline_stage("shadow_failed", request_id=request_id)
        else:
            record_pipeline_stage("shadow_completed", request_id=request_id)
        persist_record(record, jsonl_path)
        return record
    except Exception:
        record_pipeline_stage("shadow_failed", request_id=request_id)
        raise


def maybe_schedule_v213d_shadow(
    agent: Any,
    state: CurriculumQAState,
    *,
    request_id: str | None = None,
    jsonl_path: Path | None = None,
) -> threading.Thread | None:
    """Fire-and-forget; exceptions never reach the production caller.

    Invoked only after the production LangGraph response is fully determined.
    """
    settings = agent.settings
    record_pipeline_stage("request_seen", request_id=request_id)
    record_pipeline_stage("shadow_eligible", request_id=request_id)
    sampled = should_sample_v213d(settings, request_id or state.question)
    try:
        record_traffic_event(sampled=sampled)
    except Exception:
        logger.debug("v213d traffic counter skipped", exc_info=True)
    if not sampled:
        record_pipeline_stage("shadow_not_sampled", request_id=request_id)
        log_agent_event(
            logger,
            "v213d.shadow.not_sampled",
            request_id=request_id,
            extra_rate=configured_sample_rate(settings),
        )
        return None
    record_pipeline_stage("shadow_sampled", request_id=request_id)
    log_agent_event(logger, "v213d.shadow.sampled", request_id=request_id)
    snapshot = copy.deepcopy(state)
    timeout = float(getattr(settings, "v213d_shadow_timeout_seconds", 30.0))

    def _worker() -> None:
        try:
            started = time.perf_counter()
            run_production_shadow(
                agent,
                snapshot,
                request_id=request_id,
                jsonl_path=jsonl_path,
            )
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


def load_jsonl_records(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or _JSONL
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def production_shadow_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if not r.get("replay_id")]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def phase1_observation_status(metrics: dict[str, Any]) -> tuple[str, str]:
    safety = metrics.get("safety_metrics") or {}
    hard = (
        int(safety.get("wrong_context_false_acceptance") or 0)
        + int(safety.get("placeholder_false_acceptance") or 0)
        + int(safety.get("metadata_integrity_false_acceptance") or 0)
        + int(safety.get("metadata_false_acceptance") or 0)
    )
    if hard > 0 or metrics.get("safety_blocked"):
        return (
            "SAFETY_BLOCKED",
            "STOP — SAFETY ISSUE",
        )
    completed = int(metrics.get("successful_shadow_evaluations") or 0)
    worse = int(metrics.get("control_correct_shadow_worse") or 0)
    regressions = int(metrics.get("regressions") or 0)
    newly = int(metrics.get("newly_recoverable_count") or 0)
    if completed < OBSERVATION_TARGET_MIN:
        return (
            "INSUFFICIENT_SAMPLE",
            "CONTINUE SHADOW",
        )
    if worse > 0 or regressions > 0:
        return (
            "REGRESSION_DETECTED",
            "INVESTIGATE BEFORE CONTINUING",
        )
    if newly > 0 and completed >= OBSERVATION_TARGET_MIN:
        if completed >= OBSERVATION_TARGET_MAX:
            return (
                "PROMISING",
                "CONTINUE SHADOW",
            )
        return (
            "PROMISING",
            "CONTINUE SHADOW",
        )
    return (
        "OBSERVATION_READY",
        "CONTINUE SHADOW",
    )


def aggregate_records(
    records: list[dict[str, Any]],
    *,
    traffic: dict[str, int] | None = None,
    source: str = "mixed",
) -> dict[str, Any]:
    n_raw = len(records)
    n = n_raw or 1
    successful = [r for r in records if not (r.get("shadow") or {}).get("error")]
    errors = [r for r in records if (r.get("shadow") or {}).get("error")]
    pre_corpus = [
        r
        for r in records
        if r.get("corpus_epoch") == "pre_corpus"
        or (r.get("comparison") or {}).get("classification") == "DOCUMENT_CORPUS_UNAVAILABLE"
    ]
    post_corpus = [r for r in records if r not in pre_corpus]
    timeouts = [
        r
        for r in errors
        if "timeout" in str((r.get("shadow") or {}).get("shadow_error_type") or "").lower()
        or "timeout" in str((r.get("shadow") or {}).get("shadow_error_message_safe") or "").lower()
    ]
    retrieval_failures = [
        r
        for r in records
        if (r.get("comparison") or {}).get("classification")
        in {"DOCUMENT_RETRIEVAL_FAILURE", "DOCUMENT_NO_MATCH"}
    ]
    corpus_unavailable = [
        r
        for r in records
        if (r.get("comparison") or {}).get("classification") == "DOCUMENT_CORPUS_UNAVAILABLE"
    ]
    improved = [r for r in records if r.get("comparison", {}).get("improved")]
    regressed = [r for r in records if r.get("comparison", {}).get("regressed")]
    newly = [
        r
        for r in records
        if r.get("comparison", {}).get("newly_recoverable")
        or r.get("comparison", {}).get("classification")
        in {"DOCUMENT_ADDED_MISSING_CONTEXT", "DOCUMENT_ADDED_GROUNDING"}
    ]
    worse = [
        r
        for r in records
        if r.get("comparison", {}).get("control_correct_shadow_worse")
    ]
    # Observation sample sufficiency should ignore pre-corpus infrastructure failures.
    post_successful = [
        r for r in post_corpus if not (r.get("shadow") or {}).get("error")
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
        "metadata_false_acceptance": sum(
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
        "unsupported_claims": sum(
            len((r.get("grounding") or {}).get("unsupported_claims") or [])
            for r in successful
        ),
    }
    safety_fail = any(
        int(v) > 0
        for k, v in safety.items()
        if k
        not in {
            "shadow_errors_must_not_affect_production",
            "unsupported_claims",
        }
    )
    latencies = [float(r.get("latency_ms") or 0) for r in successful]
    retrieval_lat = [
        float((r.get("shadow") or {}).get("document_retrieval_latency_ms") or 0)
        for r in successful
    ]
    passages = [
        float((r.get("shadow") or {}).get("document_evidence_count") or 0)
        for r in successful
    ]
    provenance = [r.get("grounding", {}).get("provenance_complete") for r in successful]
    classifications = _count_classifications(records)
    traffic = traffic or load_traffic_counters()
    metrics: dict[str, Any] = {
        "experiment": _EXPERIMENT_NAME,
        "schema_version": _SCHEMA_VERSION,
        "source": source,
        "total_production_requests": int(traffic.get("total_production_requests") or 0),
        "sampled_requests": int(
            traffic.get("sampled_requests") or len(records)
        ),
        "traffic_sampled": len(records),
        "shadow_completed": len(successful),
        "successful_shadow_evaluations": len(successful),
        "pre_corpus_shadow_evaluations": len(pre_corpus),
        "post_corpus_shadow_evaluations": len(post_corpus),
        "post_corpus_successful_shadow_evaluations": len(post_successful),
        "corpus_unavailable_count": len(corpus_unavailable),
        "shadow_errors": len(errors),
        "shadow_timeouts": len(timeouts),
        "shadow_error_rate": len(errors) / n,
        "retrieval_failures": len(retrieval_failures),
        "retrieval_success_rate": sum(
            1
            for r in successful
            if int((r.get("shadow") or {}).get("document_evidence_count") or 0) > 0
        )
        / max(len(successful), 1),
        "retrieval_success": sum(
            1
            for r in successful
            if int((r.get("shadow") or {}).get("document_evidence_count") or 0) > 0
        )
        / max(len(successful), 1),
        "mean_retrieval_latency": (
            sum(retrieval_lat) / len(retrieval_lat) if retrieval_lat else 0.0
        ),
        "p95_retrieval_latency": _percentile(retrieval_lat, 95),
        "mean_passages_retrieved": (
            sum(passages) / len(passages) if passages else 0.0
        ),
        "provenance_complete_rate": sum(1 for p in provenance if p)
        / max(len(provenance), 1),
        "metadata_valid_rate": sum(
            1 for r in successful if r.get("grounding", {}).get("metadata_valid")
        )
        / max(len(successful), 1),
        "wrong_context_false_acceptance_rate": safety["wrong_context_false_acceptance"]
        / max(len(successful), 1),
        "placeholder_false_acceptance_rate": safety["placeholder_false_acceptance"]
        / max(len(successful), 1),
        "metadata_false_acceptance_rate": safety["metadata_false_acceptance"]
        / max(len(successful), 1),
        "unsupported_claim_rate": safety["unsupported_claims"] / max(len(successful), 1),
        "newly_recoverable_count": len(newly),
        "newly_recoverable_rate": len(newly) / n,
        "improvements": len(improved),
        "regressions": len(regressed),
        "unchanged": max(0, n_raw - len(improved) - len(regressed) - len(errors)),
        "control_correct_shadow_worse": len(worse),
        "document_added_explanation": classifications.get("DOCUMENT_ADDED_EXPLANATION", 0),
        "document_disambiguated_context": classifications.get(
            "DOCUMENT_DISAMBIGUATED_CONTEXT", 0
        ),
        "document_provided_source": classifications.get("DOCUMENT_PROVIDED_SOURCE", 0)
        + sum(
            1
            for r in records
            if "DOCUMENT_PROVIDED_SOURCE"
            in ((r.get("comparison") or {}).get("secondary_categories") or [])
        ),
        "document_did_not_help": classifications.get("DOCUMENT_DID_NOT_HELP", 0),
        "document_noise": classifications.get("DOCUMENT_NOISE", 0),
        "classifications": classifications,
        "safety_metrics": safety,
        "safety_blocked": safety_fail,
        "latency_metrics": {
            "shadow_mean_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "shadow_p95_ms": _percentile(latencies, 95),
            "retrieval_mean_ms": (
                sum(retrieval_lat) / len(retrieval_lat) if retrieval_lat else 0.0
            ),
            "retrieval_p95_ms": _percentile(retrieval_lat, 95),
        },
        "observation_target_min": OBSERVATION_TARGET_MIN,
        "observation_target_max": OBSERVATION_TARGET_MAX,
        "v213c_comparison_note": (
            "V2.13C was a controlled harness (59.7%→90.3% grounded-correct). "
            "V2.13D Phase 1 real-traffic observations are not statistically equivalent."
        ),
    }
    status_input = {
        **metrics,
        "successful_shadow_evaluations": len(post_successful),
    }
    status, recommendation = phase1_observation_status(status_input)
    metrics["phase1_status"] = status
    metrics["phase1_recommendation"] = recommendation
    metrics["canary_recommendation"] = (
        "BLOCKED" if status == "SAFETY_BLOCKED" else "CANARY_NOT_READY"
    )
    if source == "controlled_replay":
        metrics["canary_note"] = (
            "Controlled replay only. Phase 1 requires real production shadow observations."
        )
    elif status == "INSUFFICIENT_SAMPLE":
        metrics["canary_note"] = (
            f"Only {len(post_successful)} post-corpus successful real shadow "
            f"evaluations (pre-corpus={len(pre_corpus)}); target is "
            f"{OBSERVATION_TARGET_MIN}–{OBSERVATION_TARGET_MAX} before a rollout recommendation."
        )
    elif status == "SAFETY_BLOCKED":
        metrics["canary_note"] = (
            "Safety gate failed; do not enable canary or production document retrieval."
        )
    elif status == "REGRESSION_DETECTED":
        metrics["canary_note"] = (
            "Control-correct / shadow-worse cases detected; investigate before continuing."
        )
    else:
        metrics["canary_note"] = (
            "Shadow observation continues; do not promote document retrieval yet."
        )
    return metrics


def _count_classifications(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        label = record.get("comparison", {}).get("classification") or "UNKNOWN"
        counts[label] = counts.get(label, 0) + 1
    return counts


def interpret_v213d(metrics: dict[str, Any]) -> str:
    return str(
        metrics.get("phase1_status")
        or metrics.get("canary_recommendation")
        or "INSUFFICIENT_SAMPLE"
    )


__all__ = [
    "OBSERVATION_TARGET_MIN",
    "OBSERVATION_TARGET_MAX",
    "PHASE1_CATEGORIES",
    "REPLAY_QUESTION_IDS",
    "aggregate_records",
    "classify_shadow_comparison",
    "classify_shadow_outcome",
    "configured_sample_rate",
    "format_v213d_startup_banner",
    "interpret_v213d",
    "jsonl_runtime_path",
    "load_jsonl_records",
    "load_pipeline_funnel",
    "load_traffic_counters",
    "log_v213d_startup",
    "maybe_schedule_v213d_shadow",
    "persist_record",
    "phase1_observation_status",
    "prepare_replay_corpus",
    "production_corpus_status",
    "production_shadow_records",
    "record_pipeline_stage",
    "record_traffic_event",
    "replay_fixtures",
    "run_production_shadow",
    "run_shadow_pipeline",
    "should_sample_v213d",
    "v213d_runtime_config",
    "v213d_shadow_enabled",
]
