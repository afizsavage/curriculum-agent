"""V2.13G live dual-arm document arbitration shadow (observational only).

Runs BASELINE (V2.13D merge) and ARBITRATED (V2.13F policy) arms from one
frozen retrieval snapshot. Never replaces the production answer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.v213d_phase1e_diagnostics import (
    build_phase1e_diagnostics,
    classify_regression_cause,
)
from app.agent.v213d_shadow import (
    _control_snapshot,
    _document_passage_summaries,
    _evidence_summary,
    _provenance_complete,
    classify_shadow_outcome,
    infer_question_category,
    production_corpus_status,
    question_hash,
    retrieve_document_evidence,
)
from app.agent.v213f_arbitration import (
    ArbitrationPolicy,
    DocumentRole,
    DocumentUse,
    StructuredSufficiency,
    arbitrate_documents,
    select_generation_evidence,
)
from app.agent.v211_metadata_integrity import (
    PipelineVariant,
    apply_metadata_policy,
    validate_metadata_integrity,
)
from app.agent.v212_langchain import infer_mapper_fixture_class
from app.agent.v25_experiment import _CLEAN_PLACEHOLDER
from app.agent.v26_experiment import answer_hash
from app.agent.v28_recommendation_mapping import map_recommendation
from app.agent.v29_evidence_normalization import NormalizationVariant, normalize_evidence
from app.agent.evidence_snapshot import evidence_snapshot_hash
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus, merge_evidence_bundles
from app.logging_utils import log_agent_event

logger = logging.getLogger(__name__)

_EXPERIMENT = "v2.13g"
_SCHEMA = "v213g.1"
_ANALYTICAL_THRESHOLD = 0.85
_JSONL = Path("data/diagnostics/v213g_arbitration_shadow.jsonl")
_SUMMARY = Path("data/diagnostics/v213g_arbitration_summary.json")
_WRITE_LOCK = threading.Lock()
_BATCH_HEALTHY = "batch_2_healthy_structured_api"
_BATCH_HISTORICAL = "batch_1_api_unavailable"

MILESTONE_FIRST = 50
MILESTONE_STRONG = 100
MILESTONE_PREFERRED = 200
SUFFICIENT_WITH_DOCS_TARGET = 20


G_REGRESSION_CAUSES = (
    "RETRIEVAL_IRRELEVANCE",
    "ARBITRATION_MISCLASSIFICATION",
    "GENERATOR_DOCUMENT_DRIFT",
    "VERIFIER_STRICTNESS",
    "MAPPER_ROUTING_CHANGE",
    "METADATA_EFFECT",
    "CORPUS_COVERAGE_GAP",
    "UNKNOWN",
)


def v213g_enabled(settings: Settings) -> bool:
    return bool(getattr(settings, "v213g_live_arbitration_shadow", False))


def v213g_jsonl_path() -> Path:
    return _JSONL.resolve()


def persist_v213g_record(record: dict[str, Any], path: Path | None = None) -> None:
    target = path or _JSONL
    target.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()


def load_v213g_records(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or _JSONL
    if not target.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def probe_structured_api(settings: Any | None = None) -> dict[str, Any]:
    """Lightweight Curriculum Structure API health probe for V2.13G validity."""
    base = "http://127.0.0.1:8000"
    if settings is not None:
        try:
            base = str(settings.resolved_curriculum_api_url())
        except Exception:  # noqa: BLE001
            base = getattr(settings, "curriculum_api_base_url", base) or base
    url = f"{str(base).rstrip('/')}/health"
    started = time.perf_counter()
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=3.0) as resp:
            ok = int(getattr(resp, "status", 200) or 200) < 400
            body = resp.read()[:200]
            latency = (time.perf_counter() - started) * 1000
            return {
                "structured_api_available": bool(ok),
                "structured_api_latency_ms": round(latency, 3),
                "structured_request_success": bool(ok),
                "health_url": url,
                "health_body_prefix": body.decode("utf-8", errors="replace"),
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "structured_api_available": False,
            "structured_api_latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "structured_request_success": False,
            "health_url": url,
            "error": type(exc).__name__,
        }


def classify_row_validity(
    *,
    structured_api_available: bool,
    structured_count: int,
    evidence_status: str | None = None,
) -> dict[str, Any]:
    """Distinguish API outage zeros from genuine no-structured-evidence."""
    status = str(evidence_status or "").lower()
    if not structured_api_available and structured_count <= 0:
        return {
            "validity": "INFRASTRUCTURE_INVALID",
            "zero_structured_reason": "api_failure",
            "structured_count_zero_due_to_api_failure": True,
            "structured_count_zero_due_to_genuine_no_evidence": False,
        }
    if structured_count <= 0:
        # Production evidence_status error with API up can still be genuine miss
        # or tool failure; treat as genuine_no_evidence for cohort math unless
        # evidence_status explicitly signals transport/api errors.
        apiish = any(tok in status for tok in ("error", "timeout", "unavailable", "connection"))
        if apiish and not structured_api_available:
            reason = "api_failure"
            validity = "INFRASTRUCTURE_INVALID"
        else:
            reason = "genuine_no_evidence"
            validity = "VALID"
        return {
            "validity": validity,
            "zero_structured_reason": reason,
            "structured_count_zero_due_to_api_failure": reason == "api_failure",
            "structured_count_zero_due_to_genuine_no_evidence": reason == "genuine_no_evidence",
        }
    return {
        "validity": "VALID",
        "zero_structured_reason": None,
        "structured_count_zero_due_to_api_failure": False,
        "structured_count_zero_due_to_genuine_no_evidence": False,
    }


def execute_shadow_arm(
    agent: Any,
    production_state: Any,
    *,
    structured: list[CurriculumEvidence],
    documents: list[CurriculumEvidence],
    generation_evidence: list[CurriculumEvidence],
    retrieval_meta: dict[str, Any],
    request_id: str | None,
    arm_name: str,
    arbitration_info: dict[str, Any] | None = None,
    provenance_evidence: list[CurriculumEvidence] | None = None,
) -> dict[str, Any]:
    """Run normalize→metadata→generate→verify→map→route for one evidence policy.

    generation_evidence feeds the LLM. provenance_evidence is recorded only and
    must not be merged into generation when document_use is PROVENANCE_ONLY.
    """
    settings = agent.settings
    provenance_evidence = list(provenance_evidence or [])
    stage = f"{arm_name}:normalization"
    try:
        normalized = normalize_evidence(
            generation_evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION
        )
        stage = f"{arm_name}:metadata_guard"
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

        stage = f"{arm_name}:generation"
        rid = f"{request_id or 'v213g'}-{arm_name}"
        shadow_state = agent.answer(shadow_state, request_id=rid)
        stage = f"{arm_name}:verification"
        shadow_state = agent.verify(shadow_state, request_id=rid)
        stage = f"{arm_name}:mapper"
        verifier_result = shadow_state.verification
        mapping = map_recommendation(
            verifier_result,
            fixture_class=infer_mapper_fixture_class(verify_evidence, verifier_result),  # type: ignore[arg-type]
            evidence=verify_evidence,
            answer=shadow_state.final_answer or shadow_state.draft_answer or "",
            threshold=_ANALYTICAL_THRESHOLD,
        )
        stage = f"{arm_name}:routing"
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
        if metadata_blocked or wrong_context or placeholder:
            final_accepted = False

        gen_doc_n = sum(
            1
            for e in generation_evidence
            if getattr(e, "entity_type", None) == "document_passage"
        )
        arm: dict[str, Any] = {
            "arm": arm_name,
            "outcome_source": "regenerated",
            "generation_attempted": True,
            "structured_evidence_count": len(structured),
            "document_evidence_count": len(documents),
            "generation_evidence_count": len(generation_evidence),
            "provenance_evidence_count": len(provenance_evidence),
            "generation_document_count": gen_doc_n,
            "merged_evidence_count": len(verify_evidence),
            "evidence_count": len(verify_evidence),
            "evidence_snapshot": evidence_snapshot_hash(verify_evidence),
            "generation_evidence_fingerprint": evidence_snapshot_hash(verify_evidence),
            "provenance_evidence_fingerprint": (
                evidence_snapshot_hash(provenance_evidence) if provenance_evidence else ""
            ),
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
            "metadata_valid": integrity.valid,
            "metadata_blocked": metadata_blocked,
            "metadata_policy": metadata_policy,
            "metadata_violations": [v.to_dict() for v in integrity.violations],
            "blocked_evidence": len(normalized.evidence) - len(verify_evidence),
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
            "answer_length": len(answer),
            "answer_text_for_diagnostics": answer,
            "provenance_complete": _provenance_complete(documents),
            "wrong_context": wrong_context,
            "placeholder_evidence": placeholder,
            "error": None,
            "shadow_stage": stage,
            "generation_config": {
                "provider": getattr(settings, "llm_provider", None),
                "model": getattr(settings, "llm_model", None),
                "temperature": 0.0,
            },
        }
        if arbitration_info is not None:
            arm["arbitration"] = arbitration_info
        return arm
    except Exception as exc:  # noqa: BLE001
        return {
            "arm": arm_name,
            "error": type(exc).__name__,
            "shadow_error_type": type(exc).__name__,
            "shadow_error_message_safe": str(exc)[:200],
            "shadow_stage": stage,
            "final_accepted": False,
            "metadata_valid": False,
            "document_evidence_count": len(documents),
            "structured_evidence_count": len(structured),
            "unsupported_claims": [],
            "answer_hash": "",
            "answer_length": 0,
            "answer_text_for_diagnostics": "",
            "wrong_context": False,
            "placeholder_evidence": False,
            "provenance_complete": False,
            "corpus_available": retrieval_meta.get("corpus_available"),
            "retrieval_failure_kind": retrieval_meta.get("retrieval_failure_kind"),
        }


def _arm_for_comparison(arm: dict[str, Any]) -> dict[str, Any]:
    """Drop diagnostic-only answer text before classify/persist helpers."""
    out = {k: v for k, v in arm.items() if k != "answer_text_for_diagnostics"}
    return out


def _safety_false_accepts(arm: dict[str, Any]) -> dict[str, int]:
    accepted = bool(arm.get("final_accepted"))
    return {
        "wrong_context_false_acceptance": int(
            accepted and bool(arm.get("wrong_context"))
        ),
        "placeholder_false_acceptance": int(
            accepted and bool(arm.get("placeholder_evidence"))
        ),
        "metadata_false_acceptance": int(
            accepted and bool(arm.get("metadata_blocked"))
        ),
    }


def classify_g_regression_cause(
    control: dict[str, Any],
    arm: dict[str, Any],
    arbitration: dict[str, Any] | None,
    *,
    question_grade: str | None = None,
    question_subject: str | None = None,
) -> str:
    """Extend Phase 1E causes with arbitration-specific labels."""
    if not (
        bool(control.get("final_accepted")) and not bool(arm.get("final_accepted"))
    ):
        return "UNKNOWN"
    role = str((arbitration or {}).get("document_role") or "")
    use = str((arbitration or {}).get("document_use") or "")
    # Misclassification: decisive docs withheld, or redundant/irrelevant still generating.
    if role == DocumentRole.DECISIVE.value and use != DocumentUse.INCLUDE_IN_GENERATION.value:
        return "ARBITRATION_MISCLASSIFICATION"
    if role in {DocumentRole.REDUNDANT.value, DocumentRole.IRRELEVANT.value} and use == (
        DocumentUse.INCLUDE_IN_GENERATION.value
    ):
        return "ARBITRATION_MISCLASSIFICATION"
    if arm.get("retrieval_failure_kind") == "no_match" and int(
        arm.get("document_evidence_count") or 0
    ) == 0:
        return "CORPUS_COVERAGE_GAP"
    cause = classify_regression_cause(
        control,
        arm,
        question_grade=question_grade,
        question_subject=question_subject,
    )
    if cause in G_REGRESSION_CAUSES:
        return cause
    return cause if cause else "UNKNOWN"


def build_investigation(
    *,
    production_state: Any,
    control: dict[str, Any],
    baseline: dict[str, Any],
    arbitrated: dict[str, Any],
    arbitration: dict[str, Any],
    cause: str,
) -> dict[str, Any]:
    return {
        "question_hash": question_hash(production_state.question or ""),
        "grade": production_state.grade,
        "subject": production_state.subject,
        "topic": production_state.topic,
        "structured_evidence_summary": control.get("evidence_summary"),
        "document_passages": baseline.get("document_passages")
        or arbitrated.get("document_passages"),
        "document_role": arbitration.get("document_role"),
        "structured_sufficiency": arbitration.get("structured_sufficiency"),
        "document_use": arbitration.get("document_use"),
        "arbitration_reasons": arbitration.get("reasons"),
        "control": {
            "final_accepted": control.get("final_accepted"),
            "final_route": control.get("final_route"),
            "verifier_decision": control.get("verifier_decision"),
            "mapper_recommendation": control.get("mapper_recommendation"),
            "unsupported_claims": control.get("unsupported_claims"),
            "answer_hash": control.get("answer_hash"),
            "answer_length": control.get("answer_length"),
        },
        "baseline": {
            "final_accepted": baseline.get("final_accepted"),
            "final_route": baseline.get("final_route"),
            "verifier_decision": baseline.get("verifier_decision"),
            "mapper_recommendation": baseline.get("mapper_recommendation"),
            "unsupported_claims": baseline.get("unsupported_claims"),
            "answer_hash": baseline.get("answer_hash"),
            "answer_length": baseline.get("answer_length"),
        },
        "arbitrated": {
            "final_accepted": arbitrated.get("final_accepted"),
            "final_route": arbitrated.get("final_route"),
            "verifier_decision": arbitrated.get("verifier_decision"),
            "mapper_recommendation": arbitrated.get("mapper_recommendation"),
            "unsupported_claims": arbitrated.get("unsupported_claims"),
            "answer_hash": arbitrated.get("answer_hash"),
            "answer_length": arbitrated.get("answer_length"),
        },
        "regression_cause": cause,
        # Answers are not stored in production JSONL; keep hashes only.
        "note": "Answer bodies omitted from persisted investigation for privacy.",
    }


def compare_control_to_arm(
    control: dict[str, Any],
    arm: dict[str, Any],
    *,
    question_category: str,
    question_grade: str | None,
    question_subject: str | None,
    question_topic: str | None,
    control_answer: str,
    arbitration: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arm_cmp = _arm_for_comparison(arm)
    comparison = classify_shadow_outcome(
        control, arm_cmp, question_category=question_category
    )
    diagnostics = build_phase1e_diagnostics(
        control,
        arm_cmp,
        comparison,
        question_grade=question_grade,
        question_subject=question_subject,
        question_topic=question_topic,
        control_answer=control_answer,
        shadow_answer=str(arm.get("answer_text_for_diagnostics") or ""),
    )
    comparison = dict(comparison)
    comparison["document_effect"] = diagnostics["document_value"]["document_effect"]
    comparison["retrieval_quality"] = diagnostics["document_value"]["retrieval_quality"]
    if comparison.get("control_correct_shadow_worse") or comparison.get("regressed"):
        comparison["regression_cause"] = classify_g_regression_cause(
            control,
            arm_cmp,
            arbitration,
            question_grade=question_grade,
            question_subject=question_subject,
        )
    else:
        comparison["regression_cause"] = diagnostics.get("regression_cause")
    comparison["structured_sufficient_document_retrieved"] = diagnostics[
        "document_value"
    ]["structured_sufficient_document_retrieved"]
    comparison["control_correct_arm_worse"] = bool(
        comparison.get("control_correct_shadow_worse")
    )
    comparison["control_insufficient_arm_accepted"] = bool(
        comparison.get("newly_recoverable")
    ) or (
        (not bool(control.get("final_accepted")))
        and bool(arm_cmp.get("final_accepted"))
    )
    return comparison, diagnostics


def run_dual_arm_shadow_pipeline(
    agent: Any,
    production_state: Any,
    *,
    request_id: str | None = None,
    retrieval: Any | None = None,
    retrieve_documents: Any | None = None,
) -> dict[str, Any]:
    """Retrieve once; run baseline merge + arbitrated arms independently."""
    settings = agent.settings
    structured = copy.deepcopy(production_state.evidence)
    control = _control_snapshot(production_state, settings)
    started = time.perf_counter()
    control_answer = production_state.final_answer or production_state.draft_answer or ""

    infra_probe = probe_structured_api(settings)
    evidence_status = None
    try:
        evidence_status = getattr(production_state.evidence_status, "value", None) or str(
            production_state.evidence_status or ""
        )
    except Exception:  # noqa: BLE001
        evidence_status = str(getattr(production_state, "evidence_status", "") or "")
    validity = classify_row_validity(
        structured_api_available=bool(infra_probe.get("structured_api_available")),
        structured_count=len(structured),
        evidence_status=evidence_status,
    )
    batch_id = (
        _BATCH_HEALTHY
        if infra_probe.get("structured_api_available")
        else "batch_2_api_degraded"
    )

    retriever = retrieve_documents or retrieve_document_evidence
    retrieval_started = time.perf_counter()
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
    retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000
    if retrieval_meta.get("latency_ms") is None:
        retrieval_meta["latency_ms"] = retrieval_latency_ms

    policy_raw = str(
        getattr(settings, "v213f_arbitration_policy", "C_ARBITRATED") or "C_ARBITRATED"
    )
    try:
        policy = ArbitrationPolicy(policy_raw)
    except ValueError:
        policy = ArbitrationPolicy.ARBITRATED

    arb_started = time.perf_counter()
    decision = arbitrate_documents(
        documents,
        structured=structured,
        question=production_state.question or "",
        grade=production_state.grade,
        subject=production_state.subject,
        topic=production_state.topic,
        control_accepted=bool(control.get("final_accepted")),
        control_route=control.get("final_route"),
        control_verifier_decision=control.get("verifier_decision"),
        policy=policy,
    )
    generation_bundle, provenance_only = select_generation_evidence(
        structured, documents, decision
    )
    arbitration_latency_ms = (time.perf_counter() - arb_started) * 1000
    gen_doc_n = sum(
        1
        for e in generation_bundle
        if getattr(e, "entity_type", None) == "document_passage"
    )
    arbitration_info = {
        **decision.to_dict(),
        "generation_document_count": gen_doc_n,
        "provenance_document_count": len(provenance_only),
        "arbitration_latency_ms": round(arbitration_latency_ms, 3),
        "provenance_only_semantics": None,
    }

    baseline_bundle = merge_evidence_bundles(structured, documents)

    baseline_started = time.perf_counter()
    baseline = execute_shadow_arm(
        agent,
        production_state,
        structured=structured,
        documents=documents,
        generation_evidence=baseline_bundle,
        provenance_evidence=[],
        retrieval_meta=retrieval_meta,
        request_id=request_id,
        arm_name="baseline",
    )
    baseline_latency_ms = (time.perf_counter() - baseline_started) * 1000

    from app.agent.v213g1_provenance_only import (
        build_preserved_control_arm,
        resolved_provenance_only_semantics,
        should_preserve_control_answer,
    )

    semantics = resolved_provenance_only_semantics(settings)
    arbitration_info["provenance_only_semantics"] = semantics
    preserve = should_preserve_control_answer(
        document_use=decision.document_use.value,
        control_accepted=bool(control.get("final_accepted")),
        semantics=semantics,
    )

    arb_arm_started = time.perf_counter()
    if preserve:
        arbitrated = build_preserved_control_arm(
            control,
            structured=structured,
            documents=documents,
            generation_evidence=generation_bundle,
            provenance_evidence=provenance_only,
            retrieval_meta=retrieval_meta,
            arbitration_info=arbitration_info,
            arm_name="arbitrated",
        )
    else:
        arbitrated = execute_shadow_arm(
            agent,
            production_state,
            structured=structured,
            documents=documents,
            generation_evidence=generation_bundle,
            provenance_evidence=provenance_only,
            retrieval_meta=retrieval_meta,
            request_id=request_id,
            arm_name="arbitrated",
            arbitration_info=arbitration_info,
        )
    arbitrated_latency_ms = (time.perf_counter() - arb_arm_started) * 1000

    category = infer_question_category(production_state, len(documents))
    baseline_cmp, baseline_diag = compare_control_to_arm(
        control,
        baseline,
        question_category=category,
        question_grade=production_state.grade,
        question_subject=production_state.subject,
        question_topic=production_state.topic,
        control_answer=control_answer,
    )
    arb_cmp, arb_diag = compare_control_to_arm(
        control,
        arbitrated,
        question_category=category,
        question_grade=production_state.grade,
        question_subject=production_state.subject,
        question_topic=production_state.topic,
        control_answer=control_answer,
        arbitration=arbitration_info,
    )

    c_ok = bool(control.get("final_accepted"))
    b_ok = bool(baseline.get("final_accepted"))
    a_ok = bool(arbitrated.get("final_accepted"))
    hypothesis = {
        "control_correct_baseline_worse": c_ok and not b_ok,
        "control_correct_arbitrated_worse": c_ok and not a_ok,
        "control_insufficient_baseline_accepted": (not c_ok) and b_ok,
        "control_insufficient_arbitrated_accepted": (not c_ok) and a_ok,
        "baseline_unsupported_claims": len(baseline.get("unsupported_claims") or []),
        "arbitrated_unsupported_claims": len(arbitrated.get("unsupported_claims") or []),
        "generator_document_drift_baseline": baseline_cmp.get("regression_cause")
        == "GENERATOR_DOCUMENT_DRIFT",
        "generator_document_drift_arbitrated": arb_cmp.get("regression_cause")
        == "GENERATOR_DOCUMENT_DRIFT",
    }

    sufficient_cohort = (
        arbitration_info.get("structured_sufficiency")
        == StructuredSufficiency.SUFFICIENT.value
        and int(baseline.get("document_evidence_count") or 0) > 0
    )
    recovery_cohort = (
        arbitration_info.get("structured_sufficiency")
        == StructuredSufficiency.INSUFFICIENT.value
        and arbitration_info.get("document_role") == DocumentRole.DECISIVE.value
    )
    cohorts = {
        "structured_sufficient_with_docs": {
            "in_cohort": sufficient_cohort,
            "control_correct_baseline_same": bool(
                sufficient_cohort and c_ok and b_ok
            ),
            "control_correct_baseline_worse": bool(
                sufficient_cohort and c_ok and not b_ok
            ),
            "control_correct_arbitrated_same": bool(
                sufficient_cohort and c_ok and a_ok
            ),
            "control_correct_arbitrated_worse": bool(
                sufficient_cohort and c_ok and not a_ok
            ),
        },
        "structured_insufficient_decisive_docs": {
            "in_cohort": recovery_cohort,
            "baseline_accepted": bool(recovery_cohort and b_ok),
            "arbitrated_accepted": bool(recovery_cohort and a_ok),
        },
    }

    classification_diagnostics = {
        "decisive_arbitrated_rejection": bool(
            arbitration_info.get("document_role") == DocumentRole.DECISIVE.value
            and not a_ok
            and b_ok
        ),
        "redundant_harmful_inclusion": bool(
            arbitration_info.get("document_role") == DocumentRole.REDUNDANT.value
            and arbitration_info.get("document_use")
            == DocumentUse.INCLUDE_IN_GENERATION.value
            and c_ok
            and not a_ok
        ),
        "irrelevant_generation_influence": bool(
            arbitration_info.get("document_role") == DocumentRole.IRRELEVANT.value
            and arbitration_info.get("document_use")
            == DocumentUse.INCLUDE_IN_GENERATION.value
        ),
        "conflicting_accepted": bool(
            arbitration_info.get("document_role") == DocumentRole.CONFLICTING.value
            and a_ok
        ),
    }

    investigation = None
    if hypothesis["control_correct_arbitrated_worse"]:
        investigation = build_investigation(
            production_state=production_state,
            control=control,
            baseline=_arm_for_comparison(baseline),
            arbitrated=_arm_for_comparison(arbitrated),
            arbitration=arbitration_info,
            cause=str(arb_cmp.get("regression_cause") or "UNKNOWN"),
        )

    corpus_epoch = (
        "post_corpus" if retrieval_meta.get("corpus_available") else "pre_corpus"
    )
    if (
        int(baseline.get("document_evidence_count") or 0) == 0
        and retrieval_meta.get("retrieval_failure_kind") == "no_match"
        and not c_ok
        and not b_ok
        and not a_ok
    ):
        coverage_gap = "CORPUS_COVERAGE_GAP"
    else:
        coverage_gap = None

    baseline_out = _arm_for_comparison(baseline)
    arbitrated_out = _arm_for_comparison(arbitrated)
    baseline_out["latency_ms"] = round(baseline_latency_ms, 3)
    arbitrated_out["latency_ms"] = round(arbitrated_latency_ms, 3)

    record = {
        "experiment": _EXPERIMENT,
        "schema_version": _SCHEMA,
        "phase": "live_arbitration_shadow",
        "batch_id": batch_id,
        "source": "LIVE_TRAFFIC",
        "corpus_epoch": corpus_epoch,
        "request_id": hashlib.sha256((request_id or "").encode()).hexdigest()[:16]
        if request_id
        else "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sampling": {
            "enabled": True,
            "rate": float(getattr(settings, "v213d_shadow_sample_rate", 0.0) or 0.0),
            "sampled": True,
            "v213g_live_arbitration_shadow": True,
            "v213f_document_arbitration_experiment": bool(
                getattr(settings, "v213f_document_arbitration_experiment", False)
            ),
        },
        "infrastructure": {
            **infra_probe,
            **validity,
            "structured_count": len(structured),
            "evidence_status": evidence_status,
        },
        "question": {
            "hash": question_hash(production_state.question),
            "category": category,
            "grade": production_state.grade,
            "subject": production_state.subject,
            "topic": production_state.topic,
        },
        "control": control,
        "retrieval": {
            "variant": retrieval_meta.get("variant"),
            "latency_ms": retrieval_meta.get("latency_ms"),
            "document_evidence_count": len(documents),
            "corpus_available": retrieval_meta.get("corpus_available"),
            "retrieval_failure_kind": retrieval_meta.get("retrieval_failure_kind"),
            "passages": retrieval_meta.get("passages")
            or _document_passage_summaries(documents),
            "provenance_complete": _provenance_complete(documents),
        },
        "arbitration": arbitration_info,
        "baseline_shadow": baseline_out,
        "arbitrated_shadow": arbitrated_out,
        "comparisons": {
            "control_vs_baseline": baseline_cmp,
            "control_vs_arbitrated": arb_cmp,
            "route_transitions": {
                "control_to_baseline": (
                    f"{control.get('final_route')}→{baseline.get('final_route')}"
                ),
                "control_to_arbitrated": (
                    f"{control.get('final_route')}→{arbitrated.get('final_route')}"
                ),
                "baseline_to_arbitrated": (
                    f"{baseline.get('final_route')}→{arbitrated.get('final_route')}"
                ),
            },
        },
        "hypothesis": hypothesis,
        "cohorts": cohorts,
        "classification_diagnostics": classification_diagnostics,
        "diagnostics": {
            "baseline": baseline_diag,
            "arbitrated": arb_diag,
        },
        "safety": {
            "baseline": _safety_false_accepts(baseline_out),
            "arbitrated": _safety_false_accepts(arbitrated_out),
        },
        "latency": {
            "retrieval_ms": round(float(retrieval_meta.get("latency_ms") or 0.0), 3),
            "arbitration_ms": round(arbitration_latency_ms, 3),
            "baseline_shadow_ms": round(baseline_latency_ms, 3),
            "arbitrated_shadow_ms": round(arbitrated_latency_ms, 3),
            "total_ms": round((time.perf_counter() - started) * 1000, 3),
            "structured_api_ms": infra_probe.get("structured_api_latency_ms"),
        },
        "coverage_gap": coverage_gap,
        "investigation": investigation,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    log_agent_event(
        logger,
        "v213g.dual_arm.completed",
        request_id=request_id,
        extra_baseline_worse=hypothesis["control_correct_baseline_worse"],
        extra_arbitrated_worse=hypothesis["control_correct_arbitrated_worse"],
    )
    return record


def baseline_compatible_v213d_record(g_record: dict[str, Any]) -> dict[str, Any]:
    """Project dual-arm result into a V2.13D JSONL row (baseline arm only)."""
    baseline = dict(g_record.get("baseline_shadow") or {})
    comparison = dict((g_record.get("comparisons") or {}).get("control_vs_baseline") or {})
    diagnostics = (g_record.get("diagnostics") or {}).get("baseline")
    return {
        "experiment": "v2.13d",
        "schema_version": "v213d.2",
        "phase": "phase1",
        "observation_phase": "phase1e",
        "corpus_epoch": g_record.get("corpus_epoch"),
        "request_id": g_record.get("request_id"),
        "timestamp": g_record.get("timestamp"),
        "sampling": {
            **(g_record.get("sampling") or {}),
            "v213g_projected_baseline": True,
        },
        "question": g_record.get("question"),
        "control": g_record.get("control"),
        "shadow": baseline,
        "grounding": {
            "metadata_valid": baseline.get("metadata_valid"),
            "provenance_complete": baseline.get("provenance_complete"),
            "wrong_context": baseline.get("wrong_context"),
            "placeholder_evidence": baseline.get("placeholder_evidence"),
            "unsupported_claims": baseline.get("unsupported_claims"),
        },
        "comparison": comparison,
        "diagnostics": diagnostics,
        "latency_ms": (g_record.get("latency") or {}).get("baseline_shadow_ms")
        or g_record.get("latency_ms"),
        "v213g_crossref": True,
    }


def _successful(record: dict[str, Any]) -> bool:
    if record.get("corpus_epoch") == "pre_corpus":
        return False
    infra = record.get("infrastructure") or {}
    if infra.get("validity") == "INFRASTRUCTURE_INVALID":
        return False
    # Historical batch 1 (API unavailable) is preserved but not used for H1.
    if record.get("batch_id") == _BATCH_HISTORICAL:
        return False
    if not record.get("batch_id") and not infra:
        # Untagged historical rows from batch 1: treat as confounded for hypothesis n.
        return False
    b = record.get("baseline_shadow") or {}
    a = record.get("arbitrated_shadow") or {}
    if b.get("error") or a.get("error"):
        return False
    return True


def count_sufficient_with_docs(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count healthy-batch cohort coverage for live validation targets."""
    out = {
        "structured_sufficient_with_docs": 0,
        "structured_sufficient_redundant_docs": 0,
        "structured_sufficient_irrelevant_docs": 0,
        "structured_insufficient_decisive_docs": 0,
        "infrastructure_invalid": 0,
        "historical_batch_1": 0,
        "valid_healthy": 0,
    }
    for r in records:
        if not r.get("batch_id") or r.get("batch_id") == _BATCH_HISTORICAL:
            out["historical_batch_1"] += 1
            continue
        infra = r.get("infrastructure") or {}
        if infra.get("validity") == "INFRASTRUCTURE_INVALID":
            out["infrastructure_invalid"] += 1
            continue
        if (r.get("baseline_shadow") or {}).get("error") or (
            r.get("arbitrated_shadow") or {}
        ).get("error"):
            continue
        out["valid_healthy"] += 1
        arb = r.get("arbitration") or {}
        docs = int(arb.get("document_count") or 0)
        suff = str(arb.get("structured_sufficiency") or "")
        role = str(arb.get("document_role") or "")
        if suff == "SUFFICIENT" and docs > 0:
            out["structured_sufficient_with_docs"] += 1
            if role == "REDUNDANT":
                out["structured_sufficient_redundant_docs"] += 1
            if role == "IRRELEVANT":
                out["structured_sufficient_irrelevant_docs"] += 1
        if suff == "INSUFFICIENT" and role == "DECISIVE":
            out["structured_insufficient_decisive_docs"] += 1
    return out



def aggregate_v213g_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [r for r in records if _successful(r)]
    n = len(successful)
    role_counts: Counter[str] = Counter()
    use_counts: Counter[str] = Counter()
    sufficiency_counts: Counter[str] = Counter()
    baseline_helped = baseline_neutral = baseline_hurt = 0
    arb_helped = arb_neutral = arb_hurt = 0
    baseline_reg = arb_reg = 0
    baseline_rec = arb_rec = 0
    baseline_drift = arb_drift = 0
    baseline_unsup = arb_unsup = 0
    safety_b = Counter()
    safety_a = Counter()
    class_diag = Counter()
    sufficient_cohort = {
        "n": 0,
        "control_correct_baseline_same": 0,
        "control_correct_baseline_worse": 0,
        "control_correct_arbitrated_same": 0,
        "control_correct_arbitrated_worse": 0,
    }
    recovery_cohort = {"n": 0, "baseline_accepted": 0, "arbitrated_accepted": 0}
    lat_ret = lat_arb = lat_b = lat_a = 0.0
    retrieval_success = no_match = tech_fail = 0
    provenance_ok = metadata_ok = 0
    investigations: list[dict[str, Any]] = []

    for r in successful:
        arb = r.get("arbitration") or {}
        role_counts[str(arb.get("document_role") or "UNKNOWN")] += 1
        use_counts[str(arb.get("document_use") or "UNKNOWN")] += 1
        sufficiency_counts[str(arb.get("structured_sufficiency") or "UNKNOWN")] += 1
        hyp = r.get("hypothesis") or {}
        baseline_reg += int(bool(hyp.get("control_correct_baseline_worse")))
        arb_reg += int(bool(hyp.get("control_correct_arbitrated_worse")))
        baseline_rec += int(bool(hyp.get("control_insufficient_baseline_accepted")))
        arb_rec += int(bool(hyp.get("control_insufficient_arbitrated_accepted")))
        baseline_drift += int(bool(hyp.get("generator_document_drift_baseline")))
        arb_drift += int(bool(hyp.get("generator_document_drift_arbitrated")))
        baseline_unsup += int(hyp.get("baseline_unsupported_claims") or 0)
        arb_unsup += int(hyp.get("arbitrated_unsupported_claims") or 0)

        be = ((r.get("comparisons") or {}).get("control_vs_baseline") or {}).get(
            "document_effect"
        )
        ae = ((r.get("comparisons") or {}).get("control_vs_arbitrated") or {}).get(
            "document_effect"
        )
        if be == "helped":
            baseline_helped += 1
        elif be == "hurt":
            baseline_hurt += 1
        else:
            baseline_neutral += 1
        if ae == "helped":
            arb_helped += 1
        elif ae == "hurt":
            arb_hurt += 1
        else:
            arb_neutral += 1

        for k, v in ((r.get("safety") or {}).get("baseline") or {}).items():
            safety_b[k] += int(v or 0)
        for k, v in ((r.get("safety") or {}).get("arbitrated") or {}).items():
            safety_a[k] += int(v or 0)
        for k, v in (r.get("classification_diagnostics") or {}).items():
            if v:
                class_diag[k] += 1

        sc = (r.get("cohorts") or {}).get("structured_sufficient_with_docs") or {}
        if sc.get("in_cohort"):
            sufficient_cohort["n"] += 1
            for key in (
                "control_correct_baseline_same",
                "control_correct_baseline_worse",
                "control_correct_arbitrated_same",
                "control_correct_arbitrated_worse",
            ):
                sufficient_cohort[key] += int(bool(sc.get(key)))
        rc = (r.get("cohorts") or {}).get("structured_insufficient_decisive_docs") or {}
        if rc.get("in_cohort"):
            recovery_cohort["n"] += 1
            recovery_cohort["baseline_accepted"] += int(bool(rc.get("baseline_accepted")))
            recovery_cohort["arbitrated_accepted"] += int(
                bool(rc.get("arbitrated_accepted"))
            )

        lat = r.get("latency") or {}
        lat_ret += float(lat.get("retrieval_ms") or 0.0)
        lat_arb += float(lat.get("arbitration_ms") or 0.0)
        lat_b += float(lat.get("baseline_shadow_ms") or 0.0)
        lat_a += float(lat.get("arbitrated_shadow_ms") or 0.0)

        ret = r.get("retrieval") or {}
        if int(ret.get("document_evidence_count") or 0) > 0:
            retrieval_success += 1
        elif ret.get("retrieval_failure_kind") == "no_match":
            no_match += 1
        elif ret.get("retrieval_failure_kind"):
            tech_fail += 1
        if ret.get("provenance_complete"):
            provenance_ok += 1
        if (r.get("baseline_shadow") or {}).get("metadata_valid"):
            metadata_ok += 1
        if r.get("investigation"):
            investigations.append(r["investigation"])

    safety_blocked = any(int(v) > 0 for v in safety_a.values()) or any(
        int(v) > 0 for v in safety_b.values()
    )
    sufficient_with_docs = int(sufficient_cohort.get("n") or 0)
    # Prefer explicit healthy-batch cohort counter when available.
    coverage = count_sufficient_with_docs(records)
    if coverage.get("structured_sufficient_with_docs") is not None:
        sufficient_with_docs = int(coverage["structured_sufficient_with_docs"])
    status = decide_v213g_status(
        n=n,
        baseline_reg=baseline_reg,
        arb_reg=arb_reg,
        baseline_rec=baseline_rec,
        arb_rec=arb_rec,
        safety_blocked=safety_blocked,
        class_diag=dict(class_diag),
        structured_sufficient_with_docs=sufficient_with_docs,
    )

    def mean(total: float) -> float:
        return round(total / max(n, 1), 3)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": _EXPERIMENT,
        "schema_version": _SCHEMA,
        "n_total": len(records),
        "n_successful": n,
        "milestones": {
            "first": MILESTONE_FIRST,
            "stronger": MILESTONE_STRONG,
            "preferred": MILESTONE_PREFERRED,
            "reached_first": n >= MILESTONE_FIRST,
            "reached_stronger": n >= MILESTONE_STRONG,
            "reached_preferred": n >= MILESTONE_PREFERRED,
        },
        "retrieval": {
            "retrieval_success": retrieval_success,
            "no_match": no_match,
            "technical_retrieval_failure": tech_fail,
            "provenance_complete": provenance_ok,
            "metadata_valid": metadata_ok,
            "retrieval_success_rate": round(retrieval_success / max(n, 1), 4),
        },
        "arbitration": {
            "sufficiency_counts": dict(sufficiency_counts),
            "role_counts": dict(role_counts),
            "use_counts": dict(use_counts),
        },
        "document_value": {
            "baseline": {
                "helped": baseline_helped,
                "neutral": baseline_neutral,
                "hurt": baseline_hurt,
            },
            "arbitrated": {
                "helped": arb_helped,
                "neutral": arb_neutral,
                "hurt": arb_hurt,
            },
        },
        "recovery": {
            "baseline_recovery": baseline_rec,
            "arbitrated_recovery": arb_rec,
        },
        "regression": {
            "baseline_regression": baseline_reg,
            "arbitrated_regression": arb_reg,
        },
        "generator_drift": {
            "baseline_drift": baseline_drift,
            "arbitrated_drift": arb_drift,
            "baseline_unsupported_claims": baseline_unsup,
            "arbitrated_unsupported_claims": arb_unsup,
        },
        "safety": {
            "baseline": dict(safety_b),
            "arbitrated": dict(safety_a),
            "safety_blocked": safety_blocked,
        },
        "cohorts": {
            "structured_sufficient_with_docs": sufficient_cohort,
            "structured_insufficient_decisive_docs": recovery_cohort,
        },
        "healthy_batch_coverage": count_sufficient_with_docs(records),
        "sufficient_with_docs_target": SUFFICIENT_WITH_DOCS_TARGET,
        "classification_diagnostics": dict(class_diag),
        "latency": {
            "mean_retrieval_ms": mean(lat_ret),
            "mean_arbitration_ms": mean(lat_arb),
            "mean_baseline_shadow_ms": mean(lat_b),
            "mean_arbitrated_shadow_ms": mean(lat_a),
        },
        "investigations": investigations[:50],
        "comparison_table": {
            "recoveries": {"baseline": baseline_rec, "arbitrated": arb_rec},
            "neutral": {"baseline": baseline_neutral, "arbitrated": arb_neutral},
            "hurt": {"baseline": baseline_hurt, "arbitrated": arb_hurt},
            "regressions": {"baseline": baseline_reg, "arbitrated": arb_reg},
            "generator_drift": {"baseline": baseline_drift, "arbitrated": arb_drift},
            "unsupported_claims": {
                "baseline": baseline_unsup,
                "arbitrated": arb_unsup,
            },
            "wrong_context_false_accepts": {
                "baseline": int(safety_b.get("wrong_context_false_acceptance") or 0),
                "arbitrated": int(safety_a.get("wrong_context_false_acceptance") or 0),
            },
            "placeholder_false_accepts": {
                "baseline": int(safety_b.get("placeholder_false_acceptance") or 0),
                "arbitrated": int(safety_a.get("placeholder_false_acceptance") or 0),
            },
            "metadata_false_accepts": {
                "baseline": int(safety_b.get("metadata_false_acceptance") or 0),
                "arbitrated": int(safety_a.get("metadata_false_acceptance") or 0),
            },
        },
        "status": status["status"],
        "recommendation_text": status["recommendation_text"],
        "production_isolation": {
            "v213f_document_arbitration_experiment": False,
            "v213e": "disabled",
            "v213g_live_only": True,
            "note": "Production answers remain control; arbitration never user-facing.",
        },
        "batch_boundaries": {
            "batch_1_api_unavailable": {
                "n": coverage.get("historical_batch_1", 0),
                "validity": "INFRASTRUCTURE_INVALID",
                "confounder": (
                    "Curriculum Structure API unavailable during collection "
                    "→ 52/52 structured-insufficient; H1 untestable"
                ),
                "preserved_summary": str(
                    Path("data/diagnostics/v213g_arbitration_summary_batch1_api_unavailable.json")
                ),
            },
            "batch_2_healthy_structured_api": {
                "n_valid": coverage.get("valid_healthy", 0),
                "structured_sufficient_with_docs": coverage.get(
                    "structured_sufficient_with_docs", 0
                ),
                "structured_insufficient_decisive_docs": coverage.get(
                    "structured_insufficient_decisive_docs", 0
                ),
                "infrastructure_invalid": coverage.get("infrastructure_invalid", 0),
                "target_structured_sufficient_with_docs": SUFFICIENT_WITH_DOCS_TARGET,
            },
        },
        "historical_batch_1_prior": {
            "n": 52,
            "source": "LIVE_TRAFFIC",
            "validity": "INFRASTRUCTURE_INVALID",
            "recovery": {"baseline_recovery": 30, "arbitrated_recovery": 36},
            "unsupported_claims": {"baseline": 30, "arbitrated": 17},
            "structured_sufficient_with_docs": 0,
            "note": (
                "Historical prior only; excluded from healthy-batch hypothesis n. "
                "JSONL rows preserved unchanged."
            ),
        },
        "preserved_priors": {
            "v213f_freeze_jsonl_sha256_prefix": "4dcbca7a46ca11e6",
            "v213f_decision": "ARBITRATION_SUPPORTED",
            "note": "V2.13D/F datasets are not overwritten by V2.13G.",
        },
    }


def decide_v213g_status(
    *,
    n: int,
    baseline_reg: int,
    arb_reg: int,
    baseline_rec: int,
    arb_rec: int,
    safety_blocked: bool,
    class_diag: dict[str, int],
    structured_sufficient_with_docs: int | None = None,
) -> dict[str, str]:
    if safety_blocked:
        return {
            "status": "SAFETY_BLOCKED",
            "recommendation_text": "Hard safety gate non-zero on baseline or arbitrated arm. Stop.",
        }
    # Do not treat zero regressions as confirmation or rejection when the
    # primary V2.13F sufficient+doc cohort was never observed live.
    if structured_sufficient_with_docs is not None and structured_sufficient_with_docs <= 0:
        return {
            "status": "INVESTIGATE_BEFORE_PROMOTION",
            "recommendation_text": (
                "PRIMARY_REGRESSION_HYPOTHESIS_UNTESTED: live sample has "
                "structured_sufficient_with_docs=0. Do not conclude "
                "ARBITRATION_NOT_CONFIRMED solely from zero regressions. "
                "Obtain targeted/live coverage of sufficient+redundant cases "
                "before promotion decisions."
            ),
        }
    if n < MILESTONE_FIRST:
        if (
            structured_sufficient_with_docs is not None
            and structured_sufficient_with_docs >= SUFFICIENT_WITH_DOCS_TARGET
        ):
            return {
                "status": "INSUFFICIENT_SAMPLE",
                "recommendation_text": (
                    f"Stage-1 sufficient_with_docs target met "
                    f"({structured_sufficient_with_docs}>="
                    f"{SUFFICIENT_WITH_DOCS_TARGET}); continue healthy live "
                    f"collection toward n>={MILESTONE_FIRST} (have {n}). "
                    "Keep sample_rate=0.01; do not raise sampling; "
                    "do not enable production arbitration."
                ),
            }
        return {
            "status": "INSUFFICIENT_SAMPLE",
            "recommendation_text": (
                f"Need ≥{MILESTONE_FIRST} successful dual-arm comparisons "
                f"(have {n}) and ≥{SUFFICIENT_WITH_DOCS_TARGET} "
                f"structured_sufficient_with_docs "
                f"(have {structured_sufficient_with_docs}). "
                "Keep sample_rate=0.01; do not raise sampling."
            ),
        }
    mis = int(class_diag.get("decisive_arbitrated_rejection") or 0) + int(
        class_diag.get("redundant_harmful_inclusion") or 0
    ) + int(class_diag.get("irrelevant_generation_influence") or 0) + int(
        class_diag.get("conflicting_accepted") or 0
    )
    if mis > 0 and arb_reg > 0:
        return {
            "status": "INVESTIGATE_BEFORE_PROMOTION",
            "recommendation_text": (
                "Diagnostic misclassification patterns present with residual "
                "arbitrated regressions — investigate before any canary."
            ),
        }
    rec_ok = arb_rec >= max(0, baseline_rec - max(1, baseline_rec // 10))
    reg_improved = arb_reg < baseline_reg
    if reg_improved and rec_ok and arb_reg == 0 and mis == 0:
        return {
            "status": "ARBITRATION_CONFIRMED",
            "recommendation_text": (
                "Live traffic confirms arbitration reduces regressions while "
                "preserving recoveries; safety gates remain zero. Still do not "
                "enable production arbitration automatically."
            ),
        }
    if reg_improved and rec_ok:
        return {
            "status": "ARBITRATION_PARTIALLY_CONFIRMED",
            "recommendation_text": (
                "Live traffic shows regression improvement with recoveries "
                "preserved, but residual risk or diagnostics remain."
            ),
        }
    if not rec_ok:
        return {
            "status": "ARBITRATION_NOT_CONFIRMED",
            "recommendation_text": (
                "Arbitrated recoveries fell materially vs baseline — fails "
                "structured-first (not structured-only) requirement."
            ),
        }
    if baseline_reg == 0 and arb_reg == 0 and rec_ok:
        return {
            "status": "ARBITRATION_PARTIALLY_CONFIRMED",
            "recommendation_text": (
                "No live regressions on either arm and recoveries preserved; "
                "continue until structured-sufficient-with-docs coverage is adequate."
            ),
        }
    return {
        "status": "ARBITRATION_NOT_CONFIRMED",
        "recommendation_text": (
            "No meaningful live reduction in document-induced regressions."
        ),
    }


def write_v213g_report(summary: dict[str, Any], *, out_md: Path) -> None:
    table = summary.get("comparison_table") or {}
    b = lambda key: (table.get(key) or {}).get("baseline", "")
    a = lambda key: (table.get(key) or {}).get("arbitrated", "")
    lines = [
        "# V2.13G Live Document Arbitration Shadow",
        "",
        f"Generated: `{summary.get('generated_at')}`",
        "",
        f"**Status: `{summary.get('status')}`**",
        "",
        summary.get("recommendation_text", ""),
        "",
        "## Objective",
        "",
        "Validate the V2.13F deterministic document-arbitration policy against",
        "fresh real QA traffic before any production promotion.",
        "",
        "Production answers remain the control. V2.13F production arbitration",
        "remains disabled. V2.13E remains disabled.",
        "",
        "## Sample",
        "",
        f"- Successful dual-arm comparisons: **{summary.get('n_successful')}**",
        f"- Total rows: **{summary.get('n_total')}**",
        f"- Milestones: `{json.dumps(summary.get('milestones') or {})}`",
        "",
        "## Batch boundaries",
        "",
        "```json",
        json.dumps(summary.get("batch_boundaries") or {}, indent=2),
        "```",
        "",
        "### Historical confounder (batch 1)",
        "",
        "```json",
        json.dumps(summary.get("historical_batch_1_prior") or {}, indent=2),
        "```",
        "",
        "### Healthy-batch coverage",
        "",
        "```json",
        json.dumps(summary.get("healthy_batch_coverage") or {}, indent=2),
        "```",
        "",
        "## Comparison table",
        "",
        "| Metric                      | V2.13D baseline | V2.13F arbitrated |",
        "| --------------------------- | --------------: | ----------------: |",
        f"| Recoveries                  | {b('recoveries'):>16} | {a('recoveries'):>17} |",
        f"| Neutral                     | {b('neutral'):>16} | {a('neutral'):>17} |",
        f"| Hurt                        | {b('hurt'):>16} | {a('hurt'):>17} |",
        f"| Regressions                 | {b('regressions'):>16} | {a('regressions'):>17} |",
        f"| Generator drift             | {b('generator_drift'):>16} | {a('generator_drift'):>17} |",
        f"| Unsupported claims          | {b('unsupported_claims'):>16} | {a('unsupported_claims'):>17} |",
        f"| Wrong-context false accepts | {b('wrong_context_false_accepts'):>16} | {a('wrong_context_false_accepts'):>17} |",
        f"| Placeholder false accepts   | {b('placeholder_false_accepts'):>16} | {a('placeholder_false_accepts'):>17} |",
        f"| Metadata false accepts      | {b('metadata_false_accepts'):>16} | {a('metadata_false_accepts'):>17} |",
        "",
        "## Arbitration distributions",
        "",
        "```json",
        json.dumps(summary.get("arbitration") or {}, indent=2),
        "```",
        "",
        "## Cohorts",
        "",
        "```json",
        json.dumps(summary.get("cohorts") or {}, indent=2),
        "```",
        "",
        "## Classification diagnostics",
        "",
        "```json",
        json.dumps(summary.get("classification_diagnostics") or {}, indent=2),
        "```",
        "",
        "## Latency",
        "",
        "```json",
        json.dumps(summary.get("latency") or {}, indent=2),
        "```",
        "",
        "## Safety",
        "",
        "```json",
        json.dumps(summary.get("safety") or {}, indent=2),
        "```",
        "",
        "## Production isolation",
        "",
        "```json",
        json.dumps(summary.get("production_isolation") or {}, indent=2),
        "```",
        "",
        "## Preserved V2.13D/F artifacts",
        "",
        "```json",
        json.dumps(summary.get("preserved_priors") or {}, indent=2),
        "```",
        "",
        "Do not enable production arbitration based on this report alone.",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "SUFFICIENT_WITH_DOCS_TARGET",
    "aggregate_v213g_records",
    "baseline_compatible_v213d_record",
    "classify_row_validity",
    "count_sufficient_with_docs",
    "decide_v213g_status",
    "execute_shadow_arm",
    "load_v213g_records",
    "persist_v213g_record",
    "probe_structured_api",
    "run_dual_arm_shadow_pipeline",
    "v213g_enabled",
    "v213g_jsonl_path",
    "write_v213g_report",
]
