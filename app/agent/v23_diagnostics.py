"""V2.3 generation/verifier diagnostic analysis helpers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from app.agent.context_boundary import missing_evidence_covered_by_boundary, get_boundary
from app.agent.evidence_snapshot import v23_diagnostic_enabled
from app.agent.state import CurriculumQAState
from app.config import Settings
from app.schemas.verification import ClaimVerdict, VerificationResult

_SPECULATIVE_RE = re.compile(
    r"\blikely\b|\bprobably\b|\bmight\b|\bperhaps\b|"
    r"\bthis means\b|\bthe curriculum appears to\b|\bappears to mean\b",
    re.I,
)
_TRUNCATION_MISHANDLE_RE = re.compile(
    r"\blikely means\b|\bprobably means\b|\bcan be inferred\b|\bimplies that\b",
    re.I,
)


def configure_v23_experiment(
    qa: CurriculumQAState,
    *,
    settings: Settings,
    generation_mode: str | None = None,
    enabled: bool | None = None,
) -> None:
    """Apply V2.3 experiment flags (isolated; default off)."""
    if enabled is not None:
        qa.metadata["v23_diagnostic_experiment"] = enabled
    elif v23_diagnostic_enabled(settings, qa):
        qa.metadata["v23_diagnostic_experiment"] = True

    if qa.metadata.get("v23_diagnostic_experiment"):
        qa.metadata["context_boundary_experiment"] = True
        qa.metadata["v23_frozen_retrieval"] = True
        qa.metadata["v23_single_pass"] = True
    if generation_mode in {"current", "constrained"}:
        qa.metadata["generation_mode"] = generation_mode


def classify_claim_grounding(verdict: ClaimVerdict) -> str:
    if verdict == ClaimVerdict.SUPPORTED:
        return "SUPPORTED"
    if verdict == ClaimVerdict.MISSING:
        return "PARTIALLY_SUPPORTED"
    if verdict == ClaimVerdict.UNSUPPORTED:
        return "UNSUPPORTED"
    if verdict == ClaimVerdict.CONTRADICTED:
        return "UNSUPPORTED"
    return "OTHER"


def detect_speculative_answer(answer: str) -> bool:
    return bool(_SPECULATIVE_RE.search(answer or ""))


def detect_truncation_mishandling(answer: str) -> bool:
    return bool(_TRUNCATION_MISHANDLE_RE.search(answer or ""))


def verifier_input_hash(state: CurriculumQAState) -> str:
    answer = state.final_answer or state.draft_answer or ""
    evidence_ids = sorted(e.entity_id for e in state.evidence if e.entity_id)
    payload = json.dumps(
        {"answer": answer[:4000], "evidence_ids": evidence_ids},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def evidence_already_present_for_rejection(
    qa: CurriculumQAState,
    result: VerificationResult | None,
) -> bool:
    if result is None:
        return False
    boundary = get_boundary(qa.retrieval_state)
    pending = list(result.missing_evidence or qa.pending_missing_evidence or [])
    if not pending:
        return bool(boundary and boundary.learning_outcome_ids)
    return missing_evidence_covered_by_boundary(
        pending,
        boundary=boundary,
        evidence=qa.evidence,
    )


def classify_v23_failure(
    qa: CurriculumQAState,
    *,
    result: VerificationResult | None,
) -> str:
    answer = qa.final_answer or qa.draft_answer or ""
    if detect_speculative_answer(answer):
        return "GENERATION_SPECULATION"
    if detect_truncation_mishandling(answer):
        return "GENERATION_TRUNCATION_ERROR"
    if result is None:
        return "OTHER"
    if result.unsupported_claims:
        return "GENERATION_UNSUPPORTED_CLAIM"
    if result.missing_evidence:
        if evidence_already_present_for_rejection(qa, result):
            return "VERIFIER_MISSING_EVIDENCE"
        return "VERIFIER_GROUNDING_FAILURE"
    if not result.passed and result.score < 0.7:
        return "VERIFIER_SCORE_THRESHOLD"
    if not result.passed:
        return "VERIFIER_CRITERIA_MISMATCH"
    return "OTHER"


def build_generation_diagnostics(
    qa: CurriculumQAState,
    *,
    generation_latency_ms: float | None = None,
) -> dict[str, Any]:
    answer = qa.final_answer or qa.draft_answer or ""
    return {
        "generation_mode": qa.metadata.get("generation_mode", "current"),
        "evidence_snapshot_hash": qa.metadata.get("evidence_snapshot_hash"),
        "generation_evidence_ids": qa.metadata.get("generation_evidence_ids") or [],
        "generation_evidence_count": qa.metadata.get("generation_evidence_count"),
        "answer_length": len(answer),
        "generation_confidence": (
            qa.answer_confidence.value if qa.answer_confidence else None
        ),
        "generation_latency_ms": generation_latency_ms
        or qa.metadata.get("generation_latency_ms"),
        "speculative_wording": detect_speculative_answer(answer),
        "truncation_mishandling": detect_truncation_mishandling(answer),
    }


def build_verification_diagnostics(
    qa: CurriculumQAState,
    *,
    result: VerificationResult | None,
    verification_latency_ms: float | None = None,
) -> dict[str, Any]:
    claims = []
    if result and result.claims:
        for c in result.claims:
            claims.append(
                {
                    "claim": c.claim,
                    "verdict": c.verdict.value,
                    "grounding_class": classify_claim_grounding(c.verdict),
                    "evidence_ids": list(c.evidence_ids or []),
                }
            )
    overlap = qa.metadata.get("generation_to_verifier_evidence_overlap")
    if overlap is None:
        from app.agent.evidence_snapshot import generation_to_verifier_overlap

        overlap = generation_to_verifier_overlap(qa)
    return {
        "verifier_input_hash": verifier_input_hash(qa),
        "verifier_score": result.score if result else None,
        "verifier_decision": (
            result.recommendation.value if result else None
        ),
        "verifier_passed": result.passed if result else None,
        "verifier_issues": list(result.issues) if result else [],
        "missing_evidence": [
            m.model_dump() if hasattr(m, "model_dump") else m
            for m in (result.missing_evidence if result else [])
        ],
        "unsupported_claims": list(result.unsupported_claims) if result else [],
        "truncation_flags": detect_truncation_mishandling(
            qa.final_answer or qa.draft_answer or ""
        ),
        "verifier_latency_ms": verification_latency_ms
        or (result.metadata or {}).get("verification_latency_ms")
        if result
        else None,
        "claim_grounding": claims,
        "evidence_already_present": evidence_already_present_for_rejection(
            qa, result
        ),
        "generation_to_verifier_evidence_overlap": overlap,
        "v23_failure_class": classify_v23_failure(qa, result=result)
        if result and not result.passed
        else None,
    }
