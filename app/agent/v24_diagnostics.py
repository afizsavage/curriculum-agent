"""V2.4 routing / verifier isolation experiment helpers."""

from __future__ import annotations

import re
from typing import Any, Literal

from app.agent.context_boundary import (
    context_boundary_experiment_enabled,
    get_boundary,
    missing_evidence_covered_by_boundary,
)
from app.agent.evidence_snapshot import evidence_snapshot_hash, record_evidence_snapshot
from app.agent.state import CurriculumQAState
from app.agent.v23_diagnostics import evidence_already_present_for_rejection
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence
from app.schemas.verification import VerificationResult

V24Arm = Literal["A", "B", "C", "D"]

_GARBLED_LO_CODES = frozenset({"C4U06-LO02", "C4U04-LO04"})


def v24_experiment_enabled(settings: Settings, qa: CurriculumQAState) -> bool:
    if qa.metadata.get("v24_experiment_arm"):
        return True
    return bool(getattr(settings, "v24_routing_verifier_experiment", False))


def configure_v24_experiment(
    qa: CurriculumQAState,
    *,
    settings: Settings,
    arm: V24Arm | None = None,
    enabled: bool | None = None,
) -> None:
    """Configure V2.4 arm flags without changing production defaults globally."""
    if arm is not None:
        qa.metadata["v24_experiment_arm"] = arm
    if enabled is not None:
        qa.metadata["v24_routing_verifier_experiment"] = enabled
    elif v24_experiment_enabled(settings, qa):
        qa.metadata["v24_routing_verifier_experiment"] = True

    if not qa.metadata.get("v24_experiment_arm"):
        return

    qa.metadata["context_boundary_experiment"] = True
    qa.metadata["generation_mode"] = "constrained"

    arm_code = str(qa.metadata["v24_experiment_arm"]).upper()
    if arm_code in {"A", "B"}:
        qa.metadata["v24_frozen_retrieval"] = True
        qa.metadata["v23_frozen_retrieval"] = True
        qa.metadata["v23_diagnostic_experiment"] = True  # reuse frozen resolve path
    if arm_code in {"A", "C"}:
        qa.metadata["v24_single_pass"] = True
        qa.metadata["v23_single_pass"] = True
    if arm_code in {"B", "D"}:
        qa.metadata["v24_single_pass"] = False
        qa.metadata["v23_single_pass"] = False


def frozen_retrieval_enabled(settings: Settings, qa: CurriculumQAState) -> bool:
    if qa.metadata.get("v24_frozen_retrieval"):
        return True
    if qa.metadata.get("v23_frozen_retrieval"):
        return True
    from app.agent.evidence_snapshot import v23_diagnostic_enabled

    return v23_diagnostic_enabled(settings, qa)


def single_pass_enabled(qa: CurriculumQAState) -> bool:
    if qa.metadata.get("v24_single_pass") is False:
        return False
    if qa.metadata.get("v24_single_pass"):
        return True
    return bool(qa.metadata.get("v23_single_pass"))


def evidence_record_from_state(qa: CurriculumQAState) -> dict[str, Any]:
    boundary = get_boundary(qa.retrieval_state)
    lo_ids: list[str] = []
    lo_codes: list[str] = []
    unit_ids: list[str] = []
    unit_codes: list[str] = []
    for item in qa.evidence:
        et = (item.entity_type or "").lower()
        code = (item.metadata or {}).get("code")
        if et == "learning_outcome" and item.entity_id:
            lo_ids.append(item.entity_id)
            if code:
                lo_codes.append(str(code))
            elif item.name:
                lo_codes.append(str(item.name))
        if et in {"unit", "topic", "curriculum_content"} and item.entity_id:
            unit_ids.append(item.entity_id)
            if code:
                unit_codes.append(str(code))
    if boundary:
        lo_ids = list(boundary.learning_outcome_ids or lo_ids)
        lo_codes = list(boundary.lo_codes or lo_codes)
        unit_ids = list(boundary.unit_ids or unit_ids)
        unit_codes = list(boundary.unit_codes or unit_codes)
    snap_hash = qa.metadata.get("evidence_snapshot_hash") or evidence_snapshot_hash(
        qa.evidence
    )
    return {
        "evidence_snapshot_hash": snap_hash,
        "evidence_count": len(qa.evidence),
        "learning_outcome_count": len(lo_ids),
        "learning_outcome_ids": sorted(set(lo_ids)),
        "learning_outcome_codes": sorted(set(lo_codes)),
        "unit_ids": sorted(set(unit_ids)),
        "unit_codes": sorted(set(unit_codes)),
    }


def classify_evidence_presence(
    qa: CurriculumQAState,
    *,
    result: VerificationResult | None,
) -> str:
    """EVIDENCE_MISSING | EVIDENCE_PRESENT_BUT_IMPERFECT | EVIDENCE_PRESENT_AND_SUFFICIENT"""
    if result is None:
        return "EVIDENCE_MISSING"
    if result.passed:
        return "EVIDENCE_PRESENT_AND_SUFFICIENT"
    if evidence_already_present_for_rejection(qa, result):
        answer = (qa.final_answer or qa.draft_answer or "").lower()
        if any(code.lower() in answer for code in _GARBLED_LO_CODES):
            return "EVIDENCE_PRESENT_BUT_IMPERFECT"
        if result.unsupported_claims:
            return "EVIDENCE_PRESENT_BUT_IMPERFECT"
        return "EVIDENCE_PRESENT_BUT_IMPERFECT"
    if result.missing_evidence:
        return "EVIDENCE_MISSING"
    return "EVIDENCE_PRESENT_BUT_IMPERFECT"


def classify_verifier_issue(
    qa: CurriculumQAState,
    *,
    result: VerificationResult | None,
) -> str:
    if result is None:
        return "OTHER"
    answer = qa.final_answer or qa.draft_answer or ""
    issues_text = " ".join(result.issues or []).lower()
    if result.unsupported_claims:
        if any(code.lower() in issues_text or code.lower() in answer.lower() for code in _GARBLED_LO_CODES):
            return "TRUNCATED_SOURCE"
        return "UNSUPPORTED_CLAIM"
    if result.missing_evidence:
        if evidence_already_present_for_rejection(qa, result):
            return "GROUNDING_FAILURE"
        return "MISSING_EVIDENCE"
    if re.search(r"\bno\s+learning\s+outcomes?\b|\bdoes not include any\b", answer, re.I):
        return "ABSENCE_CLAIM"
    if not result.passed:
        return "GROUNDING_FAILURE"
    return "OTHER"


def classify_retrieval_delta(
    *,
    evidence_before_count: int,
    evidence_after_count: int,
    new_evidence_count: int,
    duplicate_evidence_count: int,
    new_evidence_ids: list[str],
) -> str:
    if new_evidence_count <= 0:
        if duplicate_evidence_count > 0:
            return "DUPLICATE_ONLY"
        return "NO_NEW_EVIDENCE"
    if not new_evidence_ids:
        return "NO_NEW_EVIDENCE"
    return "NEW_RELEVANT_EVIDENCE"


def record_retrieval_delta(
    qa: CurriculumQAState,
    *,
    evidence_before_count: int,
    evidence_hash_before: str,
    new_evidence_count: int,
    duplicate_evidence_count: int,
    new_evidence_ids: list[str],
    legacy_retrieval_attempted: bool = False,
) -> dict[str, Any]:
    evidence_after_count = len(qa.evidence)
    evidence_hash_after = evidence_snapshot_hash(qa.evidence)
    delta = {
        "evidence_before_count": evidence_before_count,
        "evidence_after_count": evidence_after_count,
        "new_evidence_count": new_evidence_count,
        "new_evidence_ids": list(new_evidence_ids),
        "duplicate_evidence_count": duplicate_evidence_count,
        "evidence_hash_before": evidence_hash_before,
        "evidence_hash_after": evidence_hash_after,
        "evidence_changed": evidence_hash_before != evidence_hash_after,
        "legacy_retrieval_attempted": legacy_retrieval_attempted,
        "retrieval_result_class": classify_retrieval_delta(
            evidence_before_count=evidence_before_count,
            evidence_after_count=evidence_after_count,
            new_evidence_count=new_evidence_count,
            duplicate_evidence_count=duplicate_evidence_count,
            new_evidence_ids=new_evidence_ids,
        ),
    }
    events = qa.metadata.setdefault("v24_retrieval_deltas", [])
    events.append(delta)
    return delta


def build_routing_sequence(qa: CurriculumQAState) -> dict[str, Any]:
    result = qa.verification
    verification_history = qa.verification_history or []
    first = verification_history[0] if verification_history else result
    last = verification_history[-1] if verification_history else result

    routes = list(qa.metadata.get("v24_route_events") or [])
    retrieve_more_requested = bool(
        first and first.recommendation.value == "retrieve_more"
    )
    regeneration_executed = "generate_answer" in list(
        qa.metadata.get("visited_nodes") or []
    ) and qa.verification_attempts > 1
    regeneration_requested = any(
        r.get("decision") == "regenerate" for r in routes
    )

    boundary = get_boundary(qa.retrieval_state)
    context_boundary_covered = bool(
        qa.metadata.get("evidence_already_present")
        or (
            result
            and missing_evidence_covered_by_boundary(
                list(result.missing_evidence or []),
                boundary=boundary,
                evidence=qa.evidence,
            )
        )
    )

    retrieval_deltas = qa.metadata.get("v24_retrieval_deltas") or []
    post_verify_deltas = retrieval_deltas  # only follow-up rounds are recorded
    last_delta = post_verify_deltas[-1] if post_verify_deltas else {}
    legacy_attempted = bool(
        last_delta.get("legacy_retrieval_attempted")
        or qa.metadata.get("v24_legacy_retrieval_attempted")
    )

    final_decision = (
        last.recommendation.value if last and last.passed else None
    )
    if qa.status.value == "completed":
        final_decision = final_decision or "accept"
    elif result and not result.passed:
        final_decision = result.recommendation.value

    seq = {
        "verifier_score": first.score if first else None,
        "verifier_decision": first.recommendation.value if first else None,
        "retrieve_more_requested": retrieve_more_requested,
        "evidence_already_present": evidence_already_present_for_rejection(
            qa, first
        ),
        "context_boundary_covered": context_boundary_covered,
        "legacy_retrieval_attempted": legacy_attempted,
        "new_evidence_count": last_delta.get("new_evidence_count", 0),
        "evidence_changed": last_delta.get("evidence_changed", False),
        "regeneration_requested": regeneration_requested,
        "regeneration_executed": regeneration_executed,
        "final_verifier_score": last.score if last else None,
        "final_decision": final_decision,
        "final_failure_reason": qa.metadata.get("termination_reason")
        or qa.metadata.get("fallback_reason"),
        "evidence_presence_class": classify_evidence_presence(qa, result=first),
        "verifier_issue_class": classify_verifier_issue(qa, result=first),
        "terminal_status": qa.status.value,
        "production_routing_intervention": bool(
            retrieve_more_requested
            and (
                regeneration_requested
                or legacy_attempted
                or last_delta.get("retrieval_result_class")
                in {"NO_NEW_EVIDENCE", "DUPLICATE_ONLY"}
                or qa.status.value == "insufficient_evidence"
            )
        ),
    }
    qa.metadata["v24_routing_sequence"] = seq
    return seq


def emit_v24_route_event(
    qa: CurriculumQAState,
    *,
    route: str,
    reason: str | None,
    result: VerificationResult | None,
) -> None:
    event = {
        "decision": route,
        "reason": reason,
        "verifier_score": result.score if result else None,
        "verifier_decision": result.recommendation.value if result else None,
        "retrieve_more_requested": bool(
            result and result.recommendation.value == "retrieve_more"
        ),
        "evidence_already_present": evidence_already_present_for_rejection(
            qa, result
        ),
        "context_boundary_covered": bool(qa.metadata.get("evidence_already_present")),
        "evidence_presence_class": classify_evidence_presence(qa, result=result),
        "verifier_issue_class": classify_verifier_issue(qa, result=result),
    }
    events = qa.metadata.setdefault("v24_route_events", [])
    events.append(event)

    from app.agent.trace import get_current_trace

    trace = get_current_trace()
    if trace is not None:
        trace.emit("agent.v24.routing", arm=qa.metadata.get("v24_experiment_arm"), **event)


def finalize_v24_diagnostics(qa: CurriculumQAState) -> dict[str, Any]:
    if not qa.metadata.get("v24_experiment_arm"):
        return {}
    if not qa.metadata.get("evidence_snapshot_hash") and qa.evidence:
        record_evidence_snapshot(qa)
    evidence = evidence_record_from_state(qa)
    routing = build_routing_sequence(qa)
    diag = {
        "arm": qa.metadata.get("v24_experiment_arm"),
        "evidence": evidence,
        "routing": routing,
        "route_events": list(qa.metadata.get("v24_route_events") or []),
        "retrieval_deltas": list(qa.metadata.get("v24_retrieval_deltas") or []),
    }
    qa.metadata["v24_diagnostics"] = diag
    from app.agent.trace import get_current_trace

    trace = get_current_trace()
    if trace is not None:
        trace.emit("agent.v24.diagnostics", **diag)
    return diag
