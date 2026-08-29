"""V2.5 verifier / imperfect-evidence quality experiment helpers."""

from __future__ import annotations

import copy
from typing import Any, Literal

from app.agent.answer_generator import _looks_garbled_source_text
from app.agent.evidence_snapshot import evidence_snapshot_hash, record_evidence_snapshot
from app.agent.state import CurriculumQAState
from app.agent.v23_diagnostics import evidence_already_present_for_rejection
from app.agent.v24_diagnostics import (
    classify_evidence_presence,
    classify_verifier_issue,
    configure_v24_experiment,
)
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence
from app.schemas.verification import ClaimVerdict, VerificationResult

V25Arm = Literal["A", "B", "C", "D"]
EvidenceCondition = Literal[
    "clean",
    "original_imperfect",
    "clean_annotated",
    "original_annotated",
]

_CLEAN_PLACEHOLDER = "[CLEAN_EVIDENCE_PLACEHOLDER]"
_EXPERIMENT_NAME = "v2.5_verifier_evidence_quality"


def v25_experiment_enabled(settings: Settings, qa: CurriculumQAState) -> bool:
    if qa.metadata.get("v25_experiment_arm"):
        return True
    return bool(getattr(settings, "v25_verifier_evidence_quality_experiment", False))


def evidence_condition_for_arm(arm: V25Arm) -> EvidenceCondition:
    mapping: dict[V25Arm, EvidenceCondition] = {
        "A": "clean",
        "B": "original_imperfect",
        "C": "clean_annotated",
        "D": "original_annotated",
    }
    return mapping[arm.upper()]  # type: ignore[index]


def configure_v25_experiment(
    qa: CurriculumQAState,
    *,
    settings: Settings,
    arm: V25Arm,
) -> None:
    """Configure V2.5 arm on top of frozen + single-pass V2.4 arm A controls."""
    qa.metadata["v25_experiment_arm"] = arm.upper()
    qa.metadata["v25_evidence_condition"] = evidence_condition_for_arm(arm)
    configure_v24_experiment(qa, settings=settings, arm="A")


def _lo_code(item: CurriculumEvidence) -> str | None:
    code = (item.metadata or {}).get("code")
    if code:
        return str(code)
    if item.name:
        return str(item.name)
    return None


def is_imperfect_learning_outcome(item: CurriculumEvidence) -> bool:
    if (item.entity_type or "").lower() != "learning_outcome":
        return False
    content = (item.content or "").strip()
    if not content:
        return True
    return _looks_garbled_source_text(content)


def build_evidence_inventory(evidence: list[CurriculumEvidence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in evidence:
        if not is_imperfect_learning_outcome(item):
            continue
        code = _lo_code(item) or "unknown"
        content = (item.content or "").strip()
        issue = "truncated or garbled source wording"
        if content.endswith((" greater than", " up to", " to")):
            issue = "truncated source wording"
        elif "denominators up to" in content.lower():
            issue = "repetitive or garbled source wording"
        rows.append(
            {
                "lo_code": code,
                "quality_status": "GARBLED",
                "original_text_length": len(content),
                "issue": issue,
                "entity_id": item.entity_id,
            }
        )
    return rows


def _quality_annotation(code: str) -> dict[str, Any]:
    return {
        "status": "truncated",
        "original_text_present": True,
        "source_record_id": code,
    }


def transform_evidence_for_condition(
    evidence: list[CurriculumEvidence],
    *,
    condition: EvidenceCondition,
) -> list[CurriculumEvidence]:
    """Harness-only evidence transforms; never mutates production source records."""
    transformed: list[CurriculumEvidence] = []
    for item in evidence:
        cloned = copy.deepcopy(item)
        meta = dict(cloned.metadata or {})
        code = _lo_code(cloned) or "unknown"
        imperfect = is_imperfect_learning_outcome(cloned)

        if condition in {"clean", "clean_annotated"} and imperfect:
            cloned.content = f"{code} — {_CLEAN_PLACEHOLDER}"
            meta.pop("evidence_quality", None)

        if condition in {"clean_annotated", "original_annotated"} and imperfect:
            meta["evidence_quality"] = _quality_annotation(code)

        cloned.metadata = meta
        transformed.append(cloned)
    return transformed


def _serialize_evidence(evidence: list[CurriculumEvidence]) -> list[dict[str, Any]]:
    return [item.model_dump() for item in evidence]


def _deserialize_evidence(rows: list[dict[str, Any]]) -> list[CurriculumEvidence]:
    return [CurriculumEvidence.model_validate(row) for row in rows]


def apply_v25_evidence_transform(qa: CurriculumQAState) -> None:
    """Apply experiment evidence condition after retrieval, before generation."""
    arm = qa.metadata.get("v25_experiment_arm")
    if not arm:
        return
    if qa.metadata.get("v25_transform_applied"):
        return

    original = copy.deepcopy(qa.evidence)
    qa.metadata["v25_original_evidence_serialized"] = _serialize_evidence(original)
    qa.metadata["v25_baseline_evidence_hash"] = evidence_snapshot_hash(original)

    condition = qa.metadata.get("v25_evidence_condition") or evidence_condition_for_arm(
        arm  # type: ignore[arg-type]
    )
    qa.evidence = transform_evidence_for_condition(original, condition=condition)  # type: ignore[arg-type]
    qa.metadata["v25_transformed_evidence_hash"] = evidence_snapshot_hash(qa.evidence)
    qa.metadata["v25_imperfect_evidence_count"] = len(build_evidence_inventory(original))
    qa.metadata["v25_evidence_inventory"] = build_evidence_inventory(original)
    qa.metadata["v25_transform_applied"] = True
    record_evidence_snapshot(qa)


def _classify_claim(claim: str, *, result: VerificationResult) -> str:
    lowered = claim.lower()
    if any(token in lowered for token in ("speculat", "likely", "probably", "may be")):
        return "SPECULATIVE"
    if any(
        token in lowered
        for token in (
            "no learning outcome",
            "not include any",
            "does not include",
            "no evidence",
        )
    ):
        return "ABSENCE_CLAIM"
    unsupported = [c.lower() for c in (result.unsupported_claims or [])]
    if any(claim.lower() in u or u in claim.lower() for u in unsupported):
        return "UNSUPPORTED"
    if _CLEAN_PLACEHOLDER.lower() in lowered:
        return "SUPPORTED_BY_CLEAN_EVIDENCE"
    if any(
        token in lowered
        for token in ("garbled", "truncated", "incomplete", "placeholder")
    ):
        return "SUPPORTED_BY_IMPERFECT_EVIDENCE"
    return "UNSUPPORTED"


def _rejected_claim_rows(
    qa: CurriculumQAState,
    result: VerificationResult | None,
) -> list[dict[str, Any]]:
    if result is None:
        return []
    rows: list[dict[str, Any]] = []
    assessments = result.claims or []
    if assessments:
        for item in assessments:
            if item.verdict == ClaimVerdict.SUPPORTED:
                continue
            claim = item.claim or ""
            rows.append(
                {
                    "claim": claim,
                    "verdict": item.verdict.value,
                    "classification": _classify_claim(claim, result=result),
                }
            )
        return rows
    for claim in result.unsupported_claims or []:
        rows.append(
            {
                "claim": claim,
                "verdict": "unsupported",
                "classification": _classify_claim(claim, result=result),
            }
        )
    return rows


def build_v25_run_diagnostics(qa: CurriculumQAState) -> dict[str, Any]:
    result = qa.verification
    inventory = qa.metadata.get("v25_evidence_inventory") or build_evidence_inventory(
        _deserialize_evidence(qa.metadata.get("v25_original_evidence_serialized") or [])
        if qa.metadata.get("v25_original_evidence_serialized")
        else qa.evidence
    )
    already_present = bool(evidence_already_present_for_rejection(qa, result))
    return {
        "experiment": _EXPERIMENT_NAME,
        "arm": qa.metadata.get("v25_experiment_arm"),
        "evidence_condition": qa.metadata.get("v25_evidence_condition"),
        "evidence_hash": qa.metadata.get("v25_baseline_evidence_hash")
        or qa.metadata.get("evidence_snapshot_hash"),
        "transformed_evidence_hash": qa.metadata.get("v25_transformed_evidence_hash")
        or qa.metadata.get("evidence_snapshot_hash"),
        "imperfect_evidence_count": qa.metadata.get("v25_imperfect_evidence_count")
        or len(inventory),
        "verifier_score": result.score if result else None,
        "accepted": bool(result and result.passed),
        "failure_reason": qa.metadata.get("termination_reason")
        or qa.metadata.get("fallback_reason"),
        "issue_codes": list(result.issues or []) if result else [],
        "evidence_already_present": already_present,
        "retrieve_more_requested": bool(
            result and result.recommendation.value == "retrieve_more"
        ),
        "insufficient_evidence": qa.status.value == "insufficient_evidence",
        "regeneration": qa.verification_attempts > 1,
        "evidence_presence_class": classify_evidence_presence(qa, result=result),
        "verifier_issue_class": classify_verifier_issue(qa, result=result),
        "unsupported_claims": list(result.unsupported_claims or []) if result else [],
        "speculative_claims": any(
            c.get("classification") == "SPECULATIVE"
            for c in _rejected_claim_rows(qa, result)
        ),
        "rejected_claims": _rejected_claim_rows(qa, result),
        "evidence_inventory": inventory,
        "answer_for_counterfactual": qa.final_answer or qa.draft_answer,
    }


def replay_verifier_with_evidence(
    qa: CurriculumQAState,
    *,
    evidence: list[CurriculumEvidence],
    answer: str,
    verifier: Any,
    request_id: str | None = None,
) -> VerificationResult:
    """Counterfactual verifier replay with fixed answer and swapped evidence."""
    replay = CurriculumQAState.initial(question=qa.question)
    replay.evidence = evidence
    replay.final_answer = answer
    replay.draft_answer = answer
    replay.metadata.update(
        {
            "v25_experiment_arm": qa.metadata.get("v25_experiment_arm"),
            "v25_evidence_condition": qa.metadata.get("v25_evidence_condition"),
            "v25_counterfactual_replay": True,
        }
    )
    return verifier.verify(replay, request_id=request_id)


def build_counterfactual_pair(
    qa: CurriculumQAState,
    *,
    verifier: Any,
    request_id: str | None = None,
) -> dict[str, Any]:
    answer = qa.final_answer or qa.draft_answer
    if not answer:
        return {"skipped": True, "reason": "no_answer"}

    original = _deserialize_evidence(
        qa.metadata.get("v25_original_evidence_serialized") or _serialize_evidence(qa.evidence)
    )
    if not original:
        return {"skipped": True, "reason": "no_evidence"}

    clean = transform_evidence_for_condition(original, condition="clean")
    original_result = replay_verifier_with_evidence(
        qa,
        evidence=original,
        answer=answer,
        verifier=verifier,
        request_id=request_id,
    )
    clean_result = replay_verifier_with_evidence(
        qa,
        evidence=clean,
        answer=answer,
        verifier=verifier,
        request_id=f"{request_id}-clean" if request_id else None,
    )
    return {
        "original_evidence_accepted": original_result.passed,
        "clean_evidence_accepted": clean_result.passed,
        "original_score": original_result.score,
        "clean_score": clean_result.score,
        "score_delta": round((clean_result.score or 0) - (original_result.score or 0), 3),
        "acceptance_delta": int(clean_result.passed) - int(original_result.passed),
    }


def finalize_v25_diagnostics(qa: CurriculumQAState) -> dict[str, Any]:
    if not qa.metadata.get("v25_experiment_arm"):
        return {}
    diag = build_v25_run_diagnostics(qa)
    original = _deserialize_evidence(
        qa.metadata.get("v25_original_evidence_serialized") or []
    )
    diag["original_learning_outcomes"] = [
        {
            "entity_id": item.entity_id,
            "code": _lo_code(item),
            "content": item.content,
            "entity_type": item.entity_type,
            "name": item.name,
            "metadata": dict(item.metadata or {}),
            "grade": item.grade,
            "subject": item.subject,
            "topic": item.topic,
        }
        for item in original
        if (item.entity_type or "").lower() == "learning_outcome"
    ]
    qa.metadata["v25_diagnostics"] = diag
    from app.agent.trace import get_current_trace

    trace = get_current_trace()
    if trace is not None:
        trace.emit("agent.v25.diagnostics", **diag)
    return diag
