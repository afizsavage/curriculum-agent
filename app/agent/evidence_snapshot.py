"""Resolved evidence snapshot + hash for V2.3 diagnostic experiments."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.agent.context_boundary import ContextBoundarySnapshot, get_boundary
from app.agent.state import CurriculumQAState
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence


def v23_diagnostic_enabled(settings: Settings, qa: CurriculumQAState) -> bool:
    override = qa.metadata.get("v23_diagnostic_experiment")
    if override is not None:
        return bool(override)
    return bool(settings.v23_generation_verifier_experiment)


def generation_mode(qa: CurriculumQAState) -> str:
    mode = qa.metadata.get("generation_mode")
    if mode in {"current", "constrained"}:
        return str(mode)
    return "current"


def is_constrained_generation(qa: CurriculumQAState) -> bool:
    return generation_mode(qa) == "constrained"


def evidence_snapshot_hash(evidence: list[CurriculumEvidence]) -> str:
    rows: list[dict[str, Any]] = []
    for item in sorted(evidence, key=lambda e: str(e.entity_id or "")):
        if not item.entity_id:
            continue
        meta = item.metadata or {}
        rows.append(
            {
                "entity_id": item.entity_id,
                "entity_type": item.entity_type,
                "code": meta.get("code"),
                "content": item.content,
                "name": item.name,
            }
        )
    payload = json.dumps(rows, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_resolved_context_record(
    qa: CurriculumQAState,
) -> dict[str, Any]:
    boundary: ContextBoundarySnapshot | None = get_boundary(qa.retrieval_state)
    units = [
        {
            "entity_id": e.entity_id,
            "code": (e.metadata or {}).get("code"),
            "name": e.name,
        }
        for e in qa.evidence
        if (e.entity_type or "").lower() in {"unit", "topic", "curriculum_content"}
    ]
    outcomes = [
        {
            "entity_id": e.entity_id,
            "code": (e.metadata or {}).get("code"),
            "content": e.content,
        }
        for e in qa.evidence
        if (e.entity_type or "").lower() == "learning_outcome"
    ]
    return {
        "curriculum_id": boundary.curriculum_id if boundary else None,
        "curriculum_version": boundary.curriculum_version if boundary else None,
        "grade_id": boundary.grade_id if boundary else None,
        "grade_code": boundary.grade_code if boundary else qa.grade,
        "subject_id": boundary.subject_id if boundary else None,
        "subject_code": boundary.subject_code if boundary else qa.subject,
        "topic": qa.topic,
        "units": units,
        "learning_outcomes": outcomes,
        "unit_count": len(units),
        "learning_outcome_count": len(outcomes),
    }


def record_evidence_snapshot(qa: CurriculumQAState) -> dict[str, Any]:
    snap_hash = evidence_snapshot_hash(qa.evidence)
    record = {
        "evidence_snapshot_hash": snap_hash,
        "resolved_context": build_resolved_context_record(qa),
        "evidence_count": len(qa.evidence),
    }
    qa.metadata["evidence_snapshot_hash"] = snap_hash
    qa.metadata["resolved_context_snapshot"] = record["resolved_context"]
    return record


def generation_to_verifier_overlap(qa: CurriculumQAState) -> dict[str, Any]:
    gen_ids = set(qa.metadata.get("generation_evidence_ids") or [])
    answer_ids = {ref.entity_id for ref in (qa.answer_evidence or []) if ref.entity_id}
    bag_ids = {e.entity_id for e in qa.evidence if e.entity_id}
    overlap = gen_ids & answer_ids if gen_ids and answer_ids else answer_ids & bag_ids
    return {
        "generation_evidence_ids": sorted(gen_ids),
        "answer_evidence_ids": sorted(answer_ids),
        "generation_to_verifier_evidence_overlap": len(overlap),
        "overlap_ids": sorted(overlap),
    }
