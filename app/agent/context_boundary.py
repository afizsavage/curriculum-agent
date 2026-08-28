"""V2.2 experiment: authoritative context boundary after resolve_curriculum_context.

Isolated behind CURRICULUM_V2_CONTEXT_BOUNDARY_EXPERIMENT / per-request override.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.agent.retrieval_state import (
    _LO_CODE_RE,
    _UNIT_CODE_RE,
    normalize_tool_arguments,
)
from app.agent.state import CurriculumQAState
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence
from app.schemas.verification import MissingEvidenceRequest

LEGACY_EXPLORATORY_TOOLS = frozenset(
    {
        "search_curriculum",
        "get_curriculum_structure",
        "get_topic",
        "get_learning_objectives",
    }
)

_NEW_ENTITY_RE = re.compile(
    r"lesson\s+plan|teaching\s+guide|pedagog|assessment\s+guide|scheme\s+of\s+work",
    re.I,
)


class ContextBoundarySnapshot(BaseModel):
    """Authoritative curriculum context established by a successful resolver call."""

    context_resolved: bool = False
    curriculum_id: Optional[str] = None
    curriculum_version: Optional[str] = None
    curriculum_version_id: Optional[str] = None
    grade_id: Optional[str] = None
    grade_code: Optional[str] = None
    subject_id: Optional[str] = None
    subject_code: Optional[str] = None
    topic_ids: list[str] = Field(default_factory=list)
    unit_ids: list[str] = Field(default_factory=list)
    learning_outcome_ids: list[str] = Field(default_factory=list)
    unit_codes: list[str] = Field(default_factory=list)
    lo_codes: list[str] = Field(default_factory=list)
    resolution_source: str = "resolve_curriculum_context"
    resolution_status: Optional[str] = None


def get_boundary(retrieval_state: Any) -> ContextBoundarySnapshot | None:
    raw = getattr(retrieval_state, "context_boundary", None)
    if raw is None:
        return None
    if isinstance(raw, ContextBoundarySnapshot):
        return raw
    if isinstance(raw, dict):
        return ContextBoundarySnapshot.model_validate(raw)
    return None


def context_boundary_experiment_enabled(
    settings: Settings,
    qa: CurriculumQAState,
) -> bool:
    override = qa.metadata.get("context_boundary_experiment")
    if override is not None:
        return bool(override)
    return bool(settings.curriculum_v2_context_boundary_experiment)


def capture_context_boundary(payload: dict[str, Any]) -> ContextBoundarySnapshot | None:
    resolution = payload.get("resolution") or {}
    if resolution.get("status") != "resolved":
        return None
    curriculum = payload.get("curriculum") or {}
    grade = payload.get("grade") or {}
    subject = payload.get("subject") or {}
    topics = payload.get("topics") or []
    units = payload.get("units") or []
    outcomes = payload.get("learning_outcomes") or []
    if not outcomes and not units:
        return None
    curriculum_id = str(curriculum["id"]) if curriculum.get("id") else None
    version = str(curriculum.get("version") or "") or None
    version_id = f"{curriculum_id}:{version}" if curriculum_id and version else curriculum_id
    return ContextBoundarySnapshot(
        context_resolved=True,
        curriculum_id=curriculum_id,
        curriculum_version=version,
        curriculum_version_id=version_id,
        grade_id=str(grade["id"]) if grade.get("id") else None,
        grade_code=grade.get("code"),
        subject_id=str(subject["id"]) if subject.get("id") else None,
        subject_code=subject.get("code"),
        topic_ids=[
            str(t["id"]) for t in topics if isinstance(t, dict) and t.get("id")
        ],
        unit_ids=[str(u["id"]) for u in units if isinstance(u, dict) and u.get("id")],
        learning_outcome_ids=[
            str(o["id"]) for o in outcomes if isinstance(o, dict) and o.get("id")
        ],
        unit_codes=[
            str(u.get("code")).upper()
            for u in units
            if isinstance(u, dict) and u.get("code")
        ],
        lo_codes=[
            str(o.get("code")).upper()
            for o in outcomes
            if isinstance(o, dict) and o.get("code")
        ],
        resolution_status="resolved",
    )


def _missing_text(item: MissingEvidenceRequest | str) -> str:
    if isinstance(item, str):
        return item
    return " ".join(
        [
            str(item.type or ""),
            str(item.topic or ""),
            str(item.query or ""),
            str(item.detail or ""),
            str(item.grade or ""),
            str(item.subject or ""),
        ]
    )


def _evidence_index(evidence: list[CurriculumEvidence]) -> tuple[set[str], set[str]]:
    lo_codes: set[str] = set()
    unit_codes: set[str] = set()
    for item in evidence:
        meta = item.metadata or {}
        code = str(meta.get("code") or "").upper()
        if (item.entity_type or "").lower() == "learning_outcome" and code:
            lo_codes.add(code)
        if (item.entity_type or "").lower() in {"unit", "topic", "curriculum_content"}:
            if code:
                unit_codes.add(code)
    return lo_codes, unit_codes


def _normalize_unit_code(code: str) -> str:
    raw = code.upper().replace("-", "")
    m = re.match(r"(C\d+)(U\d+)", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return code.upper()


def _lo_unit_code(lo_code: str) -> Optional[str]:
    raw = lo_code.upper().replace("-", "")
    m = re.match(r"(C\d+)(U\d+)", raw)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def missing_evidence_covered_by_boundary(
    pending_missing: list[MissingEvidenceRequest | str] | None,
    *,
    boundary: ContextBoundarySnapshot | None,
    evidence: list[CurriculumEvidence],
) -> bool:
    """True when verifier gaps refer only to entities already in the resolved boundary."""
    if boundary is None or not boundary.context_resolved:
        return False
    if not pending_missing:
        return False

    ev_lo, ev_units = _evidence_index(evidence)
    all_lo = {c.upper() for c in boundary.lo_codes} | ev_lo
    all_units = {_normalize_unit_code(c) for c in boundary.unit_codes} | {
        _normalize_unit_code(c) for c in ev_units
    }

    for item in pending_missing:
        text = _missing_text(item)
        if _NEW_ENTITY_RE.search(text):
            return False

        unit_codes = [_normalize_unit_code(c) for c in _UNIT_CODE_RE.findall(text)]
        lo_codes = [c.upper() for c in _LO_CODE_RE.findall(text)]

        if unit_codes and not all(c in all_units for c in unit_codes):
            return False
        if lo_codes:
            for lc in lo_codes:
                if lc in all_lo:
                    continue
                unit_for_lo = _lo_unit_code(lc)
                if unit_for_lo and _normalize_unit_code(unit_for_lo) in all_units and all_lo:
                    continue
                return False

        wants_lo = bool(
            lo_codes
            or re.search(r"learning\s+object|learning\s+outcome|\blo\b", text, re.I)
        )
        if wants_lo and not boundary.learning_outcome_ids and not ev_lo:
            return False

    return bool(boundary.learning_outcome_ids or all_lo or all_units)


def is_redundant_legacy_retrieval(
    tool_name: str,
    arguments: dict[str, Any] | None,
    *,
    boundary: ContextBoundarySnapshot | None,
    evidence: list[CurriculumEvidence],
    pending_missing: list[MissingEvidenceRequest | str] | None = None,
) -> bool:
    """Skip legacy exploratory tools when resolver boundary already holds the answer."""
    if boundary is None or not boundary.context_resolved:
        return False
    if tool_name not in LEGACY_EXPLORATORY_TOOLS:
        return False

    if pending_missing and not missing_evidence_covered_by_boundary(
        pending_missing, boundary=boundary, evidence=evidence
    ):
        return False

    if not boundary.learning_outcome_ids:
        return False

    args = normalize_tool_arguments(arguments)
    if args.get("topic_id"):
        tid = str(args["topic_id"])
        if tid in set(boundary.topic_ids) | set(boundary.unit_ids):
            return True

    topic_arg = str(args.get("topic") or "").upper()
    if topic_arg:
        if topic_arg in set(boundary.unit_codes):
            return True
        if any(topic_arg in code for code in boundary.unit_codes):
            return True

    query = str(args.get("query") or "").lower()
    if tool_name == "search_curriculum" and query:
        if any(code.lower() in query for code in boundary.unit_codes):
            return True
        if boundary.subject_code and str(boundary.subject_code).lower() in query:
            return True

    grade = str(args.get("grade") or "").upper()
    subject = str(args.get("subject") or "").upper()
    grade_match = not grade or (
        boundary.grade_code and grade == str(boundary.grade_code).upper()
    )
    subject_match = not subject or (
        boundary.subject_code and subject == str(boundary.subject_code).upper()
    )
    if grade_match and subject_match:
        if tool_name in {
            "get_learning_objectives",
            "get_topic",
            "get_curriculum_structure",
            "search_curriculum",
        }:
            return True
    return False


def record_boundary_metrics(qa: CurriculumQAState) -> dict[str, Any]:
    """Snapshot boundary + evidence counters for traces and eval."""
    boundary = get_boundary(qa.retrieval_state)
    unique_ids = {e.entity_id for e in qa.evidence if e.entity_id}
    return {
        "context_resolved": bool(boundary and boundary.context_resolved),
        "context_resolution_status": (
            boundary.resolution_status if boundary else None
        ),
        "curriculum_version_id": boundary.curriculum_version_id if boundary else None,
        "grade_id": boundary.grade_id if boundary else None,
        "subject_id": boundary.subject_id if boundary else None,
        "topic_ids": list(boundary.topic_ids) if boundary else [],
        "unit_ids": list(boundary.unit_ids) if boundary else [],
        "learning_outcome_ids": list(boundary.learning_outcome_ids)
        if boundary
        else [],
        "resolver_evidence_count": len(boundary.learning_outcome_ids)
        if boundary
        else 0,
        "generation_evidence_count": qa.metadata.get("generation_evidence_count"),
        "unique_evidence_count": len(unique_ids),
        "duplicate_evidence_count": qa.retrieval_state.duplicate_evidence_prevented,
    }
