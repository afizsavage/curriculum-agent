"""Scope stamps for a grade-curriculum content tree.

The content endpoint returns the published tree for one grade curriculum.
That tree is exhaustive for its subject, grade, and curriculum version.
A search miss is not this stamp, and a syllabus fallback is not this tree.
"""

from __future__ import annotations

import re

from app.curriculum.codes import normalize_grade_code, normalize_subject_code
from app.curriculum.evidence import CurriculumEvidence
from app.schemas.verification import MissingEvidenceRequest

STRUCTURE_SCOPE = "grade_curriculum_content"
_UNIT_REQUEST_TYPES = frozenset({"unit", "content", "content_record"})
_CONTENT_CODE = re.compile(r"^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+$")
_NEGATION = re.compile(
    r"\b(?:not|no|absent|without|missing|exclude[sd]?|excluding)\b",
    re.IGNORECASE,
)


def should_stamp_grade_curriculum_content(
    source: dict[str, object],
    evidence: list[CurriculumEvidence],
) -> bool:
    """True only for a non-empty grade-curriculum content tree that has codes."""
    if source.get("source") != "grade_curriculum" or not source.get("id"):
        return False
    return any((item.metadata or {}).get("code") for item in evidence)


def stamp_exhaustive_grade_curriculum(
    evidence: list[CurriculumEvidence],
    *,
    curriculum_code: str,
    curriculum_version: str,
    grade_curriculum_id: str,
) -> None:
    """Record the tree's identity. Does not invent codes or parents."""
    for item in evidence:
        metadata = dict(item.metadata or {})
        metadata["curriculum_code"] = curriculum_code
        metadata["curriculum_version"] = curriculum_version
        metadata["grade_curriculum_id"] = grade_curriculum_id
        metadata["structure_scope"] = STRUCTURE_SCOPE
        metadata["structure_exhaustive"] = True
        item.metadata = metadata


def drop_resolved_absent_code_requests(
    *,
    evidence: list[CurriculumEvidence],
    grade: str | None,
    subject: str | None,
    curriculum_version: str | None,
    answer: str,
    missing: list[MissingEvidenceRequest | str],
    unsupported_claims: list[str] | None = None,
    incorrect_claims: list[str] | None = None,
) -> tuple[list[MissingEvidenceRequest | str], list[str]]:
    """Drop a unit-code request only when this scope's tree excludes that code.

    Other missing evidence stays. Unsupported or incorrect claims stay
    blocking by leaving every request in place. The caller does not turn
    the remaining result into an acceptance.
    """
    if unsupported_claims or incorrect_claims:
        return list(missing), []
    kept: list[MissingEvidenceRequest | str] = []
    resolved: list[str] = []
    for item in missing:
        code = _requested_unit_code(item)
        if code is None or not isinstance(item, MissingEvidenceRequest):
            kept.append(item)
            continue
        if _answer_asserts_code(answer, code):
            kept.append(item)
            continue
        rows = _exhaustive_rows(
            evidence,
            grade=_scope_value(item.grade, grade, normalize_grade_code),
            subject=_scope_value(item.subject, subject, normalize_subject_code),
            curriculum_version=curriculum_version,
        )
        if not rows or _code_in_rows(rows, code):
            kept.append(item)
            continue
        resolved.append(code)
    return kept, resolved


def _requested_unit_code(item: MissingEvidenceRequest | str) -> str | None:
    if not isinstance(item, MissingEvidenceRequest):
        return None
    if (item.type or "").strip().lower() not in _UNIT_REQUEST_TYPES:
        return None
    for raw in (item.topic, item.query):
        code = (raw or "").strip().upper()
        if _CONTENT_CODE.match(code):
            return code
    return None


def _scope_value(request_value: str | None, state_value: str | None, normalize) -> str | None:
    """One scope value. Disagreeing request and state values match nothing."""
    requested = normalize(request_value)
    state = normalize(state_value)
    if requested and state and requested != state:
        return None
    return requested or state


def _answer_asserts_code(answer: str, code: str) -> bool:
    """True when any sentence names the code without saying it is absent."""
    if not answer or code.lower() not in answer.lower():
        return False
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer)
    mentions = [sentence for sentence in sentences if code.lower() in sentence.lower()]
    if not mentions:
        return False
    return any(_NEGATION.search(sentence) is None for sentence in mentions)


def _exhaustive_rows(
    evidence: list[CurriculumEvidence],
    *,
    grade: str | None,
    subject: str | None,
    curriculum_version: str | None,
) -> list[CurriculumEvidence]:
    if not grade or not subject:
        return []
    rows: list[CurriculumEvidence] = []
    for item in evidence:
        metadata = item.metadata or {}
        if metadata.get("structure_exhaustive") is not True:
            continue
        if metadata.get("structure_scope") != STRUCTURE_SCOPE:
            continue
        if item.source_reference != "grade_curriculum.content":
            continue
        if normalize_grade_code(item.grade) != grade:
            continue
        if normalize_subject_code(item.subject) != subject:
            continue
        version = str(metadata.get("curriculum_version") or "")
        code = str(metadata.get("curriculum_code") or "")
        if not version or not code or not metadata.get("grade_curriculum_id"):
            continue
        if not metadata.get("code"):
            continue
        if curriculum_version and version != str(curriculum_version):
            continue
        rows.append(item)
    versions = {str((item.metadata or {}).get("curriculum_version")) for item in rows}
    if len(versions) != 1:
        return []
    return rows


def _code_in_rows(rows: list[CurriculumEvidence], code: str) -> bool:
    return any(
        str((item.metadata or {}).get("code") or "").upper() == code for item in rows
    )
