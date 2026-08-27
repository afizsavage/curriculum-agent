"""State-aware retrieval helpers: fingerprints, coverage, gain, targeting.

Observability + orchestration efficiency only — no new Curriculum API tools.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.curriculum.codes import normalize_subject_code
from app.curriculum.evidence import CurriculumEvidence
from app.llm.base import ToolCallRequest
from app.schemas.verification import MissingEvidenceRequest


def normalize_tool_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Stable, comparable argument dict for fingerprints."""
    raw = arguments or {}
    out: dict[str, Any] = {}
    for key in sorted(raw.keys(), key=str):
        value = raw[key]
        if value is None or value == "":
            continue
        if isinstance(value, str):
            text = value.strip()
            if key in {"subject", "grade", "level"}:
                # Normalize subject codes when possible; keep original casing for topics.
                if key == "subject":
                    text = normalize_subject_code(text) or text.upper()
                elif key == "grade":
                    text = text.upper()
                elif key == "level":
                    text = text.lower()
            elif key == "query":
                text = " ".join(text.lower().split())
            out[key] = text
        else:
            out[key] = value
    return out


def tool_fingerprint(name: str, arguments: dict[str, Any] | None) -> str:
    """Deterministic fingerprint: tool + normalized args."""
    normalized = normalize_tool_arguments(arguments)
    payload = json.dumps(normalized, sort_keys=True, default=str)
    digest = hashlib.sha256(f"{name}|{payload}".encode()).hexdigest()[:16]
    # Human-readable prefix for traces.
    parts = [f"{k}={normalized[k]}" for k in sorted(normalized)]
    readable = "|".join(parts) if parts else ""
    return f"{name}|{readable}|{digest}" if readable else f"{name}|{digest}"


# Back-compat alias used by retrieve.py / tests.
def tool_call_key(name: str, arguments: dict[str, Any] | None) -> str:
    return tool_fingerprint(name, arguments)


class RetrievalCoverage(BaseModel):
    """Which request dimensions have been covered by successful retrieval."""

    grade: Optional[str] = None
    subject: Optional[str] = None
    topic: Optional[str] = None
    entity_types: list[str] = Field(default_factory=list)
    structure_grades: list[str] = Field(default_factory=list)
    structure_subjects: list[str] = Field(default_factory=list)
    learning_outcome_topic_ids: list[str] = Field(default_factory=list)


class RetrievalState(BaseModel):
    """Per-turn retrieval memory for novelty-driven tool selection."""

    fingerprints: dict[str, int] = Field(default_factory=dict)
    tools_executed: list[str] = Field(default_factory=list)
    queries_executed: list[str] = Field(default_factory=list)
    evidence_ids_seen: list[str] = Field(default_factory=list)
    evidence_ids_by_type: dict[str, list[str]] = Field(default_factory=dict)
    coverage: RetrievalCoverage = Field(default_factory=RetrievalCoverage)
    objectives: list[str] = Field(default_factory=list)
    rounds: list[dict[str, Any]] = Field(default_factory=list)

    last_retrieval_gain: int = 0
    last_relevant_gain: int = 0
    cumulative_retrieval_gain: int = 0
    cumulative_relevant_gain: int = 0

    no_progress: bool = False
    no_progress_reason: Optional[str] = None

    duplicate_tool_calls_prevented: int = 0
    duplicate_evidence_prevented: int = 0
    retrieval_rounds_with_progress: int = 0
    retrieval_rounds_without_progress: int = 0
    targeted_retrievals: int = 0
    broad_retrievals: int = 0
    tools_skipped: int = 0

    resolved_subject: Optional[str] = None

    def has_fingerprint(self, fingerprint: str) -> bool:
        return fingerprint in self.fingerprints

    def remember_fingerprint(self, fingerprint: str, tool_call_number: int) -> None:
        if fingerprint not in self.fingerprints:
            self.fingerprints[fingerprint] = tool_call_number

    def previous_call_number(self, fingerprint: str) -> int | None:
        return self.fingerprints.get(fingerprint)

    def note_evidence(self, item: CurriculumEvidence, *, is_new: bool) -> None:
        eid = item.entity_id
        if not eid:
            return
        if eid not in self.evidence_ids_seen:
            self.evidence_ids_seen.append(eid)
        et = item.entity_type or "unknown"
        bucket = self.evidence_ids_by_type.setdefault(et, [])
        if eid not in bucket:
            bucket.append(eid)
        if not is_new:
            self.duplicate_evidence_prevented += 1

    def update_coverage_from_evidence(self, evidence: list[CurriculumEvidence]) -> None:
        cov = self.coverage
        for item in evidence:
            if item.grade and not cov.grade:
                cov.grade = item.grade
            if item.subject:
                code = normalize_subject_code(item.subject) or item.subject
                self.resolved_subject = self.resolved_subject or code
                if not cov.subject:
                    cov.subject = code
            if item.topic and not cov.topic:
                # Prefer human topic names over raw UUIDs when available.
                if item.entity_type in {"topic", "unit"} and item.name:
                    cov.topic = item.name
                elif not cov.topic:
                    cov.topic = item.topic
            et = item.entity_type or "unknown"
            if et not in cov.entity_types:
                cov.entity_types.append(et)
            if (
                et == "learning_outcome"
                and item.topic
                and item.topic not in cov.learning_outcome_topic_ids
            ):
                cov.learning_outcome_topic_ids.append(str(item.topic))

    def note_structure_call(self, arguments: dict[str, Any] | None) -> None:
        args = normalize_tool_arguments(arguments)
        grade = args.get("grade")
        subject = args.get("subject")
        if grade and grade not in self.coverage.structure_grades:
            self.coverage.structure_grades.append(str(grade))
        if grade and subject:
            key = f"{grade}|{subject}"
            if key not in self.coverage.structure_subjects:
                self.coverage.structure_subjects.append(key)

    def structure_already_fetched(self, arguments: dict[str, Any] | None) -> bool:
        args = normalize_tool_arguments(arguments)
        grade = args.get("grade")
        subject = args.get("subject")
        if not grade:
            return bool(self.coverage.structure_grades)
        if subject:
            key = f"{grade}|{subject}"
            if key in self.coverage.structure_subjects:
                return True
            # Grade+subject already covered under any subject casing/normalization.
            return any(
                s.startswith(f"{grade}|") for s in self.coverage.structure_subjects
            )
        return grade in self.coverage.structure_grades

    def metrics_snapshot(self) -> dict[str, Any]:
        return {
            "duplicate_tool_calls_prevented": self.duplicate_tool_calls_prevented,
            "duplicate_evidence_prevented": self.duplicate_evidence_prevented,
            "retrieval_rounds_with_progress": self.retrieval_rounds_with_progress,
            "retrieval_rounds_without_progress": self.retrieval_rounds_without_progress,
            "targeted_retrievals": self.targeted_retrievals,
            "broad_retrievals": self.broad_retrievals,
            "tools_skipped": self.tools_skipped,
            "last_retrieval_gain": self.last_retrieval_gain,
            "last_relevant_gain": self.last_relevant_gain,
            "cumulative_retrieval_gain": self.cumulative_retrieval_gain,
            "cumulative_relevant_gain": self.cumulative_relevant_gain,
            "no_progress": self.no_progress,
            "no_progress_reason": self.no_progress_reason,
            "resolved_subject": self.resolved_subject,
            "fingerprints": len(self.fingerprints),
            "evidence_ids_seen": len(self.evidence_ids_seen),
        }


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)
_UNIT_CODE_RE = re.compile(r"\b(C\d+-?U\d+)\b", re.I)
_LO_CODE_RE = re.compile(r"\b(C\d+-?U\d+-LO\d+)\b", re.I)
_INCOMPLETE_RE = re.compile(
    r"truncat|incomplet|garbled|repetitiv|full text|source (content|wording|record)",
    re.I,
)


def is_incomplete_source_gap(
    missing: list[MissingEvidenceRequest | str] | None,
    *,
    verification_issues: list[str] | None = None,
) -> bool:
    """True when verifier is asking to re-fetch already-known incomplete text."""
    texts: list[str] = []
    for item in missing or []:
        if isinstance(item, str):
            texts.append(item)
        else:
            texts.extend(
                [
                    str(item.query or ""),
                    str(item.detail or ""),
                    str(item.type or ""),
                ]
            )
    texts.extend(verification_issues or [])
    blob = " ".join(texts)
    return bool(_INCOMPLETE_RE.search(blob))


def build_retrieval_objective(
    *,
    pending_missing: list[MissingEvidenceRequest | str] | None,
    grade: str | None,
    subject: str | None,
    topic: str | None,
) -> str:
    if not pending_missing:
        parts = ["Retrieve curriculum evidence"]
        if grade:
            parts.append(f"for {grade}")
        if subject:
            parts.append(str(subject))
        if topic:
            parts.append(f"topic={topic}")
        return " ".join(parts)

    bits: list[str] = []
    for item in pending_missing[:3]:
        if isinstance(item, str):
            bits.append(item)
            continue
        chunk = item.query or item.detail or item.type or "missing evidence"
        if item.topic:
            chunk = f"{chunk} (topic={item.topic})"
        bits.append(str(chunk))
    return "Find: " + "; ".join(bits)


def objective_key(objective: str) -> str:
    return " ".join(objective.lower().split())


def is_relevant_evidence(
    item: CurriculumEvidence,
    *,
    grade: str | None,
    subject: str | None,
    topic: str | None,
    pending_missing: list[MissingEvidenceRequest | str] | None = None,
) -> bool:
    """Deterministic relevance using metadata only (no LLM)."""
    if grade and item.grade and item.grade != grade:
        return False
    if subject:
        want = normalize_subject_code(subject) or subject.upper()
        got = normalize_subject_code(item.subject or "") or (item.subject or "").upper()
        if got and want and got != want:
            # Subjects in bag from structure listing may still be useful context;
            # only treat as irrelevant when both sides are set and differ AND
            # entity is not a learning outcome matching the topic.
            if item.entity_type not in {"learning_outcome", "topic", "unit"}:
                return False

    hay = " ".join(
        [
            str(item.name or ""),
            str(item.content or ""),
            str(item.topic or ""),
            str(item.entity_type or ""),
        ]
    ).lower()

    if pending_missing:
        for miss in pending_missing:
            if isinstance(miss, str):
                tokens = [t for t in re.findall(r"[a-z0-9-]{3,}", miss.lower()) if t]
            else:
                tokens = [
                    t
                    for t in re.findall(
                        r"[a-z0-9-]{3,}",
                        " ".join(
                            [
                                str(miss.type or ""),
                                str(miss.topic or ""),
                                str(miss.query or ""),
                                str(miss.detail or ""),
                            ]
                        ).lower(),
                    )
                    if t
                ]
            if tokens and any(tok in hay for tok in tokens[:8]):
                return True

    if topic:
        topic_l = topic.lower()
        if topic_l in hay or any(
            tok in hay for tok in topic_l.split() if len(tok) > 3
        ):
            return True
        if item.entity_type == "learning_outcome":
            return True  # LO under matching grade/subject is relevant for LO questions

    if item.entity_type in {"learning_outcome", "topic", "unit"}:
        return True
    return False


def targeted_tool_calls_from_missing(
    pending_missing: list[MissingEvidenceRequest | str] | None,
    *,
    available_tools: set[str],
    grade: str | None,
    subject: str | None,
    topic: str | None,
    retrieval_state: RetrievalState,
) -> list[ToolCallRequest]:
    """Map verifier gaps to the narrowest existing tools (no LLM)."""
    if not pending_missing:
        return []

    calls: list[ToolCallRequest] = []
    seen_fp: set[str] = set()

    def add(name: str, arguments: dict[str, Any]) -> None:
        if name not in available_tools:
            return
        fp = tool_fingerprint(name, arguments)
        if fp in seen_fp or retrieval_state.has_fingerprint(fp):
            return
        # Also skip structure repeats via coverage.
        if name == "get_curriculum_structure" and retrieval_state.structure_already_fetched(
            arguments
        ):
            return
        seen_fp.add(fp)
        calls.append(
            ToolCallRequest(id=str(uuid4()), name=name, arguments=arguments)
        )

    for item in pending_missing:
        if isinstance(item, str):
            text = item
            miss_type = None
            miss_topic = topic
            miss_grade = grade
            miss_subject = subject or retrieval_state.resolved_subject
            miss_query = item
        else:
            text = " ".join(
                [
                    str(item.type or ""),
                    str(item.topic or ""),
                    str(item.query or ""),
                    str(item.detail or ""),
                ]
            )
            miss_type = (item.type or "").lower() or None
            miss_topic = item.topic or topic
            miss_grade = item.grade or grade
            miss_subject = (
                item.subject
                or subject
                or retrieval_state.resolved_subject
            )
            miss_query = item.query or item.detail

        unit_codes = _UNIT_CODE_RE.findall(text)
        lo_codes = _LO_CODE_RE.findall(text)
        wants_lo = bool(
            miss_type
            and ("learning" in miss_type or "objective" in miss_type or "outcome" in miss_type)
        ) or bool(re.search(r"learning\s+object", text, re.I)) or bool(lo_codes)

        # Prefer topic_id when verifier supplied a UUID topic.
        topic_id = None
        if miss_topic and _UUID_RE.match(str(miss_topic).strip()):
            topic_id = str(miss_topic).strip()

        added_before = len(calls)
        if wants_lo and "get_learning_objectives" in available_tools:
            if topic_id:
                add("get_learning_objectives", {"topic_id": topic_id})
            else:
                topics_for_lo: list[str] = []
                for code in unit_codes:
                    if code not in topics_for_lo:
                        topics_for_lo.append(code)
                if (
                    miss_topic
                    and str(miss_topic) not in topics_for_lo
                    and not unit_codes
                ):
                    topics_for_lo.append(str(miss_topic))
                if not topics_for_lo and topic:
                    topics_for_lo.append(topic)
                for t in topics_for_lo[:3]:
                    add(
                        "get_learning_objectives",
                        {
                            "topic": t,
                            "grade": miss_grade,
                            "subject": miss_subject,
                        },
                    )
            # LO gap: never broaden to get_topic/search for the same item,
            # even when the preferred call is already a known duplicate.
            continue

        if unit_codes and "get_topic" in available_tools:
            for code in unit_codes[:3]:
                add(
                    "get_topic",
                    {
                        "topic": code,
                        "grade": miss_grade,
                        "subject": miss_subject,
                    },
                )

        if topic_id and "get_topic" in available_tools:
            add("get_topic", {"topic_id": topic_id})

        wants_topic = bool(
            miss_type and miss_type in {"topic", "unit", "structure"}
        ) or bool(re.search(r"\btopic\b|\bunit\b", text, re.I))
        if (
            wants_topic
            and miss_topic
            and not topic_id
            and "get_topic" in available_tools
        ):
            add(
                "get_topic",
                {
                    "topic": str(miss_topic),
                    "grade": miss_grade,
                    "subject": miss_subject,
                },
            )

        # If still nothing concrete for this gap, allow one narrow search.
        if len(calls) == added_before and miss_query and "search_curriculum" in available_tools:
            add(
                "search_curriculum",
                {
                    "query": str(miss_query)[:120],
                    "grade": miss_grade,
                    "subject": miss_subject,
                },
            )

    return calls


def has_credible_retrieval_path(
    *,
    retrieval_state: RetrievalState,
    pending_missing: list[MissingEvidenceRequest | str] | None,
    available_tools: set[str],
    grade: str | None,
    subject: str | None,
    topic: str | None,
    verification_issues: list[str] | None = None,
) -> bool:
    """Whether another retrieve round can execute a non-duplicate, goal-directed call."""
    if is_incomplete_source_gap(
        pending_missing, verification_issues=verification_issues
    ):
        # Incomplete source already retrieved — no meaningful new path.
        targeted = targeted_tool_calls_from_missing(
            pending_missing,
            available_tools=available_tools,
            grade=grade,
            subject=subject,
            topic=topic,
            retrieval_state=retrieval_state,
        )
        if not targeted:
            return False

    targeted = targeted_tool_calls_from_missing(
        pending_missing,
        available_tools=available_tools,
        grade=grade,
        subject=subject,
        topic=topic,
        retrieval_state=retrieval_state,
    )
    return bool(targeted)


def is_low_value_broad_call(
    name: str,
    arguments: dict[str, Any] | None,
    retrieval_state: RetrievalState,
    *,
    follow_up_round: bool,
) -> bool:
    """Conservative skip for repeated broad structure/search on follow-up rounds."""
    if not follow_up_round:
        return False
    args = normalize_tool_arguments(arguments)
    if name == "get_curriculum_structure":
        return retrieval_state.structure_already_fetched(args)
    if name == "search_curriculum":
        # If we already have LOs for this grade/subject/topic, broad search is low value.
        cov = retrieval_state.coverage
        has_lo = "learning_outcome" in cov.entity_types
        query = str(args.get("query") or "").lower()
        topic_l = (cov.topic or "").lower()
        if has_lo and topic_l and topic_l.split()[0] in query:
            return True
        # Exact prior query already covered via fingerprints; near-dup handled above.
    return False
