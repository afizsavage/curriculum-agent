"""V2.13F document evidence arbitration (shadow / replay only).

Deterministic, non-LLM arbitration over whether retrieved documents should
influence generation. Production answers are never affected unless a caller
explicitly enables the experiment flag in a shadow/replay harness.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from app.curriculum.evidence import CurriculumEvidence


class DocumentRole(str, Enum):
    DECISIVE = "DECISIVE"
    SUPPORTING = "SUPPORTING"
    REDUNDANT = "REDUNDANT"
    IRRELEVANT = "IRRELEVANT"
    CONFLICTING = "CONFLICTING"


class StructuredSufficiency(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    UNCERTAIN = "UNCERTAIN"


class DocumentUse(str, Enum):
    INCLUDE_IN_GENERATION = "INCLUDE_IN_GENERATION"
    PROVENANCE_ONLY = "PROVENANCE_ONLY"
    DO_NOT_USE = "DO_NOT_USE"
    REQUIRE_REVIEW = "REQUIRE_REVIEW"


class ArbitrationPolicy(str, Enum):
    """Experimental evidence policies."""

    BASELINE_MERGE = "A_BASELINE_MERGE"  # V2.13D: always merge docs into generation
    STRUCTURED_FIRST = "B_STRUCTURED_FIRST"  # gate on structured sufficiency
    ARBITRATED = "C_ARBITRATED"  # full role → use mapping


@dataclass(frozen=True)
class ArbitrationDecision:
    document_role: DocumentRole
    structured_sufficiency: StructuredSufficiency
    document_use: DocumentUse
    policy: ArbitrationPolicy
    reasons: tuple[str, ...]
    relevant_passage_count: int
    conflicting_passage_count: int
    document_count: int
    structured_count: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["document_role"] = self.document_role.value
        payload["structured_sufficiency"] = self.structured_sufficiency.value
        payload["document_use"] = self.document_use.value
        payload["policy"] = self.policy.value
        return payload


def _norm(value: Any) -> str:
    return str(value or "").strip().upper()


def _passage_meta(passage: CurriculumEvidence | dict[str, Any]) -> dict[str, Any]:
    if isinstance(passage, CurriculumEvidence):
        meta = dict(passage.metadata or {})
        return {
            "grade": passage.grade,
            "subject": passage.subject,
            "topic": passage.topic,
            "unit": passage.unit if hasattr(passage, "unit") else meta.get("unit"),
            "retrieval_score": meta.get("retrieval_score"),
            "retrieval_rank": meta.get("retrieval_rank"),
            "content": passage.content or "",
            "entity_type": passage.entity_type,
            "metadata_valid": not bool(meta.get("metadata_blocked")),
        }
    return {
        "grade": passage.get("grade"),
        "subject": passage.get("subject"),
        "topic": passage.get("topic"),
        "unit": passage.get("unit"),
        "retrieval_score": passage.get("retrieval_score"),
        "retrieval_rank": passage.get("retrieval_rank"),
        "content": passage.get("content") or "",
        "entity_type": passage.get("entity_type") or "document_passage",
        "metadata_valid": passage.get("metadata_valid", True),
    }


def classify_structured_sufficiency(
    *,
    control_accepted: bool | None = None,
    structured_count: int = 0,
    control_route: str | None = None,
    control_verifier_decision: str | None = None,
) -> StructuredSufficiency:
    """Prefer frozen control outcome when available; else heuristic on counts."""
    if control_accepted is True:
        return StructuredSufficiency.SUFFICIENT
    if control_accepted is False:
        route = str(control_route or "").lower()
        verd = str(control_verifier_decision or "").lower()
        if route in {"fallback", "retrieve_more", "clarify"} or verd in {
            "reject",
            "retrieve_more",
            "fallback",
            "clarify",
        }:
            return StructuredSufficiency.INSUFFICIENT
        if structured_count <= 0:
            return StructuredSufficiency.INSUFFICIENT
        return StructuredSufficiency.UNCERTAIN
    if structured_count <= 0:
        return StructuredSufficiency.INSUFFICIENT
    if structured_count >= 8:
        return StructuredSufficiency.UNCERTAIN
    return StructuredSufficiency.UNCERTAIN


def _passage_relevant(
    meta: dict[str, Any],
    *,
    grade: str | None,
    subject: str | None,
    topic: str | None,
) -> bool:
    if subject and meta.get("subject") and _norm(meta.get("subject")) == _norm(subject):
        return True
    if grade and meta.get("grade") and _norm(meta.get("grade")) == _norm(grade):
        return True
    if topic and meta.get("topic"):
        pt = str(meta.get("topic") or "").lower()
        qt = str(topic or "").lower()
        if qt and (qt in pt or pt in qt):
            return True
    return False


def _passage_conflicting(
    meta: dict[str, Any],
    *,
    grade: str | None,
    subject: str | None,
) -> bool:
    conflict = False
    if grade and meta.get("grade") and _norm(meta.get("grade")) != _norm(grade):
        conflict = True
    if subject and meta.get("subject") and _norm(meta.get("subject")) != _norm(subject):
        # Subject mismatch alone is conflict only when grade also mismatches or grade unset.
        if not grade or (meta.get("grade") and _norm(meta.get("grade")) != _norm(grade)):
            conflict = True
    return conflict


def _lexical_overlap(question: str, content: str) -> float:
    q = {t for t in str(question or "").lower().split() if len(t) > 2}
    c = {t for t in str(content or "").lower().split() if len(t) > 2}
    if not q or not c:
        return 0.0
    return len(q & c) / max(len(q), 1)


def classify_document_role(
    documents: list[CurriculumEvidence] | list[dict[str, Any]],
    *,
    question: str = "",
    grade: str | None = None,
    subject: str | None = None,
    topic: str | None = None,
    structured_sufficiency: StructuredSufficiency = StructuredSufficiency.UNCERTAIN,
    structured_count: int = 0,
) -> tuple[DocumentRole, int, int, tuple[str, ...]]:
    if not documents:
        return DocumentRole.IRRELEVANT, 0, 0, ("no_documents",)

    metas = [_passage_meta(d) for d in documents]
    relevant = [
        m
        for m in metas
        if _passage_relevant(m, grade=grade, subject=subject, topic=topic)
    ]
    conflicting = [
        m for m in metas if _passage_conflicting(m, grade=grade, subject=subject)
    ]
    reasons: list[str] = []
    overlaps = [_lexical_overlap(question, m.get("content") or "") for m in metas]
    mean_overlap = sum(overlaps) / max(len(overlaps), 1)
    top_score = max(
        (float(m.get("retrieval_score") or 0.0) for m in metas),
        default=0.0,
    )

    if conflicting and len(conflicting) >= max(1, len(metas) // 2):
        # Conflict still matters when structured failed — REQUIRE_REVIEW via role.
        if structured_sufficiency != StructuredSufficiency.INSUFFICIENT:
            reasons.append("majority_hierarchy_conflict")
            return (
                DocumentRole.CONFLICTING,
                len(relevant),
                len(conflicting),
                tuple(reasons),
            )
        reasons.append("conflict_ignored_for_insufficient_recovery_path")

    # When structured already failed, do not discard retrieved docs solely for
    # missing hierarchy match — that path is the recovery candidate.
    if (
        not relevant
        and (grade or subject)
        and structured_sufficiency != StructuredSufficiency.INSUFFICIENT
    ):
        reasons.append("no_grade_subject_topic_match")
        return DocumentRole.IRRELEVANT, 0, len(conflicting), tuple(reasons)

    if structured_sufficiency == StructuredSufficiency.INSUFFICIENT and relevant:
        reasons.append("structured_insufficient_with_relevant_docs")
        if mean_overlap >= 0.08 or top_score >= 0.02 or len(relevant) >= 1:
            reasons.append("decisive_for_recovery")
            return (
                DocumentRole.DECISIVE,
                len(relevant),
                len(conflicting),
                tuple(reasons),
            )

    # Structured path failed: any retrieved docs are the recovery candidate even
    # without a clean grade/subject match (common when question hierarchy is unset).
    if structured_sufficiency == StructuredSufficiency.INSUFFICIENT and metas:
        reasons.append("structured_insufficient_docs_available_for_recovery")
        return (
            DocumentRole.DECISIVE,
            len(relevant),
            len(conflicting),
            tuple(reasons),
        )

    if structured_sufficiency == StructuredSufficiency.SUFFICIENT:
        if relevant and mean_overlap >= 0.15 and structured_count < 5:
            reasons.append("supporting_despite_sufficient_structured")
            return (
                DocumentRole.SUPPORTING,
                len(relevant),
                len(conflicting),
                tuple(reasons),
            )
        if relevant:
            reasons.append("structured_already_sufficient_docs_redundant")
            return (
                DocumentRole.REDUNDANT,
                len(relevant),
                len(conflicting),
                tuple(reasons),
            )
        reasons.append("structured_sufficient_docs_irrelevant")
        return DocumentRole.IRRELEVANT, 0, len(conflicting), tuple(reasons)

    # UNCERTAIN structured
    if relevant and mean_overlap >= 0.12:
        reasons.append("uncertain_structured_relevant_overlap")
        return DocumentRole.SUPPORTING, len(relevant), len(conflicting), tuple(reasons)
    if relevant:
        reasons.append("uncertain_structured_weak_relevance")
        return DocumentRole.SUPPORTING, len(relevant), len(conflicting), tuple(reasons)
    reasons.append("uncertain_no_clear_relevance")
    return DocumentRole.IRRELEVANT, 0, len(conflicting), tuple(reasons)


def map_document_use(
    *,
    role: DocumentRole,
    sufficiency: StructuredSufficiency,
    policy: ArbitrationPolicy,
) -> DocumentUse:
    if policy == ArbitrationPolicy.BASELINE_MERGE:
        return DocumentUse.INCLUDE_IN_GENERATION

    if policy == ArbitrationPolicy.STRUCTURED_FIRST:
        if sufficiency == StructuredSufficiency.SUFFICIENT:
            return DocumentUse.PROVENANCE_ONLY
        if role in {DocumentRole.IRRELEVANT, DocumentRole.CONFLICTING}:
            return DocumentUse.DO_NOT_USE
        return DocumentUse.INCLUDE_IN_GENERATION

    # ARBITRATED
    if role == DocumentRole.CONFLICTING:
        return DocumentUse.REQUIRE_REVIEW
    if role == DocumentRole.IRRELEVANT:
        return DocumentUse.DO_NOT_USE
    if role == DocumentRole.REDUNDANT:
        return DocumentUse.PROVENANCE_ONLY
    if role == DocumentRole.DECISIVE:
        return DocumentUse.INCLUDE_IN_GENERATION
    # SUPPORTING
    if sufficiency == StructuredSufficiency.SUFFICIENT:
        return DocumentUse.PROVENANCE_ONLY
    return DocumentUse.INCLUDE_IN_GENERATION


def arbitrate_documents(
    documents: list[CurriculumEvidence] | list[dict[str, Any]],
    *,
    structured: list[CurriculumEvidence] | list[Any] | None = None,
    question: str = "",
    grade: str | None = None,
    subject: str | None = None,
    topic: str | None = None,
    control_accepted: bool | None = None,
    control_route: str | None = None,
    control_verifier_decision: str | None = None,
    policy: ArbitrationPolicy = ArbitrationPolicy.ARBITRATED,
) -> ArbitrationDecision:
    structured_count = len(structured or [])
    sufficiency = classify_structured_sufficiency(
        control_accepted=control_accepted,
        structured_count=structured_count,
        control_route=control_route,
        control_verifier_decision=control_verifier_decision,
    )
    role, relevant_n, conflict_n, reasons = classify_document_role(
        documents,
        question=question,
        grade=grade,
        subject=subject,
        topic=topic,
        structured_sufficiency=sufficiency,
        structured_count=structured_count,
    )
    use = map_document_use(role=role, sufficiency=sufficiency, policy=policy)
    return ArbitrationDecision(
        document_role=role,
        structured_sufficiency=sufficiency,
        document_use=use,
        policy=policy,
        reasons=reasons,
        relevant_passage_count=relevant_n,
        conflicting_passage_count=conflict_n,
        document_count=len(documents or []),
        structured_count=structured_count,
    )


def select_generation_evidence(
    structured: list[CurriculumEvidence],
    documents: list[CurriculumEvidence],
    decision: ArbitrationDecision,
) -> tuple[list[CurriculumEvidence], list[CurriculumEvidence]]:
    """Return (generation_evidence, provenance_only_docs)."""
    if decision.document_use == DocumentUse.INCLUDE_IN_GENERATION:
        return list(structured) + list(documents), []
    if decision.document_use == DocumentUse.PROVENANCE_ONLY:
        return list(structured), list(documents)
    if decision.document_use == DocumentUse.REQUIRE_REVIEW:
        # Conservative: do not let conflicting docs into generation.
        return list(structured), list(documents)
    # DO_NOT_USE
    return list(structured), []


def arbitrate_from_shadow_record(
    record: dict[str, Any],
    *,
    policy: ArbitrationPolicy = ArbitrationPolicy.ARBITRATED,
) -> ArbitrationDecision:
    control = record.get("control") or {}
    shadow = record.get("shadow") or {}
    question = record.get("question") or {}
    passages = list(shadow.get("document_passages") or [])
    # Prefer any available question text; production JSONL stores hash-only.
    question_text = str(
        question.get("text")
        or question.get("question")
        or record.get("question_text")
        or ""
    )
    structured_n = int(
        shadow.get("structured_evidence_count") or control.get("evidence_count") or 0
    )
    return arbitrate_documents(
        passages,
        structured=[None] * structured_n,
        question=question_text,
        grade=question.get("grade"),
        subject=question.get("subject"),
        topic=question.get("topic"),
        control_accepted=bool(control.get("final_accepted")),
        control_route=control.get("final_route"),
        control_verifier_decision=control.get("verifier_decision"),
        policy=policy,
    )


def counterfactual_outcome(
    record: dict[str, Any],
    decision: ArbitrationDecision,
) -> dict[str, Any]:
    """Offline counterfactual without re-calling an LLM.

    INCLUDE_IN_GENERATION → keep baseline shadow outcome.
    Otherwise → keep control outcome (documents withheld from generation).
    """
    control = record.get("control") or {}
    shadow = record.get("shadow") or {}
    use_shadow = decision.document_use == DocumentUse.INCLUDE_IN_GENERATION
    source = shadow if use_shadow else control
    c_accept = bool(control.get("final_accepted"))
    s_accept = bool(source.get("final_accepted"))
    return {
        "policy": decision.policy.value,
        "document_use": decision.document_use.value,
        "document_role": decision.document_role.value,
        "structured_sufficiency": decision.structured_sufficiency.value,
        "outcome_source": "shadow" if use_shadow else "control",
        "final_accepted": s_accept,
        "final_route": source.get("final_route"),
        "verifier_decision": source.get("verifier_decision"),
        "verifier_score": source.get("verifier_score"),
        "mapper_decision": source.get("mapper_recommendation"),
        "unsupported_claim_count": len(source.get("unsupported_claims") or []),
        "answer_length": source.get("answer_length"),
        "answer_hash": source.get("answer_hash"),
        "control_correct_shadow_worse": c_accept and (not s_accept),
        "control_insufficient_shadow_accepted": (not c_accept) and s_accept,
        "document_helped": (not c_accept) and s_accept,
        "document_hurt": c_accept and (not s_accept),
        "document_neutral": c_accept == s_accept,
        "generator_drift_prevented": bool(
            c_accept
            and not bool(shadow.get("final_accepted"))
            and not use_shadow
            and (record.get("comparison") or {}).get("control_correct_shadow_worse")
        ),
    }


def aggregate_policy_comparison(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare baseline (A) vs structured-first (B) vs arbitrated (C) offline."""
    policies = (
        ArbitrationPolicy.BASELINE_MERGE,
        ArbitrationPolicy.STRUCTURED_FIRST,
        ArbitrationPolicy.ARBITRATED,
    )
    out: dict[str, Any] = {"n": len(records), "policies": {}}
    for policy in policies:
        helped = neutral = hurt = recoveries = regressions = drift_prev = 0
        unsup = 0
        safety = Counter()
        role_counts: Counter[str] = Counter()
        use_counts: Counter[str] = Counter()
        cases: list[dict[str, Any]] = []
        for record in records:
            decision = arbitrate_from_shadow_record(record, policy=policy)
            cf = counterfactual_outcome(record, decision)
            role_counts[decision.document_role.value] += 1
            use_counts[decision.document_use.value] += 1
            helped += int(cf["document_helped"])
            neutral += int(cf["document_neutral"])
            hurt += int(cf["document_hurt"])
            recoveries += int(cf["control_insufficient_shadow_accepted"])
            regressions += int(cf["control_correct_shadow_worse"])
            drift_prev += int(cf["generator_drift_prevented"])
            unsup += int(cf["unsupported_claim_count"] or 0)
            g = record.get("grounding") or {}
            sh = record.get("shadow") or {}
            # Safety gates apply only when the counterfactual still uses the
            # document-conditioned shadow outcome.
            if cf.get("outcome_source") == "shadow" and cf["final_accepted"]:
                if g.get("wrong_context"):
                    safety["wrong_context_false_acceptance"] += 1
                if g.get("placeholder_evidence"):
                    safety["placeholder_false_acceptance"] += 1
                if sh.get("metadata_blocked"):
                    safety["metadata_false_acceptance"] += 1
            cases.append(
                {
                    "question_hash": (record.get("question") or {}).get("hash"),
                    "decision": decision.to_dict(),
                    "counterfactual": cf,
                }
            )
        out["policies"][policy.value] = {
            "document_helped": helped,
            "document_neutral": neutral,
            "document_hurt": hurt,
            "recoveries": recoveries,
            "regressions": regressions,
            "generator_drift_prevented": drift_prev,
            "unsupported_claims": unsup,
            "role_counts": dict(role_counts),
            "use_counts": dict(use_counts),
            "safety": dict(safety),
            "cases": cases,
        }
    return out


__all__ = [
    "ArbitrationDecision",
    "ArbitrationPolicy",
    "DocumentRole",
    "DocumentUse",
    "StructuredSufficiency",
    "aggregate_policy_comparison",
    "arbitrate_documents",
    "arbitrate_from_shadow_record",
    "classify_document_role",
    "classify_structured_sufficiency",
    "counterfactual_outcome",
    "map_document_use",
    "select_generation_evidence",
]
