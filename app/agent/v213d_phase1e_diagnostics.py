"""V2.13D Phase 1E shadow-only diagnostic attribution.

Measurement and attribution only — does not change production answers.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from app.curriculum.evidence import CurriculumEvidence

_DOC_ID_TO_SOURCE = {
    "doc-f05cba561646": "bec-framework-2020",
    "doc-bb746307a337": "math-primary-guidance",
    "doc-a065e7c5b774": "science-guidance",
}

_URL_TO_SOURCE = {
    "bec-framework": "bec-framework-2020",
    "math-primary": "math-primary-guidance",
    "science-primary": "science-guidance",
    "science-guidance": "science-guidance",
}

REGRESSION_CAUSES = (
    "RETRIEVAL_IRRELEVANCE",
    "EVIDENCE_OVERLOAD",
    "GENERATOR_DOCUMENT_DRIFT",
    "VERIFIER_STRICTNESS",
    "MAPPER_ROUTING_CHANGE",
    "METADATA_EFFECT",
    "UNKNOWN",
)


def resolve_document_source(passage: dict[str, Any] | None) -> str | None:
    if not passage:
        return None
    doc_id = str(passage.get("document_id") or "")
    if doc_id in _DOC_ID_TO_SOURCE:
        return _DOC_ID_TO_SOURCE[doc_id]
    url = str(passage.get("source_url") or "").lower()
    for needle, source in _URL_TO_SOURCE.items():
        if needle in url:
            return source
    sid = passage.get("source_id")
    return str(sid) if sid else None


def document_sources_from_passages(passages: list[dict[str, Any]]) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for passage in passages:
        source = resolve_document_source(passage)
        if source and source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


def _passage_relevant(
    passage: dict[str, Any],
    *,
    grade: str | None,
    subject: str | None,
    topic: str | None,
) -> bool:
    if subject and passage.get("subject") and str(passage.get("subject")) == str(subject):
        return True
    if grade and passage.get("grade") and str(passage.get("grade")) == str(grade):
        return True
    if topic and passage.get("topic"):
        pt = str(passage.get("topic") or "").lower()
        qt = str(topic or "").lower()
        if qt and (qt in pt or pt in qt):
            return True
    return False


def classify_regression_cause(
    control: dict[str, Any],
    shadow: dict[str, Any],
    *,
    question_grade: str | None = None,
    question_subject: str | None = None,
) -> str:
    """Attribute control-correct → shadow-worse transitions (shadow-only)."""
    if not (
        bool(control.get("final_accepted")) and not bool(shadow.get("final_accepted"))
    ):
        return "UNKNOWN"
    if shadow.get("metadata_blocked"):
        return "METADATA_EFFECT"

    c_ver = str(control.get("verifier_decision") or "")
    s_ver = str(shadow.get("verifier_decision") or "")
    c_map = str(control.get("mapper_recommendation") or "")
    s_map = str(shadow.get("mapper_recommendation") or "")
    answer_changed = bool(control.get("answer_hash")) and bool(
        shadow.get("answer_hash")
    ) and control.get("answer_hash") != shadow.get("answer_hash")
    unsupported_up = len(shadow.get("unsupported_claims") or []) > len(
        control.get("unsupported_claims") or []
    )
    score_c = control.get("verifier_score")
    score_s = shadow.get("verifier_score")
    score_down = (
        score_c is not None
        and score_s is not None
        and float(score_s) + 1e-9 < float(score_c)
    )

    # Prefer generator/verifier attribution when the answer actually changed.
    if answer_changed and (unsupported_up or score_down or c_ver != s_ver):
        return "GENERATOR_DOCUMENT_DRIFT"

    passages = list(shadow.get("document_passages") or [])
    relevant = [
        p
        for p in passages
        if _passage_relevant(
            p, grade=question_grade, subject=question_subject, topic=None
        )
    ]
    if passages and not relevant and (question_subject or question_grade):
        return "RETRIEVAL_IRRELEVANCE"

    c_docs = int(control.get("evidence_count") or 0)
    s_docs = int(shadow.get("evidence_count") or 0)
    overload = s_docs >= c_docs + 8

    if c_ver in {"accept", "ACCEPT"} and s_ver not in {"accept", "ACCEPT"} and not answer_changed:
        return "VERIFIER_STRICTNESS"
    if c_ver == s_ver and c_map != s_map:
        return "MAPPER_ROUTING_CHANGE"
    if c_ver != s_ver and score_down and not answer_changed:
        return "VERIFIER_STRICTNESS"
    if overload:
        return "EVIDENCE_OVERLOAD"
    return "UNKNOWN"


def build_document_value_attribution(
    control: dict[str, Any],
    shadow: dict[str, Any],
    comparison: dict[str, Any],
    *,
    question_grade: str | None = None,
    question_subject: str | None = None,
    question_topic: str | None = None,
) -> dict[str, Any]:
    """Map Phase 1 categories into helped / neutral / hurt + retrieval quality."""
    primary = str(comparison.get("classification") or comparison.get("primary_category") or "")
    docs = int(shadow.get("document_evidence_count") or 0)
    passages = list(shadow.get("document_passages") or [])
    relevant_n = sum(
        1
        for p in passages
        if _passage_relevant(
            p,
            grade=question_grade,
            subject=question_subject,
            topic=question_topic,
        )
    )
    c_accept = bool(control.get("final_accepted"))
    s_accept = bool(shadow.get("final_accepted"))
    structured_sufficient = c_accept
    structured_sufficient_document_retrieved = structured_sufficient and docs > 0

    if docs <= 0:
        retrieval_quality = (
            "retrieval_failure"
            if primary
            in {
                "DOCUMENT_RETRIEVAL_FAILURE",
                "DOCUMENT_CORPUS_UNAVAILABLE",
                "DOCUMENT_NO_MATCH",
            }
            else "retrieval_failure"
        )
        if primary == "DOCUMENT_NO_MATCH":
            retrieval_quality = "retrieval_failure"
        document_effect = "neutral"
        contribution = primary or "DOCUMENT_NO_MATCH"
    elif primary in {
        "DOCUMENT_ADDED_MISSING_CONTEXT",
        "DOCUMENT_ADDED_EXPLANATION",
        "DOCUMENT_DISAMBIGUATED_CONTEXT",
    } or comparison.get("newly_recoverable"):
        retrieval_quality = "retrieval_decisive"
        document_effect = "helped"
        contribution = primary
    elif primary == "DOCUMENT_NOISE" or comparison.get("control_correct_shadow_worse"):
        retrieval_quality = "retrieval_noise"
        document_effect = "hurt"
        contribution = "DOCUMENT_NOISE"
    elif primary == "STRUCTURED_DATA_ALREADY_SUFFICIENT":
        retrieval_quality = (
            "retrieval_non_decisive" if docs > 0 else "retrieval_failure"
        )
        document_effect = "neutral"
        contribution = primary
    elif primary == "DOCUMENT_DID_NOT_HELP":
        if relevant_n == 0 and (question_subject or question_grade):
            retrieval_quality = "retrieval_irrelevant"
            contribution = "DOCUMENT_DID_NOT_HELP"
        else:
            retrieval_quality = "retrieval_non_decisive"
            contribution = "DOCUMENT_DID_NOT_HELP"
        document_effect = "neutral"
    else:
        retrieval_quality = "retrieval_success" if docs > 0 else "retrieval_failure"
        document_effect = "neutral"
        contribution = primary or "UNKNOWN"

    same = c_accept and s_accept
    better = (not c_accept) and s_accept
    worse = c_accept and (not s_accept)

    return {
        "contribution": contribution,
        "document_effect": document_effect,
        "retrieval_quality": retrieval_quality,
        "retrieval_success": docs > 0,
        "retrieval_relevant": relevant_n > 0,
        "relevant_passage_count": relevant_n,
        "structured_sufficient_document_retrieved": structured_sufficient_document_retrieved,
        "control_correct_shadow_same": same,
        "control_correct_shadow_better": better,
        "control_correct_shadow_worse": worse,
        "document_sources": document_sources_from_passages(passages),
    }


def _generator_input_summary(side: dict[str, Any], *, role: str) -> dict[str, Any]:
    """Compact generator-input fingerprint (no answer text / no production change)."""
    summary = side.get("evidence_summary") or []
    entity_types: list[str] = []
    grades: list[str] = []
    subjects: list[str] = []
    if isinstance(summary, list):
        for row in summary[:40]:
            if not isinstance(row, dict):
                continue
            et = row.get("entity_type")
            if et:
                entity_types.append(str(et))
            if row.get("grade"):
                grades.append(str(row["grade"]))
            if row.get("subject"):
                subjects.append(str(row["subject"]))
    elif isinstance(summary, dict):
        entity_types = [str(x) for x in (summary.get("entity_types") or [])][:20]
        grades = [str(x) for x in (summary.get("grades") or [])][:10]
        subjects = [str(x) for x in (summary.get("subjects") or [])][:10]
    return {
        "role": role,
        "evidence_snapshot": side.get("evidence_snapshot"),
        "evidence_count": int(
            side.get("evidence_count")
            or side.get("merged_evidence_count")
            or 0
        ),
        "structured_evidence_count": int(side.get("structured_evidence_count") or 0)
        if role == "shadow"
        else int(side.get("evidence_count") or 0),
        "document_evidence_count": int(side.get("document_evidence_count") or 0)
        if role == "shadow"
        else 0,
        "entity_types": entity_types[:20],
        "grades": sorted(set(grades))[:10],
        "subjects": sorted(set(subjects))[:10],
    }


def build_phase1e_diagnostics(
    control: dict[str, Any],
    shadow: dict[str, Any],
    comparison: dict[str, Any],
    *,
    question_grade: str | None = None,
    question_subject: str | None = None,
    question_topic: str | None = None,
    control_answer: str = "",
    shadow_answer: str = "",
) -> dict[str, Any]:
    attribution = build_document_value_attribution(
        control,
        shadow,
        comparison,
        question_grade=question_grade,
        question_subject=question_subject,
        question_topic=question_topic,
    )
    regression_cause = "UNKNOWN"
    if comparison.get("control_correct_shadow_worse") or comparison.get("regressed"):
        regression_cause = classify_regression_cause(
            control,
            shadow,
            question_grade=question_grade,
            question_subject=question_subject,
        )

    control_len = len(control_answer) if control_answer else None
    shadow_len = len(shadow_answer) if shadow_answer else None
    # Prefer explicit lengths if callers already hashed-only answers.
    if control_len is None and control.get("answer_length") is not None:
        control_len = int(control.get("answer_length") or 0)
    if shadow_len is None and shadow.get("answer_length") is not None:
        shadow_len = int(shadow.get("answer_length") or 0)

    control_input = _generator_input_summary(control, role="control")
    shadow_input = _generator_input_summary(shadow, role="shadow")

    return {
        "phase": "phase1e",
        "control": {
            "structured_evidence_count": int(control.get("evidence_count") or 0),
            "generator_input": control_input,
            "verifier_decision": control.get("verifier_decision"),
            "verifier_score": control.get("verifier_score"),
            "mapper_decision": control.get("mapper_recommendation"),
            "final_route": control.get("final_route"),
            "final_accepted": bool(control.get("final_accepted")),
            "unsupported_claim_count": len(control.get("unsupported_claims") or []),
            "answer_length": control_len,
            "answer_hash": control.get("answer_hash"),
        },
        "shadow": {
            "structured_evidence_count": int(
                shadow.get("structured_evidence_count") or 0
            ),
            "document_evidence_count": int(shadow.get("document_evidence_count") or 0),
            "merged_evidence_count": int(shadow.get("evidence_count") or 0),
            "generator_input": shadow_input,
            "verifier_decision": shadow.get("verifier_decision"),
            "verifier_score": shadow.get("verifier_score"),
            "mapper_decision": shadow.get("mapper_recommendation"),
            "final_route": shadow.get("final_route"),
            "final_accepted": bool(shadow.get("final_accepted")),
            "unsupported_claim_count": len(shadow.get("unsupported_claims") or []),
            "answer_length": shadow_len,
            "answer_hash": shadow.get("answer_hash"),
            "document_sources": attribution.get("document_sources"),
        },
        "deltas": {
            "route": f"{control.get('final_route')}→{shadow.get('final_route')}",
            "verifier": (
                f"{control.get('verifier_decision')}→{shadow.get('verifier_decision')}"
            ),
            "mapper": (
                f"{control.get('mapper_recommendation')}→"
                f"{shadow.get('mapper_recommendation')}"
            ),
            "accepted": (
                f"{bool(control.get('final_accepted'))}→"
                f"{bool(shadow.get('final_accepted'))}"
            ),
            "evidence_count_delta": int(shadow.get("evidence_count") or 0)
            - int(control.get("evidence_count") or 0),
            "unsupported_claim_delta": len(shadow.get("unsupported_claims") or [])
            - len(control.get("unsupported_claims") or []),
            "answer_changed": bool(control.get("answer_hash"))
            and bool(shadow.get("answer_hash"))
            and control.get("answer_hash") != shadow.get("answer_hash"),
            "generator_input_changed": control_input.get("evidence_snapshot")
            != shadow_input.get("evidence_snapshot")
            or int(shadow.get("document_evidence_count") or 0) > 0,
        },
        "document_value": attribution,
        "regression_cause": regression_cause,
    }


def enrich_existing_record(record: dict[str, Any]) -> dict[str, Any]:
    """Attach Phase 1E diagnostics to an existing JSONL row (offline / report)."""
    if record.get("diagnostics", {}).get("phase") == "phase1e":
        return record
    control = record.get("control") or {}
    shadow = record.get("shadow") or {}
    comparison = record.get("comparison") or {}
    question = record.get("question") or {}
    diagnostics = build_phase1e_diagnostics(
        control,
        shadow,
        comparison,
        question_grade=question.get("grade"),
        question_subject=question.get("subject"),
        question_topic=question.get("topic"),
    )
    out = dict(record)
    out["diagnostics"] = diagnostics
    # Mirror key fields onto comparison for aggregator convenience.
    comparison = dict(comparison)
    comparison["document_effect"] = diagnostics["document_value"]["document_effect"]
    comparison["retrieval_quality"] = diagnostics["document_value"]["retrieval_quality"]
    comparison["regression_cause"] = diagnostics.get("regression_cause")
    comparison["structured_sufficient_document_retrieved"] = diagnostics[
        "document_value"
    ]["structured_sufficient_document_retrieved"]
    out["comparison"] = comparison
    return out


def _transition_counts(
    records: list[dict[str, Any]], left_key: str, right_key: str
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        left = (record.get("control") or {}).get(left_key)
        right = (record.get("shadow") or {}).get(right_key)
        counts[f"{left}→{right}"] += 1
    return dict(counts)


def segment_and_transitions(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Phase 1E transition matrices and segmentations."""
    enriched = [enrich_existing_record(r) for r in records]
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    by_grade: dict[str, Counter[str]] = defaultdict(Counter)
    by_subject: dict[str, Counter[str]] = defaultdict(Counter)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    effect_counts: Counter[str] = Counter()
    retrieval_quality_counts: Counter[str] = Counter()
    regression_causes: Counter[str] = Counter()
    structured_sufficient_doc = 0
    control_correct_same = 0
    control_correct_better = 0
    control_correct_worse = 0

    for record in enriched:
        q = record.get("question") or {}
        comparison = record.get("comparison") or {}
        diagnostics = record.get("diagnostics") or {}
        value = diagnostics.get("document_value") or {}
        primary = str(comparison.get("classification") or "UNKNOWN")
        effect = str(value.get("document_effect") or "neutral")
        rq = str(value.get("retrieval_quality") or "unknown")
        effect_counts[effect] += 1
        retrieval_quality_counts[rq] += 1
        by_category[str(q.get("category") or "unknown")][primary] += 1
        by_grade[str(q.get("grade") or "unknown")][primary] += 1
        by_subject[str(q.get("subject") or "unknown")][primary] += 1
        for source in value.get("document_sources") or ["unknown"]:
            by_source[str(source)][primary] += 1
        if value.get("structured_sufficient_document_retrieved"):
            structured_sufficient_doc += 1
        if value.get("control_correct_shadow_same"):
            control_correct_same += 1
        if value.get("control_correct_shadow_better"):
            control_correct_better += 1
        if value.get("control_correct_shadow_worse"):
            control_correct_worse += 1
        cause = diagnostics.get("regression_cause")
        if comparison.get("control_correct_shadow_worse"):
            regression_causes[str(cause or "UNKNOWN")] += 1

    key_routes = {
        k: v
        for k, v in _transition_counts(enriched, "final_route", "final_route").items()
        if k
        in {
            "fallback→finish",
            "retrieve_more→finish",
            "finish→fallback",
            "finish→retrieve_more",
        }
        or True
    }

    return {
        "transitions": {
            "route": _transition_counts(enriched, "final_route", "final_route"),
            "verifier": _transition_counts(
                enriched, "verifier_decision", "verifier_decision"
            ),
            "mapper": _transition_counts(
                enriched, "mapper_recommendation", "mapper_recommendation"
            ),
            "key_route_transitions": {
                k: key_routes.get(k, 0)
                for k in (
                    "fallback→finish",
                    "retrieve_more→finish",
                    "finish→fallback",
                    "finish→retrieve_more",
                )
            },
        },
        "document_effect_counts": dict(effect_counts),
        "retrieval_quality_counts": dict(retrieval_quality_counts),
        "regression_cause_counts": dict(regression_causes),
        "structured_sufficient_document_retrieved": structured_sufficient_doc,
        "control_correct_shadow_same": control_correct_same,
        "control_correct_shadow_better": control_correct_better,
        "control_correct_shadow_worse": control_correct_worse,
        "segmentation": {
            "by_category": {k: dict(v) for k, v in by_category.items()},
            "by_grade": {k: dict(v) for k, v in by_grade.items()},
            "by_subject": {k: dict(v) for k, v in by_subject.items()},
            "by_source": {k: dict(v) for k, v in by_source.items()},
        },
        "document_helped": int(effect_counts.get("helped") or 0),
        "document_neutral": int(effect_counts.get("neutral") or 0),
        "document_hurt": int(effect_counts.get("hurt") or 0),
    }


def evidence_entity_types(evidence: list[CurriculumEvidence]) -> list[str]:
    return [e.entity_type for e in evidence[:50]]
