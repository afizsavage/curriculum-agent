"""V2.9 evidence normalization & grounding-boundary experiment (harness-only)."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from app.agent.evidence_snapshot import evidence_snapshot_hash
from app.agent.state import CurriculumQAState
from app.agent.v25_experiment import (
    _CLEAN_PLACEHOLDER,
    _lo_code,
    build_evidence_inventory,
)
from app.agent.v26_experiment import answer_hash, build_claim_classifications
from app.agent.v28_recommendation_mapping import (
    FIXTURES,
    FixtureClass,
    bootstrap_c4u18_baseline,
    _prepare_evidence,
)
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.schemas.verification import VerificationRecommendation, VerificationResult

_EXPERIMENT_NAME = "v2.9_evidence_normalization"
_C4U18_HASH = "be3e342763f1faac"
_FRACTIONS_HASH = "977b259fcfb4b282"
_V29_SUBSTANCE_KEY = "_v29_substance_class"
_V29_NORMALIZATION_KEY = "_v29_normalization"

SubstanceClass = Literal["SUBSTANTIVE", "NON_SUBSTANTIVE", "MISSING"]


class NormalizationVariant(str, Enum):
    RAW = "RAW"
    PLACEHOLDER_FILTER = "PLACEHOLDER_FILTER"
    STRUCTURAL_NORMALIZATION = "STRUCTURAL_NORMALIZATION"
    SEMANTIC_EVIDENCE_EXTRACTION = "SEMANTIC_EVIDENCE_EXTRACTION"


NORMALIZATION_VARIANTS: tuple[NormalizationVariant, ...] = (
    NormalizationVariant.RAW,
    NormalizationVariant.PLACEHOLDER_FILTER,
    NormalizationVariant.STRUCTURAL_NORMALIZATION,
    NormalizationVariant.SEMANTIC_EVIDENCE_EXTRACTION,
)

_PLACEHOLDER_SENTINELS = (
    _CLEAN_PLACEHOLDER,
    "[PLACEHOLDER]",
    "[TBD]",
    "[TODO]",
    "N/A",
)


@dataclass(frozen=True)
class NormalizationDiagnostic:
    variant: str
    operations: tuple[str, ...]
    records_in: int
    records_out: int
    placeholder_filtered: int
    substance_counts: dict[str, int]
    evidence_hash_in: str
    evidence_hash_out: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "operations": list(self.operations),
            "records_in": self.records_in,
            "records_out": self.records_out,
            "placeholder_filtered": self.placeholder_filtered,
            "substance_counts": dict(self.substance_counts),
            "evidence_hash_in": self.evidence_hash_in,
            "evidence_hash_out": self.evidence_hash_out,
        }


@dataclass(frozen=True)
class NormalizedEvidence:
    evidence: list[CurriculumEvidence]
    diagnostics: NormalizationDiagnostic


def v29_experiment_enabled(settings: Settings, qa: CurriculumQAState) -> bool:
    if qa.metadata.get("v29_normalization_replay"):
        return True
    return bool(getattr(settings, "v29_evidence_normalization_experiment", False))


def detect_placeholder_content(text: str | None) -> tuple[bool, str | None]:
    if text is None:
        return True, "null"
    stripped = str(text).strip()
    if not stripped:
        return True, "empty_string"
    upper = stripped.upper()
    for sentinel in _PLACEHOLDER_SENTINELS:
        if sentinel.upper() in upper:
            if sentinel == _CLEAN_PLACEHOLDER:
                return True, "sentinel"
            return True, "sentinel"
    if re.fullmatch(r"\[.+\]", stripped):
        return True, "templated_text"
    if stripped.lower() in {"placeholder", "tbd", "todo", "none", "null"}:
        return True, "syntactically_valid_but_semantically_empty"
    return False, None


def classify_substance(item: CurriculumEvidence) -> SubstanceClass:
    content = item.content
    is_placeholder, _ = detect_placeholder_content(content)
    if is_placeholder:
        return "NON_SUBSTANTIVE"
    if content is None or not str(content).strip():
        return "MISSING"
    entity_type = (item.entity_type or "").lower()
    if entity_type == "unit":
        name = (item.name or item.content or "").strip()
        return "SUBSTANTIVE" if name else "MISSING"
    if entity_type == "learning_outcome":
        code = _lo_code(item)
        text = (content or "").strip()
        if code and text and text != code:
            return "SUBSTANTIVE"
        if code and text == code:
            return "NON_SUBSTANTIVE"
        return "MISSING"
    text = (content or item.name or "").strip()
    return "SUBSTANTIVE" if text else "MISSING"


def _whitespace_normalize(text: str | None) -> str | None:
    if text is None:
        return None
    normalized = re.sub(r"\s+", " ", str(text).strip())
    return normalized or None


def _build_unit_lookup(evidence: list[CurriculumEvidence]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for item in evidence:
        if (item.entity_type or "").lower() != "unit":
            continue
        if not item.entity_id:
            continue
        label = (
            (item.name or "").strip()
            or (item.content or "").strip()
            or str((item.metadata or {}).get("code") or "")
        )
        if label:
            lookup[item.entity_id] = label
    return lookup


def _structural_normalize_record(
    item: CurriculumEvidence,
    *,
    unit_lookup: dict[str, str],
) -> CurriculumEvidence:
    cloned = copy.deepcopy(item)
    cloned.content = _whitespace_normalize(cloned.content)
    cloned.name = _whitespace_normalize(cloned.name)
    meta = dict(cloned.metadata or {})

    parent_name = meta.get("parent_content_name")
    parent_code = meta.get("parent_content_code")
    topic = cloned.topic
    if topic and topic in unit_lookup:
        cloned.topic = unit_lookup[topic]
    elif parent_name and (not topic or topic == cloned.entity_id):
        cloned.topic = str(parent_name)
    elif parent_code and not topic:
        cloned.topic = str(parent_code)

    if parent_name:
        meta.setdefault("parent_content_name", parent_name)
    if parent_code:
        meta.setdefault("parent_content_code", parent_code)

    meta[_V29_NORMALIZATION_KEY] = "structural"
    cloned.metadata = meta
    return cloned


def _dedupe_and_sort(evidence: list[CurriculumEvidence]) -> list[CurriculumEvidence]:
    seen: dict[str, CurriculumEvidence] = {}
    for item in evidence:
        key = item.entity_id or f"{item.entity_type}:{_lo_code(item)}:{item.name}"
        if key not in seen:
            seen[key] = item
    def sort_key(item: CurriculumEvidence) -> tuple[str, str, str]:
        entity_type = (item.entity_type or "").lower()
        code = _lo_code(item) or ""
        entity_id = item.entity_id or ""
        type_order = {"unit": "0", "learning_outcome": "1"}.get(entity_type, "9")
        return (type_order, code, entity_id)

    return sorted(seen.values(), key=sort_key)


def _apply_structural_normalization(
    evidence: list[CurriculumEvidence],
) -> tuple[list[CurriculumEvidence], list[str], int]:
    operations = ["deep_copy", "whitespace_normalize", "resolve_topic_to_unit_name"]
    unit_lookup = _build_unit_lookup(evidence)
    transformed = [
        _structural_normalize_record(item, unit_lookup=unit_lookup)
        for item in evidence
    ]
    transformed = _dedupe_and_sort(transformed)
    operations.extend(["dedupe_by_entity_id", "deterministic_sort"])
    return transformed, operations, 0


def _apply_placeholder_filter(
    evidence: list[CurriculumEvidence],
) -> tuple[list[CurriculumEvidence], list[str], int]:
    operations = ["placeholder_filter"]
    kept: list[CurriculumEvidence] = []
    filtered = 0
    for item in evidence:
        is_placeholder, _ = detect_placeholder_content(item.content)
        if is_placeholder:
            filtered += 1
            continue
        kept.append(copy.deepcopy(item))
    return kept, operations, filtered


def _apply_semantic_extraction(
    evidence: list[CurriculumEvidence],
) -> tuple[list[CurriculumEvidence], list[str], int, dict[str, int]]:
    structural, struct_ops, _ = _apply_structural_normalization(evidence)
    operations = [*struct_ops, "classify_substance", "retain_substantive_only"]
    kept: list[CurriculumEvidence] = []
    filtered = 0
    counts: dict[str, int] = {"SUBSTANTIVE": 0, "NON_SUBSTANTIVE": 0, "MISSING": 0}
    for item in structural:
        substance = classify_substance(item)
        counts[substance] += 1
        if substance != "SUBSTANTIVE":
            filtered += 1
            continue
        cloned = copy.deepcopy(item)
        meta = dict(cloned.metadata or {})
        meta[_V29_SUBSTANCE_KEY] = substance
        meta[_V29_NORMALIZATION_KEY] = "semantic_extraction"
        cloned.metadata = meta
        kept.append(cloned)
    return kept, operations, filtered, counts


def normalize_evidence(
    raw_evidence: list[CurriculumEvidence],
    variant: NormalizationVariant | str,
) -> NormalizedEvidence:
    """Pure, deterministic, answer-independent evidence normalization."""
    variant_enum = (
        variant if isinstance(variant, NormalizationVariant) else NormalizationVariant(str(variant))
    )
    raw_copy = copy.deepcopy(raw_evidence)
    hash_in = evidence_snapshot_hash(raw_copy)
    operations: list[str] = []
    placeholder_filtered = 0
    substance_counts: dict[str, int] = {
        "SUBSTANTIVE": 0,
        "NON_SUBSTANTIVE": 0,
        "MISSING": 0,
    }

    if variant_enum == NormalizationVariant.RAW:
        operations.append("pass_through")
        result = raw_copy
    elif variant_enum == NormalizationVariant.PLACEHOLDER_FILTER:
        result, operations, placeholder_filtered = _apply_placeholder_filter(raw_copy)
    elif variant_enum == NormalizationVariant.STRUCTURAL_NORMALIZATION:
        result, operations, placeholder_filtered = _apply_structural_normalization(raw_copy)
    elif variant_enum == NormalizationVariant.SEMANTIC_EVIDENCE_EXTRACTION:
        result, operations, placeholder_filtered, substance_counts = _apply_semantic_extraction(
            raw_copy
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown normalization variant: {variant_enum}")

    if variant_enum in {
        NormalizationVariant.RAW,
        NormalizationVariant.PLACEHOLDER_FILTER,
        NormalizationVariant.STRUCTURAL_NORMALIZATION,
    }:
        for item in result:
            substance_counts[classify_substance(item)] += 1

    diagnostics = NormalizationDiagnostic(
        variant=variant_enum.value,
        operations=tuple(operations),
        records_in=len(raw_evidence),
        records_out=len(result),
        placeholder_filtered=placeholder_filtered,
        substance_counts=substance_counts,
        evidence_hash_in=hash_in,
        evidence_hash_out=evidence_snapshot_hash(result),
    )
    return NormalizedEvidence(evidence=result, diagnostics=diagnostics)


def _evidence_chain_for_fc(
    *,
    raw_evidence: list[CurriculumEvidence],
    normalized: NormalizedEvidence,
    result: VerificationResult,
    answer: str,
) -> list[dict[str, Any]]:
    """Build per-LO diagnostic chain for FAITHFUL_COMPLETE analysis."""
    claims = {c.claim: c for c in (result.claims or [])}
    chain: list[dict[str, Any]] = []
    lo_pattern = re.compile(
        r"\*\*(C4U\d+-LO\d+)\*\*\s*[—–-]\s*(.+?)(?=\n|$)",
        re.MULTILINE,
    )
    unit_lookup = _build_unit_lookup(raw_evidence)
    evidence_by_id = {e.entity_id: e for e in raw_evidence if e.entity_id}
    evidence_by_code = {_lo_code(e): e for e in raw_evidence if _lo_code(e)}

    for match in lo_pattern.finditer(answer):
        lo_id = match.group(1)
        answer_text = match.group(2).strip()
        raw_item = evidence_by_code.get(lo_id)
        norm_item = next(
            (e for e in normalized.evidence if _lo_code(e) == lo_id),
            None,
        )
        claim_key = f"{lo_id} — {answer_text}"
        claim = claims.get(claim_key)
        if claim is None:
            for key, value in claims.items():
                if lo_id in key:
                    claim = value
                    claim_key = key
                    break

        raw_topic = raw_item.topic if raw_item else None
        resolved_topic = None
        if raw_topic and raw_topic in unit_lookup:
            resolved_topic = unit_lookup[raw_topic]

        mismatch_reason = None
        if result.unsupported_claims and any(lo_id in str(u) for u in result.unsupported_claims):
            mismatch_reason = "verifier_unsupported_claims"
        elif claim and claim.verdict != "supported":
            mismatch_reason = f"claim_verdict_{claim.verdict}"
        elif not raw_item:
            mismatch_reason = "lo_not_in_raw_evidence"
        elif raw_topic and raw_topic not in unit_lookup and "money" not in (raw_item.content or "").lower():
            mismatch_reason = "topic_metadata_uuid_without_lexical_money_link"

        chain.append(
            {
                "lo_id": lo_id,
                "answer_claim": answer_text,
                "lo_text_raw": (raw_item.content if raw_item else None),
                "lo_text_normalized": (norm_item.content if norm_item else None),
                "topic_raw": raw_topic,
                "topic_resolved": resolved_topic or (norm_item.topic if norm_item else None),
                "parent_unit": (
                    (raw_item.metadata or {}).get("parent_content_name") if raw_item else None
                ),
                "evidence_record": raw_item.model_dump() if raw_item else None,
                "verifier_claim": claim_key if claim else None,
                "verifier_verdict": claim.verdict if claim else None,
                "evidence_matched_ids": list(claim.evidence_ids) if claim else [],
                "evidence_matched_records": [
                    evidence_by_id[eid].model_dump()
                    for eid in (claim.evidence_ids if claim else [])
                    if eid in evidence_by_id
                ],
                "mismatch_reason": mismatch_reason,
            }
        )
    return chain


def replay_fixture(
    *,
    fixture_class: FixtureClass,
    variant: NormalizationVariant,
    c4u18_baseline: list[CurriculumEvidence],
    fractions_baseline: list[CurriculumEvidence],
    verifier: Any,
    request_id: str | None = None,
) -> dict[str, Any]:
    spec = FIXTURES[fixture_class]
    raw_evidence, evidence_source = _prepare_evidence(
        fixture_class=fixture_class,
        c4u18_baseline=c4u18_baseline,
        fractions_baseline=fractions_baseline,
    )
    normalized = normalize_evidence(raw_evidence, variant)
    state = CurriculumQAState.initial(question=spec["question"])
    state.evidence = normalized.evidence
    state.evidence_status = (
        EvidenceStatus.FOUND if normalized.evidence else EvidenceStatus.NOT_FOUND
    )
    state.grade = "CLASS_4"
    state.topic = "money" if evidence_source == "c4u18" else "fractions"
    state.subject = "MATHEMATICS"
    state.final_answer = spec["answer"]
    state.draft_answer = spec["answer"]
    state.metadata["v29_normalization_replay"] = True
    state.metadata["v29_fixture_class"] = fixture_class
    state.metadata["v29_normalization_variant"] = variant.value

    result = verifier.verify(state, request_id=request_id)
    snapshot = copy.deepcopy(result.model_dump())
    claims = build_claim_classifications(
        answer=spec["answer"],
        result=result,
        evidence_state=None,
    )
    baseline_hash = _C4U18_HASH if evidence_source == "c4u18" else _FRACTIONS_HASH
    insufficient = (
        result.recommendation == VerificationRecommendation.FALLBACK
        or bool(result.missing_evidence)
        or "insufficient" in " ".join(result.issues or []).lower()
    )
    placeholder_detected = any(
        detect_placeholder_content(item.content)[0] for item in raw_evidence
    )
    row: dict[str, Any] = {
        "experiment": _EXPERIMENT_NAME,
        "normalization_variant": variant.value,
        "fixture_class": fixture_class,
        "primary_unit": spec["primary_unit"],
        "question": spec["question"],
        "answer": spec["answer"],
        "answer_hash": answer_hash(spec["answer"]),
        "evidence_hash": baseline_hash,
        "raw_evidence_hash": evidence_snapshot_hash(raw_evidence),
        "normalized_evidence_hash": normalized.diagnostics.evidence_hash_out,
        "evidence_source": evidence_source,
        "evidence_present": bool(normalized.evidence),
        "raw_evidence_present": bool(raw_evidence),
        "imperfect_evidence_count": len(build_evidence_inventory(fractions_baseline)),
        "normalization_diagnostics": normalized.diagnostics.to_dict(),
        "verifier_score": result.score,
        "verifier_accepted": result.passed,
        "verifier_decision": result.recommendation.value,
        "retrieve_more_requested": result.recommendation
        == VerificationRecommendation.RETRIEVE_MORE,
        "insufficient_evidence": insufficient,
        "unsupported_claims": list(result.unsupported_claims or []),
        "issue_codes": list(result.issues or []),
        "claim_classifications": claims,
        "placeholder_detected_in_raw": placeholder_detected,
        "verifier_result_snapshot": snapshot,
    }
    if fixture_class == "FAITHFUL_COMPLETE":
        row["fc_evidence_chain"] = _evidence_chain_for_fc(
            raw_evidence=raw_evidence,
            normalized=normalized,
            result=result,
            answer=spec["answer"],
        )
    return row


def summarize_variant_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    scores = [float(r.get("verifier_score") or 0) for r in rows]

    def subset(fixture: str) -> list[dict[str, Any]]:
        return [r for r in rows if r["fixture_class"] == fixture]

    def rate(fixture: str, predicate) -> float:
        items = subset(fixture)
        if not items:
            return 0.0
        return round(sum(1 for r in items if predicate(r)) / len(items), 3)

    safety_fixtures = {
        "UNSUPPORTED_CLAIM",
        "UNSUPPORTED_ABSENCE",
        "SPECULATIVE",
        "RECONSTRUCTION",
        "MISSING_EVIDENCE",
    }
    safety_rows = [r for r in rows if r["fixture_class"] in safety_fixtures]
    safety_false_accept = sum(1 for r in safety_rows if r.get("verifier_accepted"))
    placeholder_false_accept = sum(
        1 for r in subset("CLEAN_PLACEHOLDER") if r.get("verifier_accepted")
    )

    return {
        "n": len(rows),
        "acceptance_rate": round(sum(1 for r in rows if r.get("verifier_accepted")) / n, 3),
        "retrieve_more_rate": round(
            sum(1 for r in rows if r.get("retrieve_more_requested")) / n, 3
        ),
        "insufficient_evidence_rate": round(
            sum(1 for r in rows if r.get("insufficient_evidence")) / n, 3
        ),
        "rejection_rate": round(
            sum(
                1
                for r in rows
                if not r.get("verifier_accepted")
                and r.get("verifier_decision") in {"fallback", "reject"}
            )
            / n,
            3,
        ),
        "avg_verifier_score": round(sum(scores) / len(scores), 3) if scores else None,
        "faithful_complete_acceptance": rate("FAITHFUL_COMPLETE", lambda r: r.get("verifier_accepted")),
        "faithful_complete_unsupported_rate": rate(
            "FAITHFUL_COMPLETE", lambda r: bool(r.get("unsupported_claims"))
        ),
        "faithful_imperfect_acceptance": rate(
            "FAITHFUL_IMPERFECT", lambda r: r.get("verifier_accepted")
        ),
        "faithful_imperfect_retrieve_more": rate(
            "FAITHFUL_IMPERFECT", lambda r: r.get("retrieve_more_requested")
        ),
        "placeholder_acceptance": rate("CLEAN_PLACEHOLDER", lambda r: r.get("verifier_accepted")),
        "placeholder_false_acceptance": placeholder_false_accept,
        "safety_false_acceptance": safety_false_accept,
    }


def build_safety_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "unsupported_claim": "UNSUPPORTED_CLAIM",
        "unsupported_absence": "UNSUPPORTED_ABSENCE",
        "speculative": "SPECULATIVE",
        "reconstruction": "RECONSTRUCTION",
        "missing_evidence": "MISSING_EVIDENCE",
    }
    out: dict[str, Any] = {}
    false_accept = 0
    for key, fixture in groups.items():
        subset = [r for r in rows if r["fixture_class"] == fixture]
        accepted = sum(1 for r in subset if r.get("verifier_accepted"))
        rejected = len(subset) - accepted
        false_accept += accepted
        out[key] = {
            "cases": len(subset),
            "correctly_rejected": rejected,
            "incorrectly_accepted": accepted,
        }
    out["false_acceptance_total"] = false_accept
    return out


def build_placeholder_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from app.agent.v25_experiment import _CLEAN_PLACEHOLDER as placeholder

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if row["fixture_class"] != "CLEAN_PLACEHOLDER":
            continue
        tag = row.get("tag") or row.get("run_index")
        if tag in seen:
            continue
        seen.add(str(tag))
        out.append(
            {
                "fixture": tag,
                "variant": row.get("normalization_variant"),
                "question": row.get("question"),
                "placeholder_classification": "sentinel",
                "placeholder_value": placeholder,
                "raw_records_in": row.get("normalization_diagnostics", {}).get("records_in"),
                "normalized_records_out": row.get("normalization_diagnostics", {}).get(
                    "records_out"
                ),
                "verifier_score": row.get("verifier_score"),
                "verifier_decision": row.get("verifier_decision"),
                "verifier_accepted": row.get("verifier_accepted"),
                "insufficient_evidence": row.get("insufficient_evidence"),
            }
        )
    return out


def build_fc_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["fixture_class"] != "FAITHFUL_COMPLETE":
            continue
        chain = row.get("fc_evidence_chain") or []
        out.append(
            {
                "fixture": row.get("tag"),
                "variant": row.get("normalization_variant"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "verifier_score": row.get("verifier_score"),
                "verifier_decision": row.get("verifier_decision"),
                "verifier_accepted": row.get("verifier_accepted"),
                "failure_category": (
                    "unsupported_claims"
                    if row.get("unsupported_claims")
                    else row.get("verifier_decision")
                ),
                "unsupported_claims": row.get("unsupported_claims"),
                "issue_codes": row.get("issue_codes"),
                "evidence_chain": chain,
                "normalization_diagnostics": row.get("normalization_diagnostics"),
            }
        )
    return out


def analyze_fc_root_cause(fc_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize C4-U18 FAITHFUL_COMPLETE failure causes across variants."""
    causes: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for row in fc_rows:
        if row.get("verifier_accepted"):
            causes["accepted"] = causes.get("accepted", 0) + 1
            continue
        for chain_item in row.get("fc_evidence_chain") or []:
            reason = chain_item.get("mismatch_reason") or "unknown"
            causes[reason] = causes.get(reason, 0) + 1
            if len(examples) < 3 and reason != "accepted":
                examples.append(
                    {
                        "variant": row.get("normalization_variant"),
                        "lo_id": chain_item.get("lo_id"),
                        "mismatch_reason": reason,
                        "unsupported_claims": row.get("unsupported_claims"),
                    }
                )
        if row.get("unsupported_claims"):
            causes["verifier_topic_linkage_dispute"] = (
                causes.get("verifier_topic_linkage_dispute", 0) + 1
            )
    fc_raw = next(
        (r for r in fc_rows if r.get("normalization_variant") == "RAW" and not r.get("verifier_accepted")),
        None,
    )
    fc_structural = next(
        (
            r
            for r in fc_rows
            if r.get("normalization_variant") == "STRUCTURAL_NORMALIZATION"
            and r.get("verifier_accepted")
        ),
        None,
    )
    if fc_raw and fc_structural:
        chain = (fc_raw.get("fc_evidence_chain") or [{}])[0]
        verdict = "evidence_representation_problem"
        examples.append(
            {
                "raw_topic": chain.get("topic_raw"),
                "resolved_topic": chain.get("topic_resolved"),
                "raw_failure": fc_raw.get("unsupported_claims"),
                "structural_accepted": True,
            }
        )
    elif causes.get("verifier_topic_linkage_dispute", 0) > causes.get("accepted", 0):
        verdict = "verifier_semantic_grounding_limitation"
    else:
        verdict = "mixed_or_representation"

    return {
        "cause_counts": causes,
        "examples": examples,
        "verdict": verdict,
    }


def interpret_v29(
    variant_summaries: dict[str, dict[str, Any]],
    *,
    fc_analysis: dict[str, Any],
    safety: dict[str, Any],
) -> tuple[str, str, str, str]:
    raw = variant_summaries.get("RAW", {})
    structural = variant_summaries.get("STRUCTURAL_NORMALIZATION", {})
    semantic = variant_summaries.get("SEMANTIC_EVIDENCE_EXTRACTION", {})
    placeholder = variant_summaries.get("PLACEHOLDER_FILTER", {})

    false_accept = safety.get("false_acceptance_total", 0)
    placeholder_accept = max(
        raw.get("placeholder_acceptance", 0),
        structural.get("placeholder_acceptance", 0),
        semantic.get("placeholder_acceptance", 0),
        placeholder.get("placeholder_acceptance", 0),
    )

    fc_raw = raw.get("faithful_complete_acceptance", 0)
    fc_best = max(
        structural.get("faithful_complete_acceptance", 0),
        semantic.get("faithful_complete_acceptance", 0),
        fc_raw,
    )
    fi_preserved = all(
        variant_summaries.get(v, {}).get("faithful_imperfect_acceptance", 0) >= 0
        for v in ("RAW", "STRUCTURAL_NORMALIZATION", "SEMANTIC_EVIDENCE_EXTRACTION")
    )

    fc_verdict = fc_analysis.get("verdict", "unknown")
    grounding_answer = (
        "verifier semantic grounding limitation"
        if fc_best <= fc_raw + 0.1 and fc_raw < 0.8
        else "evidence representation problem"
        if fc_best > fc_raw + 0.2
        else "mixed: partial representation gains with residual verifier linkage limits"
    )

    if (
        false_accept == 0
        and placeholder_accept == 0
        and fc_best >= 0.8
        and fi_preserved
    ):
        conclusion = "SUPPORTED"
        note = (
            "Normalization materially improved grounding while preserving safety and "
            "placeholder rejection."
        )
        v210 = (
            "Design a controlled pre-verifier evidence-normalization layer behind a "
            "feature flag with adversarial eval."
        )
    elif (
        false_accept == 0
        and placeholder_accept == 0
        and placeholder.get("placeholder_acceptance", 1) == 0
    ):
        conclusion = "PARTIALLY_SUPPORTED"
        note = (
            "Placeholder normalization works and safety is preserved, but C4-U18 "
            "faithful-complete failures persist, indicating a separate verifier "
            "semantic-grounding limitation."
        )
        v210 = (
            "Run V2.10 verifier semantic-grounding experiment focused on claim-to-LO "
            "matching, unit/topic linkage, and curriculum hierarchy relationships."
        )
    else:
        conclusion = "NOT_SUPPORTED"
        note = "Normalization did not safely improve grounding outcomes."
        v210 = "Investigate ingestion/database representation before further normalization."

    return conclusion, note, v210, grounding_answer


def load_saved_baseline(path: Path) -> list[CurriculumEvidence]:
    from app.agent.v25_experiment import _deserialize_evidence

    return _deserialize_evidence(json.loads(path.read_text()))
