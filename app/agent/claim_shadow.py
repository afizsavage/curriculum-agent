"""Shadow diagnostics for exact claim-span attribution.

This module measures a shadow generation. It does not accept near misses, and
it is not used by the production answer path.
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.answer_generator import (
    _claim_occurs,
    _presentation_normalize,
    _span_key,
    _substantive_segments,
)


def _punctuation_fold(text: str) -> str:
    return re.sub(r"[^\w\s]", "", _span_key(text)).strip()


def _plural_fold(text: str) -> str:
    tokens = []
    for token in _punctuation_fold(text).split():
        if len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def _candidate_spans(answer: str) -> list[str]:
    spans = [_span_key(segment) for segment in _substantive_segments(answer)]
    for sentence in re.split(r"(?<=[.])\s+", _presentation_normalize(answer)):
        key = sentence.rstrip(".").strip()
        if key:
            spans.append(key)
    return spans


def classify_claim_text(claim_text: str, answer: str) -> str:
    """Classify one returned claim span. Near misses stay failures."""
    if _claim_occurs(claim_text, answer):
        return "exact"
    folded = _punctuation_fold(claim_text)
    plural = _plural_fold(claim_text)
    punctuation_hit = False
    for span in _candidate_spans(answer):
        if folded and folded == _punctuation_fold(span):
            punctuation_hit = True
        if plural and plural == _plural_fold(span):
            return "near_miss"
    if punctuation_hit:
        return "near_miss"
    return "extra"


def classify_answer_delta(production: str, shadow: str) -> str:
    """Compare user-facing answers. This does not change either answer."""
    if (production or "").strip() == (shadow or "").strip():
        return "identical"
    if _presentation_normalize(production) == _presentation_normalize(shadow):
        return "formatting-only"
    production_spans = [_span_key(segment) for segment in _substantive_segments(production)]
    shadow_spans = [_span_key(segment) for segment in _substantive_segments(shadow)]
    if production_spans == shadow_spans:
        return "wording"
    if len(production_spans) == len(shadow_spans) and all(
        classify_claim_text(shadow_span, production) in {"exact", "near_miss"}
        for shadow_span in shadow_spans
    ):
        return "wording"
    return "substantive"


def measure_shadow_case(
    *,
    production_answer: str,
    shadow_answer: str,
    claim_report: dict[str, Any] | None,
    malformed_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize one shadow case from the existing exact-span validator."""
    report = claim_report or {}
    returned = report.get("returned_claims") or []
    mappings = [item for item in returned if isinstance(item, dict)]
    text_classes = [
        classify_claim_text(str(item.get("text") or ""), shadow_answer) for item in mappings
    ]
    exact = sum(1 for kind in text_classes if kind == "exact")
    near_miss = sum(1 for kind in text_classes if kind == "near_miss")
    extra = sum(1 for kind in text_classes if kind == "extra")
    substantive = int(report.get("substantive_claim_count") or 0)
    covered = substantive - len(report.get("unattributed_claims") or [])
    invalid_refs = list(report.get("invalid_refs") or [])
    unsupported = list(report.get("unsupported_claims") or [])
    valid_claims = list(report.get("valid_claims") or [])
    malformed_ids = set(malformed_ids or [])
    malformed_ok = 0
    if malformed_ids:
        for claim in valid_claims:
            text = str(claim.get("text") or "").lower()
            refs = set(claim.get("refs") or [])
            if refs & malformed_ids and (
                "incomplete" in text or "cannot be confirmed" in text or "evidence note" in text
            ):
                malformed_ok += 1
    ref_counts = [len(claim.get("refs") or []) for claim in mappings]
    repeated_large = False
    if len(mappings) >= 3:
        signatures = [tuple(claim.get("refs") or []) for claim in mappings]
        repeated_large = len(set(signatures)) == 1 and len(signatures[0]) >= 3
    return {
        "answer_delta": classify_answer_delta(production_answer, shadow_answer),
        "substantive_claims": substantive,
        "returned_mappings": len(mappings),
        "exact_matches": exact,
        "near_misses": near_miss,
        "missing_mappings": len(report.get("unattributed_claims") or []),
        "extra_mappings": extra,
        "invalid_refs": len(invalid_refs),
        "unsupported_refs": sum(len(item.get("refs") or []) for item in unsupported),
        "multi_record_claims": int(report.get("multi_record_claim_count") or 0),
        "malformed_evidence_mappings": malformed_ok,
        "covered_claims": covered,
        "complete": substantive > 0 and not report.get("unattributed_claims"),
        "over_attribution": repeated_large or any(count >= 4 for count in ref_counts),
        "under_attribution": substantive >= 5 and len(valid_claims) <= 2,
        "text_classes": text_classes,
        "status": report.get("status"),
    }
