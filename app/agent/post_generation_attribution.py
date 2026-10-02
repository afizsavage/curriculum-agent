"""Attribute an already-written answer. This never rewrites that answer."""

from __future__ import annotations

from typing import Any

from app.agent.answer_generator import (
    _mapping_covers_segment,
    _substantive_segments,
    attribute_claim_mappings,
)
from app.agent.claim_shadow import classify_claim_text
from app.curriculum.evidence import CurriculumEvidence
from app.llm.base import LLMMessage, LLMProvider
from app.schemas.answer import POST_GENERATION_CLAIM_SCHEMA

POST_GENERATION_ATTRIBUTION_PROMPT = """You are performing evidence attribution for an answer that has already been written.

DO NOT rewrite, improve, summarize, correct, or regenerate the answer.
DO NOT return a replacement answer.

Identify the substantive curriculum claims already present in the answer.

For each substantive claim:
1. Copy the exact claim text from the answer.
2. Identify the supplied evidence records that support that exact claim.
3. Use only evidence records supplied in the evidence context.
4. Do not infer missing evidence.
5. Do not cite an evidence record merely because it is related to the topic.
6. If no supplied record clearly supports the claim, return an empty refs list.
7. Evidence limitation notes should be attributed to the incomplete or malformed record that caused the limitation.
8. Titles, headings, and generic presentation phrases such as "Pupils learn to:" are not claims.
9. One exact sentence may cite more than one record when the sentence uses more than one.
10. Do not repair malformed source text and do not invent missing wording.

The answer is immutable. The claim text must be copied from it.
"""


def _evidence_block(evidence: list[CurriculumEvidence]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(evidence, start=1):
        content = " ".join((item.content or "").split())
        if len(content) > 280:
            content = content[:280] + "…"
        lines = [
            f"--- Record {index} ---",
            f"Entity ID: {item.entity_id or 'unknown'}",
            f"Entity Type: {item.entity_type}",
        ]
        if item.name:
            lines.append(f"Name: {item.name}")
        if content and content != (item.name or ""):
            lines.append(f"Content: {content}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(no curriculum evidence supplied)"


def build_post_generation_messages(
    *,
    question: str,
    answer: str,
    evidence: list[CurriculumEvidence],
    production_refs: list[str],
) -> list[LLMMessage]:
    """Prompt that sees a frozen answer and returns claims only."""
    refs = ", ".join(production_refs) if production_refs else "(none)"
    user = (
        f"QUESTION\n{question}\n\n"
        f"IMMUTABLE ANSWER\n{answer}\n\n"
        f"SUPPLIED EVIDENCE\n{_evidence_block(evidence)}\n\n"
        "EXISTING PRODUCTION REFS\n"
        f"{refs}\n"
        "These refs are audit context only. Do not cite a record just because it "
        "appears here, and do not omit a supporting record just because it does not.\n\n"
        "Return one JSON object with a claims array. Do not return an answer field.\n"
        f"{POST_GENERATION_CLAIM_SCHEMA}"
    )
    return [
        LLMMessage(role="system", content=POST_GENERATION_ATTRIBUTION_PROMPT),
        LLMMessage(role="user", content=user),
    ]


def measure_post_generation_attribution(
    *,
    answer: str,
    returned_claims: list[Any],
    evidence: list[CurriculumEvidence],
    production_refs: list[str],
    malformed_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Validate claim mappings against a frozen answer. The answer is not edited."""
    frozen = answer
    attribution = attribute_claim_mappings(returned_claims or [], evidence, answer=frozen)
    if answer != frozen:
        raise RuntimeError("attribution mutated the production answer")
    mappings = [item for item in (returned_claims or []) if isinstance(item, dict)]
    text_classes = [classify_claim_text(str(item.get("text") or ""), frozen) for item in mappings]
    segments = _substantive_segments(frozen)
    omitted: list[str] = []
    returned_invalid: list[str] = []
    valid_segment_count = 0
    for segment in segments:
        covering = [
            item
            for item in mappings
            if _mapping_covers_segment(str(item.get("text") or ""), segment)
        ]
        if not covering:
            omitted.append(segment)
            continue
        if any(_mapping_covers_segment(claim["text"], segment) for claim in attribution.valid_claims):
            valid_segment_count += 1
        else:
            returned_invalid.append(segment)
    buckets = {"1": 0, "2": 0, "3": 0, "4+": 0}
    wide_claims: list[dict[str, Any]] = []
    for claim in attribution.valid_claims:
        count = len(claim.get("refs") or [])
        if count <= 1:
            buckets["1"] += 1
        elif count == 2:
            buckets["2"] += 1
        elif count == 3:
            buckets["3"] += 1
        else:
            buckets["4+"] += 1
            wide_claims.append(claim)
    returned_wide = [
        {"text": str(item.get("text") or ""), "refs": list(item.get("refs") or [])}
        for item in mappings
        if len(item.get("refs") or []) >= 4
    ]
    valid_ids = {ref for claim in attribution.valid_claims for ref in claim.get("refs") or []}
    production_ids = [ref for ref in production_refs if ref]
    production_set = set(production_ids)
    malformed = set(malformed_ids or [])
    malformed_ok = 0
    if malformed:
        for claim in attribution.valid_claims:
            text = str(claim.get("text") or "").lower()
            refs = set(claim.get("refs") or [])
            if refs & malformed and (
                "incomplete" in text or "cannot be confirmed" in text or "cannot be stated" in text
            ):
                malformed_ok += 1
    substantive = len(segments)
    return {
        "substantive_claims": substantive,
        "returned_mappings": len(mappings),
        "exact_matches": sum(1 for kind in text_classes if kind == "exact"),
        "near_misses": sum(1 for kind in text_classes if kind == "near_miss"),
        "extra_mappings": sum(1 for kind in text_classes if kind == "extra"),
        "missing_mappings": len(omitted),
        "returned_invalid_mappings": len(returned_invalid),
        "invalid_refs": len(attribution.invalid_refs),
        "unsupported_refs": sum(len(item.get("refs") or []) for item in attribution.unsupported_claims),
        "record_buckets": buckets,
        "wide_valid_claims": wide_claims,
        "wide_returned_claims": returned_wide,
        "malformed_evidence_mappings": malformed_ok,
        "covered_claims": valid_segment_count,
        "complete": substantive > 0 and valid_segment_count == substantive,
        "production_refs_covered": sorted(production_set & valid_ids),
        "production_refs_unattributed": sorted(production_set - valid_ids),
        "claim_refs_not_in_production": sorted(valid_ids - production_set),
        "validation": attribution.as_dict(),
        "text_classes": text_classes,
        "omitted_claims": omitted,
        "returned_invalid_claims": returned_invalid,
    }


def attribute_existing_answer(
    llm: LLMProvider,
    *,
    question: str,
    answer: str,
    evidence: list[CurriculumEvidence],
    production_refs: list[str],
    malformed_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Call the attribution model. The returned answer text is the input answer."""
    frozen = answer
    messages = build_post_generation_messages(
        question=question,
        answer=frozen,
        evidence=evidence,
        production_refs=production_refs,
    )
    raw = llm.generate_structured(messages, schema=POST_GENERATION_CLAIM_SCHEMA, temperature=0.0)
    claims = raw.get("claims") if isinstance(raw, dict) else []
    if not isinstance(claims, list):
        claims = []
    metrics = measure_post_generation_attribution(
        answer=frozen,
        returned_claims=claims,
        evidence=evidence,
        production_refs=production_refs,
        malformed_ids=malformed_ids,
    )
    return {
        "production_answer": frozen,
        "shadow_claims": claims,
        "metrics": metrics,
        "discarded_answer_field": raw.get("answer") if isinstance(raw, dict) else None,
    }
