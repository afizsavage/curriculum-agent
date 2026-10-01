"""Attribute frozen claims from a small deterministic candidate set.

This module never rewrites an answer and is not called by production generation.
"""

from __future__ import annotations

import re
from statistics import median
from typing import Any

from app.agent.answer_generator import (
    _claim_occurs,
    _looks_garbled_source_text,
    _missing_denominator_range,
    _record_supports_text,
)
from app.curriculum.evidence import CurriculumEvidence
from app.llm.base import LLMMessage, LLMProvider
from app.schemas.answer import CANDIDATE_CLAIM_REF_SCHEMA

MAX_CANDIDATES = 8

CANDIDATE_ATTRIBUTION_PROMPT = """You are attributing one claim from an answer that has already been written.

DO NOT rewrite, improve, summarize, correct, or regenerate the answer.
DO NOT return a replacement answer or replacement claim text.

Select only evidence records that directly support this exact claim.

Do not cite records merely because:
- they are in the same topic;
- they are nearby in the curriculum;
- they have similar wording;
- they belong to the same unit.

Prefer the smallest sufficient set of records.

If one record supports the claim, return one record.
If two records are genuinely required, return two.
Do not cite a record unless you can identify what part of the claim it supports.
If none of the candidates directly supports the claim, return an empty refs list.
Do not infer missing source content.
Use only the candidate record ids listed for this claim.
When the claim is an evidence limitation note, cite the incomplete or malformed candidate that the note describes.
"""

_BULLET_RE = re.compile(r"^[\*\-]\s+(\S.*)$")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")
_PUPILS_LEARN_TO_RE = re.compile(r"^pupils learn to:?$", re.I)
_SECTION_BODY_RE = re.compile(r"^pupils learn\b", re.I)
_CANDIDATE_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "these", "those",
    "their", "into", "about", "pupils", "pupil", "learn", "learns", "learning",
    "including", "include", "work", "working", "through", "following", "covers",
    "cover", "curriculum", "primary", "note", "outcome", "outcomes", "exact",
    "wording", "source", "evidence", "incomplete", "repeated", "unreliable",
    "stated", "available", "should", "using", "used", "also", "only", "some",
    "such", "than", "then", "them", "have", "has", "been", "were", "was", "are",
    "not", "but", "its", "can", "cannot", "one", "two", "part", "parts",
}
def extract_answer_claims(answer: str) -> list[dict[str, str | None]]:
    """Pull bullet items, section learning statements, and evidence notes.

    Titles, headings, "Pupils learn to:", and catalogue introductions are omitted.
    Claim text keeps the production wording with the bullet marker removed.
    """
    claims: list[dict[str, str | None]] = []
    heading: str | None = None
    in_note = False
    note_lines: list[str] = []

    def flush_note() -> None:
        text = "\n".join(note_lines).strip()
        if text:
            claims.append({
                "text": text,
                "kind": "evidence_note",
                "heading": "Curriculum Evidence Note",
            })

    for line in (answer or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            if in_note:
                flush_note()
                note_lines = []
                in_note = False
            title = re.sub(r"^\d+\.\s*", "", heading_match.group(1).strip())
            if title.lower() == "curriculum evidence note":
                in_note = True
                heading = title
            else:
                heading = title
            continue
        if in_note:
            note_lines.append(stripped)
            continue
        bullet = _BULLET_RE.match(stripped)
        if bullet:
            claims.append({
                "text": bullet.group(1).strip(),
                "kind": "bullet",
                "heading": heading,
            })
            continue
        if _PUPILS_LEARN_TO_RE.match(stripped):
            continue
        if heading and _SECTION_BODY_RE.match(stripped):
            claims.append({
                "text": stripped,
                "kind": "section_statement",
                "heading": heading,
            })
    if in_note:
        flush_note()
    return claims


def _tokens(text: str) -> set[str]:
    folded: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if len(raw) < 3 or raw in _CANDIDATE_STOPWORDS:
            continue
        if len(raw) > 4 and raw.endswith("s") and not raw.endswith("ss"):
            raw = raw[:-1]
        folded.add(raw)
    return folded


def _limited_source(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    if _looks_garbled_source_text(text) or _missing_denominator_range(text):
        return True
    words = text.lower().split()
    if len(words) >= 8:
        phrase = " ".join(words[:4])
        if text.lower().count(phrase) >= 2:
            return True
    return False


def select_candidates(
    claim: dict[str, Any],
    evidence: list[CurriculumEvidence],
    production_refs: list[str],
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> dict[str, Any]:
    """Rank already-retrieved records for one claim. No model and no new retrieval."""
    claim_tokens = _tokens(str(claim.get("text") or ""))
    heading_tokens = _tokens(str(claim.get("heading") or ""))
    is_note = claim.get("kind") == "evidence_note"
    production = {ref for ref in production_refs if ref}
    claim_norm = " ".join(str(claim.get("text") or "").lower().split()).rstrip(".")
    scored: list[tuple[int, int, int, str, CurriculumEvidence, list[str]]] = []
    for item in evidence:
        if not item.entity_id:
            continue
        name = item.name or ""
        content = item.content or ""
        parent_name = str((item.metadata or {}).get("parent_content_name") or "")
        name_tokens = _tokens(name)
        content_tokens = _tokens(content)
        topic_tokens = _tokens(item.topic or "") | _tokens(parent_name)
        overlap = claim_tokens & (name_tokens | content_tokens | topic_tokens)
        heading_overlap = heading_tokens & (name_tokens | content_tokens | topic_tokens)
        reasons: list[str] = []
        score = 4 * len(overlap)
        if overlap:
            reasons.append("wording")
        score += 2 * len(heading_overlap)
        if heading_overlap:
            reasons.append("heading")
        name_norm = " ".join(name.lower().split())
        if name_norm and (name_norm in claim_norm or claim_norm in name_norm):
            score += 6
            reasons.append("name")
        if is_note and name_norm and len(name_norm) > 3 and name_norm in str(claim.get("text") or "").lower():
            score += 6
            if "named_in_note" not in reasons:
                reasons.append("named_in_note")
        if item.entity_id in production and (overlap or heading_overlap or is_note):
            score += 3
            reasons.append("production_ref")
        entity_type = (item.entity_type or "").lower()
        if entity_type == "learning_outcome" and content.strip() and score > 0:
            score += 1
            reasons.append("outcome_content")
        if is_note and _limited_source(content):
            score += 5
            reasons.append("limited_source")
        if (
            not content.strip()
            and "name" not in reasons
            and "named_in_note" not in reasons
            and not overlap
            and not heading_overlap
        ):
            score = 0
        if score > 0:
            content_rank = 0 if content.strip() else 1
            type_rank = 0 if entity_type == "learning_outcome" else 1
            scored.append((score, content_rank, type_rank, item.entity_id, item, reasons))
    scored.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    truncated = len(scored) > max_candidates
    kept = scored[:max_candidates]
    return {
        "candidates": [
            {
                "entity_id": item.entity_id,
                "entity_type": item.entity_type,
                "name": item.name,
                "grade": item.grade,
                "subject": item.subject,
                "topic": item.topic,
                "parent": (item.metadata or {}).get("parent_content_name")
                or (item.metadata or {}).get("parent_content_code"),
                "content": " ".join((item.content or "").split()),
                "score": score,
                "reasons": reasons,
                "empty_content": not (item.content or "").strip(),
            }
            for score, _content_rank, _type_rank, _entity_id, item, reasons in kept
        ],
        "truncated": truncated,
        "candidate_count_before_cap": len(scored),
        "candidate_count": len(kept),
        "omitted_ids": [row[3] for row in scored[max_candidates:]],
    }


def _candidate_block(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "(no candidate records)"
    blocks: list[str] = []
    for index, item in enumerate(candidates, start=1):
        content = item.get("content") or ""
        if len(content) > 400:
            content = content[:400] + "…"
        lines = [
            f"--- Candidate {index} ---",
            f"Record ID: {item.get('entity_id')}",
            f"Record type: {item.get('entity_type')}",
            f"Name: {item.get('name') or ''}",
            f"Grade: {item.get('grade') or ''}",
            f"Subject: {item.get('subject') or ''}",
            f"Topic/unit: {item.get('topic') or item.get('parent') or ''}",
            f"Source wording: {content or '(empty)'}",
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_candidate_messages(
    *,
    question: str,
    claim_text: str,
    candidates: list[dict[str, Any]],
) -> list[LLMMessage]:
    user = (
        f"QUESTION\n{question}\n\n"
        f"CLAIM\n{claim_text}\n\n"
        "CANDIDATE RECORDS\n"
        f"{_candidate_block(candidates)}\n\n"
        "Return one JSON object with a refs array. Do not return an answer field.\n"
        f"{CANDIDATE_CLAIM_REF_SCHEMA}"
    )
    return [
        LLMMessage(role="system", content=CANDIDATE_ATTRIBUTION_PROMPT),
        LLMMessage(role="user", content=user),
    ]


def classify_claim_selection(
    *,
    claim_text: str,
    model_refs: list[Any],
    candidates: list[dict[str, Any]],
    evidence: list[CurriculumEvidence],
) -> dict[str, Any]:
    """Record the model selection, then apply the lexical check without dropping it."""
    by_id = {item.entity_id: item for item in evidence if item.entity_id}
    candidate_ids = {item["entity_id"] for item in candidates if item.get("entity_id")}
    seen: set[str] = set()
    ordered: list[str] = []
    unknown: list[str] = []
    outside: list[str] = []
    lexical_valid: list[str] = []
    lexical_rejected: list[str] = []
    empty_content: list[str] = []
    for raw in model_refs or []:
        identity = str(raw).strip() if not isinstance(raw, dict) else str(raw.get("entity_id") or raw.get("id") or "")
        if isinstance(raw, dict) and not identity:
            identity = str(raw.get("ref") or "").strip()
        if not identity:
            continue
        if identity in seen:
            continue
        seen.add(identity)
        ordered.append(identity)
        if identity not in by_id:
            unknown.append(identity)
            continue
        if identity not in candidate_ids:
            outside.append(identity)
            continue
        source = by_id[identity]
        if not (source.content or "").strip():
            empty_content.append(identity)
        if _record_supports_text(source, claim_text):
            lexical_valid.append(identity)
        else:
            lexical_rejected.append(identity)
    return {
        "model_refs": ordered,
        "model-selected-valid-lexically": lexical_valid,
        "model-selected-but-lexically-rejected": lexical_rejected,
        "model-selected-unknown": unknown,
        "model-selected-outside-candidates": outside,
        "model-selected-empty": not ordered,
        "empty_content_selected": empty_content,
        "candidate_ids": [item.get("entity_id") for item in candidates],
        "candidate_snippets": [
            {
                "entity_id": item.get("entity_id"),
                "name": item.get("name"),
                "content": (item.get("content") or "")[:240],
                "empty_content": item.get("empty_content"),
            }
            for item in candidates
        ],
    }


def attribute_one_claim(
    llm: LLMProvider,
    *,
    question: str,
    claim_text: str,
    candidates: list[dict[str, Any]],
    evidence: list[CurriculumEvidence],
) -> dict[str, Any]:
    """Ask for refs only. Any returned answer text is discarded."""
    frozen = claim_text
    if not candidates:
        classification = classify_claim_selection(
            claim_text=frozen,
            model_refs=[],
            candidates=[],
            evidence=evidence,
        )
        classification["model_called"] = False
        classification["no_candidates"] = True
        classification["model-selected-empty"] = False
        return classification
    messages = build_candidate_messages(
        question=question,
        claim_text=frozen,
        candidates=candidates,
    )
    raw = llm.generate_structured(messages, schema=CANDIDATE_CLAIM_REF_SCHEMA, temperature=0.0)
    if not isinstance(raw, dict) or not isinstance(raw.get("refs"), list):
        raise ValueError("attribution response did not contain a refs array")
    if frozen != claim_text:
        raise RuntimeError("attribution mutated the claim text")
    classification = classify_claim_selection(
        claim_text=frozen,
        model_refs=list(raw.get("refs") or []),
        candidates=candidates,
        evidence=evidence,
    )
    classification["model_called"] = True
    classification["no_candidates"] = False
    classification["discarded_answer_field"] = raw.get("answer")
    classification["discarded_text_field"] = raw.get("text")
    return classification


def _bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count == 3:
        return "3"
    return "4+"


def measure_candidate_case(
    *,
    answer: str,
    claim_rows: list[dict[str, Any]],
    malformed_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize one frozen answer. The answer string is not edited."""
    frozen = answer
    claims = extract_answer_claims(frozen)
    if frozen != answer:
        raise RuntimeError("measurement mutated the production answer")
    buckets = {"1": 0, "2": 0, "3": 0, "4+": 0}
    lexical_buckets = {"1": 0, "2": 0, "3": 0, "4+": 0}
    malformed = set(malformed_ids or [])
    covered = 0
    model_covered = 0
    empty_mappings = 0
    no_candidates = 0
    unsupported_refs = 0
    unknown_refs = 0
    outside_refs = 0
    empty_content_refs = 0
    malformed_lexical = 0
    malformed_model = 0
    candidate_counts: list[int] = []
    truncations = 0
    exact = 0
    wide: list[dict[str, Any]] = []
    for row in claim_rows:
        text = str(row.get("text") or "")
        if _claim_occurs(text, frozen):
            exact += 1
        selection = row.get("selection") or {}
        candidate_counts.append(int(row.get("candidate_count") or 0))
        if row.get("truncated"):
            truncations += 1
        model_refs = list(selection.get("model_refs") or [])
        in_set = [
            ref
            for ref in model_refs
            if ref not in set(selection.get("model-selected-unknown") or [])
            and ref not in set(selection.get("model-selected-outside-candidates") or [])
        ]
        if selection.get("no_candidates"):
            no_candidates += 1
        elif selection.get("model-selected-empty") or not model_refs:
            empty_mappings += 1
        elif in_set:
            model_covered += 1
            buckets[_bucket(len(in_set))] += 1
            if len(in_set) >= 4:
                wide.append({"text": text, "refs": in_set, "kind": row.get("kind")})
        lexical = list(selection.get("model-selected-valid-lexically") or [])
        rejected = list(selection.get("model-selected-but-lexically-rejected") or [])
        unsupported_refs += len(rejected)
        unknown_refs += len(selection.get("model-selected-unknown") or [])
        outside_refs += len(selection.get("model-selected-outside-candidates") or [])
        empty_content_refs += len(selection.get("empty_content_selected") or [])
        if lexical:
            covered += 1
            lexical_buckets[_bucket(len(lexical))] += 1
        lowered = text.lower()
        note_like = row.get("kind") == "evidence_note" or any(
            token in lowered for token in ("incomplete", "cannot be confirmed", "cannot be stated", "unreliable")
        )
        chosen = set(model_refs)
        if note_like and chosen & malformed:
            malformed_model += 1
        if note_like and set(lexical) & malformed:
            malformed_lexical += 1
    substantive = len(claim_rows)
    return {
        "substantive_claims": substantive,
        "extracted_claims": len(claims),
        "returned_mappings": substantive,
        "exact_matches": exact,
        "covered_claims": covered,
        "model_covered_claims": model_covered,
        "empty_mappings": empty_mappings,
        "no_candidate_claims": no_candidates,
        "unsupported_refs": unsupported_refs,
        "unknown_refs": unknown_refs,
        "outside_candidate_refs": outside_refs,
        "empty_content_refs": empty_content_refs,
        "record_buckets": buckets,
        "lexical_record_buckets": lexical_buckets,
        "wide_claims": wide,
        "malformed_evidence_mappings": malformed_lexical,
        "malformed_model_selections": malformed_model,
        "candidate_counts": candidate_counts,
        "truncations": truncations,
        "complete": substantive > 0 and covered == substantive,
        "model_complete": substantive > 0 and model_covered == substantive,
        "exact_span_compliance": (exact / substantive) if substantive else None,
    }


def candidate_size_summary(counts: list[int], truncations: int) -> dict[str, Any]:
    if not counts:
        return {
            "average": None,
            "median": None,
            "maximum": None,
            "truncations": truncations,
        }
    return {
        "average": sum(counts) / len(counts),
        "median": float(median(counts)),
        "maximum": max(counts),
        "truncations": truncations,
        "claims": len(counts),
    }
