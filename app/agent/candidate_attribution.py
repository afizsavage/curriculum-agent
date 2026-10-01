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


_FRAMING_WORDS = {
    "pupil", "pupils", "learn", "learns", "learning", "work", "working",
    "understand", "understanding", "including", "include", "following",
}
_STEM_ALIASES = {
    "reading": "read",
    "writing": "write",
    "written": "write",
    "multiplication": "multiply",
    "division": "divide",
    "addition": "add",
    "subtraction": "subtract",
    "numbers": "number",
    "fractions": "fraction",
    "operations": "operation",
}
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_OPERATION_STEMS = {"add", "subtract", "multiply", "divide"}
_UNIT_TYPES = {"topic", "subtopic", "unit", "strand", "subject"}


def _limited_source(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    if _looks_garbled_source_text(text) or _missing_denominator_range(text):
        return True
    if re.search(r"\bup to\b(?!\s*\d)", text.lower()):
        return True
    words = text.lower().split()
    if len(words) >= 8:
        phrase = " ".join(words[:4])
        if text.lower().count(phrase) >= 2:
            return True
    return False


def _stem_token(token: str) -> str:
    token = _NUMBER_WORDS.get(token, token)
    token = _STEM_ALIASES.get(token, token)
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
        token = _STEM_ALIASES.get(token, token)
    if len(token) > 5 and token.endswith("ing"):
        token = _STEM_ALIASES.get(token[:-3], token[:-3])
    return token


def _support_stems(text: str) -> list[str]:
    raw = (text or "").lower().replace("–", "-").replace("—", "-")
    stems: list[str] = []
    for token in re.findall(r"[a-z0-9]+", raw):
        stem = _stem_token(token)
        if len(stem) < 3 or stem in _CANDIDATE_STOPWORDS or stem in _FRAMING_WORDS:
            continue
        if stem not in stems:
            stems.append(stem)
    return stems


def _norm_label(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _is_generic_structural(item: CurriculumEvidence) -> bool:
    content = (item.content or "").strip()
    name = (item.name or "").strip()
    entity_type = (item.entity_type or "").lower()
    if entity_type not in _UNIT_TYPES and entity_type != "subject":
        return False
    if not content:
        return True
    return _norm_label(content) == _norm_label(name)


def _is_limitation_claim(claim_text: str, kind: str | None) -> bool:
    if kind == "evidence_note":
        return True
    lowered = (claim_text or "").lower()
    return any(
        phrase in lowered
        for phrase in ("incomplete", "unreliable", "cannot be stated", "cannot be confirmed", "not included")
    )


def relaxed_record_supports(
    item: CurriculumEvidence,
    claim_text: str,
    *,
    kind: str | None = None,
) -> bool:
    """Deterministic support check that still rejects same-topic-only citations.

    The legacy lexical checker is unchanged. This one also accepts a claim that
    is a shorter span, a normalized paraphrase, a unit-name claim, or a limitation
    note about a damaged or generic unit record.
    """
    if _record_supports_text(item, claim_text):
        return True
    name = item.name or ""
    content = item.content or ""
    source = f"{name} {content}".strip()
    if not source:
        return False
    claim_stems = _support_stems(claim_text)
    source_stems = _support_stems(source)
    name_stems = set(_support_stems(name))
    if not claim_stems or not source_stems:
        return False
    claim_ops = set(claim_stems) & _OPERATION_STEMS
    source_ops = set(source_stems) & _OPERATION_STEMS
    conflict = bool(claim_ops and source_ops and claim_ops.isdisjoint(source_ops))
    claim_phrase = " ".join(claim_stems)
    source_phrase = " ".join(_support_stems(content or name))
    if claim_phrase and claim_phrase in source_phrase and not conflict:
        return True
    if _is_limitation_claim(claim_text, kind):
        if _limited_source(content):
            return True
        name_norm = _norm_label(name)
        if name_norm and len(name_norm) > 3 and name_norm in (claim_text or "").lower():
            return True
        if _is_generic_structural(item) and any(
            phrase in (claim_text or "").lower()
            for phrase in ("unit name", "names are repeated", "not included", "same or very similar names")
        ):
            return True
        return False
    if conflict:
        return False
    if set(claim_stems) <= set(source_stems) and len(claim_stems) >= 2:
        return True
    if len(claim_stems) == 1 and claim_stems[0] in name_stems:
        return True
    if len(claim_stems) <= 2 and set(claim_stems) <= name_stems:
        return True
    return False


def _duplicate_key(item: CurriculumEvidence) -> tuple[str, str]:
    name = _norm_label(item.name or "")
    content = _norm_label(item.content or "")
    if not content or content == name:
        return ("generic", name)
    return ("content", content)


def _candidate_rank(
    item: CurriculumEvidence,
    *,
    claim_text: str,
    claim_tokens: set[str],
    heading_tokens: set[str],
    production: set[str],
    is_note: bool,
) -> tuple[tuple, list[str]] | None:
    """Higher features sort first. Topic membership alone does not score."""
    name = item.name or ""
    content = item.content or ""
    name_tokens = _tokens(name)
    content_tokens = _tokens(content)
    record_tokens = name_tokens | content_tokens
    hits = claim_tokens & record_tokens
    distinctive = (claim_tokens - heading_tokens) & record_tokens or hits
    heading_overlap = heading_tokens & record_tokens
    claim_norm = " ".join(claim_text.lower().split()).rstrip(".")
    name_norm = " ".join(name.lower().split())
    content_norm = " ".join(content.lower().split())
    reasons: list[str] = []
    claim_in_content = bool(claim_norm and content_norm and claim_norm in content_norm)
    content_in_claim = bool(content_norm and claim_norm and len(content_norm) > 12 and content_norm in claim_norm)
    name_in_claim = bool(name_tokens) and (name_tokens <= claim_tokens or claim_tokens <= name_tokens)
    if claim_in_content:
        reasons.append("claim_in_source")
    elif content_in_claim:
        reasons.append("source_in_claim")
    elif name_in_claim:
        reasons.append("name")
    if hits:
        reasons.append("wording")
    if distinctive and distinctive != hits:
        reasons.append("distinctive")
    if heading_overlap:
        reasons.append("heading")
    entity_type = (item.entity_type or "").lower()
    is_outcome = entity_type == "learning_outcome" and bool(content.strip())
    generic = _is_generic_structural(item)
    limited = is_note and _limited_source(content)
    named_in_note = bool(is_note and name_norm and len(name_norm) > 3 and name_norm in claim_text.lower())
    if is_outcome:
        reasons.append("outcome_content")
    if generic:
        reasons.append("generic_structural")
    if limited:
        reasons.append("limited_source")
    if named_in_note:
        reasons.append("named_in_note")
    if item.entity_id in production and (hits or is_note or name_in_claim):
        reasons.append("production_ref")
    heading_only = not hits and not claim_in_content and not name_in_claim and not named_in_note and not limited
    if heading_only and not heading_overlap:
        return None
    if heading_only:
        reasons.append("heading_only")
    extra_name = name_tokens - claim_tokens
    qualifier = 1 if (name_tokens & claim_tokens) and extra_name else 0
    if qualifier:
        reasons.append("specific_name")
    containment = 3 if claim_in_content else 2 if content_in_claim else 1 if name_in_claim else 0
    rank = (
        len(distinctive),
        len(hits),
        1 if is_outcome else 0,
        containment,
        qualifier,
        1 if content.strip() else 0,
        0 if generic else 1,
        1 if "production_ref" in reasons else 0,
        1 if limited or named_in_note else 0,
        0 if heading_only else 1,
    )
    return rank, reasons


def _public_candidate(
    item: CurriculumEvidence,
    *,
    rank: tuple,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "entity_id": item.entity_id,
        "entity_type": item.entity_type,
        "name": item.name,
        "grade": item.grade,
        "subject": item.subject,
        "topic": item.topic,
        "parent": (item.metadata or {}).get("parent_content_name")
        or (item.metadata or {}).get("parent_content_code"),
        "content": " ".join((item.content or "").split()),
        "score": rank[0] * 100 + rank[1] * 10 + rank[2],
        "rank": list(rank),
        "reasons": reasons,
        "empty_content": not (item.content or "").strip(),
        "generic_structural": _is_generic_structural(item),
    }


def select_candidates(
    claim: dict[str, Any],
    evidence: list[CurriculumEvidence],
    production_refs: list[str],
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> dict[str, Any]:
    """Rank retrieved records, collapse duplicate generics, then cap at 8."""
    claim_text = str(claim.get("text") or "")
    claim_tokens = _tokens(claim_text)
    heading_tokens = _tokens(str(claim.get("heading") or ""))
    is_note = claim.get("kind") == "evidence_note"
    production = {ref for ref in production_refs if ref}
    scored: list[tuple[tuple, CurriculumEvidence, list[str]]] = []
    for item in evidence:
        if not item.entity_id:
            continue
        ranked = _candidate_rank(
            item,
            claim_text=claim_text,
            claim_tokens=claim_tokens,
            heading_tokens=heading_tokens,
            production=production,
            is_note=is_note,
        )
        if ranked is None:
            continue
        rank, reasons = ranked
        scored.append((rank, item, reasons))
    scored.sort(key=lambda row: (*([-part for part in row[0]]), row[1].entity_id or ""))
    grouped: dict[tuple[str, str], list[tuple[tuple, CurriculumEvidence, list[str]]]] = {}
    for row in scored:
        grouped.setdefault(_duplicate_key(row[1]), []).append(row)
    removed_duplicates: list[dict[str, Any]] = []
    representatives: list[tuple[tuple, CurriculumEvidence, list[str]]] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: (*([-part for part in row[0]]), row[1].entity_id or ""))
        keeper = rows[0]
        representatives.append(keeper)
        for duplicate in rows[1:]:
            removed_duplicates.append({
                "entity_id": duplicate[1].entity_id,
                "name": duplicate[1].name,
                "content": " ".join((duplicate[1].content or "").split())[:240],
                "representative_id": keeper[1].entity_id,
                "reason": "duplicate",
            })
    representatives.sort(key=lambda row: (*([-part for part in row[0]]), row[1].entity_id or ""))
    kept_rows = representatives[:max_candidates]
    overflow = representatives[max_candidates:]
    rescued: list[str] = []
    generic_slots = [index for index, row in enumerate(kept_rows) if _is_generic_structural(row[1])]
    rescued_overflow: list[tuple[tuple, CurriculumEvidence, list[str]]] = []
    for row in overflow:
        item = row[1]
        content_hits = claim_tokens & _tokens(item.content or "")
        entity_type = (item.entity_type or "").lower()
        if entity_type == "learning_outcome" and content_hits and generic_slots:
            drop_at = generic_slots.pop()
            removed = kept_rows[drop_at]
            kept_rows[drop_at] = row
            overflow_replacement = removed
            rescued.append(item.entity_id or "")
            rescued_overflow.append(overflow_replacement)
        else:
            rescued_overflow.append(row)
    kept_rows.sort(key=lambda row: (*([-part for part in row[0]]), row[1].entity_id or ""))
    removed_by_cap = [
        {
            "entity_id": row[1].entity_id,
            "name": row[1].name,
            "entity_type": row[1].entity_type,
            "content": " ".join((row[1].content or "").split())[:240],
            "generic_structural": _is_generic_structural(row[1]),
            "reason": "cap",
        }
        for row in rescued_overflow
        if row not in kept_rows
    ]
    # rescued_overflow contains both unrescued overflow and displaced generics.
    # Displaced generics are not in kept_rows. Unrescued overflow rows are not either.
    # A rescued row was moved into kept_rows and must not be listed as capped.
    kept_ids = {row[1].entity_id for row in kept_rows}
    removed_by_cap = [item for item in removed_by_cap if item["entity_id"] not in kept_ids]
    return {
        "candidates": [
            _public_candidate(item, rank=rank, reasons=reasons)
            for rank, item, reasons in kept_rows
        ],
        "truncated": bool(removed_by_cap),
        "candidate_count_before_cap": len(scored),
        "candidates_considered": len(scored),
        "candidates_retained": len(kept_rows),
        "candidate_count": len(kept_rows),
        "candidates_removed_as_duplicates": len(removed_duplicates),
        "candidates_removed_by_cap": len(removed_by_cap),
        "removed_duplicates": removed_duplicates,
        "removed_by_cap": removed_by_cap,
        "omitted_ids": [item["entity_id"] for item in removed_by_cap if item.get("entity_id")],
        "rescued_specific_ids": rescued,
        "generic_structural_retained": sum(1 for row in kept_rows if _is_generic_structural(row[1])),
        "specific_content_retained": sum(
            1
            for row in kept_rows
            if (row[1].entity_type or "").lower() == "learning_outcome" and (row[1].content or "").strip()
        ),
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


def classify_truncation(
    *,
    claim_text: str,
    kind: str | None,
    retained: list[dict[str, Any]],
    removed_duplicates: list[dict[str, Any]],
    removed_by_cap: list[dict[str, Any]],
) -> str:
    """Classify a capped candidate set by what the cap removed."""
    if not removed_by_cap:
        return "not_truncated"

    def _as_evidence(item: dict[str, Any]) -> CurriculumEvidence:
        return CurriculumEvidence(
            entity_type=str(item.get("entity_type") or "unit"),
            entity_id=item.get("entity_id"),
            name=item.get("name"),
            content=item.get("content") or "",
        )

    retained_support = [
        item for item in retained
        if relaxed_record_supports(_as_evidence(item), claim_text, kind=kind)
    ]
    harmful: list[dict[str, Any]] = []
    redundant: list[dict[str, Any]] = []
    for item in removed_by_cap:
        if item.get("generic_structural"):
            continue
        if not relaxed_record_supports(_as_evidence(item), claim_text, kind=kind):
            continue
        if retained_support:
            redundant.append(item)
        else:
            harmful.append(item)
    if harmful:
        return "genuinely_harmful"
    if redundant:
        return "potentially_harmful"
    if all(item.get("generic_structural") for item in removed_by_cap):
        return "duplicate_only"
    return "harmless"


def classify_claim_selection(
    *,
    claim_text: str,
    model_refs: list[Any],
    candidates: list[dict[str, Any]],
    evidence: list[CurriculumEvidence],
    kind: str | None = None,
) -> dict[str, Any]:
    """Record the model selection, then apply both support checks without dropping it."""
    by_id = {item.entity_id: item for item in evidence if item.entity_id}
    candidate_ids = {item["entity_id"] for item in candidates if item.get("entity_id")}
    seen: set[str] = set()
    ordered: list[str] = []
    unknown: list[str] = []
    outside: list[str] = []
    lexical_valid: list[str] = []
    lexical_rejected: list[str] = []
    relaxed_valid: list[str] = []
    relaxed_rejected: list[str] = []
    support_labels: list[dict[str, Any]] = []
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
        legacy = _record_supports_text(source, claim_text)
        relaxed = relaxed_record_supports(source, claim_text, kind=kind)
        if legacy:
            lexical_valid.append(identity)
        else:
            lexical_rejected.append(identity)
        if relaxed:
            relaxed_valid.append(identity)
        else:
            relaxed_rejected.append(identity)
        support_labels.append({
            "ref": identity,
            "legacy_lexical_result": legacy,
            "relaxed_support_result": relaxed,
            "name": source.name,
            "content": " ".join((source.content or "").split())[:240],
        })
    return {
        "model_refs": ordered,
        "model-selected-valid-lexically": lexical_valid,
        "model-selected-but-lexically-rejected": lexical_rejected,
        "relaxed-supported": relaxed_valid,
        "relaxed-rejected": relaxed_rejected,
        "support_labels": support_labels,
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
    kind: str | None = None,
) -> dict[str, Any]:
    """Ask for refs only. Any returned answer text is discarded."""
    frozen = claim_text
    if not candidates:
        classification = classify_claim_selection(
            claim_text=frozen,
            model_refs=[],
            candidates=[],
            evidence=evidence,
            kind=kind,
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
        kind=kind,
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
    relaxed_buckets = {"1": 0, "2": 0, "3": 0, "4+": 0}
    malformed = set(malformed_ids or [])
    covered = 0
    relaxed_covered = 0
    model_covered = 0
    legacy_accepted_refs = 0
    relaxed_accepted_refs = 0
    relaxed_rejected_refs = 0
    truncation_classes = {
        "not_truncated": 0,
        "harmless": 0,
        "duplicate_only": 0,
        "potentially_harmful": 0,
        "genuinely_harmful": 0,
    }
    duplicates_removed = 0
    cap_removed = 0
    specific_retained = 0
    generic_retained = 0
    rescued_specific = 0
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
        truncation_class = str(row.get("truncation_class") or "not_truncated")
        truncation_classes[truncation_class] = truncation_classes.get(truncation_class, 0) + 1
        duplicates_removed += int(row.get("candidates_removed_as_duplicates") or 0)
        cap_removed += int(row.get("candidates_removed_by_cap") or 0)
        specific_retained += int(row.get("specific_content_retained") or 0)
        generic_retained += int(row.get("generic_structural_retained") or 0)
        rescued_specific += len(row.get("rescued_specific_ids") or [])
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
        relaxed = list(selection.get("relaxed-supported") or [])
        relaxed_rejected = list(selection.get("relaxed-rejected") or [])
        legacy_accepted_refs += len(lexical)
        relaxed_accepted_refs += len(relaxed)
        relaxed_rejected_refs += len(relaxed_rejected)
        if relaxed:
            relaxed_covered += 1
            relaxed_buckets[_bucket(len(relaxed))] += 1
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
        "relaxed_covered_claims": relaxed_covered,
        "model_covered_claims": model_covered,
        "legacy_accepted_refs": legacy_accepted_refs,
        "relaxed_accepted_refs": relaxed_accepted_refs,
        "relaxed_rejected_refs": relaxed_rejected_refs,
        "empty_mappings": empty_mappings,
        "no_candidate_claims": no_candidates,
        "unsupported_refs": unsupported_refs,
        "unknown_refs": unknown_refs,
        "outside_candidate_refs": outside_refs,
        "empty_content_refs": empty_content_refs,
        "record_buckets": buckets,
        "lexical_record_buckets": lexical_buckets,
        "relaxed_record_buckets": relaxed_buckets,
        "truncation_classes": truncation_classes,
        "duplicates_removed": duplicates_removed,
        "cap_removed": cap_removed,
        "specific_content_retained": specific_retained,
        "generic_structural_retained": generic_retained,
        "rescued_specific": rescued_specific,
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
