"""Evidence-conservative curriculum answer generation policy (V2.3 production)."""

from __future__ import annotations

import re
from typing import Any

from app.curriculum.evidence import CurriculumEvidence

GENERATION_POLICY = "evidence_conservative"

EVIDENCE_CONSERVATIVE_RULES = """
EVIDENCE-CONSERVATIVE GENERATION (production policy)

A. Evidence is the boundary
- Make curriculum claims ONLY when supported by the supplied evidence.
- Do not use general model knowledge to fill MBSSE curriculum gaps.
- Do not infer curriculum content that is not present in the evidence.

B. User-facing prose is a synthesis
- For ordinary pupil and teacher questions, write a concise natural-language
  explanation of what the supplied evidence supports.
- Group related learning objectives under one concept heading. List distinct
  skills as bullets under that heading. Do not create a heading for every
  learning objective, and do not prefix bullets with LO codes.
- When several units or topics answer the question, combine them into one
  explanation. Headings name the curriculum concept, not a database node.
- State only concepts present in the supplied evidence. Do not add generally
  plausible curriculum content from model knowledge.
- Do not include learning-objective codes, unit codes, entity IDs, database IDs,
  retrieval IDs, grade_curriculum_id, or other internal identifiers in the answer
  unless the user explicitly asks for that identifier.
- Minor formatting normalization is acceptable. Do not change the meaning of
  a reliable source statement.

C. Truncated / garbled source text
- If evidence text appears truncated, repetitive, malformed, incomplete, or garbled,
  do NOT reconstruct or repair it.
- Never use: "likely", "probably", "this means", "the intended objective is",
  "the missing text appears to say", or similar speculative completion.
- Use a reliable portion only when that portion is explicitly present in the source.
- Where damage materially affects what can be stated, say that the exact wording
  is unreliable. Note the limitation in limitations and/or the answer.
- Do not quote internal identifiers while describing that limitation.

D. No unsupported absence claims
- Do not claim something is absent merely because it was not found in the evidence.
- Wrong: "There are no learning outcomes for division of fractions in Primary 4."
- Prefer positive evidence statements. If needed: "The resolved evidence does not
  include a learning outcome specifically mentioning [topic]."
- not observed ≠ does not exist

E. No speculative curriculum claims
- Do not use "likely", "probably", "appears to mean", "may refer to", "presumably",
  "the curriculum intends" to infer official curriculum content.

F. Answer only the question asked
- Do not add unsupported claims about what is not taught, pedagogy, assessment,
  prerequisites, or grade progression unless explicitly in the evidence and necessary.

G. User-facing structure (when the evidence justifies it)
- Title: `# Grade Subject — Topic` from resolved grade, subject, and topic.
  No internal identifiers in the title.
- A short introductory paragraph when several learning areas answer the question.
  Name only areas supported by the evidence.
- Numbered concept headings such as `### 1. Unit Fractions`, then `Pupils learn to:`
  and one bullet per distinct learning expectation.
- Keep a one- or two-record answer short. Do not force a multi-section document.
- `### Curriculum Evidence Note` only when damaged source text materially limits
  the answer. Omit that heading when the evidence is intact.
- Do not show evidence-quality flags, database fields, or diagnostic terms.

H. Audit and provenance stay structured
- Identifiers remain on the evidence records. Return them in refs: only
  entity_id values from the supplied evidence that the answer actually used.
- Do not invent codes. The answer field is the user-facing synthesis.
- If the user explicitly asks for an LO code, unit code, entity ID, or similar
  identifier, include that identifier in the answer.

I. Do not fix curriculum data in generation
- The generator is not an ingestion system. Preserve damaged source text and flag it.
"""

EVIDENCE_CONSERVATIVE_USER_APPENDIX = """
Apply the evidence-conservative policy above.
Answer using ONLY the curriculum evidence block.
The answer field is user-facing synthesis: a title when context supports one,
a short introduction when several areas are present, concept headings, and
bullets for distinct learning expectations. No internal identifiers unless the
question explicitly asks for a code or id.
Use `### Curriculum Evidence Note` only when source text is materially limited.
Return refs containing only entity_id values from the supplied evidence that
the answer actually used, including a damaged record cited by an evidence note.
Do not invent IDs and do not list retrieved records the answer does not use.
Set limitations when source records are incomplete, duplicated, garbled, or ambiguous.
Do not reconstruct damaged source text.
"""


# --- Answer quality analysis (observability / regression tests) ---

_SPECULATIVE_RE = re.compile(
    r"\b(?:likely|probably|might|perhaps|presumably)\b|"
    r"\bthis means\b|\bappears to mean\b|\bmay refer to\b|"
    r"\bthe curriculum (?:appears to|intends)\b",
    re.I,
)
_TRUNCATION_MISHANDLE_RE = re.compile(
    r"\blikely means\b|\bprobably means\b|\bcan be inferred\b|\bimplies that\b|"
    r"\bthe intended objective is\b|\bthe missing text appears to say\b",
    re.I,
)
_UNSUPPORTED_ABSENCE_RE = re.compile(
    r"\b(?:there are|there is)\s+no\s+(?:learning\s+outcomes?|los?|objectives?)\b|"
    r"\bno\s+learning\s+outcomes?\s+(?:for|in|on)\b|"
    r"\bdoes not include any learning outcomes\b|"
    r"\bevidence does not include any learning outcomes\b",
    re.I,
)
_SAFE_ABSENCE_RE = re.compile(
    r"\b(?:resolved evidence|supplied evidence|available evidence|the evidence)\s+"
    r"does not include\b",
    re.I,
)


def detect_speculative_wording(answer: str) -> bool:
    return bool(_SPECULATIVE_RE.search(answer or ""))


def detect_truncation_mishandling(answer: str) -> bool:
    return bool(_TRUNCATION_MISHANDLE_RE.search(answer or ""))


def detect_unsupported_absence_claim(answer: str) -> bool:
    text = answer or ""
    if _SAFE_ABSENCE_RE.search(text):
        return False
    return bool(_UNSUPPORTED_ABSENCE_RE.search(text))


def detect_truncation_warning(answer: str, limitations: list[str] | None = None) -> bool:
    combined = f"{answer or ''} {' '.join(limitations or [])}"
    return bool(
        re.search(
            r"\b(?:incomplete|truncated|garbled|repetitive|damaged|unreliable)\b",
            combined,
            re.I,
        )
    )


def count_speculative_claims(answer: str) -> int:
    return len(_SPECULATIVE_RE.findall(answer or ""))


def count_unsupported_absence_claims(answer: str) -> int:
    if not detect_unsupported_absence_claim(answer):
        return 0
    return len(_UNSUPPORTED_ABSENCE_RE.findall(answer or "")) or 1


def count_truncation_warnings(answer: str, limitations: list[str] | None = None) -> int:
    return 1 if detect_truncation_warning(answer, limitations) else 0


def analyze_answer_quality(
    answer: str,
    *,
    limitations: list[str] | None = None,
    evidence: list[CurriculumEvidence] | None = None,
) -> dict[str, Any]:
    """Return observability counters for generation traces."""
    unsupported = count_unsupported_absence_claims(answer)
    speculative = count_speculative_claims(answer)
    truncation = count_truncation_warnings(answer, limitations)
    return {
        "generation_policy": GENERATION_POLICY,
        "unsupported_claim_count": unsupported,
        "speculative_claim_count": speculative,
        "truncation_warning_count": truncation,
        "absence_claim_count": unsupported,
        "speculative_wording": speculative > 0 or detect_speculative_wording(answer),
        "unsupported_absence_claim": unsupported > 0,
        "truncation_mishandling": detect_truncation_mishandling(answer),
    }


_IDENTIFIER_REQUEST_RE = re.compile(
    r"\b("
    r"learning[-\s]?objective\s+codes?|"
    r"objective\s+codes?|"
    r"\blo\s+codes?\b|"
    r"unit\s+codes?|"
    r"topic\s+codes?|"
    r"entity[_\s-]?ids?|"
    r"database\s+ids?|"
    r"retrieval\s+ids?|"
    r"grade_curriculum_id|"
    r"curriculum\s+identifiers?|"
    r"internal\s+identifiers?"
    r")\b",
    re.I,
)

_INTERNAL_IDENTIFIER_RE = re.compile(
    r"\bC\d+-U\d+\b|\bC\d+U\d+-LO\d+\b|grade_curriculum_id",
    re.I,
)


def question_requests_identifiers(question: str | None) -> bool:
    """True when the user explicitly asks for a curriculum code or id."""
    return bool(_IDENTIFIER_REQUEST_RE.search(question or ""))


def redact_internal_identifiers(
    answer: str,
    evidence: list[CurriculumEvidence] | None = None,
) -> str:
    """Remove internal ids from user-facing prose. Evidence objects are unchanged."""
    text = answer or ""
    needles: list[str] = []
    for item in evidence or []:
        if item.entity_id and len(str(item.entity_id)) >= 4:
            needles.append(str(item.entity_id))
        code = (item.metadata or {}).get("code")
        if code:
            needles.append(str(code))
        if item.name and _INTERNAL_IDENTIFIER_RE.search(str(item.name)):
            needles.append(str(item.name).strip())
    for needle in sorted(set(needles), key=len, reverse=True):
        if len(needle) < 3:
            continue
        text = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z0-9])",
            "",
            text,
            flags=re.I,
        )
    text = _INTERNAL_IDENTIFIER_RE.sub("", text)
    text = re.sub(r"\*\*\s*\*\*", "", text)
    text = re.sub(r"(?m)^[ \t]*[-*][ \t]*[—–-][ \t]*", "- ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def source_wording_preserved(
    answer: str,
    *,
    lo_code: str,
    source_wording: str,
) -> bool:
    """Heuristic for identifier-bearing audit text, not ordinary user prose."""
    if lo_code not in answer:
        return False
    words = [w for w in re.findall(r"[a-z]{4,}", source_wording.lower()) if len(w) > 4]
    if not words:
        return lo_code in answer
    hits = sum(1 for w in words[:6] if w in answer.lower())
    return hits >= min(2, len(words))
