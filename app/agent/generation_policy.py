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

B. Preserve official learning outcome text
- For learning objectives/outcomes, prefer: LO code + source wording.
- Do not rewrite the meaning of an official LO to sound cleaner.
- Minor formatting normalization is acceptable; semantic rewriting is not.

C. Truncated / garbled source text
- If evidence text appears truncated, repetitive, malformed, incomplete, or garbled,
  do NOT reconstruct or repair it.
- Never use: "likely", "probably", "this means", "the intended objective is",
  "the missing text appears to say", or similar speculative completion.
- Report the available wording and note the limitation in limitations and/or the answer.

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

G. Evidence-first structure (when useful)
- Curriculum context (grade, subject, topic/unit)
- Learning objectives/outcomes as bullets: [LO code] — [source wording]
- Source limitations (only when evidence is incomplete/garbled)

H. Preserve identifiers
- Retain LO codes, unit codes, topic codes, and entity IDs from evidence.
- Do not invent codes.

I. Do not fix curriculum data in generation
- The generator is not an ingestion system. Preserve damaged source text and flag it.
"""

EVIDENCE_CONSERVATIVE_USER_APPENDIX = """
Apply the evidence-conservative policy above.
Answer using ONLY the curriculum evidence block.
Reference entity_id values from the evidence in your evidence array.
Set limitations when source records are incomplete or ambiguous.
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


def source_wording_preserved(
    answer: str,
    *,
    lo_code: str,
    source_wording: str,
) -> bool:
    """Heuristic: answer includes LO code and a substantive substring of source wording."""
    if lo_code not in answer:
        return False
    words = [w for w in re.findall(r"[a-z]{4,}", source_wording.lower()) if len(w) > 4]
    if not words:
        return lo_code in answer
    hits = sum(1 for w in words[:6] if w in answer.lower())
    return hits >= min(2, len(words))
