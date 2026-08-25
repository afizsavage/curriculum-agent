"""Map natural-language grade/subject/level phrases to Curriculum API codes."""

from __future__ import annotations

import re
from typing import Optional


_PRIMARY = re.compile(
    r"\b(?:primary|class|p)\s*([1-6])\b|\bclass_([1-6])\b",
    re.I,
)
_JSS = re.compile(r"\b(?:jss|junior\s+secondary)\s*([1-3])\b|\bjss_([1-3])\b", re.I)
_SSS = re.compile(r"\b(?:sss|senior\s+secondary)\s*([1-3])\b|\bsss_([1-3])\b", re.I)

_SUBJECT_ALIASES: dict[str, str] = {
    "mathematics": "MATHEMATICS",
    "maths": "MATHEMATICS",
    "math": "MATHEMATICS",
    "english": "ENGLISH",
    "english language": "ENGLISH",
    "language arts": "ENGLISH",
    "science": "SCIENCE",
    "social studies": "SOCIAL_STUDIES",
    "agricultural science": "AGRICULTURAL_SCIENCE",
    "ict": "ICT",
    "fundamentals of mathematics": "FUNDAMENTALS_MATHEMATICS",
    "fundamentals mathematics": "FUNDAMENTALS_MATHEMATICS",
}


def normalize_grade_code(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if re.fullmatch(r"(CLASS|JSS|SSS)_\d+", text, re.I):
        return text.upper()
    match = _PRIMARY.search(text)
    if match:
        n = match.group(1) or match.group(2)
        return f"CLASS_{n}"
    match = _JSS.search(text)
    if match:
        n = match.group(1) or match.group(2)
        return f"JSS_{n}"
    match = _SSS.search(text)
    if match:
        n = match.group(1) or match.group(2)
        return f"SSS_{n}"
    return None


def normalize_subject_code(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if re.fullmatch(r"[A-Z][A-Z0-9_]+", text):
        return text
    key = re.sub(r"\s+", " ", text.lower())
    if key in _SUBJECT_ALIASES:
        return _SUBJECT_ALIASES[key]
    # Soft match: contained alias
    for alias, code in _SUBJECT_ALIASES.items():
        if alias in key:
            return code
    # Fallback: slugify
    slug = re.sub(r"[^a-z0-9]+", "_", key).strip("_").upper()
    return slug or None


def infer_level(grade_code: str | None, level: str | None = None) -> str | None:
    if level:
        low = level.strip().lower()
        if "senior" in low or low in {"sss", "ss"}:
            return "senior_secondary"
        if "junior" in low or low in {"jss", "js"}:
            return "junior_secondary"
        if "primary" in low or low in {"basic", "bec"}:
            return "primary"
    if not grade_code:
        return None
    if grade_code.startswith("CLASS_"):
        return "primary"
    if grade_code.startswith("JSS_"):
        return "junior_secondary"
    if grade_code.startswith("SSS_"):
        return "senior_secondary"
    return None


def default_curriculum_for_grade(grade_code: str | None) -> tuple[str, str]:
    """Return (curriculum_code, version) defaults for MBSSE."""
    if grade_code and grade_code.startswith("SSS_"):
        return "MBSSE-SSC", "2021"
    return "MBSSE-BEC", "2020"


def extract_filters_from_question(question: str) -> dict[str, Optional[str]]:
    """Lightweight heuristic filters for understand() / stub tool calling."""
    grade = normalize_grade_code(question)
    subject = None
    lower = question.lower()
    for alias, code in sorted(_SUBJECT_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in lower:
            subject = code
            break
    level = infer_level(grade)
    topic = None
    # crude topic cue: "about X" / "fractions topic"
    about = re.search(r"\babout\s+([a-z0-9][\w\s-]{1,40})", lower)
    if about:
        topic = about.group(1).strip(" ?.!")
    elif "fractions" in lower:
        topic = "fractions"
    elif "measurement" in lower:
        topic = "measurement"
    return {
        "grade": grade,
        "subject": subject,
        "level": level,
        "topic": topic,
    }
