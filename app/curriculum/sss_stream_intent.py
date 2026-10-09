"""Deterministic detection of SSS stream subject questions.

Recognizes narrow question shapes that ask which subjects belong to a named
Senior Secondary stream. Grade, topic, and other curriculum questions stay on
their existing routes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.curriculum.codes import normalize_grade_code

INTENT_SSS_STREAM_SUBJECTS = "SSS_STREAM_SUBJECTS"
TOOL_GET_SSS_STREAM_SUBJECTS = "get_sss_stream_subjects"

_THE = r"(?:the\s+)?"
_NAME = r"(?P<name>.+?)"
_OPTIONAL_STREAM = r"(?:\s+stream)?"

# Structural patterns only. Each one asks for subjects of a named stream.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"^(?:which|what)\s+subjects?\s+are(?:\s+(?:taught|offered|included))?\s+"
        rf"in\s+{_THE}{_NAME}{_OPTIONAL_STREAM}$",
        re.I,
    ),
    re.compile(
        rf"^(?:which|what)\s+subjects?\s+belong\s+to\s+{_THE}{_NAME}{_OPTIONAL_STREAM}$",
        re.I,
    ),
    re.compile(
        rf"^what\s+are\s+(?:the\s+)?subjects?\s+in\s+{_THE}{_NAME}{_OPTIONAL_STREAM}$",
        re.I,
    ),
    re.compile(
        rf"^(?:list|show|give)(?:\s+me)?\s+(?:the\s+)?subjects?\s+"
        rf"in\s+{_THE}{_NAME}{_OPTIONAL_STREAM}$",
        re.I,
    ),
    re.compile(
        rf"^subjects?\s+in\s+{_THE}{_NAME}\s+stream$",
        re.I,
    ),
    re.compile(
        rf"^(?:which|what)\s+subjects?\s+does\s+{_THE}{_NAME}\s+stream\s+"
        rf"(?:contain|include|have)$",
        re.I,
    ),
    re.compile(
        rf"^(?:which|what)\s+subjects?\s+are\s+(?:available|offered)\s+"
        rf"in\s+{_THE}{_NAME}{_OPTIONAL_STREAM}$",
        re.I,
    ),
)

_REJECT_KEYS = frozenset(
    {
        "curriculum",
        "the curriculum",
        "mbsse",
        "mbsse curriculum",
        "senior secondary",
        "senior secondary school",
        "school",
        "this",
        "that",
        "it",
        "stream",
    }
)


FOCUS_SUBJECTS = "subjects"
FOCUS_COVERAGE = "coverage"
FOCUS_TOPICS = "topics"
FOCUS_OUTCOMES = "outcomes"
FOCUS_MEMBERSHIP = "membership"
FOCUS_STREAMS = "streams"

_CONTENT_FOCUSES = frozenset({FOCUS_COVERAGE, FOCUS_TOPICS, FOCUS_OUTCOMES})


@dataclass(frozen=True)
class SSSStreamSubjectsIntent:
    intent: str
    stream_name: str
    grade: str | None = None
    subject_name: str | None = None
    focus: str = FOCUS_SUBJECTS


def stream_lookup_key(value: str) -> str:
    """Normalize a stream or subject title for comparison.

    Case, extra whitespace, punctuation, and ``&`` versus ``and`` do not
    change the key. A trailing ``stream`` is ignored. Simple plural endings
    are folded so ``Sciences & Technologies`` matches ``Science and Technology``.
    """
    text = value.strip().lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.endswith(" stream"):
        text = text[: -len(" stream")].strip()
    return " ".join(_stem_token(token) for token in text.split())


def _stem_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def match_sss_stream(streams: list[dict], stream_name: str) -> dict | None:
    """Return the unique stream whose name matches ``stream_name``.

    Matching uses the normalized name from the SSS stream records. Stream ids
    are never assumed by the caller.
    """
    wanted = stream_lookup_key(stream_name)
    if not wanted:
        return None
    matches = [
        stream
        for stream in streams
        if stream_lookup_key(str(stream.get("name") or "")) == wanted
    ]
    if len(matches) != 1:
        return None
    return matches[0]


_SUBJECT_STOPWORDS = frozenset({"and", "the", "for", "of", "with"})


def match_stream_subject(
    assignments: list[dict], subject_name: str
) -> tuple[dict | None, list[dict], str]:
    """Match a subject phrase to stream assignments.

    Returns ``(subject, candidates, kind)``.

    ``exact`` is the official name, including the same name with a public
    parenthetical removed. ``acronym`` is a unique short form printed in that
    official name, such as ``(ICT)``. ``ambiguous`` means the phrase shares
    wording with one or more official subjects but is not itself a name or
    acronym. ``unsupported`` means nothing in the stream shares that wording.

    A partial overlap never selects a subject. No alias is invented.
    """
    wanted = stream_lookup_key(subject_name)
    if not wanted:
        return None, [], "unsupported"
    exact: list[dict] = []
    acronym: list[dict] = []
    for row in assignments:
        name = _assignment_subject_name(row)
        if wanted in _public_name_keys(name):
            exact.append(row)
            continue
        if wanted in _public_acronyms(name):
            acronym.append(row)
    if len(exact) == 1:
        return exact[0], [], "exact"
    if len(exact) > 1:
        return None, exact, "ambiguous"
    if len(acronym) == 1:
        return acronym[0], [], "acronym"
    if len(acronym) > 1:
        return None, acronym, "ambiguous"
    wanted_tokens = _distinctive_tokens(wanted)
    near = [
        row
        for row in assignments
        if wanted_tokens & _public_tokens(_assignment_subject_name(row))
    ]
    if near:
        return None, near, "ambiguous"
    return None, [], "unsupported"


def _public_name_keys(name: str) -> set[str]:
    stripped = re.sub(r"\([^)]*\)", " ", name or "")
    return {key for key in (stream_lookup_key(name), stream_lookup_key(stripped)) if key}


def _public_acronyms(name: str) -> set[str]:
    """Short forms that the official subject title already prints."""
    found: set[str] = set()
    for inner in re.findall(r"\(([^)]*)\)", name or ""):
        token = " ".join(inner.split())
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,12}", token):
            key = stream_lookup_key(token)
            if key:
                found.add(key)
    return found


def _public_tokens(name: str) -> set[str]:
    tokens: set[str] = set()
    for key in _public_name_keys(name):
        tokens.update(_distinctive_tokens(key))
    tokens.update(_public_acronyms(name))
    return tokens


def _distinctive_tokens(key: str) -> set[str]:
    return {
        token
        for token in key.split()
        if token not in _SUBJECT_STOPWORDS and len(token) >= 3
    }


def detect_sss_stream_subjects(question: str | None) -> SSSStreamSubjectsIntent | None:
    """Return a stream question when the shape keeps grade, stream, and subject."""
    if not question or not question.strip():
        return None
    prepared = _prepare_question(question)
    if not prepared:
        return None
    listed = _detect_stream_list(prepared)
    if listed is not None:
        return listed
    scoped = _detect_subject_scope(prepared)
    if scoped is not None:
        return scoped
    for pattern in _PATTERNS:
        match = pattern.match(prepared)
        if match is None:
            continue
        grade, stream_name = _split_grade_and_stream(match.group("name"))
        if not _confident_stream_name(stream_name, prepared):
            continue
        return SSSStreamSubjectsIntent(
            intent=INTENT_SSS_STREAM_SUBJECTS,
            stream_name=stream_name,
            grade=grade,
            focus=FOCUS_SUBJECTS,
        )
    return None


_STREAM_LIST = re.compile(
    r"^what streams (?:are (?:available|offered|there)|exist)"
    r"(?: at| in| for)? (?:the )?(?:sss|senior secondary)"
    r"(?: level| school)?(?:\s*[123])?$",
    re.I,
)

_SCOPED: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"^what does the (?P<stream>.+?) stream cover (?:in|for) (?P<context>.+)$",
            re.I,
        ),
        FOCUS_SUBJECTS,
    ),
    (
        re.compile(
            r"^what does (?P<subject>.+?) cover (?:for|in) (?P<context>.+)$",
            re.I,
        ),
        FOCUS_COVERAGE,
    ),
    (
        re.compile(
            r"^what topics are covered in (?P<subject>.+?) (?:for|in) (?P<context>.+)$",
            re.I,
        ),
        FOCUS_TOPICS,
    ),
    (
        re.compile(
            r"^what are the learning outcomes for (?P<subject>.+?) "
            r"(?:for|in) (?P<context>.+)$",
            re.I,
        ),
        FOCUS_OUTCOMES,
    ),
    (
        re.compile(
            r"^what should students learn in (?P<subject>.+?) (?:for|in) (?P<context>.+)$",
            re.I,
        ),
        FOCUS_OUTCOMES,
    ),
    (
        re.compile(
            r"^is (?P<subject>.+?) part of (?:the )?(?P<context>.+?)(?: stream)?$",
            re.I,
        ),
        FOCUS_MEMBERSHIP,
    ),
)


def _detect_stream_list(prepared: str) -> SSSStreamSubjectsIntent | None:
    if _STREAM_LIST.match(prepared) is None:
        return None
    return SSSStreamSubjectsIntent(
        intent=INTENT_SSS_STREAM_SUBJECTS,
        stream_name="",
        focus=FOCUS_STREAMS,
    )


def _detect_subject_scope(prepared: str) -> SSSStreamSubjectsIntent | None:
    for pattern, focus in _SCOPED:
        match = pattern.match(prepared)
        if match is None:
            continue
        if "stream" in match.groupdict() and "subject" not in match.groupdict():
            stream_name = _clean_stream_name(match.group("stream"))
            grade, _rest = _split_grade_and_stream(match.group("context"))
            if grade is None:
                grade = normalize_grade_code(match.group("context"))
            if not _confident_stream_name(stream_name, prepared):
                return None
            return SSSStreamSubjectsIntent(
                intent=INTENT_SSS_STREAM_SUBJECTS,
                stream_name=stream_name,
                grade=grade if grade and grade.startswith("SSS_") else None,
                focus=FOCUS_SUBJECTS,
            )
        subject_name = _clean_stream_name(match.group("subject"))
        if subject_name.lower().startswith("the ") and subject_name.lower().endswith(
            " stream"
        ):
            continue
        grade, stream_name = _split_grade_and_stream(match.group("context"))
        if not stream_name or not _confident_stream_name(stream_name, prepared):
            return None
        if not subject_name or normalize_grade_code(subject_name):
            return None
        return SSSStreamSubjectsIntent(
            intent=INTENT_SSS_STREAM_SUBJECTS,
            stream_name=stream_name,
            grade=grade,
            subject_name=subject_name,
            focus=focus,
        )
    return None


_GRADE_WORDS = r"(?:sss|senior\s+secondary(?:\s+school)?)\s*[-_]?\s*([123])\b"


def _split_grade_and_stream(raw: str) -> tuple[str | None, str]:
    text = _clean_stream_name(raw)
    match = re.match(
        rf"^{_GRADE_WORDS}[\s,:-]*(.*)$",
        text,
        re.I,
    )
    if match:
        return f"SSS_{match.group(1)}", _clean_stream_name(match.group(2))
    match = re.search(
        rf"(?:^|[\s,]){_GRADE_WORDS}\s*$",
        text,
        re.I,
    )
    if match:
        rest = _clean_stream_name(text[: match.start()])
        return f"SSS_{match.group(1)}", rest
    return None, text


def _assignment_subject_name(row: dict) -> str:
    subject = row.get("subject") if isinstance(row.get("subject"), dict) else {}
    return str(subject.get("name") or "")


def _prepare_question(question: str) -> str:
    text = question.strip()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n?.!;")


def _clean_stream_name(raw: str) -> str:
    name = " ".join(raw.split()).strip(" \t?.!,;:")
    name = re.sub(r"^(?:the)\s+", "", name, count=1, flags=re.I)
    name = re.sub(r"\s+stream$", "", name, count=1, flags=re.I)
    return " ".join(name.split()).strip(" \t?.!,;:")


def _confident_stream_name(name: str, question: str) -> bool:
    if not name:
        return False
    # A captured name that is only a grade stays on the grade catalogue route.
    # Search-style grade detection would also match a stream title that merely
    # contains a grade, so require the whole name to be the grade.
    if re.fullmatch(_GRADE_WORDS, name, re.I):
        return False
    key = stream_lookup_key(name)
    if not key or key in _REJECT_KEYS or len(key) < 3:
        return False
    has_stream_word = re.search(r"\bstreams?\b", question, re.I) is not None
    has_connector = re.search(r"&|\band\b", name, re.I) is not None
    # A compound title or an explicit stream word. Single-word curriculum
    # subjects without "stream" are left for the existing routers.
    return has_stream_word or has_connector
