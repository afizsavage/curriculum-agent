"""Which retrieved records an absence claim must actually have checked.

Same-name units are not interchangeable. One unit's learning-outcome text
can mention a subject that another's does not. A failed topic lookup, and a
Class 4 Mathematics tree, do not establish absence from every MBSSE document.
"""

from __future__ import annotations

import re

from app.agent.state import CurriculumQAState
from app.curriculum.codes import normalize_grade_code, normalize_subject_code
from app.curriculum.evidence import CurriculumEvidence

_ABSENCE_CLAIM = re.compile(
    r"\b(?:does not include|do not include|not include|not present|not among|"
    r"not found|not published|no unit|no topic|no evidence|no record|"
    r"absent|no mention)\b",
    re.IGNORECASE,
)
_GLOBAL_ABSENCE = re.compile(
    r"\b(?:entire|whole|every|all)\b(?:\s+\w+){0,4}\s+"
    r"(?:curriculum|mbsse|documents?)\b",
    re.IGNORECASE,
)
_DENIES_GLOBAL = re.compile(
    r"\b(?:does not claim|do not claim|not claiming|not a claim)\b",
    re.IGNORECASE,
)
_TOPIC_PROBE = re.compile(
    r"\b(?:say about|says about|said about)\b",
    re.IGNORECASE,
)


def answer_makes_absence_claim(answer: str | None) -> bool:
    return bool(answer and _ABSENCE_CLAIM.search(answer))


def question_probes_topic_presence(question: str | None) -> bool:
    """A 'what does the syllabus say about X' question can be answered by absence."""
    return bool(question and _TOPIC_PROBE.search(question))


def needs_absence_verification_context(question: str | None, answer: str | None) -> bool:
    return question_probes_topic_presence(question) or answer_makes_absence_claim(answer)


def claims_global_absence(answer: str | None) -> bool:
    """True when a sentence asserts absence beyond the retrieved scope."""
    if not answer:
        return False
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        if _DENIES_GLOBAL.search(sentence):
            continue
        if _GLOBAL_ABSENCE.search(sentence) and _ABSENCE_CLAIM.search(sentence):
            return True
    return False


def select_records_for_absence_verification(
    evidence: list[CurriculumEvidence],
    *,
    grade: str | None,
    subject: str | None,
) -> list[CurriculumEvidence]:
    """In-scope units and their outcome text, with same-name units kept apart.

    Records for another subject stay out. They are not evidence for this claim
    and they are not treated as omitted from it.
    """
    units = [
        item
        for item in evidence
        if (item.entity_type or "").lower() == "unit" and _in_scope(item, grade, subject)
    ]
    unit_ids = {item.entity_id for item in units if item.entity_id}
    selected: list[CurriculumEvidence] = list(units)
    for item in evidence:
        if (item.entity_type or "").lower() != "learning_outcome":
            continue
        if _outcome_in_scope(item, grade, subject, unit_ids):
            selected.append(item)
    return selected


def absence_finish_blocked(state: CurriculumQAState) -> bool:
    """True when an absence claim would finish without the records it depends on.

    An explicit ``verification_evidence_ids`` list is what the verifier was
    given. When that list is missing, the check is the full in-scope set,
    which is what absence verification selects. A global absence claim is
    never certified from this scope.
    """
    answer = state.final_answer or state.draft_answer or ""
    if not answer_makes_absence_claim(answer):
        return False
    if claims_global_absence(answer):
        return True
    recorded = (state.metadata or {}).get("verification_evidence_ids")
    # No explicit check list means absence verification still owes the full
    # in-scope set. Do not treat that as a failed certification here; the
    # verifier records the ids when it builds its prompt.
    if recorded is None:
        return False
    relevant = select_records_for_absence_verification(
        list(state.evidence),
        grade=state.grade,
        subject=state.subject,
    )
    if not relevant:
        return True
    checked = {str(item_id) for item_id in recorded if item_id}
    return any(item.entity_id and item.entity_id not in checked for item in relevant)


def _in_scope(item: CurriculumEvidence, grade: str | None, subject: str | None) -> bool:
    item_grade = normalize_grade_code(item.grade)
    item_subject = normalize_subject_code(item.subject)
    wanted_grade = normalize_grade_code(grade)
    wanted_subject = normalize_subject_code(subject)
    if wanted_grade and item_grade and item_grade != wanted_grade:
        return False
    if wanted_subject and item_subject and item_subject != wanted_subject:
        return False
    if wanted_grade and not item_grade:
        return False
    if wanted_subject and not item_subject:
        return False
    return True


def _outcome_in_scope(
    item: CurriculumEvidence,
    grade: str | None,
    subject: str | None,
    unit_ids: set[str],
) -> bool:
    parent = str((item.metadata or {}).get("parent_id") or item.topic or "")
    if parent and parent in unit_ids:
        return True
    return _in_scope(item, grade, subject)
