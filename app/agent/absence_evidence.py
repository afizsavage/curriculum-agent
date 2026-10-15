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


_TITLE_SCOPE = re.compile(
    r"\b(?:unit titles?|matching unit title|title-level|titles only)\b",
    re.IGNORECASE,
)
_CONTENT_SCOPE = re.compile(
    r"\b(?:curriculum statement|learning outcomes?|outcome descriptions?|"
    r"curriculum content|syllabus|records|this evidence|topics?)\b",
    re.IGNORECASE,
)
_SERIALIZED_ID = re.compile(
    r"^(?:Entity ID|Learning outcome ID): (\S+)\s*$",
    re.MULTILINE,
)


def absence_claim_scope(answer: str | None) -> str | None:
    """How wide an absence sentence is.

    ``unit_titles`` is the only scope a title search can support.
    ``curriculum_content`` also covers learning-outcome descriptions.
    ``global`` reaches beyond the retrieved grade-curriculum tree.
    """
    if not answer_makes_absence_claim(answer):
        return None
    if claims_global_absence(answer):
        return "global"
    text = answer or ""
    if _TITLE_SCOPE.search(text) and not _CONTENT_SCOPE.search(text):
        return "unit_titles"
    return "curriculum_content"


def serialized_entity_ids(prompt: str | None) -> list[str]:
    """Ids written into a verifier prompt. Selection lists are not used."""
    found: list[str] = []
    for match in _SERIALIZED_ID.finditer(prompt or ""):
        value = match.group(1)
        if value and value != "unknown" and value not in found:
            found.append(value)
    return found


def format_absence_prompt(records: list[CurriculumEvidence]) -> str:
    """Serialize absence evidence, including outcome descriptions on each unit.

    Outcome text is not truncated. A word past a cutoff would not have been checked.
    """
    if not records:
        return "(no curriculum evidence retrieved)"
    blocks: list[str] = []
    for index, item in enumerate(records, start=1):
        lines = [
            f"--- Record {index} ---",
            f"Entity ID: {item.entity_id or 'unknown'}",
            f"Entity Type: {item.entity_type}",
        ]
        if item.name:
            lines.append(f"Name: {item.name}")
        if item.content and item.content != item.name:
            lines.append(f"Content: {item.content}")
        code = (item.metadata or {}).get("code")
        if code and (item.entity_type or "").lower() != "subject":
            lines.append(f"Code: {code}")
        for outcome in (item.metadata or {}).get("learning_outcomes") or []:
            if not isinstance(outcome, dict):
                continue
            lines.append(f"Learning outcome ID: {outcome.get('id') or 'unknown'}")
            if outcome.get("code"):
                lines.append(f"Learning outcome code: {outcome['code']}")
            lines.append(
                f"Learning outcome description: {outcome.get('description') or ''}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def embedded_outcome_ids(
    evidence: list[CurriculumEvidence],
    *,
    grade: str | None,
    subject: str | None,
) -> list[str]:
    """Outcome ids carried on in-scope unit records from the content tree."""
    found: list[str] = []
    for item in evidence:
        if (item.entity_type or "").lower() != "unit" or not _in_scope(item, grade, subject):
            continue
        for outcome in (item.metadata or {}).get("learning_outcomes") or []:
            if not isinstance(outcome, dict) or not outcome.get("id"):
                continue
            outcome_id = str(outcome["id"])
            if outcome_id not in found:
                found.append(outcome_id)
    return found


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

    ``verification_prompt_ids`` is parsed from the verifier prompt. When that
    list is missing, this check does not invent coverage. A global absence
    claim is never certified from this scope. A curriculum-content claim also
    needs every in-scope outcome description in that prompt.
    """
    answer = state.final_answer or state.draft_answer or ""
    if not answer_makes_absence_claim(answer):
        return False
    if claims_global_absence(answer):
        return True
    # Only ids parsed from the verifier prompt count. A selection list recorded
    # earlier does not prove those records were serialized.
    recorded = (state.metadata or {}).get("verification_prompt_ids")
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
    if any(item.entity_id and item.entity_id not in checked for item in relevant):
        return True
    scope = absence_claim_scope(answer)
    outcome_ids = _outcome_ids_for_scope(state)
    if scope == "unit_titles":
        return False
    if not outcome_ids or any(item_id not in checked for item_id in outcome_ids):
        return True
    return False


def _outcome_ids_for_scope(state: CurriculumQAState) -> list[str]:
    evidence = list(state.evidence)
    found = embedded_outcome_ids(evidence, grade=state.grade, subject=state.subject)
    for item in select_records_for_absence_verification(
        evidence, grade=state.grade, subject=state.subject
    ):
        if (item.entity_type or "").lower() != "learning_outcome" or not item.entity_id:
            continue
        if item.entity_id not in found:
            found.append(item.entity_id)
    return found


def absence_evidence_audit(
    state: CurriculumQAState,
    *,
    selected_ids: list[str],
    prompt_ids: list[str],
) -> dict:
    """Ids and omission reasons. No record text."""
    evidence = list(state.evidence)
    units = [
        item
        for item in select_records_for_absence_verification(
            evidence, grade=state.grade, subject=state.subject
        )
        if (item.entity_type or "").lower() == "unit" and item.entity_id
    ]
    outcome_ids = _outcome_ids_for_scope(state)
    selected = set(selected_ids)
    prompted = set(prompt_ids)
    omitted: list[dict[str, str]] = []
    for item in units:
        if item.entity_id not in selected:
            omitted.append({"id": item.entity_id, "reason": "not_selected"})
        elif item.entity_id not in prompted:
            omitted.append({"id": item.entity_id, "reason": "not_serialized"})
    for outcome_id in outcome_ids:
        if outcome_id not in prompted:
            omitted.append({"id": outcome_id, "reason": "not_serialized"})
    versions = {
        str((item.metadata or {}).get("curriculum_version"))
        for item in units
        if (item.metadata or {}).get("structure_exhaustive") is True
        and (item.metadata or {}).get("curriculum_version")
    }
    exhaustive = bool(units) and len(versions) == 1 and all(
        (item.metadata or {}).get("structure_exhaustive") is True for item in units
    )
    return {
        "in_scope_unit_ids": [item.entity_id for item in units],
        "in_scope_outcome_ids": outcome_ids,
        "selected_ids": list(selected_ids),
        "prompt_ids": list(prompt_ids),
        "omitted": omitted,
        "claim_scope": absence_claim_scope(state.final_answer or state.draft_answer),
        "source_exhaustive": exhaustive,
    }


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
