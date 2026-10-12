"""Record-level coverage checks for curriculum answers.

A verifier score does not establish that an answer includes every retrieved
record. This module confirms representation only from structured bullets and
cited record IDs. Uncertain prose is not counted as coverage. A completeness
claim fails closed when that proof is missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.agent.state import CurriculumQAState
from app.curriculum.codes import normalize_grade_code
from app.curriculum.evidence import CurriculumEvidence
from app.schemas.answer import AnswerEvidenceRef, GroundedAnswer

_ABSOLUTE_CLAIM = re.compile(
    r"("
    r"\bhere are all the\b"
    r"|\ball the topics\b"
    r"|\ball the units\b"
    r"|\bevery topic\b"
    r"|\bevery unit\b"
    r"|\bevery learning outcome\b"
    r"|\bcomplete topics\b"
    r"|\bthe full curriculum\b"
    r"|\bfull curriculum consists\b"
    r"|\bcovers every\b"
    r"|\bcomplete coverage\b"
    r"|\bthese are the complete\b"
    r"|\bcomprehensive list\b"
    r"|\bcomplete list\b"
    r"|\bpupils learn the following\b"
    r"|\bcovers the following topics\b"
    r"|\blist the following\b.+\btopics\b"
    r"|\blist these topics\b"
    r"|\ball subjects are listed\b"
    r"|\ball the subjects\b"
    r"|\bevery subject\b"
    r")",
    re.IGNORECASE,
)
# A list can be complete for one named grade without claiming the whole retrieval.
_SCOPED_COMPLETE = re.compile(
    r"\b(?:complete|full|comprehensive) list for (class|primary|jss|sss)\s*(\d+)\b",
    re.IGNORECASE,
)
_NEGATED_CLAIM = re.compile(
    r"\b(does not|doesn't|do not|not a complete|not the complete|not every|not all)\b",
    re.IGNORECASE,
)
_FAILURE_WORDING = (
    "could not be completed",
    "timed out",
    "unavailable",
    "retrieval did not finish",
    "lookup failed",
)
_ABSENCE_WORDING = ("no syllabus was found", "no syllabus found")
_TITLE_TYPES = {"unit", "topic", "subtopic", "strand"}
_COUNT_BULLET = re.compile(
    r"^(?P<title>.+?) \((?P<count>\d+) separate records share this exact title "
    r"and have no further description\)$"
)
_EXTRA_BULLET = re.compile(r"^(?P<title>.+?) — (?P<extra>.+)$")
_INCLUDED_COUNT = re.compile(
    r"\bincludes (\d+) of (\d+) retrieved records\b",
    re.IGNORECASE,
)
_GRADE_COUNT = re.compile(
    r"\b((?:Primary|JSS|SSS) \d+): (\d+) of (\d+)"
)
_SCOPE_LINE = re.compile(r"^This answer includes \d+ of \d+ retrieved records\b")


@dataclass
class CoverageAssessment:
    """Independent coverage decision for one answer."""

    retrieved_record_count: int
    distinct_retrieved_record_count: int
    prompt_record_count: int | None
    represented_record_count: int
    omitted_record_count: int
    by_grade: dict[str, dict[str, int]]
    represented_ids: list[str]
    omitted_ids: list[str]
    completeness_requested: bool
    claims_complete_coverage: bool
    discloses_partial_scope: bool
    stated_counts_match: bool | None
    multiplicity_ok: bool
    unsupported_outcomes: list[str] = field(default_factory=list)
    unsupported_headings: list[str] = field(default_factory=list)
    title_retrieved_count: int = 0
    title_represented_count: int = 0
    outcome_retrieved_count: int = 0
    outcome_represented_count: int = 0
    grade_scope_ok: bool = True
    retrieval_disclosure_ok: bool = True
    unsafe_retrieval_presentation: bool = False
    catalogue_inventory: bool = False
    silent_catalogue_omission: bool = False
    retrieval_status: str = "found"
    decision: str = "pass"
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieved_record_count": self.retrieved_record_count,
            "distinct_retrieved_record_count": self.distinct_retrieved_record_count,
            "prompt_record_count": self.prompt_record_count,
            "represented_record_count": self.represented_record_count,
            "omitted_record_count": self.omitted_record_count,
            "by_grade": self.by_grade,
            "represented_ids": self.represented_ids,
            "omitted_ids": self.omitted_ids,
            "completeness_requested": self.completeness_requested,
            "claims_complete_coverage": self.claims_complete_coverage,
            "discloses_partial_scope": self.discloses_partial_scope,
            "stated_counts_match": self.stated_counts_match,
            "multiplicity_ok": self.multiplicity_ok,
            "unsupported_outcomes": self.unsupported_outcomes,
            "unsupported_headings": self.unsupported_headings,
            "title_retrieved_count": self.title_retrieved_count,
            "title_represented_count": self.title_represented_count,
            "outcome_retrieved_count": self.outcome_retrieved_count,
            "outcome_represented_count": self.outcome_represented_count,
            "grade_scope_ok": self.grade_scope_ok,
            "retrieval_disclosure_ok": self.retrieval_disclosure_ok,
            "unsafe_retrieval_presentation": self.unsafe_retrieval_presentation,
            "catalogue_inventory": self.catalogue_inventory,
            "silent_catalogue_omission": self.silent_catalogue_omission,
            "retrieval_status": self.retrieval_status,
            "decision": self.decision,
            "reasons": self.reasons,
        }


def completeness_requested(question: str | None) -> bool:
    """Whether the question asks for the retrieved topic or outcome set."""
    from app.agent.answer_generator import _is_catalogue_question

    text = question or ""
    if _is_catalogue_question(text):
        return True
    lowered = text.lower()
    if re.search(r"\bwhat does\b.+\bcover\b", lowered):
        return True
    return bool(
        re.search(r"\b(all|every|complete|full)\b.+\b(topics|units|outcomes)\b", lowered)
    )


def assess_coverage(
    state: CurriculumQAState,
    *,
    answer: str,
    refs: list[AnswerEvidenceRef] | None = None,
) -> CoverageAssessment:
    """Confirm which retrieved records the answer actually represents."""
    evidence = list(state.evidence)
    cited = {
        ref.entity_id
        for ref in (refs if refs is not None else list(state.answer_evidence))
        if getattr(ref, "entity_id", None)
    }
    prompt_ids = state.metadata.get("generation_evidence_ids")
    prompt_count = len(prompt_ids) if isinstance(prompt_ids, list) else None
    subject = state.subject
    curriculum = _requested_curriculum(state)
    represented, issues = _represented_ids(
        answer or "",
        evidence,
        cited,
        audit_bullets=completeness_requested(state.question),
        subject=subject,
        curriculum=curriculum,
    )
    retrieved_ids = [item.entity_id for item in evidence if item.entity_id]
    represented_ids = [entity_id for entity_id in retrieved_ids if entity_id in represented]
    omitted_ids = [entity_id for entity_id in retrieved_ids if entity_id not in represented]
    by_grade = _coverage_by_grade(evidence, represented)
    catalogue_inventory = _has_catalogue_records(evidence)
    # Subject lists are not unit inventories. A subject sentence must not make
    # this guard replace them. Empty retrieval still honors a completeness claim.
    claims_complete = _claims_complete_coverage(answer or "") and (
        catalogue_inventory or not evidence
    )
    discloses = _discloses_partial_scope(answer or "")
    scoped_problems = _scoped_grade_problems(answer or "", by_grade)
    if not scoped_problems and _scoped_claim_is_satisfied(answer or "", by_grade):
        discloses = True
    counts_match = _stated_counts_match(answer or "", evidence, represented, by_grade)
    reasons = list(issues["reasons"])
    if claims_complete and omitted_ids:
        reasons.append("Completeness claim omits retrieved records.")
    if claims_complete and _grade_shortfall(by_grade):
        reasons.append("A grade is missing records while the answer claims complete coverage.")
    if scoped_problems:
        reasons.extend(scoped_problems)
    if counts_match is False:
        reasons.append("The stated included or omitted counts do not match the represented records.")
    if completeness_requested(state.question) and omitted_ids and not discloses and not claims_complete:
        reasons.append("A comprehensive question received a partial answer without a scope disclosure.")
    status = getattr(state.evidence_status, "value", None) or str(state.evidence_status or "found")
    retrieval_incomplete = _retrieval_incomplete(state, status)
    if claims_complete and retrieval_incomplete:
        reasons.append("A completeness claim was made although retrieval did not finish.")
    unsafe = False
    if not evidence and claims_complete:
        reasons.append("A completeness claim was made without retrieved records.")
        unsafe = True
    if not evidence and status == "error" and _presents_unretrieved_catalogue(answer or ""):
        reasons.append("An infrastructure failure was presented as curriculum content.")
        unsafe = True
    if not evidence and status == "not_found" and _presents_unretrieved_catalogue(answer or ""):
        reasons.append("Confirmed absence was presented as a topic catalogue.")
        unsafe = True
    disclosure_gaps = _retrieval_sentences_to_add(state, answer or "", status)
    if status == "error" and _uses_absence_wording(answer or "") and not _uses_failure_wording(answer or ""):
        reasons.append("A retrieval failure was described as a confirmed absence.")
        unsafe = True
    if (
        not evidence
        and status == "not_found"
        and _uses_failure_wording(answer or "")
        and not _uses_absence_wording(answer or "")
    ):
        reasons.append("Confirmed absence was described as a retrieval failure.")
        unsafe = True
    retrieval_disclosure_ok = not disclosure_gaps
    if disclosure_gaps:
        reasons.append("The answer does not disclose the retrieval status.")
    silent_catalogue_omission = bool(
        catalogue_inventory
        and state.intent != "SSS_STREAM_SUBJECTS"
        and completeness_requested(state.question)
        and omitted_ids
        and not discloses
        and not claims_complete
        and (represented_ids or _unsupported_list_markup(answer or ""))
    )
    decision = "reject" if reasons else ("pass_partial" if omitted_ids else "pass")
    title_retrieved, title_represented, outcome_retrieved, outcome_represented = _kind_counts(
        evidence, represented_ids
    )
    return CoverageAssessment(
        retrieved_record_count=len(evidence),
        distinct_retrieved_record_count=len(set(retrieved_ids)),
        prompt_record_count=prompt_count,
        represented_record_count=len(represented_ids),
        omitted_record_count=len(omitted_ids),
        by_grade=by_grade,
        represented_ids=represented_ids,
        omitted_ids=omitted_ids,
        completeness_requested=completeness_requested(state.question),
        claims_complete_coverage=claims_complete,
        discloses_partial_scope=discloses,
        stated_counts_match=counts_match,
        multiplicity_ok=issues["multiplicity_ok"],
        unsupported_outcomes=issues["unsupported_outcomes"],
        unsupported_headings=issues["unsupported_headings"],
        title_retrieved_count=title_retrieved,
        title_represented_count=title_represented,
        outcome_retrieved_count=outcome_retrieved,
        outcome_represented_count=outcome_represented,
        grade_scope_ok=not scoped_problems,
        retrieval_disclosure_ok=retrieval_disclosure_ok,
        unsafe_retrieval_presentation=unsafe,
        catalogue_inventory=catalogue_inventory,
        silent_catalogue_omission=silent_catalogue_omission,
        retrieval_status=status,
        decision=decision,
        reasons=reasons,
    )


def apply_coverage_guard(
    state: CurriculumQAState,
    result: GroundedAnswer,
) -> GroundedAnswer:
    """Keep a false completeness claim from being the answer that is returned.

    The raw decision is stored on the state. When the claim is unsupported, the
    returned text is either corrected or replaced with a scoped listing of the
    records that can be cited.
    """
    raw_answer = result.answer or ""
    state.metadata["coverage_answer_before_guard"] = raw_answer
    if not state.evidence:
        assessment = assess_coverage(state, answer=raw_answer, refs=list(result.evidence))
        state.metadata["coverage_guard_before"] = assessment.as_dict()
        if assessment.decision != "reject" or not _answer_needs_correction(assessment):
            state.metadata["coverage_guard"] = assessment.as_dict()
            return result
        if assessment.unsafe_retrieval_presentation or assessment.claims_complete_coverage:
            cleaned = _safe_empty_answer(state, raw_answer)
        else:
            cleaned = _append_sentences(
                raw_answer,
                _retrieval_sentences_to_add(state, raw_answer, assessment.retrieval_status),
            )
        after = assess_coverage(state, answer=cleaned, refs=[])
        state.metadata["coverage_guard"] = after.as_dict()
        return result.model_copy(
            update={
                "answer": cleaned,
                "evidence": [],
                "summary": _one_line(cleaned),
            }
        )
    before = assess_coverage(state, answer=raw_answer, refs=list(result.evidence))
    state.metadata["coverage_guard_before"] = before.as_dict()
    if before.decision != "reject" or not _answer_needs_correction(before):
        state.metadata["coverage_guard"] = before.as_dict()
        return result
    if _only_retrieval_disclosure(before):
        corrected = _append_sentences(
            raw_answer,
            _retrieval_sentences_to_add(state, raw_answer, before.retrieval_status),
        )
        corrected_refs = list(result.evidence)
        after = assess_coverage(state, answer=corrected, refs=corrected_refs)
        if after.decision != "reject":
            state.metadata["coverage_guard"] = after.as_dict()
            return result.model_copy(
                update={
                    "answer": corrected,
                    "evidence": corrected_refs,
                    "summary": _one_line(corrected),
                }
            )
    corrected, corrected_refs = _correct_answer(state, result, before)
    after = assess_coverage(state, answer=corrected, refs=corrected_refs)
    if after.decision == "reject":
        corrected, corrected_refs = _fallback_listing(state)
        after = assess_coverage(state, answer=corrected, refs=corrected_refs)
    limitations = [
        item
        for item in result.limitations
        if not item.startswith("This answer includes ")
    ]
    disclosure = _disclosure_sentence(corrected)
    if disclosure and disclosure not in limitations:
        limitations.append(disclosure)
    state.metadata["coverage_guard"] = after.as_dict()
    return result.model_copy(
        update={
            "answer": corrected,
            "evidence": corrected_refs,
            "limitations": limitations,
            "summary": _one_line(corrected),
        }
    )


def _correct_answer(
    state: CurriculumQAState,
    result: GroundedAnswer,
    before: CoverageAssessment,
) -> tuple[str, list[AnswerEvidenceRef]]:
    """Drop unsupported claims when the remaining bullets still prove coverage."""
    if before.represented_record_count <= 0:
        return _fallback_listing(state)
    kept: list[str] = []
    for line in (result.answer or "").splitlines():
        if (
            _is_absolute_claim_line(line)
            or _SCOPED_COMPLETE.search(line)
            or _SCOPE_LINE.match(line.strip())
        ):
            continue
        if _is_unsupported_markup(line, result.answer or "", state.evidence, result.evidence):
            continue
        kept.append(line)
    body = "\n".join(kept).strip()
    represented = [
        item
        for item in state.evidence
        if item.entity_id in set(before.represented_ids)
    ]
    note = _scope_note(state.evidence, represented)
    if note:
        body = f"{body}\n\n{note}".strip()
    body = _append_sentences(
        body,
        _retrieval_sentences_to_add(state, body, before.retrieval_status),
    )
    return body, list(result.evidence)


def _fallback_listing(
    state: CurriculumQAState,
) -> tuple[str, list[AnswerEvidenceRef]]:
    """List only the records the generator can cite, and state what is left out."""
    from app.agent.answer_generator import (
        _as_sentence,
        _display_grade,
        _evidence_grade_codes,
        _evidence_refs_from_items,
        _source_group_heading,
        _title_groups,
    )

    selected = [
        item
        for item in _selected_records(state)
        if _record_eligible(item, state.subject, _requested_curriculum(state))
    ]
    lines = ["The following records are the ones this answer accounts for."]
    grades = _evidence_grade_codes(selected)
    for grade in grades:
        grade_rows = [
            item
            for item in selected
            if (normalize_grade_code(item.grade) or item.grade) == grade
        ]
        lines.append("")
        lines.append(f"### {_display_grade(grade) or grade}")
        units = [
            item
            for item in grade_rows
            if (item.entity_type or "").lower() in {"unit", "topic", "subtopic", "strand"}
        ]
        outcomes = [
            item
            for item in grade_rows
            if (item.entity_type or "").lower() == "learning_outcome"
        ]
        current = None
        for group in _title_groups(units):
            heading = group["heading"]
            if heading != current:
                current = heading
                if heading:
                    lines.append("")
                    lines.append(f"**{heading}**")
            records = group["records"]
            title = group["title"]
            extra = group["extra"]
            if extra:
                lines.append(f"* {title} — {extra}")
            elif len(records) == 1:
                lines.append(f"* {title}")
            else:
                lines.append(
                    f"* {title} ({len(records)} separate records share this exact "
                    "title and have no further description)"
                )
        if outcomes:
            lines.append("")
            lines.append("Pupils learn to:")
            for item in outcomes:
                lines.append(f"* {_as_sentence(item.content or item.name or '')}")
    note = _scope_note(state.evidence, selected)
    if note:
        lines.append("")
        lines.append(note)
    text = "\n".join(lines).strip()
    status = getattr(state.evidence_status, "value", None) or str(state.evidence_status or "found")
    text = _append_sentences(text, _retrieval_sentences_to_add(state, text, status))
    return text, _evidence_refs_from_items(selected, state.question or "")


def _selected_records(state: CurriculumQAState) -> list[CurriculumEvidence]:
    prompt_ids = state.metadata.get("generation_evidence_ids")
    if not isinstance(prompt_ids, list) or not prompt_ids:
        return list(state.evidence)
    by_id = {item.entity_id: item for item in state.evidence if item.entity_id}
    return [by_id[entity_id] for entity_id in prompt_ids if entity_id in by_id]


def _answer_needs_correction(assessment: CoverageAssessment) -> bool:
    """Rewrite only claims that contradict the records, not unmapped prose."""
    return bool(
        assessment.claims_complete_coverage
        or assessment.stated_counts_match is False
        or not assessment.multiplicity_ok
        or assessment.unsupported_outcomes
        or assessment.unsupported_headings
        or not assessment.grade_scope_ok
        or assessment.unsafe_retrieval_presentation
        or not assessment.retrieval_disclosure_ok
        or assessment.silent_catalogue_omission
    )


def _only_retrieval_disclosure(assessment: CoverageAssessment) -> bool:
    """A missing retrieval sentence can be appended without replacing the answer."""
    return bool(
        not assessment.retrieval_disclosure_ok
        and not assessment.claims_complete_coverage
        and assessment.stated_counts_match is not False
        and assessment.multiplicity_ok
        and not assessment.unsupported_outcomes
        and not assessment.unsupported_headings
        and assessment.grade_scope_ok
        and not assessment.unsafe_retrieval_presentation
    )


def _represented_ids(
    answer: str,
    evidence: list[CurriculumEvidence],
    cited: set[str],
    *,
    audit_bullets: bool,
    subject: str | None = None,
    curriculum: str | None = None,
) -> tuple[set[str], dict[str, Any]]:
    from app.agent.answer_generator import (
        _display_grade,
        _evidence_grade_codes,
        _public_name,
        _source_group_heading,
    )

    reasons: list[str] = []
    unsupported_outcomes: list[str] = []
    unsupported_headings: list[str] = []
    multiplicity_ok = True
    represented: set[str] = set()
    grades = _evidence_grade_codes(evidence)
    for grade, body in _sections(answer, grades):
        if grade is None and len(grades) > 1:
            continue
        heading = ""
        expect_outcome = False
        allowed_headings = {
            _source_group_heading(item) or ""
            for item in evidence
            if grade is None
            or (normalize_grade_code(item.grade) or item.grade) == grade
        }
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if line.startswith("**") and line.endswith("**") and line.count("**") == 2:
                heading = line.strip("*").strip()
                if audit_bullets and heading and heading not in allowed_headings:
                    unsupported_headings.append(heading)
                    reasons.append(f"Unsupported heading: {heading}.")
                expect_outcome = False
                continue
            represented.update(
                _bold_title_ids(
                    line, evidence, cited, grade, subject=subject, curriculum=curriculum
                )
            )
            if line.lower().rstrip(":") == "pupils learn to":
                expect_outcome = True
                heading = ""
                continue
            bullet_match = re.match(r"^[-*]\s+(\S.*)$", line)
            if not bullet_match:
                continue
            bullet = _visible_bullet(bullet_match.group(1))
            matched, problem = _match_bullet(
                bullet,
                evidence,
                cited,
                grade=grade,
                heading=heading,
                outcome=expect_outcome,
                subject=subject,
                curriculum=curriculum,
            )
            if problem:
                multiplicity_ok = False
                reasons.append(problem)
            if matched:
                represented.update(matched)
                continue
            if expect_outcome or bullet.lower().startswith("pupils learn"):
                unsupported_outcomes.append(bullet)
                reasons.append(f"Unsupported learning outcome: {bullet}.")
            elif (
                audit_bullets
                and _public_name(bullet)
                and not _match_any_title(
                    bullet,
                    evidence,
                    grade,
                    heading,
                    subject=subject,
                    curriculum=curriculum,
                )
            ):
                reasons.append(f"Bullet does not match a retrieved record: {bullet}.")
    if unsupported_headings or unsupported_outcomes:
        multiplicity_ok = multiplicity_ok
    return represented, {
        "reasons": reasons,
        "multiplicity_ok": multiplicity_ok,
        "unsupported_outcomes": unsupported_outcomes,
        "unsupported_headings": unsupported_headings,
    }


def _sections(answer: str, grades: list[str]) -> list[tuple[str | None, str]]:
    from app.agent.answer_generator import _display_grade

    if len(grades) <= 1:
        return [(grades[0] if grades else None, answer or "")]
    matches = list(re.finditer(r"(?m)^#{1,3}\s+(.+)$", answer or ""))

    def grade_for(heading: str) -> str | None:
        matched = [
            grade
            for grade in grades
            if (_display_grade(grade) or grade).lower() in heading.lower()
        ]
        return matched[0] if matched else None

    sections: list[tuple[str | None, str]] = []
    for index, match in enumerate(matches):
        grade = grade_for(match.group(1).strip())
        if grade is None:
            continue
        start = match.end()
        end = len(answer or "")
        for later in matches[index + 1 :]:
            if grade_for(later.group(1).strip()) is not None:
                end = later.start()
                break
        sections.append((grade, (answer or "")[start:end]))
    return sections


def _match_bullet(
    bullet: str,
    evidence: list[CurriculumEvidence],
    cited: set[str],
    *,
    grade: str | None,
    heading: str,
    outcome: bool,
    subject: str | None = None,
    curriculum: str | None = None,
) -> tuple[set[str], str | None]:
    from app.agent.answer_generator import _public_name, _source_group_heading

    bullet = _visible_bullet(bullet)
    if outcome:
        return (
            _match_outcome(
                bullet, evidence, cited, grade, subject=subject, curriculum=curriculum
            ),
            None,
        )
    exact_titles = {
        _public_name(item.name)
        for item in evidence
        if _public_name(item.name)
        and (grade is None or (normalize_grade_code(item.grade) or item.grade) == grade)
    }
    count_match = _COUNT_BULLET.match(bullet)
    extra_match = _EXTRA_BULLET.match(bullet)
    paren_match = re.match(r"^(?P<title>.+?) \((?P<extra>[^)]+)\)$", bullet)
    if bullet in exact_titles:
        title = bullet
        extra = ""
        count = None
    elif count_match:
        title = count_match.group("title")
        count = int(count_match.group("count"))
        extra = ""
    elif extra_match:
        title = extra_match.group("title")
        extra = " ".join(extra_match.group("extra").split())
        count = None
    elif paren_match and paren_match.group("title") in exact_titles:
        title = paren_match.group("title")
        extra = " ".join(paren_match.group("extra").split())
        count = None
    else:
        title = bullet
        extra = ""
        count = None
    def pool_for(required_heading: str) -> list[CurriculumEvidence]:
        return [
            item
            for item in evidence
            if item.entity_id
            and (item.entity_type or "").lower() in _TITLE_TYPES
            and _record_eligible(item, subject, curriculum)
            and (grade is None or (normalize_grade_code(item.grade) or item.grade) == grade)
            and _public_name(item.name) == title
            and (_source_group_heading(item) or "") == required_heading
            and _extra_content(item) == extra
        ]

    pool = pool_for(heading or "")
    if not pool and heading:
        pool = pool_for("")
    if not pool:
        loose = [
            item
            for item in evidence
            if item.entity_id
            and (item.entity_type or "").lower() in _TITLE_TYPES
            and _record_eligible(item, subject, curriculum)
            and (grade is None or (normalize_grade_code(item.grade) or item.grade) == grade)
            and _public_name(item.name) == title
            and _extra_content(item) == extra
        ]
        if len(loose) == 1:
            pool = loose
    if not pool:
        return set(), None
    cited_ids = {item.entity_id for item in pool if item.entity_id in cited}
    pool_ids = {item.entity_id for item in pool if item.entity_id}
    if count is None:
        if len(pool) == 1 and pool[0].entity_id in cited:
            return {pool[0].entity_id or ""}, None
        if len(pool) > 1:
            return set(), (
                f"Title {title!r} stands for {len(pool)} records without a count."
            )
        return set(), None
    if count == len(pool_ids) and cited_ids == pool_ids:
        return cited_ids, None
    if count != len(pool_ids):
        return set(), (
            f"Title {title!r} reports {count} records but {len(pool_ids)} retrieved records share it."
        )
    return set(), f"Title {title!r} cites records that do not match that title."


def _bold_title_ids(
    line: str,
    evidence: list[CurriculumEvidence],
    cited: set[str],
    grade: str | None,
    *,
    subject: str | None = None,
    curriculum: str | None = None,
) -> set[str]:
    from app.agent.answer_generator import _public_name

    found: set[str] = set()
    for title in re.findall(r"\*\*(.+?)\*\*", line):
        visible = _visible_bullet(title)
        pool = [
            item
            for item in evidence
            if item.entity_id in cited
            and _record_eligible(item, subject, curriculum)
            and (grade is None or (normalize_grade_code(item.grade) or item.grade) == grade)
            and _public_name(item.name) == visible
        ]
        if len(pool) == 1 and pool[0].entity_id:
            found.add(pool[0].entity_id)
    return found


def _match_outcome(
    bullet: str,
    evidence: list[CurriculumEvidence],
    cited: set[str],
    grade: str | None,
    *,
    subject: str | None = None,
    curriculum: str | None = None,
) -> set[str]:
    wanted = _outcome_key(bullet)
    pool = [
        item
        for item in evidence
        if item.entity_id
        and (item.entity_type or "").lower() == "learning_outcome"
        and _record_eligible(item, subject, curriculum)
        and (grade is None or (normalize_grade_code(item.grade) or item.grade) == grade)
        and wanted in {_outcome_key(item.content or ""), _outcome_key(item.name or "")}
    ]
    if len(pool) == 1 and pool[0].entity_id in cited:
        return {pool[0].entity_id or ""}
    return set()


def _match_any_title(
    bullet: str,
    evidence: list[CurriculumEvidence],
    grade: str | None,
    heading: str,
    *,
    subject: str | None = None,
    curriculum: str | None = None,
) -> bool:
    matched, _problem = _match_bullet(
        bullet,
        evidence,
        {item.entity_id for item in evidence if item.entity_id},
        grade=grade,
        heading=heading,
        outcome=False,
        subject=subject,
        curriculum=curriculum,
    )
    return bool(matched)


def _extra_content(item: CurriculumEvidence) -> str:
    from app.agent.answer_generator import _public_name

    title = _public_name(item.name) or ""
    content = " ".join(str(item.content or "").split())
    return "" if content == title else content


def _outcome_key(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .;")


def _coverage_by_grade(
    evidence: list[CurriculumEvidence],
    represented: set[str],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for item in evidence:
        grade = normalize_grade_code(item.grade) or item.grade or "unspecified"
        bucket = counts.setdefault(grade, {"retrieved": 0, "represented": 0, "omitted": 0})
        bucket["retrieved"] += 1
        if item.entity_id in represented:
            bucket["represented"] += 1
        else:
            bucket["omitted"] += 1
    return counts


def _grade_shortfall(by_grade: dict[str, dict[str, int]]) -> bool:
    return any(bucket["omitted"] for bucket in by_grade.values())


def _has_catalogue_records(evidence: list[CurriculumEvidence]) -> bool:
    """Unit and outcome records are the inventory this guard can account for."""
    for item in evidence:
        kind = (item.entity_type or "").lower()
        if kind in _TITLE_TYPES or kind == "learning_outcome":
            return True
    return False


def _unsupported_list_markup(answer: str) -> bool:
    """Numbered lists and tables are not accepted as record proof."""
    for line in (answer or "").splitlines():
        stripped = line.strip()
        if re.match(r"^\d+[.)]\s+\S", stripped):
            return True
        if stripped.startswith("|") and "|" in stripped[1:]:
            return True
    return False


def _visible_bullet(bullet: str) -> str:
    """Collapse whitespace and one wrapping emphasis marker. Punctuation stays."""
    text = " ".join((bullet or "").split()).strip()
    wrapped = re.fullmatch(r"\*\*(.+)\*\*", text) or re.fullmatch(r"_(.+)_", text)
    if wrapped:
        text = " ".join(wrapped.group(1).split()).strip()
    return text


def _record_eligible(
    item: CurriculumEvidence,
    subject: str | None,
    curriculum: str | None,
) -> bool:
    if subject and item.subject:
        from app.curriculum.codes import normalize_subject_code

        left = normalize_subject_code(subject)
        right = normalize_subject_code(item.subject)
        if left and right and left != right:
            return False
    if curriculum:
        meta = item.metadata or {}
        found = meta.get("curriculum_code") or meta.get("curriculum_id")
        if found and str(found) != str(curriculum):
            return False
    return True


def _requested_curriculum(state: CurriculumQAState) -> str | None:
    meta = state.metadata or {}
    value = meta.get("curriculum_code") or meta.get("curriculum_id")
    return str(value) if value else None


def _kind_counts(
    evidence: list[CurriculumEvidence],
    represented_ids: list[str],
) -> tuple[int, int, int, int]:
    represented = set(represented_ids)
    titles = [item for item in evidence if (item.entity_type or "").lower() in _TITLE_TYPES]
    outcomes = [
        item
        for item in evidence
        if (item.entity_type or "").lower() == "learning_outcome"
    ]
    return (
        len(titles),
        sum(1 for item in titles if item.entity_id in represented),
        len(outcomes),
        sum(1 for item in outcomes if item.entity_id in represented),
    )


def _scoped_grades(answer: str) -> list[str]:
    found: list[str] = []
    for kind, number in _SCOPED_COMPLETE.findall(answer or ""):
        family = kind.lower()
        if family in {"class", "primary"}:
            grade = f"CLASS_{number}"
        elif family == "jss":
            grade = f"JSS_{number}"
        else:
            grade = f"SSS_{number}"
        if grade not in found:
            found.append(grade)
    return found


def _scoped_grade_problems(answer: str, by_grade: dict[str, dict[str, int]]) -> list[str]:
    problems: list[str] = []
    for grade in _scoped_grades(answer):
        bucket = by_grade.get(grade)
        if bucket is None or bucket["retrieved"] == 0 or bucket["omitted"]:
            label = grade.replace("CLASS_", "Primary ").replace("_", " ")
            problems.append(
                f"The answer calls {label} complete but that grade's records are not all represented."
            )
    return problems


def _scoped_claim_is_satisfied(answer: str, by_grade: dict[str, dict[str, int]]) -> bool:
    grades = _scoped_grades(answer)
    if not grades or _scoped_grade_problems(answer, by_grade):
        return False
    return all(
        by_grade.get(grade, {}).get("retrieved", 0) > 0
        and by_grade.get(grade, {}).get("omitted", 1) == 0
        for grade in grades
    )


def _uses_failure_wording(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(phrase in lowered for phrase in _FAILURE_WORDING)


def _uses_absence_wording(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(phrase in lowered for phrase in _ABSENCE_WORDING)


def _retrieval_incomplete(state: CurriculumQAState, status: str) -> bool:
    if status in {"error", "partial"}:
        return True
    from app.agent.answer_generator import _scope_retrieval_limitations

    return bool(_scope_retrieval_limitations(state))


def _presents_unretrieved_catalogue(answer: str) -> bool:
    for line in (answer or "").splitlines():
        stripped = line.strip()
        if re.match(r"^[-*]\s+\S", stripped) or re.match(r"^\d+\.\s+\S", stripped):
            return True
        if stripped.startswith("|") and "|" in stripped[1:]:
            return True
    return False


def _retrieval_sentences_to_add(
    state: CurriculumQAState,
    answer: str,
    status: str,
) -> list[str]:
    from app.agent.answer_generator import _scope_retrieval_limitations

    pending: list[str] = []
    for note in _scope_retrieval_limitations(state):
        if note not in (answer or ""):
            pending.append(note)
    lowered = (answer or "").lower()
    failure = _uses_failure_wording(answer)
    absence = _uses_absence_wording(answer)
    if (
        status == "error"
        and not failure
        and not any(_uses_failure_wording(note) for note in pending)
    ):
        pending.append(
            "The curriculum service timed out or was unavailable, so "
            "this question could not be checked against the syllabus."
        )
    if status == "not_found" and not state.evidence and not absence:
        pending.append(
            "No syllabus was found for the requested grades, so there is no topic list to present."
        )
    if (
        status == "partial"
        and not failure
        and "could not be completed" not in lowered
        and not any("could not be completed" in note for note in pending)
    ):
        pending.append(
            "This list is not a complete catalogue because retrieval did not finish "
            "for every requested grade."
        )
    return list(dict.fromkeys(pending))


def _append_sentences(body: str, sentences: list[str]) -> str:
    text = (body or "").strip()
    for sentence in sentences:
        if sentence and sentence not in text:
            text = f"{text}\n\n{sentence}".strip() if text else sentence
    return text


def _safe_empty_answer(state: CurriculumQAState, answer: str) -> str:
    """Drop invented topics when nothing was retrieved."""
    kept: list[str] = []
    for line in (answer or "").splitlines():
        stripped = line.strip()
        if _is_absolute_claim_line(line) or _SCOPED_COMPLETE.search(line):
            continue
        if re.match(r"^[-*]\s+\S", stripped) or re.match(r"^\d+\.\s+\S", stripped):
            continue
        if stripped.startswith("#") or (stripped.startswith("|") and stripped.endswith("|")):
            continue
        if stripped.startswith("**") and stripped.endswith("**"):
            continue
        kept.append(line)
    body = "\n".join(kept).strip()
    status = getattr(state.evidence_status, "value", None) or str(state.evidence_status or "")
    if status == "error" and (not body or (_uses_absence_wording(body) and not _uses_failure_wording(body))):
        body = (
            "The curriculum service timed out or was unavailable, so "
            "this question could not be checked against the syllabus."
        )
    elif status == "not_found" and (
        not body or (_uses_failure_wording(body) and not _uses_absence_wording(body))
    ):
        body = (
            "No syllabus was found for the requested grades, so there is no topic list to present."
        )
    return _append_sentences(body, _retrieval_sentences_to_add(state, body, status))


def _claims_complete_coverage(answer: str) -> bool:
    return any(_is_absolute_claim_line(line) for line in answer.splitlines()) or any(
        _is_absolute_claim_line(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", answer)
    )


def _is_absolute_claim_line(text: str) -> bool:
    if _NEGATED_CLAIM.search(text):
        return False
    # Drop a grade-scoped list claim before looking for a whole-catalogue claim.
    residual = _SCOPED_COMPLETE.sub(" ", text)
    return bool(_ABSOLUTE_CLAIM.search(residual))


def _discloses_partial_scope(answer: str) -> bool:
    lowered = answer.lower()
    return any(
        phrase in lowered
        for phrase in (
            "does not include every retrieved record",
            "not a complete",
            "not the complete",
            "partial selection",
            "this is a sample",
            "this is a partial",
            "only a sample",
            "here are the main topics",
            "here are some examples",
            "some examples",
            "have no public title and are not listed",
        )
    )


def _stated_counts_match(
    answer: str,
    evidence: list[CurriculumEvidence],
    represented: set[str],
    by_grade: dict[str, dict[str, int]],
) -> bool | None:
    from app.agent.answer_generator import _display_grade

    overall = _INCLUDED_COUNT.search(answer)
    grades = list(_GRADE_COUNT.finditer(answer))
    if not overall and not grades:
        return None
    if overall:
        included = int(overall.group(1))
        retrieved = int(overall.group(2))
        if included != len(represented) or retrieved != len(evidence):
            return False
    if grades:
        stated = {match.group(1): (int(match.group(2)), int(match.group(3))) for match in grades}
        for grade, bucket in by_grade.items():
            label = _display_grade(grade) or grade
            pair = stated.get(label)
            if pair is None:
                return False
            if pair != (bucket["represented"], bucket["retrieved"]):
                return False
        if len(stated) != len(by_grade):
            return False
    return True


def _is_unsupported_markup(
    line: str,
    answer: str,
    evidence: list[CurriculumEvidence],
    refs: list[AnswerEvidenceRef],
) -> bool:
    from app.agent.answer_generator import _evidence_grade_codes

    stripped = line.strip()
    if not stripped.startswith("* ") and not (
        stripped.startswith("**") and stripped.endswith("**")
    ):
        return False
    cited = {ref.entity_id for ref in refs if ref.entity_id}
    _represented, issues = _represented_ids(
        answer, evidence, cited, audit_bullets=True
    )
    bullet = stripped[2:].strip() if stripped.startswith("* ") else stripped.strip("*").strip()
    if stripped.startswith("* "):
        return any(bullet in reason for reason in issues["reasons"])
    return any(bullet in item for item in issues["unsupported_headings"])


def _scope_note(
    evidence: list[CurriculumEvidence],
    represented: list[CurriculumEvidence],
) -> str | None:
    from app.agent.answer_generator import _display_grade, _grade_rank

    if not evidence or len(represented) >= len(evidence):
        return None
    retrieved = _tally(evidence)
    kept = _tally(represented)
    parts = [
        f"{_display_grade(grade) or grade}: {kept.get(grade, 0)} of {retrieved[grade]}"
        for grade in sorted(retrieved, key=_grade_rank)
    ]
    return (
        f"This answer includes {len(represented)} of {len(evidence)} retrieved records "
        f"({'; '.join(parts)}). It does not include every retrieved record."
    )


def _tally(evidence: list[CurriculumEvidence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in evidence:
        grade = normalize_grade_code(item.grade) or item.grade or "unspecified"
        counts[grade] = counts.get(grade, 0) + 1
    return counts


def _disclosure_sentence(answer: str) -> str | None:
    for line in answer.splitlines():
        if _SCOPE_LINE.match(line.strip()):
            return line.strip()
    return None


def _one_line(answer: str) -> str | None:
    for line in answer.splitlines():
        text = line.strip("#* ").strip()
        if text:
            return text[:240]
    return None
