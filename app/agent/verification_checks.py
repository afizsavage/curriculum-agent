"""Deterministic verification checks (no LLM)."""

from __future__ import annotations

import re
from typing import Any

from app.agent.state import CurriculumQAState
from app.curriculum.codes import normalize_grade_code, normalize_subject_code
from app.curriculum.evidence import EvidenceStatus
from app.schemas.verification import (
    ClaimAssessment,
    ClaimVerdict,
    MissingEvidenceRequest,
    VerificationRecommendation,
    VerificationResult,
)


def run_deterministic_checks(state: CurriculumQAState) -> VerificationResult:
    """Compare answer + refs against retrieved evidence without LLM judgment."""
    issues: list[str] = []
    unsupported: list[str] = []
    incorrect: list[str] = []
    missing: list[MissingEvidenceRequest | str] = []
    claims: list[ClaimAssessment] = []

    answer = (state.final_answer or state.draft_answer or "").strip()
    evidence_ids = {e.entity_id for e in state.evidence if e.entity_id}
    evidence_grades = {
        normalize_grade_code(e.grade) for e in state.evidence if e.grade
    }
    evidence_grades.discard(None)
    evidence_subjects = {
        normalize_subject_code(e.subject) for e in state.evidence if e.subject
    }
    evidence_subjects.discard(None)

    requested_grade = normalize_grade_code(state.grade)
    requested_subject = normalize_subject_code(state.subject)

    if not answer:
        issues.append("No answer was generated.")
        missing.append(
            MissingEvidenceRequest(
                type="curriculum_content",
                grade=state.grade,
                subject=state.subject,
                topic=state.topic,
                detail="Answer text is empty.",
            )
        )
        return _result(
            passed=False,
            score=0.0,
            issues=issues,
            unsupported=unsupported,
            incorrect=incorrect,
            missing=missing,
            claims=claims,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            metadata={"source": "deterministic", "empty_answer": True},
        )

    if not state.evidence or state.evidence_status == EvidenceStatus.NOT_FOUND:
        issues.append("No curriculum evidence supports the answer.")
        missing.append(
            MissingEvidenceRequest(
                type="curriculum_content",
                grade=state.grade,
                subject=state.subject,
                topic=state.topic,
                query=state.topic or state.question[:80],
                detail="No retrieved evidence.",
            )
        )
        return _result(
            passed=False,
            score=0.1,
            issues=issues,
            unsupported=unsupported,
            incorrect=incorrect,
            missing=missing,
            claims=claims,
            recommendation=_clarify_or_retrieve(state),
            metadata={"source": "deterministic", "no_evidence": True},
        )

    # Grade consistency
    if requested_grade and evidence_grades and requested_grade not in evidence_grades:
        issues.append(
            f"Requested grade {requested_grade} is not reflected in retrieved evidence "
            f"({', '.join(sorted(str(g) for g in evidence_grades))})."
        )
        incorrect.append(f"Answer/evidence grade mismatch for {requested_grade}.")
        missing.append(
            MissingEvidenceRequest(
                type="grade_placement",
                grade=state.grade,
                subject=state.subject,
                topic=state.topic,
                detail=f"Need evidence for grade {requested_grade}.",
            )
        )

    # Subject consistency
    if (
        requested_subject
        and evidence_subjects
        and requested_subject not in evidence_subjects
    ):
        issues.append(
            f"Requested subject {requested_subject} is not reflected in retrieved evidence."
        )
        incorrect.append(f"Subject mismatch for {requested_subject}.")
        missing.append(
            MissingEvidenceRequest(
                type="subject",
                grade=state.grade,
                subject=state.subject,
                topic=state.topic,
                detail=f"Need evidence for subject {requested_subject}.",
            )
        )

    # Answer evidence refs must exist in retrieved evidence
    for ref in state.answer_evidence:
        claim_text = ref.claim or ref.name or ref.entity_id
        if ref.entity_id not in evidence_ids:
            unsupported.append(claim_text)
            issues.append(
                f"Answer references entity_id '{ref.entity_id}' which was not retrieved."
            )
            claims.append(
                ClaimAssessment(
                    claim=claim_text,
                    verdict=ClaimVerdict.UNSUPPORTED,
                    notes="entity_id not in evidence",
                )
            )
        else:
            claims.append(
                ClaimAssessment(
                    claim=claim_text,
                    verdict=ClaimVerdict.SUPPORTED,
                    evidence_ids=[ref.entity_id],
                )
            )

    # Ambiguous question without grade/subject when question asks broadly
    if _looks_ambiguous(state) and not state.grade:
        issues.append("Question is ambiguous about grade/level.")
        return _result(
            passed=False,
            score=0.4,
            issues=issues,
            unsupported=unsupported,
            incorrect=incorrect,
            missing=missing,
            claims=claims,
            recommendation=VerificationRecommendation.CLARIFY,
            clarification="Which grade or level would you like me to check?",
            metadata={"source": "deterministic", "ambiguous": True},
        )

    # Hallucinated grade mentions conflicting with request
    mentioned_grades = _grades_mentioned_in_text(answer)
    if requested_grade and mentioned_grades:
        for mentioned in mentioned_grades:
            if mentioned != requested_grade and mentioned not in evidence_grades:
                issues.append(
                    f"Answer mentions {mentioned} which is not supported by retrieved evidence."
                )
                incorrect.append(f"Unsupported grade mention: {mentioned}")

    hard_fail = bool(incorrect) or (
        bool(unsupported) and not any(c.verdict == ClaimVerdict.SUPPORTED for c in claims)
    )
    soft_missing = bool(missing) and not hard_fail and not claims

    if hard_fail:
        recommendation = VerificationRecommendation.RETRIEVE_MORE
        if not missing:
            missing.append(
                MissingEvidenceRequest(
                    type="learning_objective" if state.topic else "topic",
                    grade=state.grade,
                    subject=state.subject,
                    topic=state.topic,
                    query=state.topic,
                    detail="Need supporting curriculum evidence for incorrect claims.",
                )
            )
        score = max(0.1, 0.55 - 0.1 * len(incorrect) - 0.05 * len(unsupported))
        return _result(
            passed=False,
            score=score,
            issues=issues,
            unsupported=unsupported,
            incorrect=incorrect,
            missing=missing,
            claims=claims,
            recommendation=recommendation,
            metadata={"source": "deterministic", "hard_fail": True},
        )

    if soft_missing and not state.answer_evidence:
        # Evidence exists but answer not linked — may still be acceptable for stub
        # Leave decision to LLM layer / stub pass path.
        return _result(
            passed=True,
            score=0.75,
            issues=issues,
            unsupported=unsupported,
            incorrect=incorrect,
            missing=missing,
            claims=claims,
            recommendation=VerificationRecommendation.ACCEPT,
            metadata={"source": "deterministic", "soft_pass": True},
        )

    if issues and missing and not unsupported and not incorrect:
        return _result(
            passed=False,
            score=0.55,
            issues=issues,
            unsupported=unsupported,
            incorrect=incorrect,
            missing=missing,
            claims=claims,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            metadata={"source": "deterministic", "needs_more": True},
        )

    score = 0.9 if claims and not issues else (0.8 if not issues else 0.7)
    return _result(
        passed=not issues or (bool(claims) and not incorrect and not unsupported),
        score=score,
        issues=issues,
        unsupported=unsupported,
        incorrect=incorrect,
        missing=missing,
        claims=claims,
        recommendation=(
            VerificationRecommendation.ACCEPT
            if not incorrect and not unsupported
            else VerificationRecommendation.RETRIEVE_MORE
        ),
        metadata={"source": "deterministic"},
    )


def _result(
    *,
    passed: bool,
    score: float,
    issues: list[str],
    unsupported: list[str],
    incorrect: list[str],
    missing: list[MissingEvidenceRequest | str],
    claims: list[ClaimAssessment],
    recommendation: VerificationRecommendation,
    clarification: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> VerificationResult:
    return VerificationResult(
        passed=passed,
        score=max(0.0, min(1.0, score)),
        issues=issues,
        unsupported_claims=unsupported,
        incorrect_claims=incorrect,
        missing_evidence=missing,
        claims=claims,
        recommendation=recommendation,
        clarification=clarification,
        metadata=metadata or {},
    )


def _clarify_or_retrieve(state: CurriculumQAState) -> VerificationRecommendation:
    if _looks_ambiguous(state) and not state.grade:
        return VerificationRecommendation.CLARIFY
    return VerificationRecommendation.RETRIEVE_MORE


def _looks_ambiguous(state: CurriculumQAState) -> bool:
    q = state.question.lower()
    if state.grade:
        return False
    broad = any(
        phrase in q
        for phrase in (
            "what does the curriculum say",
            "where does the curriculum",
            "about fractions",
            "about measurement",
            "in the curriculum",
        )
    )
    return broad and not state.subject


_GRADE_PATTERNS = [
    (re.compile(r"\bprimary\s*([1-6])\b", re.I), "CLASS_{}"),
    (re.compile(r"\bclass\s*([1-6])\b", re.I), "CLASS_{}"),
    (re.compile(r"\bjss\s*([1-3])\b", re.I), "JSS_{}"),
    (re.compile(r"\bsss\s*([1-3])\b", re.I), "SSS_{}"),
]


def _grades_mentioned_in_text(text: str) -> set[str]:
    found: set[str] = set()
    for pattern, template in _GRADE_PATTERNS:
        for match in pattern.finditer(text):
            found.add(template.format(match.group(1)))
    return found
