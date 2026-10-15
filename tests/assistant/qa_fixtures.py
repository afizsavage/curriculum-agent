"""Fake QA outcomes for assistant router and composer tests.

These are not live agent results. They mirror statuses the public AskResponse
already uses.
"""

from __future__ import annotations

from app.assistant.contract import CurriculumExecutionFailure, CurriculumOutcome
from app.schemas.agent import (
    AnswerEvidenceSummary,
    AskMetadata,
    AskResponse,
    EvidenceSummary,
    VerificationSummary,
)
from app.schemas.answer import AnswerConfidence

_QUESTION = "What topics are taught in Primary 4 Mathematics?"
_CONVERSATION = "conv-fixture"


def completed_with_evidence() -> CurriculumOutcome:
    answer = (
        "Primary 4 Mathematics includes Number and Numeration and "
        "Everyday Arithmetic."
    )
    return CurriculumOutcome(
        http_status=200,
        coverage_decision="accept",
        response=AskResponse(
            conversation_id=_CONVERSATION,
            question=_QUESTION,
            answer=answer,
            status="completed",
            evidence=[
                EvidenceSummary(
                    entity_type="topic",
                    entity_id="topic-number",
                    name="Number and Numeration",
                    grade="CLASS_4",
                    subject="MATHEMATICS",
                )
            ],
            answer_evidence=[
                AnswerEvidenceSummary(
                    entity_id="topic-number",
                    entity_type="topic",
                    claim="Number and Numeration is taught in Primary 4 Mathematics.",
                    name="Number and Numeration",
                    grade="CLASS_4",
                    subject="MATHEMATICS",
                )
            ],
            confidence=AnswerConfidence.HIGH,
            verification=VerificationSummary(
                passed=True,
                score=1.0,
                recommendation="accept",
            ),
            metadata=AskMetadata(
                evidence_status="found",
                evidence_count=1,
                verification_status="passed",
                verification_attempts=1,
                termination_reason="verification_passed",
            ),
        ),
    )


def needs_clarification() -> CurriculumOutcome:
    return CurriculumOutcome(
        http_status=200,
        response=AskResponse(
            conversation_id=_CONVERSATION,
            question="What topics are taught in mathematics?",
            answer=None,
            status="needs_clarification",
            clarification="Which grade or level would you like me to check?",
            verification=VerificationSummary(
                passed=False,
                score=0.0,
                recommendation="clarify",
            ),
            metadata=AskMetadata(
                evidence_status="not_found",
                verification_status="needs_clarification",
                termination_reason="clarify",
            ),
        ),
    )


def insufficient_evidence() -> CurriculumOutcome:
    answer = (
        "I couldn't find sufficient MBSSE curriculum evidence to answer "
        "this reliably."
    )
    return CurriculumOutcome(
        http_status=200,
        response=AskResponse(
            conversation_id=_CONVERSATION,
            question=_QUESTION,
            answer=answer,
            status="insufficient_evidence",
            confidence=AnswerConfidence.LOW,
            limitations=[
                "Available MBSSE curriculum records were insufficient for a reliable answer."
            ],
            verification=VerificationSummary(
                passed=False,
                score=0.0,
                recommendation="fallback",
                issues=["Evidence did not establish the requested curriculum claim."],
            ),
            metadata=AskMetadata(
                evidence_status="not_found",
                verification_status="insufficient_evidence",
                termination_reason="verification_fallback",
            ),
        ),
    )


def partial_incomplete_coverage() -> CurriculumOutcome:
    answer = (
        "Some Class 4 Mathematics topics were found, but the records do not "
        "establish a complete topic list."
    )
    return CurriculumOutcome(
        http_status=200,
        coverage_decision="unavailable",
        response=AskResponse(
            conversation_id=_CONVERSATION,
            question="List every topic in the Class 4 Mathematics syllabus.",
            answer=answer,
            status="insufficient_evidence",
            confidence=AnswerConfidence.LOW,
            evidence=[
                EvidenceSummary(
                    entity_type="topic",
                    entity_id="topic-number",
                    name="Number and Numeration",
                    grade="CLASS_4",
                    subject="MATHEMATICS",
                )
            ],
            limitations=[
                "Syllabus coverage could not be validated (structured_blocks_unavailable)."
            ],
            verification=VerificationSummary(
                passed=False,
                score=0.0,
                recommendation="fallback",
                issues=[
                    "Syllabus coverage could not be validated (structured_blocks_unavailable)."
                ],
            ),
            metadata=AskMetadata(
                evidence_status="partial",
                evidence_count=1,
                verification_status="insufficient_evidence",
                termination_reason="no_retrieval_progress_incomplete_source",
            ),
        ),
    )


def evidence_status_error_http_200() -> CurriculumOutcome:
    return CurriculumOutcome(
        http_status=200,
        response=AskResponse(
            conversation_id=_CONVERSATION,
            question=_QUESTION,
            answer=None,
            status="insufficient_evidence",
            error="Curriculum API request timed out",
            limitations=["Curriculum retrieval failed before evidence was collected."],
            verification=VerificationSummary(
                passed=False,
                score=0.0,
                recommendation="fallback",
            ),
            metadata=AskMetadata(
                evidence_status="error",
                verification_status="insufficient_evidence",
                termination_reason="verification_fallback",
            ),
        ),
    )


def execution_failure() -> CurriculumOutcome:
    return CurriculumOutcome(
        failure=CurriculumExecutionFailure(
            kind="execution_failure",
            code="AGENT_EXECUTION_FAILURE",
            message="Agent failed while processing the question",
            http_status=500,
        )
    )


def timeout_failure() -> CurriculumOutcome:
    return CurriculumOutcome(
        failure=CurriculumExecutionFailure(
            kind="timeout",
            code="LLM_TIMEOUT",
            message="LLM request timed out",
            http_status=504,
        )
    )
