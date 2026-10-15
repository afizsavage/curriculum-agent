"""Evidence-first LLM verifier (Sprint 4)."""

from __future__ import annotations

import json
import math
import time
from typing import Any, Optional

from app.agent.absence_evidence import (
    absence_evidence_audit,
    format_absence_prompt,
    needs_absence_verification_context,
    select_records_for_absence_verification,
    serialized_entity_ids,
)
from app.agent.answer_generator import format_evidence_for_prompt, select_evidence_for_prompt
from app.agent.trace import get_current_trace
from app.agent.verifier_diagnostics import (
    classify_schema,
    field_report,
    fingerprint,
    sanitize_exception,
)
from app.agent.coverage_validator import assess_coverage, classify_verifier_payload
from app.agent.state import CurriculumQAState
from app.agent.verification_checks import run_deterministic_checks
from app.config import Settings
from app.exceptions import LLMProviderError
from app.llm.base import LLMMessage, LLMProvider
from app.logging_utils import get_logger, log_agent_event
from app.schemas.answer import AnswerConfidence
from app.schemas.verification import (
    VERIFICATION_RESULT_JSON_SCHEMA,
    ClaimAssessment,
    ClaimVerdict,
    MissingEvidenceRequest,
    VerificationRecommendation,
    VerificationResult,
)

logger = get_logger(__name__)


def _verification_evidence(
    state: CurriculumQAState, answer: str
) -> tuple[str, list[str]]:
    """Evidence text the verifier checks, and the ids included in it.

    Absence claims see every in-scope unit and learning-outcome description.
    Ordinary questions keep the existing deduplicated window.
    """
    if needs_absence_verification_context(state.question, answer):
        ranked = select_records_for_absence_verification(
            list(state.evidence),
            grade=state.grade,
            subject=state.subject,
        )
        return format_absence_prompt(ranked), [
            item.entity_id for item in ranked if item.entity_id
        ]
    block = format_evidence_for_prompt(state.evidence, question=state.question)
    _ranked, ids = select_evidence_for_prompt(state.evidence, question=state.question)
    return block, ids

_MISSING_EVIDENCE_KEYS = ("type", "grade", "subject", "topic", "query", "detail")


def _uninterpretable_verifier_result(fields: list[str]) -> VerificationResult:
    """Conservative result for a payload whose verdict cannot be read.

    This is the existing fallback outcome. It must not be promoted to accept.
    """
    return VerificationResult(
        passed=False,
        score=0.0,
        recommendation=VerificationRecommendation.FALLBACK,
        metadata={
            "source": "llm",
            "verifier_result_invalid": True,
            "verifier_invalid_reason": "uninterpretable:" + ",".join(fields),
        },
    )


def _json_boolean(raw: dict[str, Any], key: str, problems: list[str]) -> bool:
    value = raw.get(key)
    if isinstance(value, bool):
        return value
    problems.append(key)
    return False


def _finite_score(raw: dict[str, Any], problems: list[str]) -> float:
    value = raw.get("score")
    if (
        "score" not in raw
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= float(value) <= 1
    ):
        problems.append("score")
        return 0.0
    return float(value)


def _recommendation_value(
    raw: dict[str, Any], problems: list[str]
) -> VerificationRecommendation:
    value = raw.get("recommendation")
    if isinstance(value, str):
        try:
            return VerificationRecommendation(value)
        except ValueError:
            pass
    problems.append("recommendation")
    return VerificationRecommendation.FALLBACK


def _string_list(raw: dict[str, Any], key: str, problems: list[str]) -> list[str]:
    if key not in raw:
        return []
    value = raw[key]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    problems.append(key)
    return []


def _missing_evidence_list(
    raw: dict[str, Any], problems: list[str]
) -> list[MissingEvidenceRequest | str]:
    if "missing_evidence" not in raw:
        return []
    value = raw["missing_evidence"]
    if not isinstance(value, list):
        problems.append("missing_evidence")
        return []
    parsed: list[MissingEvidenceRequest | str] = []
    for item in value:
        if isinstance(item, str):
            parsed.append(item)
            continue
        if isinstance(item, dict) and all(
            item_value is None or isinstance(item_value, str)
            for item_value in item.values()
        ):
            parsed.append(
                MissingEvidenceRequest(
                    **{key: item.get(key) for key in _MISSING_EVIDENCE_KEYS}
                )
            )
            continue
        problems.append("missing_evidence")
        return []
    return parsed


def _optional_text(raw: dict[str, Any], key: str, problems: list[str]) -> str | None:
    if key not in raw or raw[key] is None:
        return None
    value = raw[key]
    if isinstance(value, str):
        return value or None
    problems.append(key)
    return None


def _claim_assessments(raw: dict[str, Any], problems: list[str]) -> list[ClaimAssessment]:
    if "claims" not in raw:
        return []
    items = raw["claims"]
    if not isinstance(items, list):
        problems.append("claims")
        return []
    claims: list[ClaimAssessment] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("claim"):
            continue
        if not isinstance(item.get("claim"), str):
            problems.append("claims")
            return []
        try:
            verdict = ClaimVerdict(str(item.get("verdict") or "unsupported"))
        except ValueError:
            verdict = ClaimVerdict.UNSUPPORTED
        evidence_ids = item.get("evidence_ids") or []
        if not isinstance(evidence_ids, list):
            problems.append("claims")
            return []
        claims.append(
            ClaimAssessment(
                claim=item["claim"],
                verdict=verdict,
                evidence_ids=[str(entry) for entry in evidence_ids],
                notes=item["notes"] if isinstance(item.get("notes"), str) else None,
            )
        )
    return claims


VERIFIER_SYSTEM_PROMPT = """You are the MBSSE Curriculum Q&A Agent verifier.

Your role is to find reasons an answer may be unsupported or incorrect relative
ONLY to the retrieved MBSSE curriculum evidence.

Rules:
1. Compare the Generated Answer against Curriculum Evidence — NOT general knowledge.
2. Break the answer into curriculum claims where practical.
3. Mark each claim as supported, unsupported, contradicted, or missing.
4. Flag grade/subject/topic mismatches against the structured intent and evidence.
5. Prefer retrieve_more when evidence is incomplete but the question is specific.
6. Prefer clarify when the question is genuinely ambiguous (e.g. no grade) and
   multiple placements could apply.
7. Prefer fallback when evidence cannot support a reliable answer.
8. Prefer accept only when important claims are supported and there are no
   contradictions or hallucinated curriculum facts.
9. Be conservative. When unsure, do not pass.
10. Do not invent curriculum facts while verifying.
"""


class AnswerVerifier:
    """Deterministic checks first, then evidence-first LLM verification."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings

    def verify(
        self,
        state: CurriculumQAState,
        *,
        request_id: str | None = None,
    ) -> VerificationResult:
        started = time.perf_counter()
        deterministic = run_deterministic_checks(state)
        coverage = assess_coverage(state)

        # Hard deterministic failures that already prescribe clarify/fallback
        # skip the LLM when recommendation is clear and fail-closed.
        if not deterministic.passed and (
            deterministic.metadata.get("no_evidence")
            or deterministic.metadata.get("ambiguous")
            or deterministic.metadata.get("empty_answer")
        ):
            result = deterministic
        elif coverage.status == "authoritative" and coverage.result is not None:
            result = self._coverage_decision(state, deterministic, coverage.result)
        elif coverage.status == "unavailable":
            result = self._coverage_unavailable(state, deterministic, coverage.reason)
        elif self.llm.name == "stub":
            result = self._stub_verify(state, deterministic)
        else:
            result = self._merge(
                deterministic,
                self._llm_verify(state),
            )

        before = result.recommendation.value
        result = self._normalize_recommendation(state, result)
        self._record_normalization(before, result.recommendation.value)
        latency_ms = (time.perf_counter() - started) * 1000
        result.metadata = {
            **(result.metadata or {}),
            "verification_latency_ms": round(latency_ms, 2),
            "verifier_model": self.llm.model,
        }
        log_agent_event(
            logger,
            "agent.verify.completed",
            request_id=request_id,
            conversation_id=state.conversation_id,
            question=state.question,
            verification_passed=result.passed,
            verification_score=result.score,
            recommendation=result.recommendation.value,
            latency_ms=round(latency_ms, 2),
            model=self.llm.model,
        )
        return result

    def confidence_from_verification(
        self,
        result: VerificationResult,
        *,
        previous: AnswerConfidence | None = None,
    ) -> AnswerConfidence:
        if result.passed and result.score >= 0.9 and not result.issues:
            return AnswerConfidence.HIGH
        if result.passed and result.score >= 0.7:
            return AnswerConfidence.MEDIUM
        if result.recommendation == VerificationRecommendation.ACCEPT:
            return previous or AnswerConfidence.MEDIUM
        return AnswerConfidence.LOW

    def build_messages(self, state: CurriculumQAState) -> list[LLMMessage]:
        filters = {
            "intent": state.intent,
            "level": state.level,
            "grade": state.grade,
            "subject": state.subject,
            "topic": state.topic,
        }
        answer = state.final_answer or state.draft_answer or ""
        refs = [r.model_dump() for r in state.answer_evidence]
        evidence_block, selected_ids = _verification_evidence(state, answer)
        state.metadata["verification_evidence_ids"] = selected_ids
        user = (
            f"ORIGINAL QUESTION\n{state.question}\n\n"
            f"STRUCTURED INTENT\n{json.dumps({k: v for k, v in filters.items() if v}, indent=2)}\n\n"
            f"GENERATED ANSWER\n{answer}\n\n"
            f"ANSWER EVIDENCE REFS\n{json.dumps(refs, indent=2)}\n\n"
            f"CURRICULUM EVIDENCE\n{evidence_block}\n\n"
            "Verify every curriculum-specific claim against the evidence only.\n"
            "Respond with a single JSON object matching this schema:\n"
            f"{json.dumps(VERIFICATION_RESULT_JSON_SCHEMA, indent=2)}"
        )
        if needs_absence_verification_context(state.question, answer):
            prompt_ids = serialized_entity_ids(user)
            state.metadata["verification_prompt_ids"] = prompt_ids
            state.metadata["absence_evidence"] = absence_evidence_audit(
                state, selected_ids=selected_ids, prompt_ids=prompt_ids
            )
        return [
            LLMMessage(
                role="system",
                content=VERIFIER_SYSTEM_PROMPT
                + self._experimental_prompt_suffix(state),
            ),
            LLMMessage(role="user", content=user),
        ]

    def _experimental_prompt_suffix(self, state: CurriculumQAState) -> str:
        from app.agent.v26_experiment import get_v26_system_prompt_suffix

        suffix = get_v26_system_prompt_suffix(state)
        return f"\n\n{suffix}" if suffix else ""

    def _llm_verify(self, state: CurriculumQAState) -> VerificationResult:
        messages = self.build_messages(state)
        self._record_verifier_input(state, messages)
        try:
            raw = self.llm.generate_structured(
                messages, schema=VERIFICATION_RESULT_JSON_SCHEMA, temperature=0.0
            )
        except LLMProviderError as exc:
            self._record_verifier_failure(exc)
            # Prefer deterministic verdict over failing the whole turn on bad JSON.
            if "invalid json" in str(exc).lower() or "empty structured" in str(exc).lower():
                return run_deterministic_checks(state)
            raise
        except Exception as exc:
            self._record_verifier_failure(exc)
            raise LLMProviderError(f"Verification failed: {exc}") from exc
        self._record_parsed_payload(raw)
        result = self._parse(raw)
        shape = classify_verifier_payload(raw)
        result.metadata = {
            **(result.metadata or {}),
            "verifier_payload_shape": shape,
        }
        if shape == "schema_envelope":
            result.metadata["verifier_result_invalid"] = True
            result.metadata["verifier_invalid_reason"] = "schema_envelope"
        return result

    def _coverage_diagnostics(
        self,
        state: CurriculumQAState,
        deterministic: VerificationResult,
    ) -> dict[str, Any]:
        """Record the generic verifier without letting it decide coverage."""
        try:
            if self.llm.name == "stub":
                diagnosed = self._stub_verify(state, deterministic)
            else:
                diagnosed = self._llm_verify(state)
        except LLMProviderError as exc:
            return {
                "verifier_result_invalid": True,
                "verifier_invalid_reason": "llm_error",
                "error": str(exc),
            }
        return {
            "passed": diagnosed.passed,
            "score": diagnosed.score,
            "recommendation": diagnosed.recommendation.value,
            "issues": list(diagnosed.issues),
            "verifier_payload_shape": (diagnosed.metadata or {}).get(
                "verifier_payload_shape"
            ),
            "verifier_result_invalid": bool(
                (diagnosed.metadata or {}).get("verifier_result_invalid")
            ),
            "verifier_invalid_reason": (diagnosed.metadata or {}).get(
                "verifier_invalid_reason"
            ),
            "authoritative": False,
        }

    def _coverage_decision(
        self,
        state: CurriculumQAState,
        deterministic: VerificationResult,
        structural,
    ) -> VerificationResult:
        diagnostics = self._coverage_diagnostics(state, deterministic)
        codes = [error.code for error in structural.errors]
        if structural.passed:
            return VerificationResult(
                passed=True,
                score=1.0,
                issues=[],
                recommendation=VerificationRecommendation.ACCEPT,
                notes="Syllabus coverage structure matches the source records.",
                metadata={
                    "source": "coverage_validator",
                    "coverage_decision": "accept",
                    "coverage_reason_codes": codes,
                    "coverage_latency_ms": round(structural.latency_ms, 3),
                    "llm_verification": diagnostics,
                    "llm_authoritative": False,
                },
            )
        messages = [error.message for error in structural.errors]
        return VerificationResult(
            passed=False,
            score=0.0,
            issues=messages,
            incorrect_claims=messages,
            recommendation=VerificationRecommendation.FALLBACK,
            notes="Syllabus coverage structure does not match the source records.",
            metadata={
                "source": "coverage_validator",
                "coverage_decision": "reject",
                "coverage_reason_codes": codes,
                "coverage_errors": [
                    {
                        "code": error.code,
                        "entity_id": error.entity_id,
                        "message": error.message,
                    }
                    for error in structural.errors
                ],
                "coverage_latency_ms": round(structural.latency_ms, 3),
                "llm_verification": diagnostics,
                "llm_authoritative": False,
            },
        )

    def _coverage_unavailable(
        self,
        state: CurriculumQAState,
        deterministic: VerificationResult,
        reason: str | None,
    ) -> VerificationResult:
        diagnostics = self._coverage_diagnostics(state, deterministic)
        detail = reason or "coverage_validation_unavailable"
        return VerificationResult(
            passed=False,
            score=0.0,
            issues=[f"Syllabus coverage could not be validated ({detail})."],
            recommendation=VerificationRecommendation.FALLBACK,
            notes="Structured coverage validation did not run.",
            metadata={
                "source": "coverage_validator",
                "coverage_decision": "unavailable",
                "coverage_reason_codes": [detail],
                "llm_verification": diagnostics,
                "llm_authoritative": False,
            },
        )

    def _stub_verify(
        self,
        state: CurriculumQAState,
        deterministic: VerificationResult,
    ) -> VerificationResult:
        """Conservative stub: pass when evidence and refs align; else fail closed."""
        if not deterministic.passed and deterministic.recommendation in {
            VerificationRecommendation.CLARIFY,
            VerificationRecommendation.RETRIEVE_MORE,
            VerificationRecommendation.FALLBACK,
        }:
            # Keep deterministic hard fails; soft_pass may still accept.
            if deterministic.metadata.get("soft_pass"):
                return deterministic
            if deterministic.incorrect_claims or deterministic.metadata.get(
                "no_evidence"
            ):
                return deterministic
            if deterministic.metadata.get("ambiguous"):
                return deterministic
            if deterministic.metadata.get("sss_stream_resolution") in {
                "subject_not_in_stream",
                "ambiguous_subject",
                "grade_content_missing",
            }:
                return deterministic

        if not state.evidence:
            return VerificationResult(
                passed=False,
                score=0.2,
                issues=["No evidence available for verification."],
                recommendation=VerificationRecommendation.RETRIEVE_MORE,
                missing_evidence=[
                    MissingEvidenceRequest(
                        type="curriculum_content",
                        grade=state.grade,
                        subject=state.subject,
                        topic=state.topic,
                        query=state.topic or state.question[:80],
                    )
                ],
                metadata={"source": "stub"},
            )

        if deterministic.incorrect_claims:
            return deterministic

        # Supported path for stub grounded answers with evidence.
        claims = list(deterministic.claims)
        if not claims and state.answer_evidence:
            claims = [
                ClaimAssessment(
                    claim=ref.claim,
                    verdict=ClaimVerdict.SUPPORTED,
                    evidence_ids=[ref.entity_id],
                )
                for ref in state.answer_evidence
            ]
        return VerificationResult(
            passed=True,
            score=0.92 if claims else 0.8,
            issues=[],
            unsupported_claims=[],
            incorrect_claims=[],
            missing_evidence=[],
            claims=claims,
            recommendation=VerificationRecommendation.ACCEPT,
            notes="Stub verifier accepted grounded evidence-backed answer.",
            metadata={"source": "stub"},
        )

    def _parse(self, raw: dict[str, Any]) -> VerificationResult:
        if not isinstance(raw, dict):
            raise LLMProviderError("Verifier returned non-object JSON")
        problems: list[str] = []
        passed = _json_boolean(raw, "passed", problems)
        score = _finite_score(raw, problems)
        recommendation = _recommendation_value(raw, problems)
        issues = _string_list(raw, "issues", problems)
        unsupported = _string_list(raw, "unsupported_claims", problems)
        incorrect = _string_list(raw, "incorrect_claims", problems)
        missing = _missing_evidence_list(raw, problems)
        claims = _claim_assessments(raw, problems)
        clarification = _optional_text(raw, "clarification", problems)
        notes = _optional_text(raw, "notes", problems)
        if problems:
            # Required verdict fields that cannot be read must not fall through
            # to defaults that normalization can promote to accept.
            return _uninterpretable_verifier_result(problems)
        return VerificationResult(
            passed=passed,
            score=score,
            issues=issues,
            unsupported_claims=unsupported,
            incorrect_claims=incorrect,
            missing_evidence=missing,
            claims=claims,
            recommendation=recommendation,
            clarification=clarification,
            notes=notes,
            metadata={"source": "llm"},
        )

    def _payload_event(self) -> dict[str, Any] | None:
        trace = get_current_trace()
        if trace is None:
            return None
        event = trace.latest_event("agent.verification.payload")
        if event is None:
            event = trace.record_trace_only("agent.verification.payload")
        return event

    def _record_verifier_input(
        self, state: CurriculumQAState, messages: list[LLMMessage]
    ) -> None:
        event = self._payload_event()
        if event is None:
            return
        prompt = "\n\n".join(message.content for message in messages)
        event.update(
            {
                "prompt_sha256": fingerprint(prompt),
                "prompt_length": len(prompt),
                "evidence_ids": list(
                    (state.metadata or {}).get("verification_evidence_ids") or []
                ),
                "provider": self.llm.name,
                "model": self.llm.model,
                "temperature": 0.0,
            }
        )

    def _record_parsed_payload(self, raw: Any) -> None:
        event = self._payload_event()
        if event is None:
            return
        trace = get_current_trace()
        stash = (trace.verifier_model_stash or {}) if trace is not None else {}
        report = field_report(raw if isinstance(raw, dict) else None)
        event.update(stash)
        event.update(classify_schema(raw))
        event.update(
            {
                "fields": report,
                "parsed_recommendation": (
                    raw.get("recommendation") if isinstance(raw, dict) else None
                ),
                "parsed_passed": raw.get("passed") if isinstance(raw, dict) else None,
                "parsed_score": raw.get("score") if isinstance(raw, dict) else None,
                "parsed_notes": raw.get("notes") if isinstance(raw, dict) else None,
                "parsed_issues": raw.get("issues") if isinstance(raw, dict) else None,
                "parsed_unsupported_claims": (
                    raw.get("unsupported_claims") if isinstance(raw, dict) else None
                ),
                "parsed_incorrect_claims": (
                    raw.get("incorrect_claims") if isinstance(raw, dict) else None
                ),
                "parsed_missing_evidence": (
                    raw.get("missing_evidence") if isinstance(raw, dict) else None
                ),
            }
        )

    def _record_verifier_failure(self, exc: BaseException) -> None:
        event = self._payload_event()
        if event is None:
            return
        trace = get_current_trace()
        stash = (trace.verifier_model_stash or {}) if trace is not None else {}
        event.update(stash)
        event["exception"] = sanitize_exception(exc)
        if "json_extracted" not in event:
            event["json_extracted"] = False
        event["schema_valid"] = False
        event["safely_interpretable"] = False
        self._log_payload_summary(event)

    def _record_normalization(self, before: str, after: str) -> None:
        event = self._payload_event()
        if event is None:
            return
        event["recommendation_before_normalize"] = before
        event["recommendation_after_normalize"] = after
        self._log_payload_summary(event)

    def _log_payload_summary(self, event: dict[str, Any]) -> None:
        log_agent_event(
            logger,
            "agent.verification.payload",
            request_id=None,
            raw_response_present=event.get("raw_response_present"),
            raw_response_length=event.get("raw_response_length"),
            raw_response_fingerprint=event.get("raw_response_fingerprint"),
            json_extracted=event.get("json_extracted"),
            schema_valid=event.get("schema_valid"),
            parsed_recommendation=event.get("parsed_recommendation"),
            parsed_passed=event.get("parsed_passed"),
            recommendation_before_normalize=event.get(
                "recommendation_before_normalize"
            ),
            recommendation_after_normalize=event.get("recommendation_after_normalize"),
            exception_category=(event.get("exception") or {}).get("category")
            if isinstance(event.get("exception"), dict)
            else None,
        )

    def _merge(
        self,
        deterministic: VerificationResult,
        llm_result: VerificationResult,
    ) -> VerificationResult:
        """Fail closed: deterministic hard fails override LLM accept."""
        if (
            deterministic.incorrect_claims
            or deterministic.metadata.get("no_evidence")
            or deterministic.metadata.get("sss_stream_resolution")
            in {"subject_not_in_stream", "ambiguous_subject", "grade_content_missing"}
        ):
            # Prefer deterministic failure signals; enrich with LLM issues.
            issues = list(
                dict.fromkeys(deterministic.issues + llm_result.issues)
            )
            unsupported = list(
                dict.fromkeys(
                    deterministic.unsupported_claims + llm_result.unsupported_claims
                )
            )
            incorrect = list(
                dict.fromkeys(
                    deterministic.incorrect_claims + llm_result.incorrect_claims
                )
            )
            missing = list(deterministic.missing_evidence) + list(
                llm_result.missing_evidence
            )
            return VerificationResult(
                passed=False,
                score=min(deterministic.score, llm_result.score),
                issues=issues,
                unsupported_claims=unsupported,
                incorrect_claims=incorrect,
                missing_evidence=missing,
                claims=llm_result.claims or deterministic.claims,
                recommendation=(
                    deterministic.recommendation
                    if deterministic.recommendation
                    != VerificationRecommendation.ACCEPT
                    else llm_result.recommendation
                ),
                clarification=deterministic.clarification or llm_result.clarification,
                notes=llm_result.notes or deterministic.notes,
                metadata={
                    "source": "merged",
                    "deterministic": deterministic.metadata,
                    "llm": llm_result.metadata,
                },
            )

        # If LLM accepts but deterministic found unsupported entity refs, fail.
        if llm_result.passed and deterministic.unsupported_claims:
            return VerificationResult(
                passed=False,
                score=min(0.6, llm_result.score),
                issues=list(dict.fromkeys(deterministic.issues + llm_result.issues)),
                unsupported_claims=deterministic.unsupported_claims,
                incorrect_claims=deterministic.incorrect_claims,
                missing_evidence=deterministic.missing_evidence
                or llm_result.missing_evidence,
                claims=llm_result.claims or deterministic.claims,
                recommendation=VerificationRecommendation.RETRIEVE_MORE,
                metadata={"source": "merged", "overrode_llm_accept": True},
            )

        return llm_result

    def _normalize_recommendation(
        self,
        state: CurriculumQAState,
        result: VerificationResult,
    ) -> VerificationResult:
        from app.curriculum.structure_scope import drop_resolved_absent_code_requests

        kept, resolved = drop_resolved_absent_code_requests(
            evidence=list(state.evidence),
            grade=state.grade,
            subject=state.subject,
            curriculum_version=(state.metadata or {}).get("curriculum_version"),
            answer=state.final_answer or state.draft_answer or "",
            missing=list(result.missing_evidence),
            unsupported_claims=list(result.unsupported_claims or []),
            incorrect_claims=list(result.incorrect_claims or []),
        )
        if resolved:
            metadata = dict(result.metadata or {})
            metadata["resolved_absent_codes"] = resolved
            result = result.model_copy(
                update={"missing_evidence": kept, "metadata": metadata}
            )
        if (result.metadata or {}).get("verifier_result_invalid"):
            return result.model_copy(
                update={
                    "passed": False,
                    "recommendation": VerificationRecommendation.FALLBACK,
                }
            )
        if result.missing_evidence and (
            result.passed
            or result.recommendation == VerificationRecommendation.ACCEPT
        ):
            # A passed flag must not close an answer that still names a gap.
            return result.model_copy(
                update={
                    "passed": False,
                    "recommendation": VerificationRecommendation.RETRIEVE_MORE,
                }
            )
        if result.passed:
            return result.model_copy(
                update={"recommendation": VerificationRecommendation.ACCEPT}
            )
        if result.recommendation == VerificationRecommendation.ACCEPT and not result.passed:
            return result.model_copy(
                update={"recommendation": VerificationRecommendation.RETRIEVE_MORE}
            )
        return result
