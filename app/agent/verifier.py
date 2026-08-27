"""Evidence-first LLM verifier (Sprint 4)."""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from app.agent.answer_generator import format_evidence_for_prompt
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

        # Hard deterministic failures that already prescribe clarify/fallback
        # skip the LLM when recommendation is clear and fail-closed.
        if not deterministic.passed and (
            deterministic.metadata.get("no_evidence")
            or deterministic.metadata.get("ambiguous")
            or deterministic.metadata.get("empty_answer")
        ):
            result = deterministic
        elif self.llm.name == "stub":
            result = self._stub_verify(state, deterministic)
        else:
            result = self._merge(
                deterministic,
                self._llm_verify(state),
            )

        result = self._normalize_recommendation(state, result)
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
        user = (
            f"ORIGINAL QUESTION\n{state.question}\n\n"
            f"STRUCTURED INTENT\n{json.dumps({k: v for k, v in filters.items() if v}, indent=2)}\n\n"
            f"GENERATED ANSWER\n{answer}\n\n"
            f"ANSWER EVIDENCE REFS\n{json.dumps(refs, indent=2)}\n\n"
            f"CURRICULUM EVIDENCE\n{format_evidence_for_prompt(state.evidence, question=state.question)}\n\n"
            "Verify every curriculum-specific claim against the evidence only.\n"
            "Respond with a single JSON object matching this schema:\n"
            f"{json.dumps(VERIFICATION_RESULT_JSON_SCHEMA, indent=2)}"
        )
        return [
            LLMMessage(role="system", content=VERIFIER_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user),
        ]

    def _llm_verify(self, state: CurriculumQAState) -> VerificationResult:
        messages = self.build_messages(state)
        try:
            raw = self.llm.generate_structured(
                messages, schema=VERIFICATION_RESULT_JSON_SCHEMA, temperature=0.0
            )
        except LLMProviderError as exc:
            # Prefer deterministic verdict over failing the whole turn on bad JSON.
            if "invalid json" in str(exc).lower() or "empty structured" in str(exc).lower():
                return run_deterministic_checks(state)
            raise
        except Exception as exc:
            raise LLMProviderError(f"Verification failed: {exc}") from exc
        return self._parse(raw)

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
        recommendation_raw = str(raw.get("recommendation") or "fallback").lower()
        try:
            recommendation = VerificationRecommendation(recommendation_raw)
        except ValueError:
            recommendation = VerificationRecommendation.FALLBACK

        claims: list[ClaimAssessment] = []
        for item in raw.get("claims") or []:
            if not isinstance(item, dict) or not item.get("claim"):
                continue
            try:
                verdict = ClaimVerdict(str(item.get("verdict") or "unsupported"))
            except ValueError:
                verdict = ClaimVerdict.UNSUPPORTED
            claims.append(
                ClaimAssessment(
                    claim=str(item["claim"]),
                    verdict=verdict,
                    evidence_ids=[str(x) for x in (item.get("evidence_ids") or [])],
                    notes=str(item["notes"]) if item.get("notes") else None,
                )
            )

        missing: list[MissingEvidenceRequest | str] = []
        for item in raw.get("missing_evidence") or []:
            if isinstance(item, str):
                missing.append(item)
            elif isinstance(item, dict):
                missing.append(MissingEvidenceRequest(**{
                    k: item.get(k)
                    for k in (
                        "type",
                        "grade",
                        "subject",
                        "topic",
                        "query",
                        "detail",
                    )
                }))

        return VerificationResult(
            passed=bool(raw.get("passed")),
            score=float(raw.get("score") or 0.0),
            issues=[str(x) for x in (raw.get("issues") or []) if x],
            unsupported_claims=[
                str(x) for x in (raw.get("unsupported_claims") or []) if x
            ],
            incorrect_claims=[
                str(x) for x in (raw.get("incorrect_claims") or []) if x
            ],
            missing_evidence=missing,
            claims=claims,
            recommendation=recommendation,
            clarification=(
                str(raw["clarification"]) if raw.get("clarification") else None
            ),
            notes=str(raw["notes"]) if raw.get("notes") else None,
            metadata={"source": "llm"},
        )

    def _merge(
        self,
        deterministic: VerificationResult,
        llm_result: VerificationResult,
    ) -> VerificationResult:
        """Fail closed: deterministic hard fails override LLM accept."""
        if deterministic.incorrect_claims or deterministic.metadata.get("no_evidence"):
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
        if result.passed:
            return result.model_copy(
                update={"recommendation": VerificationRecommendation.ACCEPT}
            )
        if result.recommendation == VerificationRecommendation.ACCEPT and not result.passed:
            return result.model_copy(
                update={"recommendation": VerificationRecommendation.RETRIEVE_MORE}
            )
        return result
