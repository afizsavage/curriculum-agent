"""Grounded curriculum answer generation (Sprint 3)."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from app.agent.context import ConversationContext
from app.agent.state import CurriculumQAState
from app.curriculum.codes import normalize_grade_code
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.exceptions import LLMProviderError
from app.llm.base import LLMMessage, LLMProvider
from app.logging_utils import get_logger, log_agent_event
from app.schemas.answer import (
    GROUNDED_ANSWER_JSON_SCHEMA,
    AnswerConfidence,
    AnswerEvidenceRef,
    GroundedAnswer,
    shadow_claim_generation_schema,
)

from app.agent.generation_policy import (
    EVIDENCE_CONSERVATIVE_RULES,
    EVIDENCE_CONSERVATIVE_USER_APPENDIX,
    GENERATION_POLICY,
    analyze_answer_quality,
    question_requests_identifiers,
    redact_internal_identifiers,
)

logger = get_logger(__name__)

SYSTEM_PROMPT = f"""You are the MBSSE Curriculum Q&A Agent answer generator.

Your role is to produce grounded answers for questions about the MBSSE curriculum
using ONLY the retrieved curriculum evidence provided in the user message.

Core rules:
1. CURRICULUM AUTHORITY: Retrieved MBSSE curriculum evidence is authoritative for
   curriculum-specific claims (topics, learning objectives, grade placement,
   subject structure, progression).
2. NO UNSUPPORTED CLAIMS: Do not invent topics, objectives, grades, subjects,
   strands, or curriculum terminology not supported by the evidence.
3. EVIDENCE LIMITATIONS: If evidence is insufficient, state that clearly. Do not
   fill gaps from general knowledge for MBSSE-specific facts.
4. EVIDENCE REFERENCES: Return a refs array of entity_id values copied from
   the supplied evidence. Include only records this answer actually used,
   including a record used only to support a curriculum evidence note.
   Never invent entity IDs. Do not put those IDs in the answer prose.
5. CONFIDENCE: Assign high only when exact topic/objective evidence answers the
   question; medium when interpretation is needed; low when evidence is partial.
6. STYLE: Write for pupils, teachers, and education users. Synthesize the
   evidence into concise natural language. Do not expose internal curriculum
   identifiers unless the question asks for them.

{EVIDENCE_CONSERVATIVE_RULES}
"""

# Backward compatibility for V2.3 diagnostic experiment arm B.
CONSTRAINED_GENERATION_APPENDIX = EVIDENCE_CONSERVATIVE_USER_APPENDIX

# Shadow-only. Production generation does not append this, and it does not
# make `claims` part of the production contract.
CLAIM_SHADOW_APPENDIX = """
SHADOW CLAIM ATTRIBUTION (diagnostic only; do not change the answer to fit it)
Also return a claims array. This does not replace the natural-language answer.
- text must be copied verbatim from answer. Do not paraphrase.
- Preserve punctuation, wording, singular and plural forms, and Markdown wording.
- A leading bullet marker in answer does not need to be repeated in text.
- Include every substantive curriculum bullet.
- Include a curriculum evidence note when the answer has one.
- Do not emit titles, introductions, or "Pupils learn to:" as claims.
- One claim may cite multiple supplied records when the sentence uses more than one.
- Do not cite a record the claim did not use.
- refs must be entity IDs copied from the supplied evidence. Do not invent IDs.
- Do not rewrite the answer so that it quotes internal evidence wording.
"""


class AnswerGenerator:
    """Builds grounded prompts and produces structured answers from evidence."""

    INSUFFICIENT_EVIDENCE_ANSWER = (
        "I couldn't find sufficient MBSSE curriculum evidence in the "
        "available curriculum data to answer this reliably."
    )

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def generate(
        self,
        state: CurriculumQAState,
        *,
        conversation: ConversationContext | None = None,
        request_id: str | None = None,
    ) -> GroundedAnswer:
        started = time.perf_counter()
        log_agent_event(
            logger,
            "answer_generation_started",
            request_id=request_id,
            conversation_id=state.conversation_id,
            question=state.question,
            model=self.llm.model,
            input_evidence_count=len(state.evidence),
            evidence_status=state.evidence_status.value,
        )

        if not state.evidence or state.evidence_status == EvidenceStatus.NOT_FOUND:
            result = self._insufficient_evidence_answer(state)
        elif self.llm.name == "stub":
            result = self._stub_generate(state)
        else:
            result = self._llm_generate(state, conversation=conversation)

        result = self._apply_evidence_constraints(state, result)
        if not question_requests_identifiers(state.question):
            cleaned = redact_internal_identifiers(result.answer, state.evidence)
            if cleaned != (result.answer or "").strip():
                result = result.model_copy(
                    update={
                        "answer": cleaned,
                        "summary": _one_line_summary(cleaned) or result.summary,
                    }
                )
        if state.metadata.get("live_attribution") is not None and self.llm.name != "stub":
            returned = state.metadata["live_attribution"].get("returned_refs") or []
            state.metadata["live_attribution"] = attribute_live_model_refs(
                returned,
                state.evidence,
                answer=result.answer or "",
            ).as_dict()
        if state.metadata.get("claim_attribution") is not None and self.llm.name != "stub":
            returned_claims = state.metadata["claim_attribution"].get("returned_claims")
            if returned_claims is not None:
                claim_attribution = attribute_claim_mappings(
                    returned_claims,
                    state.evidence,
                    answer=result.answer or "",
                )
                state.metadata["claim_attribution"] = claim_attribution.as_dict()
                result = result.model_copy(update={"evidence": claim_attribution.refs})
        latency_ms = (time.perf_counter() - started) * 1000
        quality = analyze_answer_quality(
            result.answer or "",
            limitations=result.limitations,
            evidence=state.evidence,
        )
        state.metadata["generation_policy"] = GENERATION_POLICY
        state.metadata.update(quality)
        state.metadata["generation_mode"] = state.metadata.get("generation_mode", "current")
        state.metadata["generation_latency_ms"] = round(latency_ms, 2)

        from app.agent.trace import get_current_trace

        trace = get_current_trace()
        if trace is not None:
            trace.emit(
                "agent.generation.diagnostics",
                iteration=state.iteration,
                generation_policy=GENERATION_POLICY,
                generation_mode=state.metadata.get("generation_mode"),
                evidence_snapshot_hash=state.metadata.get("evidence_snapshot_hash"),
                generation_evidence_count=state.metadata.get("generation_evidence_count"),
                generation_evidence_ids=state.metadata.get("generation_evidence_ids"),
                answer_length=len(result.answer or ""),
                generation_confidence=result.confidence.value,
                generation_latency_ms=round(latency_ms, 2),
                unsupported_claim_count=quality.get("unsupported_claim_count"),
                speculative_claim_count=quality.get("speculative_claim_count"),
                truncation_warning_count=quality.get("truncation_warning_count"),
                absence_claim_count=quality.get("absence_claim_count"),
            )

        log_agent_event(
            logger,
            "answer_generation_completed",
            request_id=request_id,
            conversation_id=state.conversation_id,
            question=state.question,
            model=self.llm.model,
            input_evidence_count=len(state.evidence),
            output_evidence_references=len(result.evidence),
            confidence=result.confidence.value,
            latency_ms=round(latency_ms, 2),
            token_usage=(getattr(self.llm, "last_token_usage", None)),
        )
        return result

    def build_messages(
        self,
        state: CurriculumQAState,
        *,
        conversation: ConversationContext | None = None,
    ) -> list[LLMMessage]:
        history = self._format_conversation_history(conversation)
        filters = self._format_filters(state)
        ranked, generation_ids = select_evidence_for_prompt(
            state.evidence, question=state.question
        )
        evidence_block = format_evidence_for_prompt(
            state.evidence, question=state.question
        )
        state.metadata["retrieved_evidence_count"] = len(state.evidence)
        state.metadata["generation_evidence_count"] = len(ranked)
        state.metadata["generation_evidence_ids"] = generation_ids
        from app.agent.trace import get_current_trace

        trace = get_current_trace()
        if trace is not None:
            trace.emit(
                "agent.evidence.rank",
                iteration=state.iteration,
                retrieved_evidence_count=len(state.evidence),
                generation_evidence_count=len(ranked),
                generation_evidence_ids=generation_ids,
            )

        user_content = (
            f"USER QUESTION\n{state.question}\n\n"
            f"STRUCTURED INTENT\n{filters}\n\n"
        )
        if history:
            user_content += f"CONVERSATION CONTEXT\n{history}\n\n"
        user_content += (
            f"CURRICULUM EVIDENCE\n{evidence_block}\n\n"
            "The evidence block above includes entity IDs and codes for audit. "
            "Those identifiers stay available for the evidence array.\n\n"
            "INSTRUCTIONS\n"
            "Answer the question using ONLY the curriculum evidence above.\n"
            "Do not invent curriculum information.\n"
            "Return refs: the entity_id values of supplied records this answer "
            "actually used. Do not invent IDs and do not include unused records.\n"
            "Set limitations when evidence is partial, ambiguous, or source text is damaged.\n"
        )
        if question_requests_identifiers(state.question):
            user_content += (
                "The user explicitly asked for a curriculum identifier. Include the "
                "requested code or id from the evidence in the answer, and do not "
                "invent one.\n"
            )
        else:
            user_content += (
                "USER-FACING ANSWER\n"
                "Write a concise natural-language synthesis for pupils and teachers.\n"
                "When several learning areas are present, use this shape:\n"
                "# Grade Subject — Topic\n"
                "A short introduction that names only those areas.\n"
                "### 1. Concept name\n"
                "Pupils learn to:\n"
                "* one grounded learning expectation\n"
                "### Curriculum Evidence Note\n"
                "Include the evidence note only when source text is incomplete, "
                "duplicated, or garbled. Omit it when the evidence is intact.\n"
                "A question with only one or two relevant records stays short, "
                "without a long multi-section document.\n"
                "Do not copy learning-objective codes, unit codes, entity IDs, "
                "database IDs, retrieval IDs, or grade_curriculum_id into the answer.\n"
                "AUDIT\n"
                "Return a refs array containing only entity_id values from the "
                "supplied evidence that this answer actually used, including a "
                "record used only to support a curriculum evidence note. "
                "Do not invent IDs. Do not include retrieved records the answer "
                "does not use. Keep those IDs out of the answer prose.\n"
            )
        user_content += f"{EVIDENCE_CONSERVATIVE_USER_APPENDIX}\n"
        if state.metadata.get("conservative_regeneration"):
            user_content += (
                "\nCONSERVATIVE REGENERATION (authoritative context already resolved)\n"
                "- Regenerate using only the evidence above; remove unsupported claims.\n"
                "- Do not complete truncated or garbled text.\n"
                "- Keep the answer a user-facing synthesis unless the user asked "
                "for an identifier. Keep identifiers in the evidence array.\n"
            )
        if state.metadata.get("generation_mode") == "constrained":
            user_content += CONSTRAINED_GENERATION_APPENDIX
        schema = (
            shadow_claim_generation_schema()
            if state.metadata.get("claim_shadow")
            else GROUNDED_ANSWER_JSON_SCHEMA
        )
        if state.metadata.get("claim_shadow"):
            user_content += CLAIM_SHADOW_APPENDIX
        user_content += (
            "\nJSON OUTPUT\n"
            "Respond with a single JSON object (no markdown code fences) matching this schema:\n"
            f"{json.dumps(schema, indent=2)}"
        )
        return [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_content),
        ]

    def _llm_generate(
        self,
        state: CurriculumQAState,
        *,
        conversation: ConversationContext | None,
    ) -> GroundedAnswer:
        messages = self.build_messages(state, conversation=conversation)
        schema = (
            shadow_claim_generation_schema()
            if state.metadata.get("claim_shadow")
            else GROUNDED_ANSWER_JSON_SCHEMA
        )
        try:
            raw = self.llm.generate_structured(
                messages, schema=schema, temperature=0.0
            )
            return self._parse_structured(raw, state.evidence, state=state)
        except LLMProviderError as first_exc:
            detail = str(first_exc).lower()
            # Compact retry: large evidence / empty answers often need a second pass.
            if not any(
                token in detail
                for token in (
                    "invalid json",
                    "empty structured",
                    "empty answer",
                )
            ):
                raise
            compact = list(messages)
            compact.append(
                LLMMessage(
                    role="user",
                    content=(
                        "Your previous response was unusable (invalid JSON or empty "
                        "`answer`). Reply with ONE compact JSON object only — no "
                        "markdown fences, no prose. The `answer` field MUST be a "
                        "non-empty markdown string grounded in the evidence. Keep "
                        "`answer` under 1200 characters. Include refs for every "
                        "supplied entity_id the answer uses, and short limitations."
                    ),
                )
            )
            try:
                raw = self.llm.generate_structured(
                    compact, schema=schema, temperature=0.0
                )
                return self._parse_structured(raw, state.evidence, state=state)
            except LLMProviderError:
                # Last resort: deterministic grounded summary from evidence.
                return self._stub_generate(state)
        except Exception as exc:
            raise LLMProviderError(f"Answer generation failed: {exc}") from exc

    def _stub_generate(self, state: CurriculumQAState) -> GroundedAnswer:
        """Deterministic grounded synthesis for tests without a model call."""
        evidence = state.evidence
        grade_label = _display_grade(state.grade or _first_attr(evidence, "grade"))
        subject_label = _display_subject(
            state.subject or _first_attr(evidence, "subject")
        )
        answer_text, limitations, used = _render_stub_answer(
            state,
            grade_label=grade_label,
            subject_label=subject_label,
        )
        if state.evidence_status == EvidenceStatus.PARTIAL:
            limitations.append(
                "Some curriculum API calls failed or returned partial results."
            )
        limitations = list(dict.fromkeys(limitations))

        refs = _evidence_refs_from_items(
            _records_in_evidence_order(evidence, used),
            state.question,
        )
        confidence = (
            AnswerConfidence.HIGH
            if state.evidence_status == EvidenceStatus.FOUND and refs
            else AnswerConfidence.MEDIUM
        )
        answer_text = answer_text.strip() or self.INSUFFICIENT_EVIDENCE_ANSWER
        return GroundedAnswer(
            answer=answer_text,
            summary=_one_line_summary(answer_text),
            evidence=refs,
            limitations=limitations,
            confidence=confidence,
        )

    def _insufficient_evidence_answer(
        self, state: CurriculumQAState
    ) -> GroundedAnswer:
        limitations = [
            "No relevant curriculum evidence was retrieved from the MBSSE Curriculum API."
        ]
        if state.evidence_status == EvidenceStatus.ERROR:
            limitations.append("Curriculum retrieval encountered errors.")
        return GroundedAnswer(
            answer=self.INSUFFICIENT_EVIDENCE_ANSWER,
            summary=None,
            evidence=[],
            limitations=limitations,
            confidence=AnswerConfidence.LOW,
        )

    def _parse_structured(
        self,
        raw: dict[str, Any],
        evidence: list[CurriculumEvidence],
        *,
        state: CurriculumQAState | None = None,
    ) -> GroundedAnswer:
        if not isinstance(raw, dict):
            raise LLMProviderError("LLM returned non-object structured answer")

        confidence_raw = str(raw.get("confidence", "medium")).lower()
        try:
            confidence = AnswerConfidence(confidence_raw)
        except ValueError:
            confidence = AnswerConfidence.MEDIUM

        answer = _extract_answer_text(raw)
        if not answer:
            if evidence and state is not None:
                fallback = self._stub_generate(state)
                limitations = list(
                    dict.fromkeys(
                        fallback.limitations
                        + ["Model returned an empty answer; used evidence summary."]
                    )
                )
                return fallback.model_copy(update={"limitations": limitations})
            raise LLMProviderError("LLM returned empty answer")

        raw_refs = _raw_attribution_refs(raw)
        attribution = attribute_live_model_refs(raw_refs, evidence, answer=answer)
        refs = attribution.refs
        claim_attribution = None
        if (
            state is not None
            and state.metadata.get("claim_shadow")
            and "claims" in raw
            and raw.get("claims") is not None
        ):
            claim_attribution = attribute_claim_mappings(
                raw.get("claims") or [],
                evidence,
                answer=answer,
            )
            refs = claim_attribution.refs
        if state is not None:
            state.metadata["live_attribution"] = attribution.as_dict()
            if claim_attribution is not None:
                state.metadata["claim_attribution"] = claim_attribution.as_dict()

        limitations = [str(x) for x in (raw.get("limitations") or []) if x]
        if (
            claim_attribution is not None
            and claim_attribution.unattributed_claims
            and claim_attribution.valid_claims
        ):
            limitations.append("Claim-level attribution is incomplete.")
            if confidence == AnswerConfidence.HIGH:
                confidence = AnswerConfidence.MEDIUM
        summary = raw.get("summary")
        return GroundedAnswer(
            answer=answer,
            summary=str(summary) if summary else None,
            evidence=refs,
            limitations=limitations,
            confidence=confidence,
        )

    def _validate_evidence_refs(
        self,
        refs: list[Any],
        evidence: list[CurriculumEvidence],
    ) -> list[AnswerEvidenceRef]:
        return attribute_live_model_refs(refs, evidence, answer="").refs

    def _apply_evidence_constraints(
        self,
        state: CurriculumQAState,
        answer: GroundedAnswer,
    ) -> GroundedAnswer:
        """Post-process confidence and limitations based on evidence quality."""
        limitations = list(answer.limitations)
        confidence = answer.confidence

        if not state.evidence:
            return answer

        question_grade = normalize_grade_code(state.grade or state.question)
        evidence_grades = {
            g
            for g in (
                normalize_grade_code(e.grade) for e in state.evidence if e.grade
            )
            if g
        }

        if question_grade and evidence_grades and question_grade not in evidence_grades:
            limitations.append(
                f"Retrieved evidence is for grade(s) {', '.join(sorted(evidence_grades))}, "
                f"not {question_grade} as asked."
            )
            confidence = AnswerConfidence.LOW

        if state.evidence_status == EvidenceStatus.PARTIAL and confidence == AnswerConfidence.HIGH:
            confidence = AnswerConfidence.MEDIUM

        if not answer.evidence and state.evidence and _answer_has_substantive_claim(answer.answer):
            confidence = AnswerConfidence.LOW
            limitations.append(
                "The answer was not linked to the specific curriculum records "
                "used to write it, so attribution is incomplete."
            )

        return answer.model_copy(
            update={"limitations": limitations, "confidence": confidence}
        )

    @staticmethod
    def _format_filters(state: CurriculumQAState) -> str:
        payload = {
            "intent": state.intent,
            "level": state.level,
            "grade": state.grade,
            "subject": state.subject,
            "topic": state.topic,
        }
        return json.dumps({k: v for k, v in payload.items() if v}, indent=2)

    @staticmethod
    def _format_conversation_history(
        conversation: ConversationContext | None,
    ) -> str:
        if not conversation or len(conversation.messages) <= 1:
            return ""
        # Exclude the latest user message (already in USER QUESTION).
        prior = conversation.messages[:-1][-6:]
        lines = []
        for msg in prior:
            lines.append(f"{msg.role.value}: {msg.content[:500]}")
        return "\n".join(lines)


class LiveAttribution:
    """Deterministic reading of the refs a live model returned."""

    def __init__(
        self,
        *,
        refs: list[AnswerEvidenceRef],
        returned_refs: list[Any],
        valid_ids: list[str],
        invalid_refs: list[str],
        duplicate_count: int,
        unused_ids: list[str],
        unsupported_ids: list[str],
        substantive_claim_count: int,
        claims_with_supporting_refs: int,
    ) -> None:
        self.refs = refs
        self.returned_refs = returned_refs
        self.valid_ids = valid_ids
        self.invalid_refs = invalid_refs
        self.duplicate_count = duplicate_count
        self.unused_ids = unused_ids
        self.unsupported_ids = unsupported_ids
        self.substantive_claim_count = substantive_claim_count
        self.claims_with_supporting_refs = claims_with_supporting_refs

    def as_dict(self) -> dict[str, Any]:
        total = self.substantive_claim_count
        supported = self.claims_with_supporting_refs
        if total and not self.valid_ids:
            status = "missing"
        elif self.unsupported_ids:
            status = "unsupported"
        elif total and supported < total:
            status = "incomplete"
        elif self.valid_ids:
            status = "valid"
        else:
            status = "empty"
        completeness = (supported / total) if total else None
        return {
            "status": status,
            "returned_refs": self.returned_refs,
            "valid_refs": list(self.valid_ids),
            "invalid_refs": list(self.invalid_refs),
            "duplicate_ref_count": self.duplicate_count,
            "unused_evidence_ids": list(self.unused_ids),
            "unsupported_ref_ids": list(self.unsupported_ids),
            "substantive_claim_count": total,
            "claims_with_supporting_refs": supported,
            "attribution_completeness": completeness,
        }


def _raw_attribution_refs(raw: dict[str, Any]) -> list[Any]:
    """Prefer an explicit refs list. Fall back to the evidence array only when refs is absent."""
    if "refs" in raw and raw.get("refs") is not None:
        value = raw.get("refs")
        if isinstance(value, list):
            return list(value)
        return [value]
    evidence = raw.get("evidence") or []
    return list(evidence) if isinstance(evidence, list) else []


def _ref_identity(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("entity_id") or "").strip()
    return ""


_ATTRIBUTION_STOPWORDS = {
    "what",
    "which",
    "learning",
    "objective",
    "objectives",
    "code",
    "codes",
    "primary",
    "junior",
    "senior",
    "class",
    "grade",
    "pupil",
    "pupils",
    "this",
    "that",
    "from",
    "with",
    "about",
    "should",
    "learn",
    "taught",
    "curriculum",
    "into",
    "main",
    "areas",
    "note",
    "outcome",
    "particular",
    "available",
    "cannot",
    "confirmed",
    "incomplete",
    "source",
    "exact",
    "wording",
    "therefore",
    "these",
    "using",
    "their",
}


def _distinctive_tokens(text: str) -> list[str]:
    seen: list[str] = []
    for raw in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if len(raw) < 5 or raw in _ATTRIBUTION_STOPWORDS:
            continue
        if raw not in seen:
            seen.append(raw)
    return seen


def _answer_has_substantive_claim(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return False
    return "couldn't find sufficient" not in text.lower()


def _substantive_segments(answer: str) -> list[str]:
    if not _answer_has_substantive_claim(answer):
        return []
    segments: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if re.match(r"^[\*\-]\s+\S", stripped):
            segments.append(stripped)
    note_match = re.search(
        r"### Curriculum Evidence Note\s*\n+(.*)\Z",
        answer,
        re.S,
    )
    if note_match and note_match.group(1).strip():
        segments.append(note_match.group(1).strip())
    if segments:
        return segments
    prose = [
        line.strip()
        for line in answer.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if prose:
        return [" ".join(prose)]
    return [answer.strip()]


def _record_supports_text(item: CurriculumEvidence, text: str) -> bool:
    """Lexical overlap only. This is not claim-level proof."""
    content = (item.content or "").strip()
    if not content:
        return False
    tokens = _distinctive_tokens(content)
    hay = (text or "").lower()
    if len(tokens) >= 2:
        hits = sum(1 for token in tokens if token in hay)
        return hits >= max(2, (len(tokens) + 1) // 2)
    short = [
        token
        for token in re.findall(r"[a-z0-9]+", content.lower())
        if token not in _ATTRIBUTION_STOPWORDS and token not in tokens and len(token) >= 3
    ]
    if len(tokens) == 1:
        if short:
            return tokens[0] in hay and all(token in hay for token in short)
        return tokens[0] in hay and content.lower().rstrip(".") in hay
    return content.lower().rstrip(".") in hay


def attribute_live_model_refs(
    raw_refs: list[Any],
    evidence: list[CurriculumEvidence],
    *,
    answer: str,
) -> LiveAttribution:
    """Keep only supplied records the model cited, and measure obvious mismatches.

    Unknown ids are dropped. Retrieved records the model did not cite stay out of
    answer evidence. A cited record whose wording does not appear in the answer
    is reported as unsupported but is not replaced with a different record.
    """
    by_id = {item.entity_id: item for item in evidence if item.entity_id}
    refs: list[AnswerEvidenceRef] = []
    valid_ids: list[str] = []
    invalid_refs: list[str] = []
    seen: set[str] = set()
    duplicate_count = 0
    for item in raw_refs:
        identity = _ref_identity(item)
        if not identity or identity not in by_id:
            label = identity or "<empty>"
            if label not in invalid_refs:
                invalid_refs.append(label)
            continue
        if identity in seen:
            duplicate_count += 1
            continue
        seen.add(identity)
        source = by_id[identity]
        claim = ""
        if isinstance(item, dict):
            claim = str(item.get("claim") or "").strip()
        if not claim:
            claim = (source.content or source.name or "").strip()
        valid_ids.append(identity)
        refs.append(
            AnswerEvidenceRef(
                entity_id=identity,
                entity_type=source.entity_type,
                claim=claim,
                name=source.name,
                grade=source.grade,
                subject=source.subject,
                topic=source.topic,
            )
        )

    unsupported_ids = [
        entity_id
        for entity_id in valid_ids
        if len(_distinctive_tokens(by_id[entity_id].content or "")) >= 2
        and not _record_supports_text(by_id[entity_id], answer)
    ]
    segments = _substantive_segments(answer)
    supported_claims = 0
    for segment in segments:
        if any(_record_supports_text(by_id[entity_id], segment) for entity_id in valid_ids):
            supported_claims += 1
    unused_ids = [item.entity_id for item in evidence if item.entity_id and item.entity_id not in seen]
    return LiveAttribution(
        refs=refs,
        returned_refs=list(raw_refs),
        valid_ids=valid_ids,
        invalid_refs=invalid_refs,
        duplicate_count=duplicate_count,
        unused_ids=unused_ids,
        unsupported_ids=unsupported_ids,
        substantive_claim_count=len(segments),
        claims_with_supporting_refs=supported_claims,
    )


class ClaimAttribution:
    """Exact claim-text mappings returned by the experimental contract."""

    def __init__(
        self,
        *,
        refs: list[AnswerEvidenceRef],
        returned_claims: list[Any],
        valid_claims: list[dict[str, Any]],
        absent_claims: list[dict[str, Any]],
        unknown_ref_claims: list[dict[str, Any]],
        unsupported_claims: list[dict[str, Any]],
        invalid_refs: list[str],
        unattributed_claims: list[str],
        substantive_claim_count: int,
    ) -> None:
        self.refs = refs
        self.returned_claims = returned_claims
        self.valid_claims = valid_claims
        self.absent_claims = absent_claims
        self.unknown_ref_claims = unknown_ref_claims
        self.unsupported_claims = unsupported_claims
        self.invalid_refs = invalid_refs
        self.unattributed_claims = unattributed_claims
        self.substantive_claim_count = substantive_claim_count

    def as_dict(self) -> dict[str, Any]:
        valid_count = len(self.valid_claims)
        if self.substantive_claim_count and not valid_count and self.unsupported_claims:
            status = "unsupported"
        elif self.substantive_claim_count and not valid_count:
            status = "missing"
        elif self.unattributed_claims:
            status = "incomplete"
        elif self.unsupported_claims:
            status = "unsupported"
        elif valid_count:
            status = "valid"
        else:
            status = "empty"
        multi_record_claims = sum(1 for claim in self.valid_claims if len(claim["refs"]) >= 2)
        return {
            "status": status,
            "returned_claims": self.returned_claims,
            "valid_claims": self.valid_claims,
            "absent_claims": self.absent_claims,
            "unknown_ref_claims": self.unknown_ref_claims,
            "unsupported_claims": self.unsupported_claims,
            "invalid_refs": list(self.invalid_refs),
            "unattributed_claims": list(self.unattributed_claims),
            "substantive_claim_count": self.substantive_claim_count,
            "valid_claim_count": valid_count,
            "multi_record_claim_count": multi_record_claims,
            "valid_refs": [ref.entity_id for ref in self.refs],
        }


def _presentation_normalize(text: str) -> str:
    """Drop bullet markers and collapse whitespace. Do not rewrite words."""
    lines: list[str] = []
    for line in (text or "").splitlines():
        stripped = re.sub(r"^[\*\-]\s+", "", line.strip())
        if stripped:
            lines.append(stripped)
    compact = " ".join(lines) if lines else (text or "").strip()
    return re.sub(r"\s+", " ", compact).strip()


def _span_key(text: str) -> str:
    return _presentation_normalize(text).rstrip(".").strip()


def _sentence_keys(text: str) -> list[str]:
    normalized = _presentation_normalize(text)
    parts = re.split(r"(?<=[.])\s+", normalized)
    return [part.rstrip(".").strip() for part in parts if part.strip()]


def _claim_occurs(claim: str, answer: str) -> bool:
    """True when the claim is an exact span of the answer, not a shortened word."""
    needle = _span_key(claim)
    if not needle:
        return False
    hay = _span_key(answer)
    if needle == hay:
        return True
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", hay) is not None


def _mapping_covers_segment(claim: str, segment: str) -> bool:
    claim_key = _span_key(claim)
    if not claim_key:
        return False
    if claim_key == _span_key(segment):
        return True
    return claim_key in _sentence_keys(segment)


def attribute_claim_mappings(
    raw_claims: list[Any],
    evidence: list[CurriculumEvidence],
    *,
    answer: str,
) -> ClaimAttribution:
    """Validate experimental claim → ref mappings by exact answer text.

    A mapping is kept only when its text occurs in the answer and at least one
    ref is a supplied record whose wording supports that text. Unknown refs and
    mappings whose text is absent are rejected. Unrelated refs are reported and
    are not treated as valid support. Retrieved records are never attached to a
    claim the model did not map.
    """
    if not isinstance(raw_claims, list):
        raw_claims = []
    by_id = {item.entity_id: item for item in evidence if item.entity_id}
    valid_claims: list[dict[str, Any]] = []
    absent_claims: list[dict[str, Any]] = []
    unknown_ref_claims: list[dict[str, Any]] = []
    unsupported_claims: list[dict[str, Any]] = []
    invalid_refs: list[str] = []
    refs: list[AnswerEvidenceRef] = []
    seen_ids: set[str] = set()

    for item in raw_claims:
        if not isinstance(item, dict):
            absent_claims.append({"text": "", "refs": []})
            continue
        text = str(item.get("text") or "").strip()
        raw_refs = item.get("refs") or []
        if not isinstance(raw_refs, list):
            raw_refs = [raw_refs]
        identities = [_ref_identity(ref) for ref in raw_refs]
        if not text or not _claim_occurs(text, answer):
            for identity in identities:
                label = identity or "<empty>"
                if label not in by_id and label not in invalid_refs:
                    invalid_refs.append(label)
            absent_claims.append({"text": text, "refs": [identity for identity in identities if identity]})
            continue

        accepted: list[str] = []
        unsupported: list[str] = []
        unknown: list[str] = []
        seen_local: set[str] = set()
        for identity in identities:
            if not identity or identity not in by_id:
                label = identity or "<empty>"
                if label not in unknown:
                    unknown.append(label)
                if label not in invalid_refs:
                    invalid_refs.append(label)
                continue
            if identity in seen_local:
                continue
            seen_local.add(identity)
            if _record_supports_text(by_id[identity], text):
                accepted.append(identity)
            else:
                unsupported.append(identity)

        if accepted:
            valid_claims.append({"text": text, "refs": accepted})
            for identity in accepted:
                if identity in seen_ids:
                    continue
                seen_ids.add(identity)
                source = by_id[identity]
                refs.append(
                    AnswerEvidenceRef(
                        entity_id=identity,
                        entity_type=source.entity_type,
                        claim=text,
                        name=source.name,
                        grade=source.grade,
                        subject=source.subject,
                        topic=source.topic,
                    )
                )
        if unsupported:
            unsupported_claims.append({"text": text, "refs": unsupported})
        if unknown and not accepted:
            unknown_ref_claims.append({"text": text, "refs": unknown})

    segments = _substantive_segments(answer)
    unattributed = [
        _span_key(segment)
        for segment in segments
        if not any(_mapping_covers_segment(claim["text"], segment) for claim in valid_claims)
    ]
    return ClaimAttribution(
        refs=refs,
        returned_claims=list(raw_claims),
        valid_claims=valid_claims,
        absent_claims=absent_claims,
        unknown_ref_claims=unknown_ref_claims,
        unsupported_claims=unsupported_claims,
        invalid_refs=invalid_refs,
        unattributed_claims=unattributed,
        substantive_claim_count=len(segments),
    )


def _looks_garbled_source_text(text: str) -> bool:
    lowered = text.lower()
    if len(text) > 120 and lowered.count("denominators up to") >= 2:
        return True
    if text.endswith((" greater than", " up to", " to")):
        return True
    return False


_UNIT_TYPES = {"topic", "subtopic", "unit", "strand"}
_LO_CODE_RE = re.compile(r"(C\d+)U(\d+)-LO\d+", re.I)
_LEADING_CODE_RE = re.compile(
    r"^(?:C\d+-U\d+|C\d+U\d+-LO\d+)\s*[—–:-]\s*",
    re.I,
)
_TRUNCATION_CUES = ("up to", "greater than", "related to")
_TRAILING_FUNCTION_WORDS = {
    "with",
    "of",
    "and",
    "or",
    "to",
    "for",
    "the",
    "a",
    "an",
    "in",
    "on",
}
_IDENTIFIER_STOPWORDS = {
    "what",
    "which",
    "learning",
    "objective",
    "objectives",
    "code",
    "codes",
    "primary",
    "junior",
    "senior",
    "class",
    "grade",
    "pupil",
    "pupils",
    "this",
    "that",
    "from",
    "with",
    "about",
    "should",
    "learn",
    "taught",
    "curriculum",
}


def _render_stub_answer(
    state: CurriculumQAState,
    *,
    grade_label: str | None,
    subject_label: str | None,
) -> tuple[str, list[str], list[CurriculumEvidence]]:
    """Group supplied evidence into user-facing prose. No model call."""
    evidence = state.evidence
    topics = [e for e in evidence if (e.entity_type or "").lower() in _UNIT_TYPES]
    subjects = [e for e in evidence if (e.entity_type or "").lower() == "subject"]
    outcomes = [
        e for e in evidence if (e.entity_type or "").lower() == "learning_outcome"
    ]
    question = state.question or ""
    limitations: list[str] = []

    if question_requests_identifiers(question) and (outcomes or topics):
        text, used = _render_identifier_answer(
            question,
            outcomes=outcomes,
            topics=topics,
            grade_label=grade_label,
            subject_label=subject_label,
        )
        return text, limitations, used

    if outcomes and not _is_catalogue_question(question):
        text, damaged, used = _render_outcome_synthesis(
            question,
            outcomes=outcomes,
            topics=topics,
            grade_label=grade_label,
            subject_label=subject_label,
            topic_hint=state.topic,
        )
        if damaged:
            limitations.append(
                "Some supplied curriculum records are incomplete or duplicated, "
                "so their exact wording cannot be confirmed."
            )
        return text, limitations, used

    lines = _heading_lines(grade_label, subject_label, focus=None)
    used: list[CurriculumEvidence] = []
    if topics:
        catalogue, used = _render_topic_catalogue(question, topics, subject_label)
        lines.extend(catalogue)
    elif subjects:
        named = [s for s in subjects if s.name and _public_name(s.name)]
        names = sorted({s.name for s in named if s.name})
        if names:
            lines.append("")
            lines.append("Subjects include:")
            lines.extend(f"* {name}" for name in names)
            used = named
    elif outcomes:
        text, damaged, used = _render_outcome_synthesis(
            question,
            outcomes=outcomes,
            topics=topics,
            grade_label=grade_label,
            subject_label=subject_label,
            topic_hint=state.topic,
        )
        if damaged:
            limitations.append(
                "Some supplied curriculum records are incomplete or duplicated, "
                "so their exact wording cannot be confirmed."
            )
        return text, limitations, used

    if len(lines) <= 1:
        named_items = [e for e in evidence if e.name and _public_name(e.name)]
        names = list(dict.fromkeys(e.name for e in named_items if e.name))[:10]
        if names:
            lines.append("")
            lines.append("Retrieved curriculum records include:")
            lines.extend(f"* {name}" for name in names)
            used = [e for e in named_items if e.name in names]
    return "\n".join(lines).strip(), limitations, used


def _render_outcome_synthesis(
    question: str,
    *,
    outcomes: list[CurriculumEvidence],
    topics: list[CurriculumEvidence],
    grade_label: str | None,
    subject_label: str | None,
    topic_hint: str | None,
) -> tuple[str, bool, list[CurriculumEvidence]]:
    units_by_code = _units_by_code(topics)
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for outcome in outcomes:
        label = _group_label(outcome, units_by_code)
        bucket = groups.get(label)
        if bucket is None:
            bucket = {"claims": [], "damaged": []}
            groups[label] = bucket
            order.append(label)
        content = _strip_leading_code(outcome.content or "")
        damaged = _is_damaged_source(content) or _evidence_marked_imperfect(outcome)
        if damaged:
            _remember(bucket["damaged"], outcome)
            portion = (
                _reliable_portion(content)
                if _is_damaged_source(content)
                else content.strip()
            )
        else:
            portion = content.strip()
        if portion:
            _append_claim(bucket["claims"], portion, outcome, from_damage=damaged)

    focus = _focus_phrase(question, topic_hint)
    where = " ".join(part for part in (grade_label, subject_label) if part) or "the supplied curriculum"
    damaged_records = [
        {"label": _display_heading(label, focus), "raw": _strip_leading_code(item.content or "")}
        for label in order
        for item in groups[label]["damaged"]
    ]
    visible = [label for label in order if groups[label]["claims"]]
    display_labels = [_display_heading(label, focus) for label in visible]
    structured = _use_structured_sections(visible, groups)
    lines = _heading_lines(grade_label, subject_label, focus)
    if structured and len(visible) >= 2:
        lines.append("")
        lines.append(_area_introduction(where, focus, display_labels))
    if structured:
        for index, label in enumerate(visible):
            heading = display_labels[index]
            lines.append("")
            if len(visible) >= 2:
                lines.append(f"### {index + 1}. {heading}")
            else:
                lines.append(f"### {heading}")
            lines.extend(
                _expectation_block([claim["text"] for claim in groups[label]["claims"]])
            )
    else:
        claims = [
            claim["text"]
            for label in order
            for claim in groups[label]["claims"]
        ]
        if claims:
            lines.append("")
            if len(claims) == 1:
                lines.append(_as_sentence(claims[0]))
            else:
                lines.extend(_expectation_block(claims))
    if damaged_records:
        lines.append("")
        lines.append("### Curriculum Evidence Note")
        lines.append("")
        lines.append(_curriculum_evidence_note(damaged_records))
    return "\n".join(lines).strip(), bool(damaged_records), _sources_for_groups(groups, order, units_by_code)


def _use_structured_sections(
    order: list[str],
    groups: dict[str, dict[str, Any]],
) -> bool:
    """Use headings when several areas, or many skills, need separating."""
    claim_count = sum(len(groups[label]["claims"]) for label in order)
    if len(order) >= 2 and claim_count >= 3:
        return True
    return claim_count >= 4


def _area_introduction(where: str, focus: str | None, labels: list[str]) -> str:
    topic = (focus or "this topic").lower()
    listed = _join_labels(labels)
    return (
        f"In {where}, {topic} work is organised into {len(labels)} main areas: "
        f"{listed}."
    )


def _join_labels(labels: list[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _expectation_block(claims: list[str]) -> list[str]:
    return ["", "Pupils learn to:", "", *[_bullet(claim) for claim in claims]]


def _bullet(text: str) -> str:
    return f"* {_as_sentence(text)}"


def _as_sentence(text: str) -> str:
    sentence = " ".join(text.split()).strip(" .;")
    if not sentence:
        return ""
    sentence = sentence[:1].upper() + sentence[1:]
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence


_FRAMING_WORDS = {"identify", "identifying", "work", "working", "with"}
_FILLER_WORDS = {"a", "an", "the", "these", "this", "using", "use"}


def _expectation_key(text: str) -> tuple[str, ...]:
    """Near-exact key: drop only a leading framing verb, keep distinguishing words."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    while words and words[0] in _FRAMING_WORDS:
        words.pop(0)
    return tuple(word for word in words if word not in _FILLER_WORDS)


def _append_claim(
    claims: list[dict[str, Any]],
    text: str,
    source: CurriculumEvidence,
    *,
    from_damage: bool,
) -> None:
    key = _expectation_key(text)
    for claim in claims:
        if key and claim["key"] == key:
            _remember(claim["sources"], source)
            if claim.get("from_damage") and not from_damage:
                claim["text"] = text
                claim["from_damage"] = False
            return
    claims.append(
        {
            "text": text,
            "sources": [source],
            "key": key,
            "from_damage": from_damage,
        }
    )


def _remember(items: list[CurriculumEvidence], item: CurriculumEvidence) -> None:
    if all(existing is not item and existing.entity_id != item.entity_id for existing in items):
        items.append(item)


def _sources_for_groups(
    groups: dict[str, dict[str, Any]],
    order: list[str],
    units_by_code: dict[str, CurriculumEvidence],
) -> list[CurriculumEvidence]:
    used: list[CurriculumEvidence] = []
    for label in order:
        bucket = groups[label]
        contributors: list[CurriculumEvidence] = []
        for claim in bucket["claims"]:
            contributors.extend(claim["sources"])
        contributors.extend(bucket["damaged"])
        for source in contributors:
            _remember(used, source)
            parent_code = _unit_code_for_outcome(source)
            parent = units_by_code.get(parent_code or "")
            if parent is not None:
                _remember(used, parent)
    return used


def _records_in_evidence_order(
    evidence: list[CurriculumEvidence],
    used: list[CurriculumEvidence],
) -> list[CurriculumEvidence]:
    used_ids = {item.entity_id for item in used if item.entity_id}
    return [item for item in evidence if item.entity_id and item.entity_id in used_ids]


def _display_heading(label: str, focus: str | None) -> str:
    """Present a unit/topic label without strand prefixes or shouty database casing."""
    text = " ".join((label or "").split()).strip()
    if not text:
        return "Related curriculum evidence"
    lowered = text.lower()
    prefix = "number and numeration"
    if lowered.startswith(prefix):
        rest = text[len(prefix) :].strip(" -–—:")
        if rest and not _looks_like_internal_id(rest):
            text = rest
            lowered = text.lower()
    if focus:
        focus_key = re.sub(r"s$", "", focus.lower())
        label_key = re.sub(r"s$", "", lowered)
        if label_key == focus_key:
            return focus[:1].upper() + focus[1:]
    return _heading_case(text)


def _heading_case(text: str) -> str:
    small = {"and", "or", "of", "on", "in", "the", "a", "for"}
    rendered: list[str] = []
    for index, word in enumerate(text.split()):
        lower = word.lower()
        if index > 0 and lower in small:
            rendered.append(lower)
        elif word.isupper() or word.islower():
            rendered.append(lower[:1].upper() + lower[1:])
        else:
            rendered.append(word)
    return " ".join(rendered)


def _curriculum_evidence_note(records: list[dict[str, str]]) -> str:
    if len(records) == 1:
        raw = records[0]["raw"]
        subject = _note_subject(records[0])
        concerning = f" concerning {subject}" if subject else ""
        if _missing_denominator_range(raw):
            return (
                f"One learning outcome{concerning} is incomplete in the source "
                "evidence. Therefore, the exact denominator range for that "
                "particular outcome cannot be confirmed from the available evidence."
            )
        return (
            f"One learning outcome{concerning} is incomplete or duplicated in "
            "the source evidence. The exact wording cannot be confirmed from "
            "the available evidence."
        )
    return (
        "Some learning outcomes are incomplete or duplicated in the source "
        "evidence. Their exact wording cannot be confirmed from the available "
        "evidence."
    )


_NOTE_VERBS = {"identify", "identifying", "work", "working"}


def _note_subject(record: dict[str, str]) -> str:
    portion = (_reliable_portion(record["raw"]) or "").strip(" .")
    words = portion.split()
    if words and words[0].lower() in _NOTE_VERBS and len(words) > 1:
        subject_words = words[1:]
        if subject_words[0].lower() == "with" and len(subject_words) > 1:
            subject_words = subject_words[1:]
        subject = " ".join(subject_words)
    elif portion and len(words) <= 8:
        subject = portion
    else:
        label = record["label"].strip()
        if not label or label.lower() == "related curriculum evidence":
            return ""
        subject = label
    return subject[:1].lower() + subject[1:]


def _missing_denominator_range(text: str) -> bool:
    lowered = " ".join((text or "").lower().split())
    if "denominator" not in lowered:
        return False
    if re.search(r"\bdenominators?\s+up to\s+\d+", lowered):
        return False
    return "up to" in lowered


def _render_identifier_answer(
    question: str,
    *,
    outcomes: list[CurriculumEvidence],
    topics: list[CurriculumEvidence],
    grade_label: str | None,
    subject_label: str | None,
) -> tuple[str, list[CurriculumEvidence]]:
    kind = _identifier_kind(question)
    lines = _heading_lines(grade_label, subject_label, focus=None)
    lines.append("")
    if kind == "unit":
        matches = _best_content_matches(question, topics) or topics[:3]
        if not matches:
            lines.append(
                "The supplied curriculum evidence does not include a unit code "
                "that matches this question."
            )
            return "\n".join(lines).strip(), []
        lines.append("The requested unit code from the supplied evidence:")
        for item in matches:
            code = _record_code(item)
            label = _public_name(item.name) or "Curriculum unit"
            if code:
                lines.append(f"* **{code}** — {label}")
        return "\n".join(lines).strip(), matches

    pool = outcomes or topics
    matches = _best_content_matches(question, pool)
    if not matches:
        lines.append(
            "The supplied curriculum evidence does not include an identifier "
            "that matches this question."
        )
        return "\n".join(lines).strip(), []
    lines.append("The requested curriculum identifier from the supplied evidence:")
    for item in matches:
        code = _record_code(item) or (
            item.entity_id if kind == "entity" else None
        )
        wording = _strip_leading_code(item.content or "")
        if _is_damaged_source(wording):
            portion = _reliable_portion(wording)
            shown = portion or "the exact source wording is unreliable"
            lines.append(
                f"* **{code or 'unknown'}** — {shown}. "
                "The exact wording of this record is unreliable."
            )
        elif code and wording:
            lines.append(f"* **{code}** — {wording.strip()}")
        elif code:
            lines.append(f"* **{code}**")
        elif wording:
            lines.append(f"* {wording.strip()}")
    return "\n".join(lines).strip(), matches


def _render_topic_catalogue(
    question: str,
    topics: list[CurriculumEvidence],
    subject_label: str | None,
) -> tuple[list[str], list[CurriculumEvidence]]:
    lines: list[str] = [""]
    used: list[CurriculumEvidence] = []
    list_mode = _is_catalogue_question(question) or len(topics) > 1
    if list_mode:
        lines.append("The MBSSE curriculum includes these units/topics:")
        seen: set[str] = set()
        for topic in topics[:40]:
            name = _display_heading(topic.name or "", focus=None) if _public_name(topic.name) else ""
            public = _public_name(topic.name)
            if not public or public in seen:
                continue
            seen.add(public)
            lines.append(f"* {name or public}")
            _remember(used, topic)
        return lines, used
    topic = topics[0]
    name = _public_name(topic.name) or "this topic"
    lines.append(
        f"The MBSSE curriculum includes **{name}**"
        + (f" under {subject_label}." if subject_label else ".")
    )
    _remember(used, topic)
    if topic.content and topic.content != topic.name and _public_name(topic.content):
        lines.append("")
        lines.append(_strip_leading_code(topic.content))
    return lines, used


def _heading_lines(
    grade_label: str | None,
    subject_label: str | None,
    focus: str | None,
) -> list[str]:
    parts = [part for part in (grade_label, subject_label) if part]
    title = " ".join(parts) if parts else "Curriculum evidence"
    if focus:
        title = f"{title} — {focus}"
    return [f"# {title}"]


def _is_catalogue_question(question: str) -> bool:
    lowered = question.lower()
    return any(
        token in lowered
        for token in (
            "what topics",
            "which topics",
            "what units",
            "which units",
            "list the units",
            "list the topics",
            "what is taught",
            "curriculum structure",
        )
    )


def _identifier_kind(question: str) -> str:
    lowered = question.lower()
    if "unit code" in lowered:
        return "unit"
    if "entity" in lowered or "database id" in lowered or "retrieval id" in lowered:
        return "entity"
    return "learning_outcome"


def _units_by_code(topics: list[CurriculumEvidence]) -> dict[str, CurriculumEvidence]:
    indexed: dict[str, CurriculumEvidence] = {}
    for item in topics:
        code = _record_code(item)
        if code:
            indexed[code.upper()] = item
    return indexed


def _record_code(item: CurriculumEvidence) -> str | None:
    code = (item.metadata or {}).get("code")
    if code and str(code).strip():
        return str(code).strip()
    if item.name and _looks_like_internal_id(item.name):
        return item.name.strip()
    return None


def _unit_code_for_outcome(item: CurriculumEvidence) -> str | None:
    parent = (item.metadata or {}).get("parent_content_code")
    if parent and str(parent).strip():
        return str(parent).strip().upper()
    code = _record_code(item) or ""
    match = _LO_CODE_RE.match(code)
    if match:
        return f"{match.group(1).upper()}-U{match.group(2)}"
    return None


def _group_label(
    item: CurriculumEvidence,
    units_by_code: dict[str, CurriculumEvidence],
) -> str:
    topic = (item.topic or "").strip()
    if topic and not _looks_like_internal_id(topic):
        return topic
    parent_name = str((item.metadata or {}).get("parent_content_name") or "").strip()
    if parent_name and not _looks_like_internal_id(parent_name):
        return parent_name
    unit_code = _unit_code_for_outcome(item)
    if unit_code:
        unit = units_by_code.get(unit_code.upper())
        if unit is not None:
            label = _public_name(unit.name)
            if label:
                return label
    return "Related curriculum evidence"


def _public_name(value: str | None) -> str | None:
    if not value:
        return None
    text = " ".join(str(value).split()).strip()
    if not text or _looks_like_internal_id(text):
        return None
    return text


def _looks_like_internal_id(value: str) -> bool:
    text = value.strip()
    if re.fullmatch(r"C\d+-U\d+", text, flags=re.I):
        return True
    if re.fullmatch(r"C\d+U\d+-LO\d+", text, flags=re.I):
        return True
    if text.lower() == "grade_curriculum_id":
        return True
    return False


def _strip_leading_code(text: str) -> str:
    return _LEADING_CODE_RE.sub("", text or "").strip()


def _evidence_marked_imperfect(item: CurriculumEvidence) -> bool:
    state = str((item.metadata or {}).get("evidence_state") or "")
    if "IMPERFECT" in state.upper():
        return True
    quality = (item.metadata or {}).get("evidence_quality")
    if isinstance(quality, dict):
        status = str(quality.get("status") or "").lower()
        if status in {"imperfect", "garbled", "truncated", "damaged", "incomplete"}:
            return True
    return False


def _is_damaged_source(text: str) -> bool:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return False
    if _looks_garbled_source_text(cleaned):
        return True
    words = re.findall(r"[A-Za-z0-9']+", cleaned)
    return _repeated_phrase_cut(words) < len(words)


def _repeated_phrase_cut(words: list[str]) -> int:
    lower = [word.lower() for word in words]
    cut = len(words)
    max_n = min(8, len(words) // 2)
    for size in range(4, max_n + 1):
        seen: dict[tuple[str, ...], int] = {}
        for index in range(0, len(lower) - size + 1):
            phrase = tuple(lower[index : index + size])
            if phrase in seen:
                return min(cut, index)
            seen[phrase] = index
    return cut


def _reliable_portion(text: str) -> str | None:
    """Return source text that can be stated without repairing damage."""
    raw = _strip_leading_code(" ".join((text or "").split()))
    if not raw:
        return None
    if not _is_damaged_source(raw):
        return raw
    words = re.findall(r"[A-Za-z0-9']+", raw)
    prefix = " ".join(words[: _repeated_phrase_cut(words)])
    lower_prefix = prefix.lower()
    for cue in _TRUNCATION_CUES:
        index = lower_prefix.find(cue)
        if index != -1:
            prefix = prefix[:index]
            break
    prefix = re.sub(r"\b(?:with|of)\s+\w+\s*$", "", prefix).strip(" ,;:-")
    parts = prefix.split()
    while parts and parts[-1].lower() in _TRAILING_FUNCTION_WORDS:
        parts.pop()
    if len(parts) < 2:
        return None
    return " ".join(parts)


def _focus_phrase(question: str, topic_hint: str | None) -> str | None:
    hint = _public_name(topic_hint)
    if hint and not _looks_like_internal_id(hint):
        return hint[:1].upper() + hint[1:]
    match = re.search(r"\babout\s+([A-Za-z][A-Za-z\s]{2,40})", question or "", re.I)
    if not match:
        return None
    phrase = re.sub(
        r"\b(?:in|for|at|under)\b.*$",
        "",
        match.group(1),
        flags=re.I,
    ).strip()
    if not phrase or _looks_like_internal_id(phrase):
        return None
    return phrase[:1].upper() + phrase[1:]


def _content_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in re.findall(r"[a-z]+", (text or "").lower()):
        if raw in _IDENTIFIER_STOPWORDS or len(raw) < 4:
            continue
        tokens.add(raw)
        if raw.endswith("ing") and len(raw) > 6:
            tokens.add(raw[:-3])
    return tokens


def _best_content_matches(
    question: str,
    items: list[CurriculumEvidence],
) -> list[CurriculumEvidence]:
    scored: list[tuple[int, CurriculumEvidence]] = []
    question_tokens = _content_tokens(question)
    for item in items:
        hay = _content_tokens(
            " ".join(
                [
                    item.content or "",
                    item.topic or "",
                    item.name or "",
                    str((item.metadata or {}).get("parent_content_name") or ""),
                ]
            )
        )
        scored.append((len(question_tokens & hay), item))
    if not scored:
        return []
    best = max(score for score, _item in scored)
    if best < 2:
        return []
    return [item for score, item in scored if score == best]


def format_evidence_for_prompt(
    evidence: list[CurriculumEvidence],
    *,
    question: str | None = None,
    max_records: int = 24,
) -> str:
    """Render retrieved evidence with hierarchy for the LLM prompt."""
    ranked, _ids = select_evidence_for_prompt(
        evidence, question=question, max_records=max_records
    )
    if not ranked:
        return "(no curriculum evidence retrieved)"

    blocks: list[str] = []
    for index, item in enumerate(ranked, start=1):
        hierarchy = _hierarchy_path(item)
        lines = [
            f"--- Record {index} ---",
            f"Entity ID: {item.entity_id or 'unknown'}",
            f"Entity Type: {item.entity_type}",
        ]
        if item.name:
            lines.append(f"Name: {item.name}")
        if hierarchy:
            lines.append(f"Hierarchy: {' → '.join(hierarchy)}")
        if item.content and item.content != item.name:
            content = str(item.content)
            if len(content) > 500:
                content = content[:500] + "…"
            lines.append(f"Content: {content}")
        if item.source_reference:
            lines.append(f"Source: {item.source_reference}")
        code = item.metadata.get("code") if item.metadata else None
        if code:
            lines.append(f"Code: {code}")
        eq = (item.metadata or {}).get("evidence_quality")
        if isinstance(eq, dict):
            lines.append(
                "Evidence quality: "
                f"status={eq.get('status')}; "
                f"original_text_present={eq.get('original_text_present')}; "
                f"source_record_id={eq.get('source_record_id')}"
            )
        es = (item.metadata or {}).get("evidence_state")
        if es:
            lines.append(f"Evidence state: {es}")
        blocks.append("\n".join(lines))
    omitted = len(evidence) - len(ranked)
    if omitted > 0:
        blocks.append(
            f"(omitted {omitted} additional evidence records; "
            "prefer the most relevant records above)"
        )
    return "\n\n".join(blocks)


def select_evidence_for_prompt(
    evidence: list[CurriculumEvidence],
    *,
    question: str | None = None,
    max_records: int = 24,
) -> tuple[list[CurriculumEvidence], list[str]]:
    """Return ranked evidence rows supplied to the generator (and their ids)."""
    if not evidence:
        return [], []
    ranked = _rank_evidence(evidence, question=question)[:max_records]
    ids = [item.entity_id for item in ranked if item.entity_id]
    return ranked, ids


def _rank_evidence(
    evidence: list[CurriculumEvidence],
    *,
    question: str | None,
) -> list[CurriculumEvidence]:
    if not question:
        return list(evidence)
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if len(token) > 2 and token not in {"the", "and", "for", "what", "are"}
    }
    if not tokens:
        return list(evidence)

    def score(item: CurriculumEvidence) -> tuple[int, int]:
        hay = " ".join(
            [
                str(item.name or ""),
                str(item.content or ""),
                str(item.topic or ""),
                str(item.entity_type or ""),
                str((item.metadata or {}).get("code") or ""),
            ]
        ).lower()
        hits = sum(1 for token in tokens if token in hay)
        # Prefer learning outcomes / units when present.
        type_bonus = 1 if (item.entity_type or "").lower() in {
            "learning_outcome",
            "unit",
            "topic",
            "subtopic",
        } else 0
        return (hits, type_bonus)

    return sorted(evidence, key=score, reverse=True)


def _extract_answer_text(raw: dict[str, Any]) -> str:
    """Pull a non-empty answer from common structured-output field names."""
    for key in ("answer", "summary", "text", "content", "response"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _hierarchy_path(item: CurriculumEvidence) -> list[str]:
    parts: list[str] = []
    if item.level:
        parts.append(item.level)
    if item.grade:
        parts.append(_display_grade(item.grade))
    if item.subject:
        parts.append(_display_subject(item.subject))
    if item.topic and item.topic != item.name:
        parts.append(item.topic)
    if item.name and (not parts or parts[-1] != item.name):
        parts.append(item.name)
    return parts


def _display_grade(code: str | None) -> str | None:
    if not code:
        return None
    if code.startswith("CLASS_"):
        return f"Primary {code.split('_', 1)[1]}"
    if code.startswith("JSS_"):
        return f"JSS {code.split('_', 1)[1]}"
    if code.startswith("SSS_"):
        return f"SSS {code.split('_', 1)[1]}"
    return code


def _display_subject(code: str | None) -> str | None:
    if not code:
        return None
    return code.replace("_", " ").title()


def _first_attr(items: list[CurriculumEvidence], attr: str) -> str | None:
    for item in items:
        value = getattr(item, attr, None)
        if value:
            return str(value)
    return None


def _one_line_summary(text: str) -> str:
    line = text.splitlines()[0].lstrip("# ").strip()
    return line[:200] if line else ""


def _evidence_refs_from_items(
    items: list[CurriculumEvidence],
    question: str,
) -> list[AnswerEvidenceRef]:
    refs: list[AnswerEvidenceRef] = []
    for item in items:
        if not item.entity_id:
            continue
        claim = item.content or item.name or item.entity_type
        refs.append(
            AnswerEvidenceRef(
                entity_id=item.entity_id,
                entity_type=item.entity_type,
                claim=str(claim),
                name=item.name,
                grade=item.grade,
                subject=item.subject,
                topic=item.topic,
            )
        )
    if not refs and question:
        return refs
    return refs
