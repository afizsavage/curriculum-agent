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
4. EVIDENCE REFERENCES: Every curriculum-specific claim in your answer should
   appear in the evidence array with a valid entity_id from the provided records.
   Never invent entity IDs.
5. CONFIDENCE: Assign high only when exact topic/objective evidence answers the
   question; medium when interpretation is needed; low when evidence is partial.
6. STYLE: Write for pupils, teachers, and education users. Synthesize the
   evidence into concise natural language. Do not expose internal curriculum
   identifiers unless the question asks for them.

{EVIDENCE_CONSERVATIVE_RULES}
"""

# Backward compatibility for V2.3 diagnostic experiment arm B.
CONSTRAINED_GENERATION_APPENDIX = EVIDENCE_CONSERVATIVE_USER_APPENDIX


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
            "Reference entity_id values from the evidence records in your evidence array.\n"
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
                "Group related objectives by unit or topic name into one explanation.\n"
                "Do not emit one bullet per learning objective.\n"
                "Do not copy learning-objective codes, unit codes, entity IDs, "
                "database IDs, retrieval IDs, or grade_curriculum_id into the answer.\n"
                "AUDIT\n"
                "Keep entity_id values in the evidence array only.\n"
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
        user_content += (
            "\nJSON OUTPUT\n"
            "Respond with a single JSON object (no markdown code fences) matching this schema:\n"
            f"{json.dumps(GROUNDED_ANSWER_JSON_SCHEMA, indent=2)}"
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
        try:
            raw = self.llm.generate_structured(
                messages, schema=GROUNDED_ANSWER_JSON_SCHEMA, temperature=0.0
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
                        "`answer` under 1200 characters, at most 8 evidence refs, "
                        "and short limitations."
                    ),
                )
            )
            try:
                raw = self.llm.generate_structured(
                    compact, schema=GROUNDED_ANSWER_JSON_SCHEMA, temperature=0.0
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
        answer_text, limitations = _render_stub_answer(
            state,
            grade_label=grade_label,
            subject_label=subject_label,
        )
        if state.evidence_status == EvidenceStatus.PARTIAL:
            limitations.append(
                "Some curriculum API calls failed or returned partial results."
            )
        limitations = list(dict.fromkeys(limitations))

        refs = _evidence_refs_from_items(evidence[:8], state.question)
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

        refs = self._validate_evidence_refs(raw.get("evidence") or [], evidence)
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

        limitations = [str(x) for x in (raw.get("limitations") or []) if x]
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
        by_id = {e.entity_id: e for e in evidence if e.entity_id}
        validated: list[AnswerEvidenceRef] = []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            entity_id = str(ref.get("entity_id") or "")
            if not entity_id or entity_id not in by_id:
                continue
            source = by_id[entity_id]
            validated.append(
                AnswerEvidenceRef(
                    entity_id=entity_id,
                    entity_type=str(ref.get("entity_type") or source.entity_type),
                    claim=str(ref.get("claim") or ""),
                    name=source.name,
                    grade=source.grade,
                    subject=source.subject,
                    topic=source.topic,
                )
            )
        return validated

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

        if not answer.evidence and state.evidence and confidence != AnswerConfidence.LOW:
            confidence = AnswerConfidence.MEDIUM
            limitations.append(
                "Answer could not be linked to specific curriculum entity references."
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
) -> tuple[str, list[str]]:
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
        return (
            _render_identifier_answer(
                question,
                outcomes=outcomes,
                topics=topics,
                grade_label=grade_label,
                subject_label=subject_label,
            ),
            limitations,
        )

    if outcomes and not _is_catalogue_question(question):
        text, damaged = _render_outcome_synthesis(
            question,
            outcomes=outcomes,
            topics=topics,
            grade_label=grade_label,
            subject_label=subject_label,
            topic_hint=state.topic,
        )
        if damaged:
            limitations.append(
                "Some supplied curriculum records are duplicated, truncated, or "
                "garbled, so their exact wording is unreliable."
            )
        return text, limitations

    lines = _heading_lines(grade_label, subject_label, focus=None)
    if topics:
        lines.extend(_render_topic_catalogue(question, topics, subject_label))
    elif subjects:
        names = sorted({s.name for s in subjects if s.name and _public_name(s.name)})
        if names:
            lines.append("")
            lines.append("Subjects include:")
            lines.extend(f"* {name}" for name in names)
    elif outcomes:
        text, damaged = _render_outcome_synthesis(
            question,
            outcomes=outcomes,
            topics=topics,
            grade_label=grade_label,
            subject_label=subject_label,
            topic_hint=state.topic,
        )
        if damaged:
            limitations.append(
                "Some supplied curriculum records are duplicated, truncated, or "
                "garbled, so their exact wording is unreliable."
            )
        return text, limitations

    if len(lines) <= 1:
        names = [
            name
            for name in dict.fromkeys(
                e.name for e in evidence if e.name and _public_name(e.name)
            )
        ][:10]
        if names:
            lines.append("")
            lines.append("Retrieved curriculum records include:")
            lines.extend(f"* {name}" for name in names)
    return "\n".join(lines).strip(), limitations


def _render_outcome_synthesis(
    question: str,
    *,
    outcomes: list[CurriculumEvidence],
    topics: list[CurriculumEvidence],
    grade_label: str | None,
    subject_label: str | None,
    topic_hint: str | None,
) -> tuple[str, bool]:
    units_by_code = _units_by_code(topics)
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for outcome in outcomes:
        label = _group_label(outcome, units_by_code)
        bucket = groups.get(label)
        if bucket is None:
            bucket = {"reliable": [], "damaged": False}
            groups[label] = bucket
            order.append(label)
        content = _strip_leading_code(outcome.content or "")
        damaged = _is_damaged_source(content) or _evidence_marked_imperfect(outcome)
        if damaged:
            bucket["damaged"] = True
            portion = (
                _reliable_portion(content)
                if _is_damaged_source(content)
                else content.strip()
            )
            if portion:
                bucket["reliable"].append(portion)
        elif content.strip():
            bucket["reliable"].append(content.strip())

    focus = _focus_phrase(question, topic_hint)
    lines = _heading_lines(grade_label, subject_label, focus)
    lines.append("")
    where = " ".join(part for part in (grade_label, subject_label) if part) or "supplied"
    topic_name = focus or "this topic"
    lines.append(
        f"Based on the {where} curriculum evidence, pupils are expected to "
        f"learn about {topic_name} through these areas:"
    )
    lines.append("")
    any_damaged = False
    for label in order:
        bucket = groups[label]
        body = _join_claims(bucket["reliable"])
        if bucket["damaged"]:
            any_damaged = True
            if body:
                lines.append(
                    f"* **{label}:** {body} The exact wording of part of this "
                    "evidence is unreliable because the source text is duplicated, "
                    "truncated, or garbled."
                )
            else:
                lines.append(
                    f"* **{label}:** The supplied source text is duplicated, "
                    "truncated, or garbled, so the exact wording is unreliable "
                    "and is not restated."
                )
        elif body:
            lines.append(f"* **{label}:** {body}")
    if any_damaged:
        lines.append("")
        lines.append(
            "The curriculum evidence contains malformed or duplicated source text, "
            "so the exact wording of those records should be verified against the "
            "source before quoting them verbatim."
        )
    return "\n".join(lines).strip(), any_damaged


def _render_identifier_answer(
    question: str,
    *,
    outcomes: list[CurriculumEvidence],
    topics: list[CurriculumEvidence],
    grade_label: str | None,
    subject_label: str | None,
) -> str:
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
            return "\n".join(lines).strip()
        lines.append("The requested unit code from the supplied evidence:")
        for item in matches:
            code = _record_code(item)
            label = _public_name(item.name) or "Curriculum unit"
            if code:
                lines.append(f"* **{code}** — {label}")
        return "\n".join(lines).strip()

    pool = outcomes or topics
    matches = _best_content_matches(question, pool)
    if not matches:
        lines.append(
            "The supplied curriculum evidence does not include an identifier "
            "that matches this question."
        )
        return "\n".join(lines).strip()
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
    return "\n".join(lines).strip()


def _render_topic_catalogue(
    question: str,
    topics: list[CurriculumEvidence],
    subject_label: str | None,
) -> list[str]:
    lines: list[str] = [""]
    list_mode = _is_catalogue_question(question) or len(topics) > 1
    if list_mode:
        lines.append("The MBSSE curriculum includes these units/topics:")
        seen: set[str] = set()
        for topic in topics[:40]:
            name = _public_name(topic.name)
            if not name or name in seen:
                continue
            seen.add(name)
            lines.append(f"* {name}")
        return lines
    topic = topics[0]
    name = _public_name(topic.name) or "this topic"
    lines.append(
        f"The MBSSE curriculum includes **{name}**"
        + (f" under {subject_label}." if subject_label else ".")
    )
    if topic.content and topic.content != topic.name and _public_name(topic.content):
        lines.append("")
        lines.append(_strip_leading_code(topic.content))
    return lines


def _heading_lines(
    grade_label: str | None,
    subject_label: str | None,
    focus: str | None,
) -> list[str]:
    parts = [part for part in (grade_label, subject_label) if part]
    title = " ".join(parts) if parts else "Curriculum evidence"
    if focus:
        title = f"{title} — {focus}"
    return [f"**{title}**"]


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


def _join_claims(parts: list[str]) -> str:
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = " ".join(part.split()).strip(" .;")
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    if not cleaned:
        return ""
    rendered: list[str] = []
    for index, text in enumerate(cleaned):
        if index == 0:
            rendered.append(text[:1].upper() + text[1:])
        else:
            rendered.append(text[:1].lower() + text[1:])
    return "; ".join(rendered) + "."


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
