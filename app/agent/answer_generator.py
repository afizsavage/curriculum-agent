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

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the MBSSE Curriculum Q&A Agent answer generator.

Your role is to produce grounded answers for questions about the MBSSE curriculum
using ONLY the retrieved curriculum evidence provided in the user message.

Rules:
1. CURRICULUM AUTHORITY: Retrieved MBSSE curriculum evidence is authoritative for
   curriculum-specific claims (topics, learning objectives, grade placement,
   subject structure, progression).
2. NO UNSUPPORTED CLAIMS: Do not invent topics, objectives, grades, subjects,
   strands, or curriculum terminology not supported by the evidence.
3. EVIDENCE LIMITATIONS: If evidence is insufficient, state that clearly. Do not
   fill gaps from general knowledge for MBSSE-specific facts.
4. EXPLANATION VS FACT: You may explain curriculum wording in simpler language,
   but do not change the underlying requirement or add new requirements.
5. EVIDENCE REFERENCES: Every curriculum-specific claim in your answer should
   appear in the evidence array with a valid entity_id from the provided records.
   Never invent entity IDs.
6. CONFIDENCE: Assign high only when exact topic/objective evidence answers the
   question; medium when interpretation is needed; low when evidence is partial.
7. STYLE: Write for teachers and education officers — concise, clear headings,
   bullet points, hierarchy, and learning objectives where relevant.
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
        latency_ms = (time.perf_counter() - started) * 1000

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
            "INSTRUCTIONS\n"
            "Answer the question using ONLY the curriculum evidence above.\n"
            "Do not invent curriculum information.\n"
            "Reference entity_id values from the evidence records in your evidence array.\n"
            "Set limitations when evidence is partial or ambiguous.\n"
            "Use markdown headings and bullet points in the answer field where helpful.\n"
        )
        if state.metadata.get("conservative_regeneration"):
            user_content += (
                "\nCONSERVATIVE REGENERATION (authoritative context already resolved)\n"
                "- Make only claims directly supported by the evidence above.\n"
                "- Preserve source wording for learning outcomes; do not complete "
                "truncated or garbled source text.\n"
                "- If source text is incomplete, state that explicitly in limitations.\n"
                "- Do not use speculative wording (e.g. 'likely', 'probably') or "
                "invent objectives not present in the evidence.\n"
            )
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
        """Deterministic grounded answers for tests without network calls."""
        evidence = state.evidence
        grade_label = _display_grade(state.grade or _first_attr(evidence, "grade"))
        subject_label = _display_subject(
            state.subject or _first_attr(evidence, "subject")
        )

        topics = [
            e
            for e in evidence
            if (e.entity_type or "").lower() in {"topic", "subtopic", "unit", "strand"}
        ]
        subjects = [e for e in evidence if (e.entity_type or "").lower() == "subject"]
        outcomes = [
            e for e in evidence if (e.entity_type or "").lower() == "learning_outcome"
        ]

        lines: list[str] = []
        if grade_label and subject_label:
            lines.append(f"### {grade_label} — {subject_label}")
        elif grade_label:
            lines.append(f"### {grade_label}")

        if topics:
            lines.append("")
            # For structure/topic-list questions, enumerate units/topics.
            q = (state.question or "").lower()
            list_mode = any(
                token in q for token in ("topics", "units", "what is taught", "structure")
            ) or len(topics) > 1
            if list_mode:
                lines.append("The MBSSE curriculum includes these units/topics:")
                seen: set[str] = set()
                for topic in topics[:40]:
                    name = (topic.name or "").strip()
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    code = (topic.metadata or {}).get("code")
                    suffix = f" ({code})" if code else ""
                    lines.append(f"- {name}{suffix}")
            else:
                topic = topics[0]
                lines.append(
                    f"The MBSSE curriculum includes **{topic.name}**"
                    + (f" under {subject_label}." if subject_label else ".")
                )
                if topic.content and topic.content != topic.name:
                    lines.append("")
                    lines.append(topic.content)
        elif subjects:
            names = sorted({s.name for s in subjects if s.name})
            if names:
                lines.append("")
                lines.append("Subjects include:")
                lines.extend(f"- {name}" for name in names)
        elif outcomes:
            lines.append("")
            lines.append("Relevant learning objectives include:")
            for outcome in outcomes[:8]:
                if outcome.content:
                    lines.append(f"- {outcome.content}")

        if not lines:
            names = sorted({e.name for e in evidence if e.name})[:10]
            if names:
                lines.append("")
                lines.append("Retrieved curriculum records include:")
                lines.extend(f"- {name}" for name in names)

        refs = _evidence_refs_from_items(evidence[:8], state.question)
        confidence = (
            AnswerConfidence.HIGH
            if state.evidence_status == EvidenceStatus.FOUND and refs
            else AnswerConfidence.MEDIUM
        )
        limitations: list[str] = []
        if state.evidence_status == EvidenceStatus.PARTIAL:
            limitations.append(
                "Some curriculum API calls failed or returned partial results."
            )

        answer_text = "\n".join(lines).strip() or self.INSUFFICIENT_EVIDENCE_ANSWER
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
