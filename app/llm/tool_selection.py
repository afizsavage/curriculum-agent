"""Heuristic tool selection for the stub LLM (tests / offline)."""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from app.curriculum.codes import extract_filters_from_question, normalize_grade_code, normalize_subject_code
from app.llm.base import LLMMessage, ToolCallRequest


def select_tool_calls(
    messages: list[LLMMessage],
    tools: list[dict[str, Any]],
) -> list[ToolCallRequest]:
    """Choose curriculum tools from the latest user question using heuristics."""
    available = {t.get("name") for t in tools}
    question = ""
    for message in reversed(messages):
        if message.role == "user" and message.content:
            # Prefer the original question line
            for line in message.content.splitlines():
                if line.lower().startswith("question:"):
                    question = line.split(":", 1)[1].strip()
                    break
            if not question:
                question = message.content
            break
    if not question:
        return []

    filters = extract_filters_from_question(question)

    # If tool results already present, optionally follow up search → get_topic
    tool_messages = [m for m in messages if m.role == "tool"]
    if tool_messages:
        return _follow_up(tool_messages[-1].content, available)

    # Verification-guided retrieval (Sprint 4 feedback in user prompt)
    for message in reversed(messages):
        if (
            message.role == "user"
            and message.content
            and "missing evidence" in message.content.lower()
        ):
            targeted = _from_missing_evidence(message.content, available, filters)
            if targeted:
                return targeted
            break

    grade = filters.get("grade")
    subject = filters.get("subject")
    topic = filters.get("topic")
    level = filters.get("level")
    lower = question.lower()

    def make(name: str, arguments: dict[str, Any]) -> ToolCallRequest:
        return ToolCallRequest(id=str(uuid4()), name=name, arguments=arguments)

    if "learning objective" in lower or "objectives" in lower or "what should" in lower:
        if "resolve_curriculum_context" in available and grade:
            args: dict[str, Any] = {"grade": grade}
            if subject:
                args["subject"] = subject
            if topic:
                args["topic"] = topic
            return [make("resolve_curriculum_context", args)]
        if "get_learning_objectives" in available:
            args = {}
            if topic:
                args["topic"] = topic
            if grade:
                args["grade"] = grade
            if subject:
                args["subject"] = subject
            return [make("get_learning_objectives", args)]

    if re.search(r"\bwhat subjects\b|\bsubjects (are )?available\b", lower):
        if "get_curriculum_structure" in available and grade:
            return [
                make(
                    "get_curriculum_structure",
                    {"grade": grade, "level": level},
                )
            ]

    if topic and grade and ("resolve_curriculum_context" in available) and subject and (
        "topic" in lower
        or "fractions" in lower
        or "what is the" in lower
        or re.search(r"\babout\b", lower)
    ) and not re.search(r"\bfind\b|\brelated to\b|\bsearch\b", lower):
        return [
            make(
                "resolve_curriculum_context",
                {"grade": grade, "subject": subject, "topic": topic},
            )
        ]

    if re.search(r"\btopics\b", lower) and grade and subject:
        if "get_curriculum_structure" in available:
            return [
                make(
                    "get_curriculum_structure",
                    {
                        "grade": grade,
                        "subject": subject,
                        "level": level,
                    },
                )
            ]

    if topic and (
        "topic" in lower
        or "about" in lower
        or "fractions" in lower
        or "what is the" in lower
    ):
        if "get_topic" in available:
            return [
                make(
                    "get_topic",
                    {
                        "topic": topic,
                        "grade": grade,
                        "subject": subject,
                    },
                )
            ]

    if re.search(r"\bfind\b|\brelated to\b|\babout\b|\bsearch\b", lower) or topic:
        if "search_curriculum" in available:
            return [
                make(
                    "search_curriculum",
                    {
                        "query": topic or question[:80],
                        "grade": grade,
                        "subject": subject,
                        "level": level,
                    },
                )
            ]

    if grade and subject and "get_subject" in available and "topic" not in lower:
        if re.search(r"\bwhat is\b|\bsubject\b", lower):
            return [make("get_subject", {"grade": grade, "subject": subject})]

    if grade and subject and "get_curriculum_structure" in available:
        return [
            make(
                "get_curriculum_structure",
                {"grade": grade, "subject": subject, "level": level},
            )
        ]

    if "search_curriculum" in available:
        return [
            make(
                "search_curriculum",
                {
                    "query": topic or question[:80],
                    "grade": grade,
                    "subject": subject,
                    "level": level,
                },
            )
        ]
    return []


def _from_missing_evidence(
    content: str,
    available: set[str | None],
    filters: dict[str, Any],
) -> list[ToolCallRequest]:
    """Prefer targeted tools based on verification missing_evidence hints."""
    lower = content.lower()
    grade = filters.get("grade")
    subject = filters.get("subject")
    topic = filters.get("topic")

    # Try to scrape topic/grade/subject from the feedback JSON blob.
    topic_match = re.search(r'"topic"\s*:\s*"([^"]+)"', content)
    grade_match = re.search(r'"grade"\s*:\s*"([^"]+)"', content)
    subject_match = re.search(r'"subject"\s*:\s*"([^"]+)"', content)
    type_match = re.search(r'"type"\s*:\s*"([^"]+)"', content)
    if topic_match:
        topic = topic_match.group(1)
    if grade_match:
        grade = normalize_grade_code(grade_match.group(1)) or grade_match.group(1)
    if subject_match:
        subject = (
            normalize_subject_code(subject_match.group(1)) or subject_match.group(1)
        )
    evidence_type = (type_match.group(1) if type_match else "").lower()

    def make(name: str, arguments: dict[str, Any]) -> ToolCallRequest:
        return ToolCallRequest(id=str(uuid4()), name=name, arguments=arguments)

    if "learning_objective" in evidence_type or "objective" in lower:
        if "resolve_curriculum_context" in available and (topic or grade):
            return [
                make(
                    "resolve_curriculum_context",
                    {"topic": topic, "grade": grade, "subject": subject},
                )
            ]
        if "get_learning_objectives" in available and (topic or grade):
            return [
                make(
                    "get_learning_objectives",
                    {"topic": topic, "grade": grade, "subject": subject},
                )
            ]
    if "grade_placement" in evidence_type or "structure" in evidence_type:
        if "get_curriculum_structure" in available and grade:
            return [
                make(
                    "get_curriculum_structure",
                    {"grade": grade, "subject": subject},
                )
            ]
    if topic and "get_topic" in available:
        return [
            make(
                "get_topic",
                {"topic": topic, "grade": grade, "subject": subject},
            )
        ]
    if "search_curriculum" in available:
        return [
            make(
                "search_curriculum",
                {
                    "query": topic or "curriculum",
                    "grade": grade,
                    "subject": subject,
                },
            )
        ]
    return []


def _follow_up(tool_content: str, available: set[str | None]) -> list[ToolCallRequest]:
    """After search_curriculum, optionally fetch the first topic.

    Also expands toward learning objectives / structure when verification
    feedback is present in earlier user messages (handled by select_tool_calls
    re-entry via tool role only here).
    """
    import json

    try:
        payload = json.loads(tool_content)
    except json.JSONDecodeError:
        return []

    # Targeted expansion after a successful search hit.
    if payload.get("tool") == "search_curriculum":
        sample = payload.get("sample_evidence") or []
        if sample and "get_topic" in available:
            first = sample[0]
            entity_id = first.get("entity_id")
            if entity_id:
                return [
                    ToolCallRequest(
                        id=str(uuid4()),
                        name="get_topic",
                        arguments={
                            "topic_id": entity_id,
                            "grade": first.get("grade"),
                            "subject": first.get("subject"),
                        },
                    )
                ]

    # After get_topic, fetch learning objectives when available.
    if payload.get("tool") == "get_topic" and "get_learning_objectives" in available:
        sample = payload.get("sample_evidence") or []
        if sample:
            first = sample[0]
            entity_id = first.get("entity_id")
            if entity_id:
                return [
                    ToolCallRequest(
                        id=str(uuid4()),
                        name="get_learning_objectives",
                        arguments={
                            "topic_id": entity_id,
                            "grade": first.get("grade"),
                            "subject": first.get("subject"),
                        },
                    )
                ]
    return []
