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

    # If tool results already present, optionally follow up search → get_topic
    tool_messages = [m for m in messages if m.role == "tool"]
    if tool_messages:
        return _follow_up(tool_messages[-1].content, available)

    filters = extract_filters_from_question(question)
    grade = filters.get("grade")
    subject = filters.get("subject")
    topic = filters.get("topic")
    level = filters.get("level")
    lower = question.lower()

    def make(name: str, arguments: dict[str, Any]) -> ToolCallRequest:
        return ToolCallRequest(id=str(uuid4()), name=name, arguments=arguments)

    if "learning objective" in lower or "objectives" in lower or "what should" in lower:
        if "get_learning_objectives" in available:
            args: dict[str, Any] = {}
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


def _follow_up(tool_content: str, available: set[str | None]) -> list[ToolCallRequest]:
    """After search_curriculum, optionally fetch the first topic."""
    import json

    try:
        payload = json.loads(tool_content)
    except json.JSONDecodeError:
        return []
    if payload.get("tool") != "search_curriculum":
        return []
    sample = payload.get("sample_evidence") or []
    if not sample or "get_topic" not in available:
        return []
    first = sample[0]
    entity_id = first.get("entity_id")
    if not entity_id:
        return []
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
