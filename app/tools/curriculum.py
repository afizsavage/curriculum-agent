"""Curriculum retrieval tools backed by the Curriculum Structure API."""

from __future__ import annotations

from typing import Any, Optional

from app.curriculum.client import CurriculumAPIClient
from app.curriculum.codes import (
    default_curriculum_for_grade,
    infer_level,
    normalize_grade_code,
    normalize_subject_code,
)
from app.curriculum.errors import (
    CurriculumAPIError,
    CurriculumInvalidQueryError,
    CurriculumNotFoundError,
)
from app.curriculum.evidence import CurriculumEvidence
from app.curriculum.normalize import (
    evidence_from_hit,
    evidence_from_outcome,
    evidence_from_structure_node,
    evidence_from_subject,
    iter_content_nodes,
    match_query,
    node_to_search_hit,
)
from app.tools.base import Tool, ToolResult


def _tool_error(exc: Exception) -> ToolResult:
    code = getattr(exc, "code", "TOOL_FAILURE")
    return ToolResult(
        success=False,
        error=str(exc),
        data={"error_code": code},
    )


class CurriculumTool(Tool):
    def __init__(self, client: CurriculumAPIClient) -> None:
        self.client = client

    def _resolve_curriculum(
        self, *, grade_code: str | None, curriculum_code: str | None = None
    ) -> tuple[str, str, str]:
        code, version = default_curriculum_for_grade(grade_code)
        if curriculum_code:
            code = curriculum_code
        curriculum_id = self.client.resolve_curriculum_id(code=code, version=version)
        if not curriculum_id:
            # retry without version
            curriculum_id = self.client.resolve_curriculum_id(code=code)
        if not curriculum_id:
            raise CurriculumNotFoundError(f"Curriculum '{code}' was not found")
        return curriculum_id, code, version

    def _find_syllabus(
        self,
        *,
        subject_code: str | None,
        grade_code: str | None,
        curriculum_id: str | None = None,
    ) -> dict[str, Any]:
        page = self.client.list_syllabuses(
            subject_code=subject_code,
            grade_code=grade_code,
            curriculum_id=curriculum_id,
            limit=50,
        )
        items = page.get("items") or []
        if not items:
            raise CurriculumNotFoundError(
                "No syllabus found for the given subject/grade filters"
            )
        return items[0]


class SearchCurriculumTool(CurriculumTool):
    @property
    def name(self) -> str:
        return "search_curriculum"

    @property
    def description(self) -> str:
        return (
            "Search MBSSE curriculum content by concept, keyword, or natural-language "
            "description. Optional filters: level, grade, subject. Use when the user "
            "asks to find topics/content related to a concept (e.g. fractions, measurement)."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Concept or keyword to search for",
                },
                "level": {
                    "type": "string",
                    "description": "Education level, e.g. primary",
                },
                "grade": {
                    "type": "string",
                    "description": "Grade label or code, e.g. Primary 4 or CLASS_4",
                },
                "subject": {
                    "type": "string",
                    "description": "Subject name or code, e.g. Mathematics",
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            query = str(kwargs.get("query") or "").strip()
            if not query:
                raise CurriculumInvalidQueryError("query is required")
            grade_code = normalize_grade_code(kwargs.get("grade"))
            subject_code = normalize_subject_code(kwargs.get("subject"))
            level = kwargs.get("level") or infer_level(grade_code)
            curriculum_id, _, _ = self._resolve_curriculum(grade_code=grade_code)
            syllabus = self._find_syllabus(
                subject_code=subject_code,
                grade_code=grade_code,
                curriculum_id=curriculum_id,
            )
            tree = self.client.get_syllabus_content_tree(
                str(syllabus["id"]), grade_code=grade_code
            )
            hits = []
            for node, parent_id in iter_content_nodes(tree):
                if match_query(node, query):
                    hits.append(
                        node_to_search_hit(
                            node,
                            parent_id=parent_id,
                            grade=grade_code or kwargs.get("grade"),
                            subject=subject_code or kwargs.get("subject"),
                            level=level,
                        )
                    )
            evidence = [evidence_from_hit(h) for h in hits]
            return ToolResult(
                success=True,
                data={
                    "results": [h.model_dump() for h in hits],
                    "evidence": [e.model_dump() for e in evidence],
                    "count": len(hits),
                },
            )
        except CurriculumAPIError as exc:
            return _tool_error(exc)


class GetCurriculumStructureTool(CurriculumTool):
    @property
    def name(self) -> str:
        return "get_curriculum_structure"

    @property
    def description(self) -> str:
        return (
            "Retrieve the curriculum hierarchy for a grade/subject "
            "(strands → topics → subtopics). Use when the user asks what topics "
            "or structure is taught in a subject at a grade."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "level": {"type": "string"},
                "grade": {
                    "type": "string",
                    "description": "Grade label or code (required)",
                },
                "subject": {
                    "type": "string",
                    "description": "Subject name or code. Omit to list subjects for the grade.",
                },
            },
            "required": ["grade"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            grade_code = normalize_grade_code(kwargs.get("grade"))
            subject_code = normalize_subject_code(kwargs.get("subject"))
            if not grade_code:
                raise CurriculumInvalidQueryError("grade is required and must be resolvable")
            level = kwargs.get("level") or infer_level(grade_code)
            curriculum_id, curr_code, version = self._resolve_curriculum(
                grade_code=grade_code
            )

            # Subject listing: return subjects from curriculum structure for the grade.
            if not subject_code:
                structure = self.client.get_curriculum_structure(curriculum_id)
                subjects = []
                evidence: list[CurriculumEvidence] = []
                for level_node in structure.get("education_levels") or []:
                    for grade in level_node.get("grades") or []:
                        if str(grade.get("code", "")).upper() != grade_code:
                            continue
                        for subject in grade.get("subjects") or []:
                            subjects.append(
                                {
                                    "id": subject.get("id"),
                                    "name": subject.get("name"),
                                    "code": subject.get("code"),
                                }
                            )
                            evidence.append(
                                evidence_from_subject(subject, grade=grade_code)
                            )
                if not subjects:
                    # Fallback: catalogue subjects on the curriculum
                    page = self.client.list_subjects(curriculum_id, limit=200)
                    for subject in page.get("items") or []:
                        subjects.append(
                            {
                                "id": subject.get("id"),
                                "name": subject.get("name"),
                                "code": subject.get("code"),
                            }
                        )
                        evidence.append(
                            evidence_from_subject(subject, grade=grade_code)
                        )
                return ToolResult(
                    success=True,
                    data={
                        "curriculum": {
                            "id": curriculum_id,
                            "code": curr_code,
                            "version": version,
                        },
                        "grade": grade_code,
                        "level": level,
                        "subjects": subjects,
                        "evidence": [e.model_dump() for e in evidence],
                    },
                )

            syllabus = self._find_syllabus(
                subject_code=subject_code,
                grade_code=grade_code,
                curriculum_id=curriculum_id,
            )
            tree = self.client.get_syllabus_content_tree(
                str(syllabus["id"]), grade_code=grade_code
            )
            evidence = []
            flat = []
            for node, parent_id in iter_content_nodes(tree):
                hit = node_to_search_hit(
                    node,
                    parent_id=parent_id,
                    grade=grade_code,
                    subject=subject_code,
                    level=level,
                )
                flat.append(hit.model_dump())
                evidence.append(evidence_from_hit(hit))
            return ToolResult(
                success=True,
                data={
                    "curriculum": {
                        "id": curriculum_id,
                        "code": curr_code,
                        "version": version,
                    },
                    "syllabus_id": str(syllabus["id"]),
                    "grade": grade_code,
                    "subject": subject_code,
                    "level": level,
                    "hierarchy": tree,
                    "nodes": flat,
                    "evidence": [e.model_dump() for e in evidence],
                },
            )
        except CurriculumAPIError as exc:
            return _tool_error(exc)


class GetSubjectTool(CurriculumTool):
    @property
    def name(self) -> str:
        return "get_subject"

    @property
    def description(self) -> str:
        return (
            "Retrieve information about a subject within a grade, including identity, "
            "metadata, and related syllabus structure when available. Use for questions "
            "about a subject itself (e.g. what is Primary 4 Mathematics)."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "grade": {"type": "string"},
                "subject": {"type": "string"},
            },
            "required": ["grade", "subject"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            grade_code = normalize_grade_code(kwargs.get("grade"))
            subject_code = normalize_subject_code(kwargs.get("subject"))
            if not subject_code:
                raise CurriculumInvalidQueryError("subject is required")
            curriculum_id, _, _ = self._resolve_curriculum(grade_code=grade_code)
            page = self.client.list_subjects(curriculum_id, limit=200)
            items = page.get("items") or []
            subject = next(
                (
                    s
                    for s in items
                    if str(s.get("code", "")).upper() == subject_code
                    or str(s.get("name", "")).lower()
                    == str(kwargs.get("subject") or "").lower()
                ),
                None,
            )
            if subject is None:
                # Fall back via syllabus listing
                syllabus = self._find_syllabus(
                    subject_code=subject_code,
                    grade_code=grade_code,
                    curriculum_id=curriculum_id,
                )
                subject = {
                    "id": syllabus.get("subject_id"),
                    "code": subject_code,
                    "name": kwargs.get("subject"),
                    "syllabus_id": syllabus.get("id"),
                }
            evidence = [evidence_from_subject(subject, grade=grade_code)]
            detail = subject
            if subject.get("id"):
                try:
                    detail = self.client.get_subject(str(subject["id"]))
                    evidence = [evidence_from_subject(detail, grade=grade_code)]
                except CurriculumNotFoundError:
                    pass
            return ToolResult(
                success=True,
                data={
                    "subject": detail,
                    "grade": grade_code,
                    "evidence": [e.model_dump() for e in evidence],
                },
            )
        except CurriculumAPIError as exc:
            return _tool_error(exc)


class GetTopicTool(CurriculumTool):
    @property
    def name(self) -> str:
        return "get_topic"

    @property
    def description(self) -> str:
        return (
            "Retrieve the canonical representation of a curriculum topic. Prefer "
            "topic_id (syllabus content UUID or structure topic UUID) when known. "
            "Otherwise provide grade, subject, and topic name."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic_id": {
                    "type": "string",
                    "description": "Preferred canonical topic/content UUID",
                },
                "topic": {"type": "string", "description": "Topic name if id unknown"},
                "grade": {"type": "string"},
                "subject": {"type": "string"},
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            topic_id = kwargs.get("topic_id")
            topic_name = kwargs.get("topic")
            grade_code = normalize_grade_code(kwargs.get("grade"))
            subject_code = normalize_subject_code(kwargs.get("subject"))

            if topic_id:
                # Prefer syllabus content via curriculum-context; fall back to structure topic.
                try:
                    ctx = self.client.get_curriculum_context(
                        syllabus_content_id=topic_id
                    )
                    auth = ctx.get("authoritative") or {}
                    topic = auth.get("topic") or {"id": topic_id}
                    outcomes = auth.get("learning_outcomes") or []
                    evidence = [
                        CurriculumEvidence(
                            entity_type="topic",
                            entity_id=str(topic.get("id") or topic_id),
                            name=topic.get("name"),
                            grade=grade_code,
                            subject=subject_code,
                            topic=topic.get("name"),
                            content=topic.get("description") or topic.get("name"),
                            metadata=topic,
                            source_reference="curriculum-context",
                        )
                    ]
                    evidence.extend(
                        evidence_from_outcome(
                            o, topic_id=str(topic_id), grade=grade_code, subject=subject_code
                        )
                        for o in outcomes
                    )
                    return ToolResult(
                        success=True,
                        data={
                            "topic": topic,
                            "learning_outcomes": outcomes,
                            "context": auth,
                            "evidence": [e.model_dump() for e in evidence],
                        },
                    )
                except CurriculumNotFoundError:
                    topic = self.client.get_topic(str(topic_id))
                    evidence = [
                        evidence_from_structure_node(
                            topic, entity_type="topic", grade=grade_code, subject=subject_code
                        )
                    ]
                    return ToolResult(
                        success=True,
                        data={
                            "topic": topic,
                            "evidence": [e.model_dump() for e in evidence],
                        },
                    )

            if not topic_name:
                raise CurriculumInvalidQueryError(
                    "Provide topic_id or topic name (with grade/subject when possible)"
                )
            # Resolve by searching syllabus tree
            search = SearchCurriculumTool(self.client).execute(
                query=str(topic_name),
                grade=kwargs.get("grade"),
                subject=kwargs.get("subject"),
            )
            if not search.success:
                return search
            results = (search.data or {}).get("results") or []
            if not results:
                raise CurriculumNotFoundError(
                    f"Topic matching '{topic_name}' was not found"
                )
            best = results[0]
            return self.execute(
                topic_id=best["id"],
                grade=kwargs.get("grade"),
                subject=kwargs.get("subject"),
            )
        except CurriculumAPIError as exc:
            return _tool_error(exc)


class GetLearningObjectivesTool(CurriculumTool):
    @property
    def name(self) -> str:
        return "get_learning_objectives"

    @property
    def description(self) -> str:
        return (
            "Retrieve authoritative learning objectives/outcomes for a curriculum topic. "
            "Prefer topic_id. Use when the user asks what learners should know or be able "
            "to do for a topic."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic_id": {
                    "type": "string",
                    "description": "Canonical topic or syllabus content UUID",
                },
                "topic": {"type": "string"},
                "grade": {"type": "string"},
                "subject": {"type": "string"},
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            topic_id = kwargs.get("topic_id")
            if not topic_id:
                # Resolve via get_topic first
                topic_tool = GetTopicTool(self.client)
                resolved = topic_tool.execute(**kwargs)
                if not resolved.success:
                    return resolved
                topic = (resolved.data or {}).get("topic") or {}
                topic_id = topic.get("id")
                if not topic_id:
                    raise CurriculumNotFoundError("Could not resolve topic_id")
                # If get_topic already returned outcomes, use them
                outcomes = (resolved.data or {}).get("learning_outcomes")
                if outcomes is not None:
                    evidence = [
                        evidence_from_outcome(
                            o,
                            topic_id=str(topic_id),
                            grade=normalize_grade_code(kwargs.get("grade")),
                            subject=normalize_subject_code(kwargs.get("subject")),
                        )
                        for o in outcomes
                    ]
                    return ToolResult(
                        success=True,
                        data={
                            "topic_id": str(topic_id),
                            "objectives": [
                                {
                                    "id": o.get("id"),
                                    "text": o.get("description") or o.get("text"),
                                    "sequence": o.get("display_order") or idx + 1,
                                    "code": o.get("code"),
                                }
                                for idx, o in enumerate(outcomes)
                            ],
                            "evidence": [e.model_dump() for e in evidence],
                        },
                    )

            grade_code = normalize_grade_code(kwargs.get("grade"))
            subject_code = normalize_subject_code(kwargs.get("subject"))
            try:
                ctx = self.client.get_curriculum_context(syllabus_content_id=topic_id)
                outcomes = (ctx.get("authoritative") or {}).get("learning_outcomes") or []
            except CurriculumNotFoundError:
                page = self.client.get_topic_learning_outcomes(str(topic_id))
                outcomes = page.get("items") if isinstance(page, dict) else page
                outcomes = outcomes or []

            evidence = [
                evidence_from_outcome(
                    o, topic_id=str(topic_id), grade=grade_code, subject=subject_code
                )
                for o in outcomes
            ]
            return ToolResult(
                success=True,
                data={
                    "topic_id": str(topic_id),
                    "objectives": [
                        {
                            "id": o.get("id"),
                            "text": o.get("description") or o.get("text"),
                            "sequence": o.get("display_order") or idx + 1,
                            "code": o.get("code"),
                        }
                        for idx, o in enumerate(outcomes)
                    ],
                    "evidence": [e.model_dump() for e in evidence],
                },
            )
        except CurriculumAPIError as exc:
            return _tool_error(exc)


def build_curriculum_tools(client: CurriculumAPIClient) -> list[Tool]:
    return [
        SearchCurriculumTool(client),
        GetCurriculumStructureTool(client),
        GetSubjectTool(client),
        GetTopicTool(client),
        GetLearningObjectivesTool(client),
    ]
