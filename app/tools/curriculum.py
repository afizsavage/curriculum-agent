"""Curriculum retrieval tools backed by the Curriculum Structure API."""

from __future__ import annotations

import time
from typing import Any, Optional

from app.curriculum.client import CurriculumAPIClient
from app.curriculum.codes import (
    default_curriculum_for_grade,
    infer_level,
    normalize_classification,
    normalize_grade_code,
    normalize_subject_code,
)
from app.curriculum.errors import (
    CurriculumAPIError,
    CurriculumInvalidQueryError,
    CurriculumNotFoundError,
    CurriculumTimeoutError,
)
from app.curriculum.evidence import CurriculumEvidence
from app.curriculum.sss_stream_intent import (
    FOCUS_COVERAGE,
    FOCUS_MEMBERSHIP,
    FOCUS_OUTCOMES,
    FOCUS_STREAMS,
    FOCUS_SUBJECTS,
    FOCUS_TOPICS,
    TOOL_GET_SSS_STREAM_SUBJECTS,
    match_sss_stream,
    match_stream_subject,
)
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

    @staticmethod
    def _outcome_matches(outcome: dict[str, Any], query: str) -> bool:
        return match_query(
            {
                "name": outcome.get("code") or "",
                "description": outcome.get("description")
                or outcome.get("text")
                or outcome.get("statement")
                or "",
                "content_type": "LEARNING_OUTCOME",
            },
            query,
        )

    def _search_syllabus_tree(
        self,
        *,
        tree: list[dict[str, Any]] | dict[str, Any],
        query: str,
        grade: str | None,
        subject: str | None,
        level: str | None,
        source_reference: str = "grade_curriculum.content",
    ) -> tuple[list[Any], list[CurriculumEvidence]]:
        hits_by_id: dict[str, Any] = {}
        evidence: list[CurriculumEvidence] = []

        for node, parent_id in iter_content_nodes(tree):
            node_matched = match_query(node, query)
            matched_outcomes = [
                outcome
                for outcome in (node.get("learning_outcomes") or [])
                if isinstance(outcome, dict) and self._outcome_matches(outcome, query)
            ]
            if not node_matched and not matched_outcomes:
                continue

            hit = node_to_search_hit(
                node,
                parent_id=parent_id,
                grade=grade,
                subject=subject,
                level=level,
            )
            if matched_outcomes:
                hit.metadata["matched_learning_outcomes"] = [
                    {
                        "id": outcome.get("id"),
                        "code": outcome.get("code"),
                        "description": outcome.get("description")
                        or outcome.get("text")
                        or outcome.get("statement"),
                        "curriculum_content_id": str(node.get("id"))
                        if node.get("id") is not None
                        else None,
                    }
                    for outcome in matched_outcomes
                ]
                for outcome in matched_outcomes:
                    evidence.append(
                        evidence_from_outcome(
                            outcome,
                            topic_id=str(node.get("id")) if node.get("id") else None,
                            grade=grade,
                            subject=subject,
                        )
                    )

            hit_id = str(hit.id)
            if hit_id not in hits_by_id:
                hits_by_id[hit_id] = hit

        hits = list(hits_by_id.values())
        evidence = [
            evidence_from_hit(h, source_reference=source_reference) for h in hits
        ] + evidence
        return hits, evidence

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

    def _grade_id_from_structure(
        self, structure: dict[str, Any], grade_code: str
    ) -> str | None:
        for level_node in structure.get("education_levels") or []:
            for grade in level_node.get("grades") or []:
                if str(grade.get("code", "")).upper() == grade_code:
                    grade_id = grade.get("id")
                    return str(grade_id) if grade_id else None
        return None

    def _subjects_from_grade_subject_items(
        self, items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        subjects: list[dict[str, Any]] = []
        for item in items:
            subject = item.get("subject") if isinstance(item.get("subject"), dict) else {}
            subjects.append(
                {
                    "id": subject.get("id") or item.get("subject_id"),
                    "name": subject.get("name"),
                    "code": subject.get("code"),
                    "classification": item.get("classification"),
                    "status": item.get("status"),
                    "grade_subject_id": item.get("id"),
                    "grade_id": item.get("grade_id"),
                }
            )
        return subjects

    def _list_grade_subjects_for_grade(
        self,
        *,
        curriculum_id: str,
        grade_code: str,
        classification: str | None = None,
        subject_code: str | None = None,
    ) -> list[dict[str, Any]]:
        """Authoritative grade-scoped subject list via GradeSubject rows.

        Classification is enforced here (Structure API filter or local NON_CORE
        exclusion), never deferred to the LLM.
        """
        structure = self.client.get_curriculum_structure(curriculum_id)
        grade_id = self._grade_id_from_structure(structure, grade_code)
        if not grade_id:
            return []

        params: dict[str, Any] = {"grade_id": grade_id, "limit": 200}
        # NON_CORE is agent-side: fetch grade set then exclude CORE.
        api_classification = (
            None if classification in (None, "NON_CORE") else classification
        )
        if api_classification:
            params["classification"] = api_classification

        page = self.client.list_grade_subjects(curriculum_id, **params)
        items = list(page.get("items") or [])
        if classification == "NON_CORE":
            items = [i for i in items if i.get("classification") != "CORE"]
        if subject_code:
            wanted = subject_code.upper()
            filtered = []
            for item in items:
                code = str((item.get("subject") or {}).get("code") or "").upper()
                name = str((item.get("subject") or {}).get("name") or "").lower()
                if code == wanted or wanted in code or wanted.lower() in name:
                    filtered.append(item)
            items = filtered
        return self._subjects_from_grade_subject_items(items)

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

    def _find_grade_curriculum(
        self,
        *,
        curriculum_id: str,
        subject_code: str | None,
        grade_code: str | None,
    ) -> dict[str, Any]:
        """Resolve the grade×subject curriculum row used by the admin UI."""
        page = self.client.list_curriculum_grade_curricula(
            curriculum_id, limit=200
        )
        items = page.get("items") or []
        matches: list[dict[str, Any]] = []
        for item in items:
            grade = item.get("grade") if isinstance(item.get("grade"), dict) else {}
            subject = (
                item.get("subject") if isinstance(item.get("subject"), dict) else {}
            )
            item_grade = str(grade.get("code") or "").upper() or None
            item_subject = str(subject.get("code") or "").upper() or None
            if grade_code and item_grade and item_grade != grade_code:
                continue
            if subject_code and item_subject and item_subject != subject_code:
                continue
            if grade_code and not item_grade:
                continue
            if subject_code and not item_subject:
                continue
            matches.append(item)
        if not matches:
            raise CurriculumNotFoundError(
                "No grade curriculum found for the given subject/grade filters"
            )
        return matches[0]

    def _load_content_tree(
        self,
        *,
        curriculum_id: str,
        subject_code: str | None,
        grade_code: str | None,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Prefer grade-curriculum content (admin UI path); fall back to syllabus."""
        try:
            grade_curriculum = self._find_grade_curriculum(
                curriculum_id=curriculum_id,
                subject_code=subject_code,
                grade_code=grade_code,
            )
            tree = self.client.get_grade_curriculum_content(
                str(grade_curriculum["id"])
            )
            if tree:
                return tree, {
                    "source": "grade_curriculum",
                    "id": str(grade_curriculum["id"]),
                    "source_reference": "grade_curriculum.content",
                }
        except CurriculumAPIError:
            pass

        syllabus = self._find_syllabus(
            subject_code=subject_code,
            grade_code=grade_code,
            curriculum_id=curriculum_id,
        )
        tree = self.client.get_syllabus_content_tree(
            str(syllabus["id"]), grade_code=grade_code
        )
        return tree, {
            "source": "syllabus",
            "id": str(syllabus["id"]),
            "source_reference": "syllabus.content.tree",
        }

    def _topic_from_content_node(
        self,
        node: dict[str, Any],
        *,
        grade_code: str | None,
        subject_code: str | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[CurriculumEvidence]]:
        outcomes = [
            o for o in (node.get("learning_outcomes") or []) if isinstance(o, dict)
        ]
        topic = {
            "id": node.get("id"),
            "name": node.get("name"),
            "code": node.get("code"),
            "description": node.get("description"),
            "content_type": node.get("content_type"),
        }
        evidence = [
            CurriculumEvidence(
                entity_type=str(node.get("content_type") or "topic").lower(),
                entity_id=str(node.get("id")) if node.get("id") else None,
                name=node.get("name"),
                grade=grade_code,
                subject=subject_code,
                topic=node.get("name"),
                content=node.get("description") or node.get("name"),
                metadata={
                    "code": node.get("code"),
                    "content_type": node.get("content_type"),
                },
                source_reference="grade_curriculum.content",
            )
        ]
        evidence.extend(
            evidence_from_outcome(
                o,
                topic_id=str(node.get("id")) if node.get("id") else None,
                grade=grade_code,
                subject=subject_code,
            )
            for o in outcomes
        )
        return topic, outcomes, evidence


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
            tree, source = self._load_content_tree(
                curriculum_id=curriculum_id,
                subject_code=subject_code,
                grade_code=grade_code,
            )
            hits, evidence = self._search_syllabus_tree(
                tree=tree,
                query=query,
                grade=grade_code or kwargs.get("grade"),
                subject=subject_code or kwargs.get("subject"),
                level=level,
                source_reference=str(
                    source.get("source_reference") or "grade_curriculum.content"
                ),
            )
            return ToolResult(
                success=True,
                data={
                    "results": [h.model_dump() for h in hits],
                    "evidence": [e.model_dump() for e in evidence],
                    "count": len(hits),
                    "content_source": source,
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
            "(units/topics and learning outcomes from the grade curriculum "
            "content tree used by the Curriculum Structure admin UI). "
            "Use when the user asks what topics or structure is taught in a "
            "subject at a grade."
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
                "classification": {
                    "type": "string",
                    "description": (
                        "Optional GradeSubject classification constraint for the "
                        "requested grade (CORE, OPTIONAL, ELECTIVE, AVAILABLE, or "
                        "NON_CORE). Enforced in structured retrieval."
                    ),
                },
            },
            "required": ["grade"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            grade_code = normalize_grade_code(kwargs.get("grade"))
            subject_code = normalize_subject_code(kwargs.get("subject"))
            classification = normalize_classification(kwargs.get("classification"))
            if not grade_code:
                raise CurriculumInvalidQueryError("grade is required and must be resolvable")
            level = kwargs.get("level") or infer_level(grade_code)
            curriculum_id, curr_code, version = self._resolve_curriculum(
                grade_code=grade_code
            )

            # Subject listing: authoritative GradeSubject rows for the grade,
            # with optional classification constraint applied in the Structure API.
            if not subject_code:
                subjects: list[dict[str, Any]] = []
                evidence: list[CurriculumEvidence] = []
                try:
                    subjects = self._list_grade_subjects_for_grade(
                        curriculum_id=curriculum_id,
                        grade_code=grade_code,
                        classification=classification,
                    )
                    evidence = [
                        evidence_from_subject(subject, grade=grade_code)
                        for subject in subjects
                    ]
                except CurriculumAPIError:
                    if classification is not None:
                        raise
                    subjects = []
                    evidence = []
                if not subjects and classification is None:
                    # Fallback: unfiltered structure subjects (no classification constraint).
                    structure = self.client.get_curriculum_structure(curriculum_id)
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
                if not subjects and classification is None:
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
                        "classification": classification,
                        "subjects": subjects,
                        "evidence": [e.model_dump() for e in evidence],
                    },
                )

            tree, source = self._load_content_tree(
                curriculum_id=curriculum_id,
                subject_code=subject_code,
                grade_code=grade_code,
            )
            evidence = []
            flat = []
            source_ref = str(
                source.get("source_reference") or "grade_curriculum.content"
            )
            for node, parent_id in iter_content_nodes(tree):
                hit = node_to_search_hit(
                    node,
                    parent_id=parent_id,
                    grade=grade_code,
                    subject=subject_code,
                    level=level,
                )
                flat.append(hit.model_dump())
                evidence.append(
                    evidence_from_hit(hit, source_reference=source_ref)
                )
            return ToolResult(
                success=True,
                data={
                    "curriculum": {
                        "id": curriculum_id,
                        "code": curr_code,
                        "version": version,
                    },
                    "grade_curriculum_id": (
                        source["id"] if source.get("source") == "grade_curriculum" else None
                    ),
                    "syllabus_id": (
                        source["id"] if source.get("source") == "syllabus" else None
                    ),
                    "content_source": source,
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
            if not grade_code:
                raise CurriculumInvalidQueryError(
                    "grade is required to resolve subject classification within a grade"
                )
            curriculum_id, _, _ = self._resolve_curriculum(grade_code=grade_code)

            # Prefer grade-scoped GradeSubject metadata (classification is per grade).
            grade_subjects = self._list_grade_subjects_for_grade(
                curriculum_id=curriculum_id,
                grade_code=grade_code,
                subject_code=subject_code,
            )
            subject: dict[str, Any] | None = (
                grade_subjects[0] if grade_subjects else None
            )

            if subject is None:
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
            detail = dict(subject)
            if subject.get("id") and subject.get("classification") is None:
                try:
                    fetched = self.client.get_subject(str(subject["id"]))
                    detail = {**fetched, **{k: v for k, v in subject.items() if v}}
                    evidence = [evidence_from_subject(detail, grade=grade_code)]
                except CurriculumNotFoundError:
                    pass
            return ToolResult(
                success=True,
                data={
                    "subject": detail,
                    "grade": grade_code,
                    "classification": detail.get("classification"),
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
                # Prefer grade-curriculum / syllabus content via curriculum-context.
                for key in ("curriculum_content_id", "syllabus_content_id"):
                    try:
                        ctx = self.client.get_curriculum_context(**{key: topic_id})
                        auth = ctx.get("authoritative") or {}
                        topic = auth.get("topic") or {"id": topic_id}
                        outcomes = auth.get("learning_outcomes") or []
                        if not outcomes:
                            # Fall back to content node outcomes embedded in detail.
                            try:
                                detail = self.client.get_curriculum_content(str(topic_id))
                                outcomes = detail.get("learning_outcomes") or []
                                topic = {
                                    "id": detail.get("id") or topic_id,
                                    "name": detail.get("name") or topic.get("name"),
                                    "code": detail.get("code") or topic.get("code"),
                                    "description": detail.get("description"),
                                    "content_type": detail.get("content_type"),
                                }
                            except CurriculumAPIError:
                                pass
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
                                o,
                                topic_id=str(topic_id),
                                grade=grade_code,
                                subject=subject_code,
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
                        continue

                try:
                    detail = self.client.get_curriculum_content(str(topic_id))
                    topic, outcomes, evidence = self._topic_from_content_node(
                        detail, grade_code=grade_code, subject_code=subject_code
                    )
                    return ToolResult(
                        success=True,
                        data={
                            "topic": topic,
                            "learning_outcomes": outcomes,
                            "evidence": [e.model_dump() for e in evidence],
                        },
                    )
                except CurriculumNotFoundError:
                    topic = self.client.get_topic(str(topic_id))
                    evidence = [
                        evidence_from_structure_node(
                            topic,
                            entity_type="topic",
                            grade=grade_code,
                            subject=subject_code,
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


class ResolveCurriculumContextTool(CurriculumTool):
    """Preferred structured lookup via V2 GradeCurriculum context resolve."""

    @property
    def name(self) -> str:
        return "resolve_curriculum_context"

    @property
    def description(self) -> str:
        return (
            "Resolve authoritative curriculum context (grade → subject → topic/units "
            "→ learning outcomes) in one call using existing GradeCurriculum "
            "relationships. Prefer this over exploratory search_curriculum / "
            "get_curriculum_structure when grade and subject (and ideally topic) "
            "are known. Does not search syllabus or instructional references. "
            "Returns resolution status resolved|ambiguous|not_found|needs_context."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "grade": {
                    "type": "string",
                    "description": "Grade code or name, e.g. CLASS_4 or Primary 4",
                },
                "subject": {
                    "type": "string",
                    "description": "Subject code or name, e.g. MATHEMATICS",
                },
                "topic": {
                    "type": "string",
                    "description": "Optional topic/unit keyword, e.g. fractions",
                },
                "unit": {
                    "type": "string",
                    "description": "Optional unit code/name to narrow matches",
                },
                "curriculum_code": {
                    "type": "string",
                    "description": "Optional curriculum code, e.g. MBSSE-BEC",
                },
                "version": {
                    "type": "string",
                    "description": "Exact curriculum version, e.g. 2020",
                },
            },
            "required": ["grade"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        started = time.perf_counter()
        try:
            grade_raw = kwargs.get("grade")
            if not grade_raw:
                raise CurriculumInvalidQueryError("grade is required")
            grade_code = normalize_grade_code(grade_raw) or str(grade_raw).strip()
            subject_code = normalize_subject_code(kwargs.get("subject"))
            topic = (kwargs.get("topic") or "").strip() or None
            unit = (kwargs.get("unit") or "").strip() or None
            curriculum_code = (kwargs.get("curriculum_code") or "").strip() or None
            version = (kwargs.get("version") or "").strip() or None

            if not curriculum_code:
                curriculum_code, inferred_version = default_curriculum_for_grade(
                    grade_code
                )
                if not version:
                    version = inferred_version

            params: dict[str, Any] = {
                "grade": grade_code,
                "subject": subject_code or kwargs.get("subject"),
                "topic": topic,
                "unit": unit,
                "curriculum_code": curriculum_code,
                "version": version,
            }
            payload = self.client.resolve_curriculum_context(**params)
            if not isinstance(payload, dict):
                raise CurriculumAPIError("Unexpected resolve_curriculum_context payload")

            resolution = payload.get("resolution") or {}
            status = resolution.get("status") or "not_found"
            curriculum = payload.get("curriculum") or {}
            grade = payload.get("grade") or {}
            subject = payload.get("subject") or {}
            topics = payload.get("topics") or []
            units = payload.get("units") or []
            outcomes = payload.get("learning_outcomes") or []
            candidates = payload.get("candidates") or []

            grade_label = grade.get("code") or grade_code
            subject_label = subject.get("code") or subject_code

            evidence: list[CurriculumEvidence] = []
            for node in list(units) + list(topics):
                if not isinstance(node, dict):
                    continue
                evidence.append(
                    CurriculumEvidence(
                        entity_type=str(
                            node.get("content_type") or "curriculum_content"
                        ).lower(),
                        entity_id=str(node["id"]) if node.get("id") else None,
                        name=node.get("name"),
                        grade=grade_label,
                        subject=subject_label,
                        topic=node.get("name"),
                        content=node.get("name"),
                        metadata={
                            "code": node.get("code"),
                            "content_type": node.get("content_type"),
                            "grade_curriculum_id": payload.get("grade_curriculum_id"),
                            "authority": resolution.get("authority")
                            or "grade_curriculum",
                        },
                        source_reference="v2.curriculum.context.resolve",
                    )
                )
            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    continue
                parent_id = outcome.get("parent_content_id")
                ev = evidence_from_outcome(
                    outcome,
                    topic_id=str(parent_id) if parent_id else None,
                    grade=grade_label,
                    subject=subject_label,
                )
                provenance = outcome.get("provenance") or {}
                if isinstance(provenance, dict) and any(provenance.values()):
                    ev.metadata = {
                        **ev.metadata,
                        "provenance": provenance,
                        "parent_content_code": outcome.get("parent_content_code"),
                        "parent_content_name": outcome.get("parent_content_name"),
                        "evidence_quality": outcome.get("evidence_quality"),
                    }
                    if provenance.get("source_reference"):
                        ev.source_reference = str(provenance["source_reference"])
                else:
                    ev.source_reference = "v2.curriculum.context.resolve"
                evidence.append(ev)

            total_ms = round((time.perf_counter() - started) * 1000, 2)
            observability = {
                "tool": self.name,
                "resolution_status": status,
                "curriculum_id": curriculum.get("id"),
                "grade_id": grade.get("id"),
                "subject_id": subject.get("id"),
                "topic_ids": [t.get("id") for t in topics if isinstance(t, dict)],
                "unit_ids": [u.get("id") for u in units if isinstance(u, dict)],
                "learning_outcome_count": len(outcomes),
                "candidate_count": len(candidates),
                "query_timing_ms": resolution.get("query_timing_ms"),
                "total_tool_latency_ms": total_ms,
                "authority": resolution.get("authority") or "grade_curriculum",
            }
            diag = resolution.get("diagnostics")
            if isinstance(diag, dict):
                observability.update(
                    {
                        "requested_grade": diag.get("requested_grade"),
                        "requested_grade_id": diag.get("requested_grade_id"),
                        "resolved_grade_code": diag.get("resolved_grade_code"),
                        "grade_curriculum_id": diag.get("grade_curriculum_id"),
                        "grade_strategy": diag.get("grade_strategy"),
                    }
                )

            # Ambiguous / needs_context / not_found remain successful tool calls
            # with structured status so the agent can follow up (no silent pick).
            return ToolResult(
                success=True,
                data={
                    "resolution": resolution,
                    "curriculum": curriculum,
                    "education_level": payload.get("education_level"),
                    "grade": grade,
                    "subject": subject,
                    "grade_curriculum_id": payload.get("grade_curriculum_id"),
                    "topics": topics,
                    "units": units,
                    "learning_outcomes": outcomes,
                    "candidates": candidates,
                    "objectives": [
                        {
                            "id": o.get("id"),
                            "text": o.get("description"),
                            "code": o.get("code"),
                            "parent_content_id": o.get("parent_content_id"),
                        }
                        for o in outcomes
                        if isinstance(o, dict)
                    ],
                    "evidence": [e.model_dump() for e in evidence],
                    "observability": observability,
                },
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
            outcomes: list[dict[str, Any]] = []
            try:
                ctx = self.client.get_curriculum_context(
                    curriculum_content_id=topic_id
                )
                outcomes = (ctx.get("authoritative") or {}).get("learning_outcomes") or []
            except CurriculumNotFoundError:
                try:
                    ctx = self.client.get_curriculum_context(
                        syllabus_content_id=topic_id
                    )
                    outcomes = (
                        (ctx.get("authoritative") or {}).get("learning_outcomes") or []
                    )
                except CurriculumNotFoundError:
                    try:
                        page = self.client.get_curriculum_content_learning_outcomes(
                            str(topic_id), limit=200
                        )
                        outcomes = (
                            page.get("items") if isinstance(page, dict) else page
                        ) or []
                    except CurriculumNotFoundError:
                        page = self.client.get_topic_learning_outcomes(str(topic_id))
                        outcomes = (
                            page.get("items") if isinstance(page, dict) else page
                        ) or []

            if not outcomes:
                try:
                    detail = self.client.get_curriculum_content(str(topic_id))
                    outcomes = detail.get("learning_outcomes") or []
                except CurriculumAPIError:
                    pass

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


class GetSSSStreamSubjectsTool(CurriculumTool):
    """SSS streams, the subjects assigned to them, and one grade's syllabus.

    Streams are recorded for Senior Secondary as a whole. A grade syllabus is
    loaded only after the subject is an exact assignment in the named stream.
    """

    @property
    def name(self) -> str:
        return TOOL_GET_SSS_STREAM_SUBJECTS

    @property
    def description(self) -> str:
        return (
            "Look up Senior Secondary (SSS) streams and the subjects assigned "
            "to them. Use for which subjects belong to a named stream, which "
            "streams exist at SSS, and what a subject covers for one SSS grade "
            "in one stream. Pass grade and subject together when the question "
            "names them. Do not use another grade's syllabus for the answer."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "stream_name": {
                    "type": "string",
                    "description": "SSS stream name, for example Sciences & Technologies",
                },
                "grade": {
                    "type": "string",
                    "description": "SSS grade when the question names one, such as SSS1",
                },
                "subject": {
                    "type": "string",
                    "description": "Subject name when the question names one",
                },
                "focus": {
                    "type": "string",
                    "description": (
                        "subjects, coverage, topics, outcomes, membership, or streams"
                    ),
                },
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        focus = str(kwargs.get("focus") or FOCUS_SUBJECTS).strip().lower()
        stream_name = str(kwargs.get("stream_name") or "").strip()
        grade = normalize_grade_code(str(kwargs.get("grade") or "")) or None
        subject_name = str(kwargs.get("subject") or "").strip()
        if focus == FOCUS_STREAMS:
            return self._list_streams()
        if not stream_name:
            return _tool_error(CurriculumInvalidQueryError("stream_name is required"))
        try:
            resolved = self._load_stream(stream_name)
        except CurriculumNotFoundError:
            return _sss_stream_not_found(stream_name)
        except CurriculumAPIError as exc:
            return _tool_error(exc)
        if resolved is None:
            return _sss_stream_not_found(stream_name)
        stream, stream_id, assignments = resolved
        if subject_name:
            return self._subject_context(
                stream=stream,
                stream_id=stream_id,
                assignments=assignments,
                subject_name=subject_name,
                grade=grade,
                focus=focus,
            )
        if grade:
            return self._subjects_for_grade(
                stream=stream,
                stream_id=stream_id,
                assignments=assignments,
                grade=grade,
            )
        return _stream_subject_result(stream, stream_id, assignments)

    def _list_streams(self) -> ToolResult:
        try:
            curriculum_id, _, _ = self._resolve_curriculum(grade_code="SSS_1")
            streams = self._collect_pages(
                lambda **params: self.client.list_sss_streams(curriculum_id, **params)
            )
        except CurriculumAPIError as exc:
            return _tool_error(exc)
        evidence = [
            _stream_evidence(stream, stream_id=str(stream["id"]))
            for stream in streams
            if stream.get("id") and stream.get("name")
        ]
        return ToolResult(
            success=True,
            data={
                "streams": [
                    {"id": item.entity_id, "name": item.name} for item in evidence
                ],
                "evidence": [item.model_dump() for item in evidence],
                "observability": {
                    "sss_stream_resolution": "found" if evidence else "no_subjects",
                    "subject_count": 0,
                    "stream_count": len(evidence),
                    "focus": FOCUS_STREAMS,
                    "grade_specific_streams": False,
                },
            },
        )

    def _load_stream(
        self, stream_name: str
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]]] | None:
        curriculum_id, _, _ = self._resolve_curriculum(grade_code="SSS_1")
        streams = self._collect_pages(
            lambda **params: self.client.list_sss_streams(curriculum_id, **params)
        )
        matched = match_sss_stream(streams, stream_name)
        if matched is None or not matched.get("id"):
            return None
        stream_id = str(matched["id"])
        stream = self.client.get_sss_stream(stream_id)
        assignments = self._collect_pages(
            lambda **params: self.client.list_sss_stream_subjects(stream_id, **params)
        )
        return stream, stream_id, assignments

    def _subjects_for_grade(
        self,
        *,
        stream: dict[str, Any],
        stream_id: str,
        assignments: list[dict[str, Any]],
        grade: str,
    ) -> ToolResult:
        try:
            offered = self._subject_ids_for_grade(grade)
        except CurriculumAPIError as exc:
            return _tool_error(exc)
        selected = [
            row
            for row in assignments
            if _assignment_subject_id(row) in offered
        ]
        result = _stream_subject_result(stream, stream_id, selected, grade=grade)
        result.data["observability"]["grade"] = grade
        result.data["observability"]["focus"] = FOCUS_SUBJECTS
        result.data["observability"]["grade_specific_streams"] = False
        return result

    def _subject_context(
        self,
        *,
        stream: dict[str, Any],
        stream_id: str,
        assignments: list[dict[str, Any]],
        subject_name: str,
        grade: str | None,
        focus: str,
    ) -> ToolResult:
        official_name = str(stream.get("name") or "")
        matched, near, match_kind = match_stream_subject(assignments, subject_name)
        if matched is None:
            return _subject_not_in_stream(
                stream,
                stream_id,
                requested_subject=subject_name,
                near=near,
                grade=grade,
                resolution=(
                    "ambiguous_subject"
                    if match_kind == "ambiguous"
                    else "subject_not_in_stream"
                ),
            )
        subject_row = _subject_from_assignment(
            matched, stream=stream, stream_id=stream_id, grade=grade
        )
        if subject_row is None:
            return _subject_not_in_stream(
                stream,
                stream_id,
                requested_subject=subject_name,
                near=[],
                grade=grade,
            )
        evidence = [
            _stream_evidence(stream, stream_id=stream_id),
            subject_row["evidence"],
        ]
        subject_id = str(subject_row["summary"]["id"])
        resolved_subject = str(subject_row["summary"]["name"])
        resolution = "found"
        source_reference = None
        coverage_expected = None
        grade_curriculum_id = None
        if focus in {FOCUS_COVERAGE, FOCUS_TOPICS, FOCUS_OUTCOMES, FOCUS_MEMBERSHIP} and grade:
            try:
                grade_row = self._grade_curriculum_for(grade, subject_id)
            except CurriculumAPIError as exc:
                return _tool_error(exc)
            if grade_row is None:
                resolution = "grade_content_missing"
            elif focus != FOCUS_MEMBERSHIP:
                source_reference = grade_row.get("source_reference")
                grade_curriculum_id = str(grade_row["id"])
                cited_source = str(source_reference) if source_reference else None
                try:
                    tree = self.client.get_grade_curriculum_content(grade_curriculum_id)
                except CurriculumAPIError as exc:
                    return _tool_error(exc)
                evidence.extend(
                    _content_evidence(
                        tree,
                        grade=grade,
                        subject=resolved_subject,
                        stream_name=official_name,
                        source_reference=cited_source,
                    )
                )
                coverage_expected = coverage_expected_from_tree(
                    tree,
                    grade=grade,
                    subject=resolved_subject,
                    stream_name=official_name,
                    source_reference=cited_source,
                    grade_curriculum_id=grade_curriculum_id,
                )
                resolution = "found"
        observability = {
            "sss_stream_resolution": resolution,
            "subject_count": 1,
            "stream_name": official_name,
            "subject_name": resolved_subject,
            "requested_subject": subject_name,
            "grade": grade,
            "focus": focus,
            "source_reference": source_reference,
            "grade_curriculum_id": grade_curriculum_id,
            "coverage_expected": coverage_expected,
        }
        return ToolResult(
            success=True,
            data={
                "stream": {"id": stream_id, "name": official_name, "code": stream.get("code")},
                "subjects": [subject_row["summary"]],
                "evidence": [item.model_dump() for item in evidence],
                "observability": observability,
            },
        )

    def _subject_ids_for_grade(self, grade: str) -> set[str]:
        found: set[str] = set()
        for row in self._grade_curriculum_rows():
            grade_code = str((row.get("grade") or {}).get("code") or "").upper()
            if grade_code != grade:
                continue
            subject_id = (row.get("subject") or {}).get("id") or row.get("subject_id")
            if subject_id:
                found.add(str(subject_id))
        return found

    def _grade_curriculum_for(
        self, grade: str, subject_id: str
    ) -> dict[str, Any] | None:
        for row in self._grade_curriculum_rows():
            grade_code = str((row.get("grade") or {}).get("code") or "").upper()
            row_subject = (row.get("subject") or {}).get("id") or row.get("subject_id")
            if grade_code == grade and str(row_subject) == subject_id:
                return row
        return None

    def _grade_curriculum_rows(self) -> list[dict[str, Any]]:
        cached = getattr(self, "_grade_curriculum_cache", None)
        if cached is not None:
            return cached
        curriculum_id, _, _ = self._resolve_curriculum(grade_code="SSS_1")
        rows = self._collect_pages(
            lambda **params: self.client.list_curriculum_grade_curricula(
                curriculum_id, **params
            )
        )
        self._grade_curriculum_cache = rows
        return rows

    def _collect_pages(self, fetch: Any) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        offset = 0
        while True:
            try:
                page = fetch(limit=200, offset=offset)
            except CurriculumTimeoutError:
                # Grade-curriculum lists are large. One slow page should not
                # drop a resolved grade, stream, and subject.
                page = fetch(limit=200, offset=offset)
            if not isinstance(page, dict):
                break
            batch = [
                row for row in (page.get("items") or []) if isinstance(row, dict)
            ]
            items.extend(batch)
            total = page.get("total")
            if not batch:
                break
            offset += len(batch)
            if isinstance(total, int) and offset >= total:
                break
            if not isinstance(total, int) and len(batch) < 200:
                break
        return items


def _stream_subject_result(
    stream: dict[str, Any],
    stream_id: str,
    assignments: list[dict[str, Any]],
    *,
    grade: str | None = None,
) -> ToolResult:
    official_name = str(stream.get("name") or "")
    evidence = [_stream_evidence(stream, stream_id=stream_id)]
    subjects: list[dict[str, Any]] = []
    for assignment in sorted(
        assignments,
        key=lambda row: (
            row.get("display_order") if isinstance(row.get("display_order"), int) else 0
        ),
    ):
        subject_row = _subject_from_assignment(
            assignment, stream=stream, stream_id=stream_id, grade=grade
        )
        if subject_row is None:
            continue
        subjects.append(subject_row["summary"])
        evidence.append(subject_row["evidence"])
    resolution = "found" if subjects else "no_subjects"
    return ToolResult(
        success=True,
        data={
            "stream": {
                "id": stream_id,
                "name": official_name,
                "code": stream.get("code"),
            },
            "subjects": subjects,
            "evidence": [item.model_dump() for item in evidence],
            "observability": {
                "sss_stream_resolution": resolution,
                "subject_count": len(subjects),
                "stream_name": official_name,
                "grade": grade,
                "focus": FOCUS_SUBJECTS,
            },
        },
    )


def _subject_not_in_stream(
    stream: dict[str, Any],
    stream_id: str,
    *,
    requested_subject: str,
    near: list[dict[str, Any]],
    grade: str | None,
    resolution: str = "subject_not_in_stream",
) -> ToolResult:
    official_name = str(stream.get("name") or "")
    evidence = [_stream_evidence(stream, stream_id=stream_id)]
    near_names: list[str] = []
    for assignment in near:
        subject_row = _subject_from_assignment(
            assignment, stream=stream, stream_id=stream_id, grade=grade
        )
        if subject_row is None:
            continue
        near_names.append(str(subject_row["summary"]["name"]))
        evidence.append(subject_row["evidence"])
    return ToolResult(
        success=True,
        data={
            "stream": {"id": stream_id, "name": official_name, "code": stream.get("code")},
            "subjects": [],
            "evidence": [item.model_dump() for item in evidence],
            "observability": {
                "sss_stream_resolution": resolution,
                "subject_count": 0,
                "stream_name": official_name,
                "requested_subject": requested_subject,
                "near_subject_names": near_names,
                "grade": grade,
            },
        },
    )


# Rows the coverage answer lists. Sections stay context. Subtopics stay off
# the evidence list so a large syllabus does not crowd the verifier window.
_CONTENT_EVIDENCE_TYPES = {"THEME", "TOPIC", "LEARNING_OUTCOME"}
# The coverage contract requires themes and topics. This census is taken from
# the content tree itself, not from the evidence list later checked.
_COVERAGE_CENSUS_TYPES = {"THEME", "TOPIC"}


def coverage_expected_from_tree(
    tree: list[Any],
    *,
    grade: str,
    subject: str,
    stream_name: str,
    source_reference: str | None,
    grade_curriculum_id: str | None,
) -> dict[str, Any]:
    """In-scope theme and topic ids from the content tree.

    Sections and subtopics are not required coverage rows. Ids are copied
    from the tree nodes. Display names are not used as identity.
    """
    entity_ids: list[str] = []

    def walk(nodes: list[Any]) -> None:
        for node in nodes or []:
            if not isinstance(node, dict):
                continue
            content_type = str(node.get("content_type") or "").upper()
            name = node.get("name") or node.get("statement") or node.get("description")
            node_id = str(node["id"]) if node.get("id") is not None else None
            if content_type in _COVERAGE_CENSUS_TYPES and name and node_id:
                entity_ids.append(node_id)
            walk(node.get("children") or [])

    walk(tree if isinstance(tree, list) else [tree])
    return {
        "grade": grade,
        "subject": subject,
        "stream_name": stream_name,
        "source_reference": str(source_reference) if source_reference else None,
        "grade_curriculum_id": str(grade_curriculum_id) if grade_curriculum_id else None,
        "entity_ids": entity_ids,
    }


def _content_evidence(
    tree: list[Any],
    *,
    grade: str,
    subject: str,
    stream_name: str,
    source_reference: str | None,
) -> list[CurriculumEvidence]:
    rows: list[CurriculumEvidence] = []

    def walk(
        nodes: list[Any],
        theme_name: str | None,
        parent_node: dict[str, Any] | None,
        section: dict[str, Any] | None,
    ) -> None:
        for node in nodes or []:
            if not isinstance(node, dict):
                continue
            content_type = str(node.get("content_type") or "").upper()
            name = node.get("name") or node.get("statement") or node.get("description")
            node_id = str(node["id"]) if node.get("id") is not None else None
            current = {
                "id": node_id,
                "content_type": content_type,
                "name": str(name) if name else None,
            }
            next_section = section
            if content_type == "SECTION" and name:
                next_section = {
                    "id": node_id,
                    "entity_type": "SECTION",
                    "name": str(name),
                }
            next_theme_name = theme_name
            if content_type == "THEME" and name:
                next_theme_name = str(name)
            if content_type in _CONTENT_EVIDENCE_TYPES and name:
                raw_parent_id = node.get("parent_id")
                parent_id = str(raw_parent_id) if raw_parent_id else None
                parent_entity_type = None
                parent_display_name = None
                parent_in_evidence = None
                # Copy the API parent only. Do not invent one from a shared name.
                if parent_id and parent_node and parent_node.get("id") == parent_id:
                    parent_entity_type = parent_node.get("content_type") or None
                    parent_display_name = parent_node.get("name")
                    parent_in_evidence = parent_entity_type in _CONTENT_EVIDENCE_TYPES
                elif parent_id:
                    parent_in_evidence = False
                omitted_children = []
                for child in node.get("children") or []:
                    if not isinstance(child, dict):
                        continue
                    child_type = str(child.get("content_type") or "").upper()
                    if not child_type or child_type in _CONTENT_EVIDENCE_TYPES or child_type == "SECTION":
                        continue
                    child_id = child.get("id")
                    omitted_children.append(
                        {
                            "id": str(child_id) if child_id is not None else None,
                            "content_type": child_type,
                        }
                    )
                metadata: dict[str, Any] = {
                    "source_type": "grade_curriculum_content",
                    "content_type": content_type,
                    "stream_name": stream_name,
                    "parent_name": theme_name,
                    "parent_id": parent_id,
                    "parent_in_evidence": parent_in_evidence,
                }
                if parent_entity_type:
                    metadata["parent_entity_type"] = parent_entity_type
                if parent_display_name:
                    metadata["parent_display_name"] = parent_display_name
                if section and section.get("id"):
                    metadata["section_id"] = section["id"]
                    metadata["section_name"] = section["name"]
                    metadata["section_entity_type"] = section["entity_type"]
                if omitted_children:
                    metadata["omitted_child_count"] = len(omitted_children)
                    metadata["omitted_child_types"] = sorted(
                        {item["content_type"] for item in omitted_children}
                    )
                    metadata["omitted_child_ids"] = [
                        item["id"] for item in omitted_children if item["id"]
                    ]
                rows.append(
                    CurriculumEvidence(
                        entity_type=content_type.lower(),
                        entity_id=node_id,
                        name=str(name),
                        grade=grade,
                        level="senior_secondary",
                        subject=subject,
                        topic=theme_name if content_type != "THEME" else None,
                        content=node.get("description") or str(name),
                        metadata=metadata,
                        source_reference=source_reference or "grade_curriculum.content",
                    )
                )
            walk(node.get("children") or [], next_theme_name, current, next_section)

    walk(tree if isinstance(tree, list) else [tree], None, None, None)
    return rows


def _assignment_subject_id(row: dict[str, Any]) -> str | None:
    subject = row.get("subject") if isinstance(row.get("subject"), dict) else {}
    subject_id = subject.get("id") or row.get("subject_id")
    return str(subject_id) if subject_id else None


def _sss_stream_not_found(stream_name: str) -> ToolResult:
    return ToolResult(
        success=False,
        error=f"SSS stream '{stream_name}' was not found",
        data={
            "error_code": "CURRICULUM_NOT_FOUND",
            "observability": {
                "sss_stream_resolution": "not_found",
                "subject_count": 0,
                "stream_name": stream_name,
            },
        },
    )


def _stream_evidence(stream: dict[str, Any], *, stream_id: str) -> CurriculumEvidence:
    name = stream.get("name")
    return CurriculumEvidence(
        entity_type="sss_stream",
        entity_id=stream_id,
        name=name,
        content=stream.get("description") or name,
        metadata={
            "code": stream.get("code"),
            "source_type": "sss_stream",
            "stream_id": stream_id,
            "stream_name": name,
        },
        source_reference="sss_stream",
    )


def _subject_from_assignment(
    assignment: dict[str, Any],
    *,
    stream: dict[str, Any],
    stream_id: str,
    grade: str | None = None,
) -> dict[str, Any] | None:
    subject = assignment.get("subject") if isinstance(assignment.get("subject"), dict) else {}
    subject_id = subject.get("id") or assignment.get("subject_id")
    name = subject.get("name")
    if not subject_id or not name:
        return None
    stream_name = stream.get("name")
    summary = {
        "id": str(subject_id),
        "name": name,
        "code": subject.get("code"),
        "subject_type": assignment.get("subject_type"),
        "stream_id": stream_id,
        "stream_name": stream_name,
        "grade": grade,
    }
    evidence = CurriculumEvidence(
        entity_type="subject",
        entity_id=str(subject_id),
        name=name,
        grade=grade,
        level="senior_secondary" if grade else None,
        subject=subject.get("code") or name,
        content=name,
        metadata={
            "code": subject.get("code"),
            "source_type": "sss_stream_subject",
            "subject_type": assignment.get("subject_type"),
            "stream_id": stream_id,
            "stream_name": stream_name,
            "assignment_id": (
                str(assignment["id"]) if assignment.get("id") is not None else None
            ),
        },
        source_reference="sss_stream.subjects",
    )
    return {"summary": summary, "evidence": evidence}


def build_curriculum_tools(client: CurriculumAPIClient) -> list[Tool]:
    return [
        ResolveCurriculumContextTool(client),
        SearchCurriculumTool(client),
        GetCurriculumStructureTool(client),
        GetSubjectTool(client),
        GetTopicTool(client),
        GetLearningObjectivesTool(client),
        GetSSSStreamSubjectsTool(client),
    ]
