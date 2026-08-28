"""Curriculum retrieval tools backed by the Curriculum Structure API."""

from __future__ import annotations

import time
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


def build_curriculum_tools(client: CurriculumAPIClient) -> list[Tool]:
    return [
        ResolveCurriculumContextTool(client),
        SearchCurriculumTool(client),
        GetCurriculumStructureTool(client),
        GetSubjectTool(client),
        GetTopicTool(client),
        GetLearningObjectivesTool(client),
    ]
