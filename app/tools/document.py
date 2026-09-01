"""V2.13A curriculum document evidence retrieval tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agent.v213_experiment import DocumentEvidencePipeline, v213_document_evidence_enabled
from app.config import Settings, get_settings
from app.curriculum.client import CurriculumAPIClient
from app.tools.base import Tool, ToolResult


class SearchCurriculumDocumentTool(Tool):
    """Read-only lexical search over trusted curriculum source documents."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        pipeline: DocumentEvidencePipeline | None = None,
        client: CurriculumAPIClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.pipeline = pipeline or DocumentEvidencePipeline()
        self.client = client

    @property
    def name(self) -> str:
        return "search_curriculum_document"

    @property
    def description(self) -> str:
        return (
            "Search authoritative curriculum source documents for narrative guidance "
            "not available as structured entities. Returns document passages with provenance."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Lexical search query"},
                "curriculum_id": {"type": "string"},
                "grade": {"type": "string"},
                "subject": {"type": "string"},
                "topic": {"type": "string"},
                "source_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        if not v213_document_evidence_enabled(self.settings):
            return ToolResult(
                success=False,
                error="v213_document_evidence_experiment is disabled",
                data={"error_code": "EXPERIMENT_DISABLED"},
            )
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return ToolResult(success=False, error="query is required")
        try:
            bundle = self.pipeline.search(
                query=query,
                curriculum_id=kwargs.get("curriculum_id"),
                grade=kwargs.get("grade"),
                subject=kwargs.get("subject"),
                topic=kwargs.get("topic"),
                source_id=kwargs.get("source_id"),
                limit=int(kwargs.get("limit") or 5),
            )
            return ToolResult(success=True, data=bundle)
        except Exception as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                data={"error_code": "DOCUMENT_RETRIEVAL_FAILED"},
            )


def build_document_tools(
    *,
    settings: Settings | None = None,
    pipeline: DocumentEvidencePipeline | None = None,
    client: CurriculumAPIClient | None = None,
) -> list[Tool]:
    settings = settings or get_settings()
    if not v213_document_evidence_enabled(settings):
        return []
    return [
        SearchCurriculumDocumentTool(
            settings=settings,
            pipeline=pipeline,
            client=client,
        )
    ]


__all__ = ["SearchCurriculumDocumentTool", "build_document_tools"]
