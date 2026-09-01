"""V2.13A document evidence integration with CurriculumEvidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agent.v213_document_contract import DocumentPassage, DocumentSearchResult
from app.agent.v213_document_parser import DocumentParser
from app.agent.v213_document_store import DocumentStore
from app.agent.v213_passage_builder import PassageBuilder
from app.agent.v213_document_retrieval import DocumentRetrievalService
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence

_EXPERIMENT_NAME = "v2.13a_document_evidence"


def v213_document_evidence_enabled(settings: Settings) -> bool:
    return bool(getattr(settings, "v213_document_evidence_experiment", False))


def document_passage_to_evidence(passage: DocumentPassage) -> CurriculumEvidence:
    """Map a document passage into the shared CurriculumEvidence abstraction."""
    provenance = {
        "source_id": passage.source_id,
        "document_id": passage.document_id,
        "page_number": passage.page_number,
        "section": passage.section,
        "heading": passage.heading,
        "source_url": passage.source_url,
        "content_hash": passage.content_hash,
        "association_method": passage.association_method.value,
        "block_id": passage.metadata.get("block_id"),
    }
    return CurriculumEvidence(
        source="document_evidence",
        entity_type="document_passage",
        entity_id=passage.passage_id,
        name=passage.heading or passage.section,
        grade=passage.grade,
        subject=passage.subject,
        topic=passage.topic,
        content=passage.text,
        metadata={
            "document_id": passage.document_id,
            "source_id": passage.source_id,
            "page_number": passage.page_number,
            "section": passage.section,
            "unit": passage.unit,
            "provenance": provenance,
            "association_method": passage.association_method.value,
            "document_passages": True,
        },
        source_reference="v213.document.search",
    )


def search_result_to_evidence_bundle(
    result: DocumentSearchResult,
) -> dict[str, Any]:
    """Structured evidence bundle for agent tools and diagnostics."""
    evidence = [document_passage_to_evidence(p) for p in result.passages]
    return {
        "experiment": _EXPERIMENT_NAME,
        "document_passages": [e.model_dump() for e in evidence],
        "structured_records": [],
        "source_references": [p.to_dict() for p in result.provenance],
        "retrieval_diagnostics": result.diagnostics.to_dict(),
        "evidence_count": len(evidence),
    }


class DocumentEvidencePipeline:
    """Acquire → parse → passage-build → cache for one trusted source record."""

    def __init__(
        self,
        *,
        store: DocumentStore | None = None,
        parser: DocumentParser | None = None,
        builder: PassageBuilder | None = None,
        retrieval: DocumentRetrievalService | None = None,
    ) -> None:
        self.store = store or DocumentStore()
        self.parser = parser or DocumentParser()
        self.builder = builder or PassageBuilder(self.store)
        self.retrieval = retrieval or DocumentRetrievalService(self.store, self.builder)

    def ingest_source(
        self,
        source_record: dict[str, Any],
        *,
        curriculum_id: str | None = None,
        structure_hints: dict[str, Any] | None = None,
        allow_local_path: str | None = None,
    ) -> dict[str, Any]:
        record = self.store.acquire(
            source_record,
            curriculum_id=curriculum_id,
            allow_local_path=allow_local_path,
        )
        source_path = None
        doc_dir = self.store.document_dir(record.document_id)
        for candidate in (doc_dir / "source.txt", doc_dir / "source.pdf", doc_dir / "source.bin"):
            if candidate.exists():
                source_path = candidate
                break
        if source_path is None:
            raise FileNotFoundError(f"acquired file missing for {record.document_id}")
        parsed = self.parser.parse_file(source_path, content_type=record.content_type)
        passages = self.builder.build_passages(
            parsed=parsed,
            record=record,
            structure_hints=structure_hints,
        )
        self.builder.persist_passages(record.document_id, passages)
        self.store.mark_parsed(
            record.document_id,
            page_count=parsed.page_count,
            passage_count=len(passages),
        )
        return {
            "document_id": record.document_id,
            "source_id": record.source_id,
            "content_hash": record.content_hash,
            "page_count": parsed.page_count,
            "passage_count": len(passages),
            "status": "parsed",
        }

    def search(
        self,
        *,
        query: str,
        curriculum_id: str | None = None,
        curriculum_version_id: str | None = None,
        grade: str | None = None,
        subject: str | None = None,
        topic: str | None = None,
        source_id: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        result = self.retrieval.search_document_evidence(
            query=query,
            curriculum_id=curriculum_id,
            curriculum_version_id=curriculum_version_id,
            grade=grade,
            subject=subject,
            topic=topic,
            source_id=source_id,
            limit=limit,
        )
        return search_result_to_evidence_bundle(result)


def benchmark_fixture_path(spec: dict[str, Any], *, project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[2]
    return root / spec["fixture_path"]


BENCHMARK_SOURCES: list[dict[str, Any]] = [
    {
        "id": "bec-framework-2020",
        "name": "MBSSE Basic Education Curriculum Framework",
        "document_url": "https://example.invalid/bec-framework.pdf",
        "version": "2020",
        "verification_status": "VERIFIED",
        "authority": "MBSSE",
        "fixture_path": "tests/fixtures/v213_documents/bec_framework.txt",
        "structure_hints": {
            "subject": "MATHEMATICS",
            "association_method": "source_metadata",
        },
    },
    {
        "id": "math-primary-guidance",
        "name": "Primary Mathematics Curriculum Guidance",
        "document_url": "https://example.invalid/math-primary.pdf",
        "version": "2020",
        "verification_status": "VERIFIED",
        "authority": "MBSSE",
        "fixture_path": "tests/fixtures/v213_documents/math_primary.txt",
        "structure_hints": {
            "grade": "CLASS_4",
            "subject": "MATHEMATICS",
            "topic": "money",
            "unit": "Everyday Arithmetic Money",
        },
    },
    {
        "id": "science-guidance",
        "name": "Primary Science Curriculum Guidance",
        "document_url": "https://example.invalid/science-primary.pdf",
        "version": "2020",
        "verification_status": "VERIFIED",
        "authority": "MBSSE",
        "fixture_path": "tests/fixtures/v213_documents/science_guidance.txt",
        "structure_hints": {
            "grade": "CLASS_5",
            "subject": "SCIENCE",
        },
    },
]

EVALUATION_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "narrative_math_purpose",
        "question": "What does the MBSSE curriculum say about the purpose of mathematics education?",
        "query": "purpose of mathematics education",
        "expect_source": "bec-framework-2020",
    },
    {
        "id": "math_principles",
        "question": "What principles does the curriculum give for teaching mathematics?",
        "query": "principles teaching mathematics",
        "expect_source": "bec-framework-2020",
    },
    {
        "id": "primary_math",
        "question": "What does the curriculum say about mathematics at the primary level?",
        "query": "mathematics primary level",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
    },
    {
        "id": "money_class4",
        "question": "What does the curriculum say about money in Class 4?",
        "query": "money class 4",
        "grade": "CLASS_4",
        "subject": "MATHEMATICS",
        "topic": "money",
        "expect_source": "math-primary-guidance",
    },
    {
        "id": "science_primary",
        "question": "What does the curriculum say about science inquiry at primary level?",
        "query": "science inquiry primary",
        "grade": "CLASS_5",
        "subject": "SCIENCE",
        "expect_source": "science-guidance",
    },
    {
        "id": "structured_control",
        "question": "What are the learning objectives for C4-U18?",
        "query": "C4-U18 learning objectives",
        "structured_only": True,
    },
]


def interpret_v213a(metrics: dict[str, Any]) -> tuple[str, str, str]:
    acquisition = metrics.get("acquisition", {})
    parsing = metrics.get("parsing", {})
    retrieval = metrics.get("retrieval", {})
    grounding = metrics.get("grounding", {})
    security = metrics.get("security", {})

    if (
        acquisition.get("failed", 0) == 0
        and parsing.get("documents_parsed", 0) >= 3
        and retrieval.get("questions_with_evidence", 0) >= 4
        and grounding.get("wrong_context_accepted", 0) == 0
        and security.get("untrusted_url_blocked", 0) >= 1
    ):
        return (
            "SUPPORTED",
            "Document evidence substrate is reliable with deterministic provenance and lexical retrieval.",
            "V2.13B — semantic/hybrid document retrieval over the validated substrate",
        )
    if acquisition.get("parsed", 0) >= 2 and retrieval.get("questions_with_evidence", 0) >= 2:
        return (
            "PARTIALLY_SUPPORTED",
            "Infrastructure works but parsing, hierarchy, or retrieval quality needs targeted improvement.",
            "Targeted V2.13A hardening before semantic retrieval",
        )
    return (
        "NOT_SUPPORTED",
        "Authoritative document acquisition or retrieval could not be made reliable.",
        "Diagnose acquisition/parser blockers before V2.13B",
    )


__all__ = [
    "BENCHMARK_SOURCES",
    "EVALUATION_QUESTIONS",
    "DocumentEvidencePipeline",
    "benchmark_fixture_path",
    "document_passage_to_evidence",
    "interpret_v213a",
    "search_result_to_evidence_bundle",
    "v213_document_evidence_enabled",
]
