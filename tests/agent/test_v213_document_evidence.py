"""V2.13A curriculum document evidence layer tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.agent.evidence_snapshot import evidence_snapshot_hash
from app.agent.v213_document_contract import (
    AssociationMethod,
    DocumentPassage,
    document_id_for_source,
)
from app.agent.v213_document_parser import DocumentParser
from app.agent.v213_document_store import (
    DocumentHashConflictError,
    DocumentStore,
    UntrustedSourceError,
)
from app.agent.v213_experiment import (
    BENCHMARK_SOURCES,
    DocumentEvidencePipeline,
    document_passage_to_evidence,
    interpret_v213a,
    v213_document_evidence_enabled,
)
from app.agent.v213_passage_builder import PassageBuilder
from app.agent.v213_document_retrieval import DocumentRetrievalService
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, is_document_passage, merge_evidence_bundles
from app.tools.document import SearchCurriculumDocumentTool
from app.tools.registry import build_default_registry

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
FIXTURES = ROOT / "fixtures" / "v213_documents"


def _source(**overrides):
    base = {
        "id": "test-source-1",
        "name": "Test Source",
        "document_url": "https://example.invalid/doc.pdf",
        "version": "2020",
        "verification_status": "VERIFIED",
        "authority": "MBSSE",
    }
    base.update(overrides)
    return base


@pytest.fixture
def temp_store(tmp_path):
    return DocumentStore(root=tmp_path / "documents")


@pytest.fixture
def pipeline(temp_store):
    return DocumentEvidencePipeline(store=temp_store)


def test_source_url_resolution():
    url = DocumentStore.validate_trusted_source(_source())
    assert url.startswith("https://")


def test_trusted_source_validation_rejects_missing_url():
    with pytest.raises(UntrustedSourceError):
        DocumentStore.validate_trusted_source(_source(document_url=""))


def test_trusted_source_validation_rejects_draft():
    with pytest.raises(UntrustedSourceError):
        DocumentStore.validate_trusted_source(_source(verification_status="DRAFT"))


def test_document_acquisition(temp_store):
    record = temp_store.acquire(
        _source(),
        allow_local_path=str(FIXTURES / "bec_framework.txt"),
    )
    assert record.status.value == "acquired"
    assert record.content_hash
    assert temp_store.metadata_path(record.document_id).exists()


def test_content_hashing(temp_store):
    record = temp_store.acquire(
        _source(),
        allow_local_path=str(FIXTURES / "bec_framework.txt"),
    )
    again = temp_store.acquire(
        _source(),
        allow_local_path=str(FIXTURES / "bec_framework.txt"),
    )
    assert record.content_hash == again.content_hash


def test_duplicate_document_detection(temp_store):
    record = temp_store.acquire(
        _source(),
        allow_local_path=str(FIXTURES / "bec_framework.txt"),
    )
    loaded = temp_store.load_record(record.document_id)
    assert loaded is not None
    assert loaded.document_id == record.document_id


def test_changed_document_detection(temp_store, tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("version one")
    temp_store.acquire(_source(), allow_local_path=str(src))
    src.write_text("version two changed")
    with pytest.raises(DocumentHashConflictError):
        temp_store.acquire(_source(), allow_local_path=str(src))


def test_pdf_parsing_text_fixture():
    parser = DocumentParser()
    parsed = parser.parse_text(FIXTURES / "math_primary.txt")
    assert parsed.page_count >= 1
    assert any(p.text for p in parsed.pages)


def test_page_preservation():
    parser = DocumentParser()
    parsed = parser.parse_text(FIXTURES / "bec_framework.txt")
    numbers = [p.page_number for p in parsed.pages]
    assert numbers == sorted(numbers)
    assert 1 in numbers


def test_section_preservation(pipeline, temp_store):
    result = pipeline.ingest_source(
        _source(id="math-src"),
        allow_local_path=str(FIXTURES / "math_primary.txt"),
        structure_hints={"grade": "CLASS_4", "subject": "MATHEMATICS"},
    )
    passages = PassageBuilder(temp_store).load_passages(result["document_id"])
    assert any(p.section or p.heading for p in passages)


def test_provenance_preservation(pipeline, temp_store):
    pipeline.ingest_source(
        _source(id="bec-src"),
        allow_local_path=str(FIXTURES / "bec_framework.txt"),
    )
    result = pipeline.search(query="purpose mathematics education")
    assert result["evidence_count"] > 0
    prov = result["source_references"][0]
    assert prov["page_number"] is not None
    assert prov["source_url"]


def test_hierarchy_association(pipeline, temp_store):
    pipeline.ingest_source(
        _source(id="math-src-2"),
        allow_local_path=str(FIXTURES / "math_primary.txt"),
        structure_hints={"grade": "CLASS_4", "subject": "MATHEMATICS", "topic": "money"},
    )
    passages = PassageBuilder(temp_store).load_passages(
        document_id_for_source(source_id="math-src-2", document_version="2020")
    )
    assert any(p.grade == "CLASS_4" and p.subject == "MATHEMATICS" for p in passages)


def test_unresolved_hierarchy_behavior(pipeline, temp_store):
    pipeline.ingest_source(
        _source(id="generic-src"),
        allow_local_path=str(FIXTURES / "bec_framework.txt"),
    )
    passages = PassageBuilder(temp_store).load_passages(
        document_id_for_source(source_id="generic-src", document_version="2020")
    )
    assert any(
        p.association_method in {
            AssociationMethod.HEADING_MATCH,
            AssociationMethod.SOURCE_METADATA,
            AssociationMethod.UNRESOLVED,
        }
        for p in passages
    )


def test_lexical_retrieval(pipeline, temp_store):
    pipeline.ingest_source(
        _source(id="bec-src-2"),
        allow_local_path=str(FIXTURES / "bec_framework.txt"),
    )
    result = pipeline.search(query="purpose mathematics education")
    assert result["evidence_count"] >= 1


def test_curriculum_filtering(pipeline, temp_store):
    pipeline.ingest_source(
        _source(id="science-src"),
        allow_local_path=str(FIXTURES / "science_guidance.txt"),
        structure_hints={"grade": "CLASS_5", "subject": "SCIENCE"},
    )
    pipeline.ingest_source(
        _source(id="math-src-3"),
        allow_local_path=str(FIXTURES / "math_primary.txt"),
        structure_hints={"grade": "CLASS_4", "subject": "MATHEMATICS"},
    )
    result = pipeline.search(
        query="science inquiry primary",
        grade="CLASS_5",
        subject="SCIENCE",
    )
    assert result["evidence_count"] >= 1


def test_wrong_grade_rejection(pipeline, temp_store):
    pipeline.ingest_source(
        _source(id="math-src-4"),
        allow_local_path=str(FIXTURES / "math_primary.txt"),
        structure_hints={"grade": "CLASS_4", "subject": "MATHEMATICS"},
    )
    service = DocumentRetrievalService(temp_store)
    result = service.search_document_evidence(
        query="money class",
        grade="CLASS_5",
        subject="MATHEMATICS",
    )
    assert result.passages == []
    assert result.diagnostics.rejected_wrong_grade >= 1


def test_wrong_subject_rejection(pipeline, temp_store):
    pipeline.ingest_source(
        _source(id="science-src-2"),
        allow_local_path=str(FIXTURES / "science_guidance.txt"),
        structure_hints={"grade": "CLASS_5", "subject": "SCIENCE"},
    )
    result = pipeline.search(
        query="inquiry observation",
        grade="CLASS_5",
        subject="MATHEMATICS",
    )
    assert result["evidence_count"] == 0


def test_wrong_source_rejection(temp_store):
    service = DocumentRetrievalService(temp_store)
    result = service.search_document_evidence(
        query="anything",
        source_id="nonexistent-source",
    )
    assert result.passages == []


def test_placeholder_handling_document_passage_to_evidence():
    passage = DocumentPassage(
        passage_id="p1",
        document_id="d1",
        source_id="s1",
        curriculum_id=None,
        curriculum_version_id=None,
        page_number=1,
        section=None,
        heading=None,
        text="",
        source_url="https://example.invalid",
        content_hash="abc",
        association_method=AssociationMethod.UNRESOLVED,
    )
    evidence = document_passage_to_evidence(passage)
    assert evidence.entity_type == "document_passage"


def test_evidence_hash_stability(pipeline, temp_store):
    pipeline.ingest_source(
        _source(id="hash-src"),
        allow_local_path=str(FIXTURES / "bec_framework.txt"),
    )
    passages = PassageBuilder(temp_store).load_passages(
        document_id_for_source(source_id="hash-src", document_version="2020")
    )
    evidence = [document_passage_to_evidence(p) for p in passages]
    h1 = evidence_snapshot_hash(evidence)
    h2 = evidence_snapshot_hash(copy.deepcopy(evidence))
    assert h1 == h2


def test_raw_evidence_immutability(pipeline, temp_store):
    pipeline.ingest_source(
        _source(id="imm-src"),
        allow_local_path=str(FIXTURES / "bec_framework.txt"),
    )
    doc_id = document_id_for_source(source_id="imm-src", document_version="2020")
    before = json.loads(temp_store.passages_path(doc_id).read_text())
    snapshot = copy.deepcopy(before)
    snapshot[0]["text"] = "mutated-copy"
    after = json.loads(temp_store.passages_path(doc_id).read_text())
    assert after[0]["text"] == before[0]["text"]
    assert snapshot[0]["text"] != before[0]["text"]


def test_curriculum_evidence_integration(pipeline, temp_store):
    pipeline.ingest_source(
        _source(id="integrate-src"),
        allow_local_path=str(FIXTURES / "math_primary.txt"),
        structure_hints={"grade": "CLASS_4", "subject": "MATHEMATICS"},
    )
    result = pipeline.search(query="money class 4", grade="CLASS_4", subject="MATHEMATICS")
    for row in result["document_passages"]:
        evidence = CurriculumEvidence.model_validate(row)
        assert is_document_passage(evidence)


def test_feature_flag_off_preserves_registry():
    registry = build_default_registry(settings=Settings(v213_document_evidence_experiment=False))
    assert "search_curriculum_document" not in registry.names()


def test_feature_flag_on_registers_tool():
    registry = build_default_registry(settings=Settings(v213_document_evidence_experiment=True))
    assert "search_curriculum_document" in registry.names()


def test_agent_tool_output_contract(pipeline):
    tool = SearchCurriculumDocumentTool(
        settings=Settings(v213_document_evidence_experiment=True),
        pipeline=pipeline,
    )
    for spec in BENCHMARK_SOURCES:
        pipeline.ingest_source(
            {
                "id": spec["id"],
                "name": spec["name"],
                "document_url": spec["document_url"],
                "version": spec["version"],
                "verification_status": spec["verification_status"],
            },
            allow_local_path=str(PROJECT_ROOT / spec["fixture_path"]),
            structure_hints=spec.get("structure_hints"),
        )
    result = tool.execute(query="purpose mathematics education")
    assert result.success
    assert "document_passages" in result.data
    assert "retrieval_diagnostics" in result.data


def test_acquisition_failure_isolation():
    with pytest.raises(UntrustedSourceError):
        DocumentStore.validate_trusted_source(_source(document_url="ftp://bad.example/doc.pdf"))


def test_retrieval_failure_isolation(pipeline):
    tool = SearchCurriculumDocumentTool(
        settings=Settings(v213_document_evidence_experiment=True),
        pipeline=pipeline,
    )
    result = tool.execute(query="nonexistent-zzzz-term-xyzzy")
    assert result.success
    assert result.data["evidence_count"] == 0


def test_v213_flag_helper():
    assert not v213_document_evidence_enabled(Settings())
    assert v213_document_evidence_enabled(Settings(v213_document_evidence_experiment=True))


def test_merge_evidence_bundles():
    structured = [
        CurriculumEvidence(source="curriculum_api", entity_type="learning_outcome", entity_id="lo1")
    ]
    document = [
        document_passage_to_evidence(
            DocumentPassage(
                passage_id="p1",
                document_id="d1",
                source_id="s1",
                curriculum_id=None,
                curriculum_version_id=None,
                page_number=1,
                section=None,
                heading=None,
                text="science inquiry",
                source_url="https://example.invalid",
                content_hash="h",
                association_method=AssociationMethod.UNRESOLVED,
            )
        )
    ]
    merged = merge_evidence_bundles(structured, document)
    assert len(merged) == 2


def test_interpret_v213a_supported_uses_documents_parsed_key():
    conclusion, _, _ = interpret_v213a(
        {
            "acquisition": {"failed": 0, "parsed": 3},
            "parsing": {"documents_parsed": 3},
            "retrieval": {"questions_with_evidence": 5},
            "grounding": {"wrong_context_accepted": 0},
            "security": {"untrusted_url_blocked": 1},
        }
    )
    assert conclusion == "SUPPORTED"
