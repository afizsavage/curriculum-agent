#!/usr/bin/env python3
"""Activate the trusted V2.13A–C corpus for V2.13D production shadow.

Ingests BENCHMARK_SOURCES into data/documents, rebuilds data/document_index,
verifies provenance, and reclassifies pre-corpus Phase 1 rows as
DOCUMENT_CORPUS_UNAVAILABLE.

Does not change sample rate or enable user-facing document retrieval.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STORE = ROOT / "data" / "documents"
INDEX = ROOT / "data" / "document_index"
MANIFEST = ROOT / "data" / "diagnostics" / "v213d_corpus_activation.json"
JSONL = ROOT / "data" / "diagnostics" / "v213d_shadow.jsonl"
EXPECTED_HASHES = {
    "bec-framework-2020": (
        "26409f8e53267603f3b446ad19ef422cff73f7116b74661daa09c77459fb08a4"
    ),
    "math-primary-guidance": (
        "3710ff81811f0fd2c436d89db7d43d9a03bdac5ed4e4e25712a0dde2c1733d70"
    ),
    "science-guidance": (
        "feef53e14b590cbd983834cd0c144d2060f88ae68f493fe20b536350df3fea13"
    ),
}


def _fixture_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def activate() -> dict:
    from app.agent.v213_document_store import DocumentStore
    from app.agent.v213_experiment import BENCHMARK_SOURCES, benchmark_fixture_path
    from app.agent.v213b_experiment import ingest_benchmark_corpus
    from app.agent.v213b_embeddings import FeatureHashEmbeddingProvider
    from app.agent.v213b_semantic_retrieval import HybridDocumentRetrievalService
    from app.agent.v213b_vector_index import PassageVectorIndex
    from app.agent.v213d_shadow import production_corpus_status
    from app.config import Settings

    provenance = []
    discrepancies = []
    for spec in BENCHMARK_SOURCES:
        fixture = benchmark_fixture_path(spec, project_root=ROOT)
        if not fixture.is_file():
            discrepancies.append(
                {
                    "source_id": spec["id"],
                    "issue": "fixture_missing",
                    "path": str(fixture),
                }
            )
            continue
        digest = _fixture_sha256(fixture)
        expected = EXPECTED_HASHES.get(spec["id"])
        if expected and digest != expected:
            discrepancies.append(
                {
                    "source_id": spec["id"],
                    "issue": "hash_mismatch",
                    "expected": expected,
                    "actual": digest,
                }
            )
        provenance.append(
            {
                "document_name": spec["name"],
                "source_id": spec["id"],
                "source_url": spec["document_url"],
                "local_source_path": str(fixture.relative_to(ROOT)),
                "sha256": digest,
                "expected_sha256": expected,
                "version": spec.get("version"),
                "verification_status": spec.get("verification_status"),
                "structure_hints": spec.get("structure_hints") or {},
            }
        )

    if discrepancies:
        raise SystemExit(
            "Corpus discrepancy; refusing activation:\n"
            + json.dumps(discrepancies, indent=2)
        )

    corpus = ingest_benchmark_corpus(store_root=STORE, index_root=INDEX)
    store = DocumentStore(root=STORE)
    provider = FeatureHashEmbeddingProvider()
    index = PassageVectorIndex(root=INDEX, store=store, provider=provider)
    service = HybridDocumentRetrievalService(
        store=store,
        provider=provider,
        index=index,
        settings=Settings(_env_file=None),
    )
    probe = service.search(
        query="purpose of mathematics education",
        variant="context_hybrid",
        subject="MATHEMATICS",
        limit=3,
    )

    # Provenance integrity across all passages
    orphaned = 0
    missing_prov = 0
    passage_count = 0
    hierarchy = {"with_grade": 0, "with_subject": 0, "with_topic": 0}
    for doc_dir in sorted(STORE.glob("doc-*")):
        passages_path = doc_dir / "passages.json"
        meta_path = doc_dir / "metadata.json"
        if not passages_path.is_file() or not meta_path.is_file():
            orphaned += 1
            continue
        meta = json.loads(meta_path.read_text())
        passages = json.loads(passages_path.read_text())
        passage_count += len(passages)
        for p in passages:
            if not (
                p.get("document_id")
                and p.get("source_url")
                and p.get("page_number") is not None
                and (p.get("content_hash") or meta.get("content_hash"))
            ):
                missing_prov += 1
            if p.get("grade"):
                hierarchy["with_grade"] += 1
            if p.get("subject"):
                hierarchy["with_subject"] += 1
            if p.get("topic"):
                hierarchy["with_topic"] += 1

    status = production_corpus_status(STORE, INDEX)
    report = {
        "activation": "v213d_phase1c_corpus",
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_family": "V2.13A–C BENCHMARK_SOURCES",
        "store_root": str(STORE),
        "index_root": str(INDEX),
        "provenance": provenance,
        "discrepancies": discrepancies,
        "ingestion": corpus,
        "counts": {
            "documents": status["document_count"],
            "passages": passage_count,
            "index_entries": status["index_entry_count"],
            "orphaned_document_dirs": orphaned,
            "passages_missing_provenance": missing_prov,
        },
        "hierarchy": hierarchy,
        "corpus_available": status["available"],
        "probe_context_hybrid_hits": len(probe.hits),
        "expected_hashes_matched": True,
    }
    if (
        report["counts"]["documents"] < 3
        or report["counts"]["passages"] < 1
        or report["counts"]["index_entries"] < 1
        or missing_prov
        or orphaned
        or len(probe.hits) < 1
    ):
        report["activation_ok"] = False
    else:
        report["activation_ok"] = True

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(report, indent=2))

    reclassified = reclassify_pre_corpus_jsonl(JSONL)
    report["pre_corpus_rows_reclassified"] = reclassified
    MANIFEST.write_text(json.dumps(report, indent=2))
    return report


def reclassify_pre_corpus_jsonl(path: Path) -> int:
    if not path.exists() or not path.read_text().strip():
        return 0
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    changed = 0
    out = []
    for line in lines:
        row = json.loads(line)
        shadow = row.get("shadow") or {}
        comparison = row.get("comparison") or {}
        docs = int(shadow.get("document_evidence_count") or 0)
        label = comparison.get("classification")
        # Pre-corpus infrastructure failures only.
        if docs == 0 and label in {
            "DOCUMENT_RETRIEVAL_FAILURE",
            "DOCUMENT_CORPUS_UNAVAILABLE",
            None,
        }:
            if not row.get("corpus_epoch"):
                row["corpus_epoch"] = "pre_corpus"
            row["shadow"] = {
                **shadow,
                "corpus_available": False,
                "retrieval_failure_kind": "corpus_unavailable",
            }
            row["comparison"] = {
                **comparison,
                "classification": "DOCUMENT_CORPUS_UNAVAILABLE",
                "primary_category": "DOCUMENT_CORPUS_UNAVAILABLE",
                "secondary_categories": comparison.get("secondary_categories") or [],
                "improved": False,
                "regressed": False,
                "newly_recoverable": False,
                "control_correct_shadow_worse": False,
            }
            changed += 1
        elif not row.get("corpus_epoch"):
            # Leave other rows alone but mark epoch if already post?
            pass
        out.append(json.dumps(row))
    path.write_text("\n".join(out) + ("\n" if out else ""))
    return changed


def main() -> int:
    report = activate()
    print(json.dumps(report, indent=2))
    return 0 if report.get("activation_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
