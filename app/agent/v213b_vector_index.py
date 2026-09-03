"""V2.13B reproducible local passage vector index."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agent.v213_document_contract import DocumentPassage
from app.agent.v213_passage_builder import PassageBuilder
from app.agent.v213_document_store import DocumentStore
from app.agent.v213b_embeddings import EmbeddingProvider, cosine_similarity, embedding_identity


@dataclass
class IndexedPassage:
    passage_id: str
    document_id: str
    source_id: str
    content_hash: str
    document_content_hash: str
    embedding: list[float]
    passage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "document_id": self.document_id,
            "source_id": self.source_id,
            "content_hash": self.content_hash,
            "document_content_hash": self.document_content_hash,
            "embedding": self.embedding,
            "passage": self.passage,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "IndexedPassage":
        return cls(
            passage_id=row["passage_id"],
            document_id=row["document_id"],
            source_id=row["source_id"],
            content_hash=row["content_hash"],
            document_content_hash=row["document_content_hash"],
            embedding=list(row["embedding"]),
            passage=dict(row["passage"]),
        )


@dataclass
class IndexManifest:
    embedding_model: str
    embedding_dimension: int
    indexed_at: str
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "indexed_at": self.indexed_at,
            "documents": dict(self.documents),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "IndexManifest":
        return cls(
            embedding_model=row["embedding_model"],
            embedding_dimension=int(row["embedding_dimension"]),
            indexed_at=row["indexed_at"],
            documents=dict(row.get("documents") or {}),
        )


class PassageVectorIndex:
    """File-backed experimental vector index keyed by embedding model identity."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        store: DocumentStore | None = None,
        builder: PassageBuilder | None = None,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self.store = store or DocumentStore()
        self.builder = builder or PassageBuilder(self.store)
        self.provider = provider
        self.root = root or Path("data/document_index")
        self.root.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, provider: EmbeddingProvider) -> Path:
        slug = provider.model_name.replace("/", "_").replace(" ", "-")
        return self.root / slug

    def _index_path(self, provider: EmbeddingProvider) -> Path:
        return self._model_dir(provider) / "index.json"

    def _manifest_path(self, provider: EmbeddingProvider) -> Path:
        return self._model_dir(provider) / "manifest.json"

    def _document_needs_reindex(
        self,
        *,
        document_id: str,
        document_content_hash: str,
        passage_count: int,
        manifest: IndexManifest,
    ) -> bool:
        row = manifest.documents.get(document_id)
        if not row:
            return True
        return (
            row.get("document_content_hash") != document_content_hash
            or int(row.get("passage_count") or 0) != passage_count
        )

    def build_index(
        self,
        provider: EmbeddingProvider | None = None,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        provider = provider or self.provider
        if provider is None:
            raise ValueError("embedding provider required")

        model_dir = self._model_dir(provider)
        model_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._index_path(provider)
        manifest_path = self._manifest_path(provider)

        existing_rows: dict[str, IndexedPassage] = {}
        manifest = IndexManifest(
            embedding_model=provider.model_name,
            embedding_dimension=provider.dimension,
            indexed_at="",
        )
        if index_path.exists() and manifest_path.exists() and not force:
            manifest = IndexManifest.from_dict(json.loads(manifest_path.read_text()))
            if (
                manifest.embedding_model == provider.model_name
                and manifest.embedding_dimension == provider.dimension
            ):
                for row in json.loads(index_path.read_text()):
                    item = IndexedPassage.from_dict(row)
                    existing_rows[item.passage_id] = item

        rebuilt_documents: list[str] = []
        kept_documents: list[str] = []
        rows: dict[str, IndexedPassage] = {}

        for document_id in sorted(self._list_document_ids()):
            record = self.store.load_record(document_id)
            if not record:
                continue
            passages = self.builder.load_passages(document_id)
            needs = force or self._document_needs_reindex(
                document_id=document_id,
                document_content_hash=record.content_hash,
                passage_count=len(passages),
                manifest=manifest,
            )
            if not needs:
                for passage in passages:
                    kept = existing_rows.get(passage.passage_id)
                    if kept and kept.content_hash == passage.content_hash:
                        rows[passage.passage_id] = kept
                kept_documents.append(document_id)
                continue

            rebuilt_documents.append(document_id)
            texts = [p.text for p in passages]
            embeddings = provider.embed_batch(texts)
            for passage, embedding in zip(passages, embeddings):
                rows[passage.passage_id] = IndexedPassage(
                    passage_id=passage.passage_id,
                    document_id=passage.document_id,
                    source_id=passage.source_id,
                    content_hash=passage.content_hash,
                    document_content_hash=record.content_hash,
                    embedding=embedding,
                    passage=passage.to_dict(),
                )

        manifest = IndexManifest(
            embedding_model=provider.model_name,
            embedding_dimension=provider.dimension,
            indexed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            documents={
                doc_id: {
                    "document_content_hash": self.store.load_record(doc_id).content_hash,
                    "passage_count": len(self.builder.load_passages(doc_id)),
                    "indexed_at": manifest.indexed_at or time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                }
                for doc_id in sorted({row.document_id for row in rows.values()})
            },
        )
        index_path.write_text(
            json.dumps([row.to_dict() for row in rows.values()], indent=2)
        )
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))
        return {
            "embedding_model": provider.model_name,
            "embedding_dimension": provider.dimension,
            "passages_indexed": len(rows),
            "documents_rebuilt": rebuilt_documents,
            "documents_kept": kept_documents,
            "indexed_at": manifest.indexed_at,
        }

    def load_index(self, provider: EmbeddingProvider) -> list[IndexedPassage]:
        index_path = self._index_path(provider)
        if not index_path.exists():
            return []
        return [IndexedPassage.from_dict(row) for row in json.loads(index_path.read_text())]

    def search(
        self,
        *,
        query: str,
        provider: EmbeddingProvider,
        limit: int = 10,
    ) -> list[tuple[float, IndexedPassage]]:
        query_vec = provider.embed_text(query)
        ranked: list[tuple[float, IndexedPassage]] = []
        for row in self.load_index(provider):
            score = cosine_similarity(query_vec, row.embedding)
            if score > 0:
                ranked.append((score, row))
        ranked.sort(key=lambda item: (-item[0], item[1].passage_id))
        return ranked[:limit]

    def _list_document_ids(self) -> list[str]:
        root = self.store.root
        if not root.exists():
            return []
        return sorted(
            p.name for p in root.iterdir() if p.is_dir() and (p / "metadata.json").exists()
        )

    def invalidate_document(self, document_id: str, provider: EmbeddingProvider) -> bool:
        manifest_path = self._manifest_path(provider)
        if not manifest_path.exists():
            return False
        manifest = IndexManifest.from_dict(json.loads(manifest_path.read_text()))
        if document_id not in manifest.documents:
            return False
        del manifest.documents[document_id]
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))
        rows = [
            row
            for row in self.load_index(provider)
            if row.document_id != document_id
        ]
        self._index_path(provider).write_text(
            json.dumps([row.to_dict() for row in rows], indent=2)
        )
        return True


__all__ = ["IndexedPassage", "IndexManifest", "PassageVectorIndex"]
