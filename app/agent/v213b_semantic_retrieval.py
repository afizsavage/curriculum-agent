"""V2.13B hybrid semantic document retrieval service."""

from __future__ import annotations

import time
from typing import Any

from app.agent.v213_document_contract import (
    AssociationMethod,
    DocumentPassage,
    DocumentSearchResult,
)
from app.agent.v213_document_retrieval import DocumentRetrievalService, _lexical_score, _tokenize
from app.agent.v213_document_store import DocumentStore
from app.agent.v213_passage_builder import PassageBuilder
from app.agent.v213b_embeddings import EmbeddingProvider, build_embedding_provider
from app.agent.v213b_retrieval_contract import (
    CurriculumContext,
    HybridRetrievalDiagnostics,
    HybridSearchResult,
    RetrievalVariant,
    RetrievedPassageHit,
    hits_to_provenance,
)
from app.agent.v213b_vector_index import PassageVectorIndex
from app.config import Settings

_RRF_K = 60


def _passage_from_dict(row: dict[str, Any]) -> DocumentPassage:
    return DocumentPassage(
        passage_id=row["passage_id"],
        document_id=row["document_id"],
        source_id=row["source_id"],
        curriculum_id=row.get("curriculum_id"),
        curriculum_version_id=row.get("curriculum_version_id"),
        page_number=int(row["page_number"]),
        section=row.get("section"),
        heading=row.get("heading"),
        text=row["text"],
        source_url=row["source_url"],
        content_hash=row["content_hash"],
        grade=row.get("grade"),
        subject=row.get("subject"),
        unit=row.get("unit"),
        topic=row.get("topic"),
        association_method=AssociationMethod(row.get("association_method", "unresolved")),
        metadata=dict(row.get("metadata") or {}),
    )


def _context_from_kwargs(**kwargs: Any) -> CurriculumContext:
    has_any = any(
        kwargs.get(key)
        for key in (
            "curriculum_id",
            "curriculum_version_id",
            "grade",
            "subject",
            "unit",
            "topic",
        )
    )
    return CurriculumContext(
        curriculum_id=kwargs.get("curriculum_id"),
        curriculum_version_id=kwargs.get("curriculum_version_id"),
        grade=kwargs.get("grade"),
        subject=kwargs.get("subject"),
        unit=kwargs.get("unit"),
        topic=kwargs.get("topic"),
        resolved=bool(has_any),
    )


def _soft_context_boost(passage: DocumentPassage, context: CurriculumContext) -> float:
    boost = 0.0
    if context.unit and passage.unit and passage.unit.lower() == context.unit.lower():
        boost += 0.08
    if context.topic:
        topic = context.topic.lower()
        if passage.topic and passage.topic.lower() == topic:
            boost += 0.12
        elif topic in passage.text.lower():
            boost += 0.05
        elif passage.heading and topic in passage.heading.lower():
            boost += 0.04
    if context.subject and passage.subject == context.subject:
        boost += 0.03
    return boost


def _hard_filter(
    passage: DocumentPassage,
    *,
    context: CurriculumContext,
    strict: bool,
    diagnostics: HybridRetrievalDiagnostics,
) -> bool:
    if context.curriculum_version_id and passage.curriculum_version_id:
        if passage.curriculum_version_id != context.curriculum_version_id:
            diagnostics.rejected_wrong_version += 1
            return False
    if strict:
        if context.grade and passage.grade and passage.grade != context.grade:
            diagnostics.rejected_wrong_grade += 1
            return False
        if context.subject and passage.subject and passage.subject != context.subject:
            diagnostics.rejected_wrong_subject += 1
            return False
    return True


def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float, str]]],
    *,
    limit: int,
) -> list[tuple[str, float, str]]:
    scores: dict[str, float] = {}
    methods: dict[str, str] = {}
    for ranked in ranked_lists:
        for rank, (passage_id, _score, method) in enumerate(ranked, start=1):
            scores[passage_id] = scores.get(passage_id, 0.0) + 1.0 / (_RRF_K + rank)
            methods[passage_id] = method
    merged = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [(pid, score, methods[pid]) for pid, score in merged[:limit]]


class HybridDocumentRetrievalService:
    """Lexical, semantic, hybrid, and context-filtered hybrid retrieval."""

    def __init__(
        self,
        *,
        store: DocumentStore | None = None,
        builder: PassageBuilder | None = None,
        lexical: DocumentRetrievalService | None = None,
        index: PassageVectorIndex | None = None,
        provider: EmbeddingProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.store = store or DocumentStore()
        self.builder = builder or PassageBuilder(self.store)
        self.lexical = lexical or DocumentRetrievalService(self.store, self.builder)
        self.provider = provider or build_embedding_provider(self.settings)
        self.index = index or PassageVectorIndex(
            store=self.store,
            builder=self.builder,
            provider=self.provider,
        )

    def ensure_index(self, *, force: bool = False) -> dict[str, Any]:
        return self.index.build_index(self.provider, force=force)

    def search(
        self,
        *,
        query: str,
        variant: RetrievalVariant | str = RetrievalVariant.LEXICAL,
        curriculum_id: str | None = None,
        curriculum_version_id: str | None = None,
        grade: str | None = None,
        subject: str | None = None,
        unit: str | None = None,
        topic: str | None = None,
        source_id: str | None = None,
        limit: int = 5,
    ) -> HybridSearchResult:
        started = time.perf_counter()
        if isinstance(variant, str):
            variant = RetrievalVariant(variant)
        context = _context_from_kwargs(
            curriculum_id=curriculum_id,
            curriculum_version_id=curriculum_version_id,
            grade=grade,
            subject=subject,
            unit=unit,
            topic=topic,
        )
        diagnostics = HybridRetrievalDiagnostics(
            query=query,
            variant=variant.value,
            filters_applied={
                k: v
                for k, v in {
                    "curriculum_id": curriculum_id,
                    "curriculum_version_id": curriculum_version_id,
                    "grade": grade,
                    "subject": subject,
                    "unit": unit,
                    "topic": topic,
                    "source_id": source_id,
                }.items()
                if v
            },
            unresolved_context=not context.resolved,
            embedding_model=self.provider.model_name,
        )

        if variant == RetrievalVariant.LEXICAL:
            result = self._search_lexical(
                query=query,
                context=context,
                curriculum_id=curriculum_id,
                curriculum_version_id=curriculum_version_id,
                grade=grade,
                subject=subject,
                topic=topic,
                source_id=source_id,
                limit=limit,
                diagnostics=diagnostics,
            )
            result.diagnostics.latency_ms = (time.perf_counter() - started) * 1000
            return result

        self.ensure_index()
        diagnostics.index_passages = len(self.index.load_index(self.provider))

        if variant == RetrievalVariant.SEMANTIC:
            result = self._search_semantic(
                query=query,
                context=context,
                curriculum_id=curriculum_id,
                source_id=source_id,
                limit=limit,
                diagnostics=diagnostics,
                strict=False,
            )
        elif variant == RetrievalVariant.HYBRID:
            result = self._search_hybrid(
                query=query,
                context=context,
                curriculum_id=curriculum_id,
                curriculum_version_id=curriculum_version_id,
                grade=grade,
                subject=subject,
                topic=topic,
                source_id=source_id,
                limit=limit,
                diagnostics=diagnostics,
                strict=False,
            )
        else:
            result = self._search_hybrid(
                query=query,
                context=context,
                curriculum_id=curriculum_id,
                curriculum_version_id=curriculum_version_id,
                grade=grade,
                subject=subject,
                topic=topic,
                unit=unit,
                source_id=source_id,
                limit=limit,
                diagnostics=diagnostics,
                strict=True,
                apply_soft_boost=True,
            )
        result.diagnostics.latency_ms = (time.perf_counter() - started) * 1000
        return result

    def _search_lexical(
        self,
        *,
        query: str,
        context: CurriculumContext,
        curriculum_id: str | None,
        curriculum_version_id: str | None,
        grade: str | None,
        subject: str | None,
        topic: str | None,
        source_id: str | None,
        limit: int,
        diagnostics: HybridRetrievalDiagnostics,
    ) -> HybridSearchResult:
        lexical_result: DocumentSearchResult = self.lexical.search_document_evidence(
            query=query,
            curriculum_id=curriculum_id,
            curriculum_version_id=curriculum_version_id,
            grade=grade,
            subject=subject,
            topic=topic,
            source_id=source_id,
            limit=limit,
        )
        diagnostics.documents_searched = lexical_result.diagnostics.documents_searched
        diagnostics.passages_scanned = lexical_result.diagnostics.passages_scanned
        diagnostics.passages_matched = lexical_result.diagnostics.passages_matched
        diagnostics.rejected_wrong_grade = lexical_result.diagnostics.rejected_wrong_grade
        diagnostics.rejected_wrong_subject = lexical_result.diagnostics.rejected_wrong_subject
        diagnostics.rejected_wrong_source = lexical_result.diagnostics.rejected_wrong_source
        diagnostics.lexical_candidates = len(lexical_result.passages)
        query_tokens = _tokenize(query)
        hits: list[RetrievedPassageHit] = []
        for rank, passage in enumerate(lexical_result.passages, start=1):
            hits.append(
                RetrievedPassageHit(
                    passage=passage,
                    retrieval_method="lexical",
                    retrieval_score=_lexical_score(query_tokens, passage),
                    retrieval_rank=rank,
                    lexical_score=_lexical_score(query_tokens, passage),
                    curriculum_context=context,
                    metadata_valid=True,
                )
            )
        return HybridSearchResult(
            hits=hits,
            diagnostics=diagnostics,
            provenance=hits_to_provenance(hits),
        )

    def _collect_passages(
        self,
        *,
        curriculum_id: str | None,
        source_id: str | None,
        diagnostics: HybridRetrievalDiagnostics,
    ) -> list[DocumentPassage]:
        passages: list[DocumentPassage] = []
        for document_id in self.lexical.list_document_ids():
            record = self.store.load_record(document_id)
            if not record:
                continue
            if curriculum_id and record.curriculum_id and record.curriculum_id != curriculum_id:
                diagnostics.rejected_wrong_source += 1
                continue
            if source_id and record.source_id != source_id:
                diagnostics.rejected_wrong_source += 1
                continue
            diagnostics.documents_searched += 1
            passages.extend(self.builder.load_passages(document_id))
        diagnostics.passages_scanned = len(passages)
        return passages

    def _search_semantic(
        self,
        *,
        query: str,
        context: CurriculumContext,
        curriculum_id: str | None,
        source_id: str | None,
        limit: int,
        diagnostics: HybridRetrievalDiagnostics,
        strict: bool,
    ) -> HybridSearchResult:
        ranked = self.index.search(query=query, provider=self.provider, limit=max(limit * 4, 20))
        diagnostics.semantic_candidates = len(ranked)
        passage_lookup = {
            p.passage_id: p for p in self._collect_passages(
                curriculum_id=curriculum_id,
                source_id=source_id,
                diagnostics=diagnostics,
            )
        }
        hits: list[RetrievedPassageHit] = []
        for semantic_score, indexed in ranked:
            passage = passage_lookup.get(indexed.passage_id)
            if passage is None:
                passage = _passage_from_dict(indexed.passage)
            if not _hard_filter(passage, context=context, strict=strict, diagnostics=diagnostics):
                continue
            diagnostics.passages_matched += 1
            hits.append(
                RetrievedPassageHit(
                    passage=passage,
                    retrieval_method="semantic",
                    retrieval_score=semantic_score,
                    retrieval_rank=len(hits) + 1,
                    semantic_score=semantic_score,
                    curriculum_context=context,
                    metadata_valid=True,
                )
            )
            if len(hits) >= limit:
                break
        return HybridSearchResult(
            hits=hits,
            diagnostics=diagnostics,
            provenance=hits_to_provenance(hits),
        )

    def _search_hybrid(
        self,
        *,
        query: str,
        context: CurriculumContext,
        curriculum_id: str | None,
        curriculum_version_id: str | None,
        grade: str | None,
        subject: str | None,
        topic: str | None,
        unit: str | None = None,
        source_id: str | None,
        limit: int,
        diagnostics: HybridRetrievalDiagnostics,
        strict: bool,
        apply_soft_boost: bool = False,
    ) -> HybridSearchResult:
        query_tokens = _tokenize(query)
        passages = self._collect_passages(
            curriculum_id=curriculum_id,
            source_id=source_id,
            diagnostics=diagnostics,
        )
        lexical_ranked: list[tuple[str, float, str]] = []
        for passage in passages:
            if not _hard_filter(passage, context=context, strict=strict, diagnostics=diagnostics):
                continue
            score = _lexical_score(query_tokens, passage)
            if score <= 0:
                continue
            lexical_ranked.append((passage.passage_id, score, "lexical"))
        lexical_ranked.sort(key=lambda item: (-item[1], item[0]))
        diagnostics.lexical_candidates = len(lexical_ranked)

        semantic_ranked_raw = self.index.search(
            query=query, provider=self.provider, limit=max(limit * 4, 20)
        )
        semantic_ranked: list[tuple[str, float, str]] = []
        passage_lookup = {p.passage_id: p for p in passages}
        for score, indexed in semantic_ranked_raw:
            passage = passage_lookup.get(indexed.passage_id) or _passage_from_dict(
                indexed.passage
            )
            if not _hard_filter(passage, context=context, strict=strict, diagnostics=diagnostics):
                continue
            semantic_ranked.append((passage.passage_id, score, "semantic"))
        diagnostics.semantic_candidates = len(semantic_ranked)

        fused = _reciprocal_rank_fusion(
            [
                [(pid, score, method) for pid, score, method in lexical_ranked[: limit * 4]],
                [(pid, score, method) for pid, score, method in semantic_ranked[: limit * 4]],
            ],
            limit=limit * 2,
        )
        hits: list[RetrievedPassageHit] = []
        for rank, (passage_id, fused_score, method) in enumerate(fused, start=1):
            passage = passage_lookup.get(passage_id)
            if passage is None:
                continue
            lexical_score = next((s for pid, s, _ in lexical_ranked if pid == passage_id), None)
            semantic_score = next((s for pid, s, _ in semantic_ranked if pid == passage_id), None)
            score = fused_score
            boost = 0.0
            if apply_soft_boost:
                boost = _soft_context_boost(passage, context)
                score += boost
            hits.append(
                RetrievedPassageHit(
                    passage=passage,
                    retrieval_method="context_hybrid" if apply_soft_boost else "hybrid",
                    retrieval_score=score,
                    retrieval_rank=rank,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                    context_boost=boost,
                    curriculum_context=context,
                    metadata_valid=True,
                )
            )
            diagnostics.passages_matched += 1
        hits = sorted(hits, key=lambda item: (-item.retrieval_score, item.passage.page_number))[
            :limit
        ]
        hits = [
            RetrievedPassageHit(
                passage=h.passage,
                retrieval_method=h.retrieval_method,
                retrieval_score=h.retrieval_score,
                retrieval_rank=idx,
                lexical_score=h.lexical_score,
                semantic_score=h.semantic_score,
                context_boost=h.context_boost,
                curriculum_context=h.curriculum_context,
                metadata_valid=h.metadata_valid,
            )
            for idx, h in enumerate(hits, start=1)
        ]
        return HybridSearchResult(
            hits=hits,
            diagnostics=diagnostics,
            provenance=hits_to_provenance(hits),
        )


def v213b_semantic_retrieval_enabled(settings: Settings) -> bool:
    return bool(getattr(settings, "v213b_semantic_retrieval_experiment", False))


def resolved_retrieval_variant(settings: Settings) -> RetrievalVariant:
    raw = getattr(settings, "v213b_retrieval_variant", "lexical")
    try:
        return RetrievalVariant(str(raw))
    except ValueError:
        return RetrievalVariant.LEXICAL


__all__ = [
    "HybridDocumentRetrievalService",
    "resolved_retrieval_variant",
    "v213b_semantic_retrieval_enabled",
]
