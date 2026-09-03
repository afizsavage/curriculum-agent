# V2.13B — Hybrid Semantic Document Retrieval

Generated: `2026-09-03T06:41:13.643730+00:00`

**Conclusion: SUPPORTED**

Semantic and context-filtered hybrid retrieval improve document recall while preserving provenance and safety.

## 1. Objective

Evaluate semantic and hybrid document retrieval over the V2.13A substrate while preserving provenance, metadata integrity, and production safety.

## 2. Architecture

```text
USER QUERY → Curriculum Context
  ├─ Structured Retrieval (Curriculum API) — unchanged
  └─ Document Retrieval (V2.13B)
       ├─ Lexical (A)
       ├─ Semantic (B)
       ├─ Hybrid RRF (C)
       └─ Context-filtered Hybrid (D)
            → Document Evidence → V2.9 → V2.11 → Verifier → V2.8 → Routing
```

## 3. Existing infrastructure reused

- `DocumentStore`, `DocumentParser`, `PassageBuilder`
- `DocumentRetrievalService` (Variant A lexical control)
- `CurriculumEvidence` + `merge_evidence_bundles`
- V2.9 normalization and V2.11 metadata guard (integration subset only)

## 4. Embedding architecture

- Model: `feature-hash-v1`
- Dimension: `128`
- Default: deterministic feature-hash provider (no network / no secrets)
- Optional OpenAI-compatible `/embeddings` provider via `v213b_embedding_provider=openai`

## 5. Index architecture

- Documents: **3**
- Passages indexed: **16**
- Local JSON index under `data/diagnostics/v213b_semantic_retrieval/index/`
- Keyed by embedding model + document content hash + passage identity
- Rebuilt when document hash or passage count changes

## 6. Passage strategy

V2.13A page/section passages preserved; no recursive chunking. Anonymous chunks forbidden.

## 7. Retrieval variants

| Variant | Method |
|---------|--------|
| A lexical | V2.13A token overlap (unchanged semantics) |
| B semantic | Cosine similarity over passage embeddings |
| C hybrid | Reciprocal rank fusion (k=60) |
| D context_hybrid | Hybrid + hard metadata filters + soft context boost |

## 8. Context filtering

- **Hard constraints:** grade, subject, curriculum_version when passage metadata is present
- **Soft ranking signals:** topic, unit, heading (boost without eliminating framework passages)
- Unresolved context is marked explicitly; retrieval broadens conservatively

## 9. Ranking strategy

Hybrid uses RRF across lexical and semantic candidate lists; context_hybrid adds soft boosts. No LLM reranker.

## 10. Provenance

Every hit retains source, document, page, URL, content hash, and passage ID.

Provenance complete rates: `{"lexical": 1.0, "semantic": 1.0, "hybrid": 1.0, "context_hybrid": 1.0}`

## 11. Evaluation dataset

- Questions: **10** across narrative, specific fact, structured overlap, cross-context negatives, broad, and safety categories
- Gold passages derived from V2.13A fixture documents (not LLM-invented)

## 12. Retrieval metrics

| Metric | Lexical | Semantic | Hybrid | Context Hybrid |
|--------|--------:|---------:|-------:|---------------:|
| Evidence found | 0.889 | 1.000 | 1.000 | 1.000 |
| Recall@1 | 0.889 | 0.556 | 0.778 | 0.778 |
| Recall@3 | 0.889 | 0.889 | 1.000 | 1.000 |
| Recall@5 | 0.889 | 1.000 | 1.000 | 1.000 |
| Recall@10 | 0.889 | 1.000 | 1.000 | 1.000 |
| MRR | 0.889 | 0.750 | 0.889 | 0.889 |
| Mean latency (ms) | 4.325 | 32.019 | 51.525 | 44.716 |

## 13. Grounding metrics

Integration subset through V2.9 normalization + V2.11 metadata guard (verifier/mapper unchanged; observational only):

```json
[
  {
    "id": "narrative_math_purpose",
    "evidence_count": 5,
    "metadata_valid": true,
    "violations": []
  },
  {
    "id": "math_principles",
    "evidence_count": 5,
    "metadata_valid": true,
    "violations": []
  },
  {
    "id": "primary_math",
    "evidence_count": 5,
    "metadata_valid": true,
    "violations": []
  },
  {
    "id": "money_class4",
    "evidence_count": 5,
    "metadata_valid": true,
    "violations": []
  },
  {
    "id": "science_primary",
    "evidence_count": 5,
    "metadata_valid": true,
    "violations": []
  }
]
```

## 14. Safety results

```json
{
  "lexical": {
    "wrong_subject_retrieval": 0,
    "wrong_grade_retrieval": 0,
    "placeholder_retrieval": 0
  },
  "semantic": {
    "wrong_subject_retrieval": 0,
    "wrong_grade_retrieval": 0,
    "placeholder_retrieval": 0
  },
  "hybrid": {
    "wrong_subject_retrieval": 0,
    "wrong_grade_retrieval": 0,
    "placeholder_retrieval": 0
  },
  "context_hybrid": {
    "wrong_subject_retrieval": 0,
    "wrong_grade_retrieval": 0,
    "placeholder_retrieval": 0
  }
}
```

## 15. Latency

```json
{
  "lexical": 4.325131110413673,
  "semantic": 32.01921833331451,
  "hybrid": 51.52547588901749,
  "context_hybrid": 44.71642955549113
}
```

## 16. Failure analysis

- Semantic-only Recall@1 trails lexical on some keyword-heavy questions (expected for feature-hash embeddings)
- Hybrid/context hybrid recover Recall@3/@5 to 1.0 over the gold set
- Lexical misses one broad/safety probe where evidence_found_rate < 1.0

## 17. Structured/document evidence integration

`hits_to_evidence_bundle` maps into `CurriculumEvidence` with `entity_type=document_passage`. `merge_evidence_bundles` keeps structured + document coexistence.

## 18. Production impact

- LangGraph (`graph.py`) and orchestrator unchanged
- Verifier, V2.11 guard, V2.8 mapper unchanged
- Production retrieval unchanged
- Flags OFF by default: `v213b_semantic_retrieval_experiment=false`, `v213b_retrieval_variant=lexical`

## 19. Architectural recommendation

V2.13C — controlled hybrid retrieval + real curriculum QA evaluation

Do **not** automatically promote V2.13B to production.