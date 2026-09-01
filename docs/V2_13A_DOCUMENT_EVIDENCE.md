# V2.13A — Curriculum Document Evidence Layer

Generated: `2026-09-01T11:15:16.668964+00:00`

**Conclusion: SUPPORTED**

Document evidence substrate is reliable with deterministic provenance and lexical retrieval.

## 1. Objective

Establish a trusted, reproducible document-evidence substrate for curriculum PDFs (acquire → parse → passages → lexical retrieval → `CurriculumEvidence`) without vector RAG. Production LangGraph path unchanged; feature flag `v213_document_evidence_experiment=false`.

## 2. Existing architecture findings

See `docs/V2_13A_EXISTING_DOCUMENT_ARCHITECTURE.md`.

Key finding: `CurriculumSource.document_url` and provenance fields already exist in curriculum-structure; V2.13A reuses them via the Structure API client without new v2 routes.

## 3. Source metadata architecture

Agent reads registered sources through `CurriculumAPIClient.get_curriculum_source()` / `list_curriculum_sources()`. Benchmark corpus uses three fixture-backed sources:

| source_id | passages | pages |
|-----------|----------|-------|
| `bec-framework-2020` | 7 | 3 |
| `math-primary-guidance` | 6 | 2 |
| `science-guidance` | 3 | 2 |

## 4. Document acquisition architecture

```json
{
  "attempted": 3,
  "parsed": 3,
  "failed": 0
}
```

Trusted acquisition only: `DocumentStore` validates `verification_status` and rejects arbitrary URLs (`UntrustedSourceError`). Local benchmark fixtures allowed via `allow_local_path`.

## 5. Document storage/versioning

Cached under `data/documents/<document_id>/` with `content_hash` conflict detection (`DocumentHashConflictError`). Immutable `DocumentPassage` records persisted in `passages.json`.

## 6. PDF parsing

```json
{
  "documents_parsed": 3,
  "pages_extracted": 7,
  "passages_built": 16
}
```

Parser: `pypdf` for PDF, plain-text for fixtures. Page boundaries and block IDs preserved.

## 7. Curriculum hierarchy association

```json
{
  "grade_resolved": 9,
  "subject_resolved": 16,
  "unit_resolved": 6,
  "topic_resolved": 6,
  "unresolved": 0
}
```

Association methods: `source_metadata`, `heading_match`, `known_page_range`, `structure_entity`.

## 8. Document evidence schema

`CurriculumEvidence` extended additively: `entity_type=document_passage`, `source=document_evidence`. Contract in `app/agent/v213_document_contract.py`; merge helper `merge_evidence_bundles()`.

## 9. Retrieval API

`DocumentRetrievalService.search()` — lexical token overlap, grade/subject/topic filters, diagnostics (`passages_scanned`, `rejected_wrong_grade`, etc.).

## 10. Agent retrieval tool

`search_curriculum_document` registered only when `v213_document_evidence_experiment=true`. See `docs/V2_13A_DOCUMENT_EVIDENCE_API.md`.

## 11. Provenance model

```json
{
  "passages_with_page": 16,
  "passages_with_source_url": 16,
  "passages_with_hash": 16
}
```

Every passage carries `source_id`, `document_id`, `page_number`, `section`, `heading`, `block_id`, `content_hash`, and `association_method`.

## 12. Evaluation corpus

- Benchmark sources: 3
- Evaluation questions: 6 (1 structured-only control skipped)
- Fixtures: `tests/fixtures/v213_documents/*.txt`

## 13. Retrieval results

Questions with evidence: **5 / 5** (excluding structured-only control).

```json
[
  {
    "id": "narrative_math_purpose",
    "question": "What does the MBSSE curriculum say about the purpose of mathematics education?",
    "evidence_count": 5,
    "skipped": false
  },
  {
    "id": "math_principles",
    "question": "What principles does the curriculum give for teaching mathematics?",
    "evidence_count": 5,
    "skipped": false
  },
  {
    "id": "primary_math",
    "question": "What does the curriculum say about mathematics at the primary level?",
    "evidence_count": 4,
    "skipped": false
  },
  {
    "id": "money_class4",
    "question": "What does the curriculum say about money in Class 4?",
    "evidence_count": 5,
    "skipped": false
  },
  {
    "id": "science_primary",
    "question": "What does the curriculum say about science inquiry at primary level?",
    "evidence_count": 2,
    "skipped": false
  },
  {
    "id": "structured_control",
    "question": "What are the learning objectives for C4-U18?",
    "evidence_count": 0,
    "skipped": true
  }
]
```

Full bundles: `data/diagnostics/v213a_document_evidence.json`.

## 14. Grounding results

```json
{
  "wrong_context_accepted": 0
}
```

Wrong-context acceptance: 0 (grade/subject/topic filters enforced).

## 15. Failure cases

- Untrusted URL blocked (security probe): `ftp://evil.example/x.pdf`
- Structured-only control (`C4-U18 learning objectives`): correctly returns 0 document passages
- No acquisition failures in benchmark run

## 16. Security/trust-boundary results

```json
{
  "untrusted_url_blocked": 1
}
```

## 17. Regression results

V2.13A tests: `tests/agent/test_v213_document_evidence.py` (29 tests). Prior experiment suites (V2.7–V2.12B) unchanged; production graph untouched.

## 18. Architectural recommendations

V2.13B — semantic/hybrid document retrieval over the validated substrate

Next: V2.13B semantic/hybrid retrieval over this validated substrate; optional live MBSSE PDF acquisition.