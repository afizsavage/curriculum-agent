# V2.13A — Existing Document Architecture Findings

Generated as part of the Curriculum Document Evidence Layer experiment.

## 1. Where source metadata currently lives

| Location | Fields |
|----------|--------|
| `curriculum-structure` → `CurriculumSource` table | `document_url`, `document_name`, `version`, `content_hash`, `verification_status`, `authority`, `retrieved_at` |
| `Curriculum` row (denormalized) | `source_document_name`, `source_document_url`, `source_reference` |
| `Syllabus.source_id` | FK to `CurriculumSource` |
| `InstructionalReference.source_id` | Required FK to `CurriculumSource` |
| Per-entity provenance (`SourceProvenanceMixin`) | `source_document_id`, `source_page`, `source_section`, `source_reference`, `source_text` |
| Import JSON metadata | `metadata.source_url` on instructional imports (not always promoted to `document_url`) |

**Agent:** no document-source model; `CurriculumEvidence.metadata["provenance"]` when V2 resolve returns LO provenance.

## 2. How source URLs are stored

- **Authoritative column:** `CurriculumSource.document_url` (max 1024 chars).
- **API:** `GET /api/v1/curriculum-sources/{id}` returns `CurriculumSourceRead`.
- **Syllabus shortcut:** `GET /api/v1/syllabuses/{id}/source`.
- **Context snippet:** `GET /api/v1/curriculum-context` may include `source` when instructional references match.

URLs in import JSON (`metadata.source_url`) are informational until a `CurriculumSource` row is created and linked.

## 3. Authoritative vs informational URLs

| Type | Treatment |
|------|-----------|
| `CurriculumSource.document_url` with `verification_status=VERIFIED` | Authoritative for V2.13A acquisition |
| `metadata.source_url` in import JSON only | Informational until registered |
| User/agent-provided arbitrary URLs | **Not trusted** — must not become evidence |

## 4. Curriculum version relationships

Three version concepts coexist:

1. **Framework version** — `Curriculum(code, version)` e.g. `MBSSE-BEC` / `2020`
2. **Syllabus document version** — `Syllabus.version` under a curriculum
3. **Source document version** — `CurriculumSource.version` (PDF edition, not framework version)

V2.13A treats `source_id` + `CurriculumSource.version` + `content_hash` as document identity.

## 5. Whether PDFs are already retrievable

- **Runtime API:** No PDF download or parse endpoints. Design explicitly states PDF parsing is offline (`docs/DESIGN.md`, `README.md`).
- **Offline scripts:** `scripts/parse_*.py`, `scripts/build_mbsse_*.py` transcribe PDFs → canonical JSON.
- **Committed data:** JSON import payloads in `data/`; PDFs are not stored in the repo.

**Gap:** No blob store or runtime document acquisition in either service.

## 6. Existing document extraction capabilities

- Offline markdown/text parsers for lesson plans.
- Manual transcription builders for syllabus JSON.
- No in-API `DOCUMENT_EXTRACT` execution despite enum existing on import types.

## 7. Existing provenance capabilities

- DB columns on curriculum entities via `SourceProvenanceMixin`.
- `ImportProvenance` in canonical JSON (`source_page`, `source_section`, `source_reference`, `source_text`, `source_document_id`).
- V2 resolve returns `LearningOutcomeRef.provenance` when populated.
- Agent maps provenance into `CurriculumEvidence.metadata["provenance"]`.

## 8. Existing evidence abstractions

**`CurriculumEvidence`** (`app/curriculum/evidence.py`):

- Structured entity fields: `entity_type`, `entity_id`, `grade`, `subject`, `topic`, `content`
- `source` = retrieval path label (e.g. `curriculum_api`), not document authority
- `source_reference` = tool/API path (e.g. `v2.curriculum.context.resolve`)
- `metadata` = extensible dict (provenance nested here today)

No first-class `document_passage` type or document URL on the model.

## 9. What must be added (V2.13A)

1. Document evidence contract (`DocumentPassage`, provenance, content hash)
2. Trusted acquisition from registered `CurriculumSource` only
3. Local deterministic document cache (`data/documents/<document_id>/`)
4. PDF/text parser with page boundaries
5. Curriculum-aware passage construction (deterministic hierarchy association)
6. Lexical retrieval with curriculum-context filters
7. `search_curriculum_document` agent tool (feature-flagged)
8. `CurriculumEvidence` mapping for `entity_type=document_passage`
9. Evaluation harness and security/trust-boundary tests

## 10. What can be reused

| Component | Reuse |
|-----------|-------|
| `CurriculumSource` API | Fetch trusted source metadata |
| `CurriculumAPIClient` | Extend with `get_curriculum_source` |
| `CurriculumEvidence` | Additive `document_passage` entity type |
| V2.9 normalization | Unchanged; document evidence passes through when integrated |
| V2.11 metadata guard | Unchanged in V2.13A |
| V2.12 contract | Unchanged |
| Feature-flag pattern | `v213_document_evidence_experiment` |
| Offline import JSON | Source URL references for benchmark corpus |

## Production constraint

V2.13A does **not** modify the Curriculum Structure API schema, LangGraph production path, or structured retrieval semantics. Document evidence is experimental and feature-flagged in the agent only.
