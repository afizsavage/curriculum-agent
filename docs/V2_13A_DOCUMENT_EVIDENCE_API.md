# V2.13A — Document Evidence API (Agent-Side)

V2.13A adds an **experimental agent-side document evidence layer**. The Curriculum Structure API (`/api/v1`) remains unchanged.

## Design principle

Document evidence is acquired from registered `CurriculumSource` records via the existing Structure API, cached locally in the agent, and exposed through a read-only retrieval tool. No vector database or new Structure API routes are introduced in V2.13A.

## Existing Structure API (reused)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/curriculum-sources` | List registered sources |
| GET | `/api/v1/curriculum-sources/{source_id}` | Fetch trusted source metadata (`document_url`, `version`, `content_hash`) |
| GET | `/api/v1/syllabuses/{syllabus_id}/source` | Syllabus-linked source |

## Agent service API (Python, not HTTP)

Equivalent conceptual endpoints implemented in `DocumentEvidencePipeline`:

| Operation | Function | Description |
|-----------|----------|-------------|
| Acquire | `DocumentStore.acquire(source_record)` | Download/cache from registered `document_url` |
| Parse | `DocumentParser.parse_file(path)` | Extract pages/blocks |
| Build passages | `PassageBuilder.build_passages(...)` | Curriculum-aware passages |
| Search | `DocumentRetrievalService.search_document_evidence(...)` | Lexical retrieval + filters |
| Ingest | `DocumentEvidencePipeline.ingest_source(...)` | Acquire → parse → passages |

## Agent tool (LLM-facing)

| Tool | Name | When enabled |
|------|------|--------------|
| `search_curriculum_document` | `search_curriculum_document` | `v213_document_evidence_experiment=true` |

### Input

```json
{
  "query": "purpose of mathematics education",
  "grade": "CLASS_4",
  "subject": "MATHEMATICS",
  "topic": "money",
  "source_id": "<optional>",
  "limit": 5
}
```

### Output (conceptual)

```json
{
  "experiment": "v2.13a_document_evidence",
  "document_passages": [ "...CurriculumEvidence..." ],
  "structured_records": [],
  "source_references": [ "...provenance..." ],
  "retrieval_diagnostics": { "...": "..." },
  "evidence_count": 2
}
```

## Local storage layout

```text
data/documents/
  <document_id>/
    source.pdf | source.txt
    metadata.json
    passages.json
```

## Feature flag

```env
v213_document_evidence_experiment=false   # default — production unchanged
```

## Future V2 API (not implemented in V2.13A)

If document evidence moves to curriculum-structure later, equivalent routes might be:

- `GET /api/v2/sources/{source_id}/document`
- `POST /api/v2/evidence/search`

V2.13A intentionally keeps document acquisition in the agent to avoid turning the relational DB into a document store.

## Security boundary

- Only `CurriculumSource.document_url` from registered API records
- Reject `DRAFT` / `SUPERSEDED` sources
- Reject arbitrary user URLs
- Content hash conflict detection without silent overwrite
