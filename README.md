# Curriculum Q&A Agent

Independent FastAPI service for MBSSE Curriculum Q&A.

The **Curriculum Structure API** remains the sole source of truth. This agent is
read-only and does not duplicate or write curriculum data.

## Phases

| Phase | Status |
| --- | --- |
| 1 Foundation (state, LLM, tools, ask API) | Done |
| 2 Curriculum retrieval & tools | Done |
| 3 Answer generation + verification | Next |

```text
User Question → Understand → Retrieve (tools → Curriculum API) → Evidence
Answer / Verify arrive in Phase 3 (`answer` remains null).
```

## Requirements

- Python 3.10+
- Running Curriculum Structure API (default `http://127.0.0.1:8000`) for live retrieval

## Setup

```bash
cd curriculum-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Run

```bash
uvicorn app.main:app --reload --port 8001
```

- Health: http://127.0.0.1:8001/health
- OpenAPI: http://127.0.0.1:8001/docs

## Ask

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/agent/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What topics are taught in Primary 4 Mathematics?",
    "conversation_id": null
  }'
```

Phase 2 response shape:

```json
{
  "conversation_id": "...",
  "question": "What topics are taught in Primary 4 Mathematics?",
  "answer": null,
  "status": "retrieved",
  "evidence": [
    {"entity_type": "topic", "entity_id": "...", "name": "Fractions"}
  ],
  "metadata": {
    "iterations": 2,
    "tool_calls": 1,
    "tools_used": ["get_curriculum_structure"],
    "evidence_status": "found",
    "evidence_count": 1,
    "model": "stub-model",
    "provider": "stub"
  },
  "error": null
}
```

## Curriculum tools

| Tool | Use when |
| --- | --- |
| `search_curriculum` | Find content by concept/keyword |
| `get_curriculum_structure` | List topics (or subjects if subject omitted) for a grade |
| `get_subject` | Subject identity/metadata for a grade |
| `get_topic` | Canonical topic by `topic_id` or name |
| `get_learning_objectives` | Authoritative outcomes for a topic |

Primary Curriculum API routes used:

- `GET /api/v1/curricula`
- `GET /api/v1/curricula/{id}/structure`
- `GET /api/v1/curricula/{id}/subjects`
- `GET /api/v1/subjects/{id}`
- `GET /api/v1/syllabuses`
- `GET /api/v1/syllabuses/{id}/content/tree`
- `GET /api/v1/curriculum-context`
- `GET /api/v1/topics/{id}` (+ learning-outcomes)

There is no full-text search endpoint; `search_curriculum` filters syllabus trees client-side.

## Configuration

| Variable | Purpose |
| --- | --- |
| `CURRICULUM_API_URL` | Curriculum Structure API base URL |
| `CURRICULUM_API_TIMEOUT` | HTTP timeout seconds |
| `LLM_PROVIDER` | `stub`, `openai` (Chat Completions), or `deepseek` (Responses API) |
| `LLM_MODEL` / `LLM_API_KEY` / `LLM_BASE_URL` | Provider settings |
| `AGENT_MAX_ITERATIONS` | Retrieval loop iteration cap |
| `AGENT_MAX_TOOL_CALLS` | Hard tool-call cap |

## Tests

```bash
pytest
```

Uses mocked Curriculum API responses. No real LLM key required for `stub`.

## Architecture

```text
app/
├── api/v1/agent.py
├── agent/          # state, context, orchestrator, retrieve
├── curriculum/     # API client, evidence, code normalization
├── llm/            # providers + stub tool selection
└── tools/          # registry + curriculum tools
```
