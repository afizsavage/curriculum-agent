# Curriculum Q&A Agent — Sprint 1 Foundation

Independent FastAPI service for an MBSSE Curriculum Q&A agent.

The **Curriculum Structure API** (`curriculum-structure`) remains the authoritative
source of curriculum data. This service does **not** duplicate curriculum storage and
does **not** write to the curriculum API. Sprint 1 is infrastructure only: typed
state, LLM abstraction, tool registry, conversation context, observability, and
`POST /api/v1/agent/ask`. Curriculum tools and the full agent loop arrive later.

```text
User Question
     ↓
Understand      (Sprint 2+)
     ↓
Retrieve MBSSE Curriculum  (Sprint 2+)
     ↓
Reason / Answer (Sprint 2+)
     ↓
Verify          (Sprint 2+)
     ↺
```

## Requirements

- Python 3.10+

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

The Curriculum Structure API typically runs on port `8000`. This agent uses `8001`
by default so both can run side by side.

## Ask endpoint

```http
POST /api/v1/agent/ask
Content-Type: application/json
```

### Request

```json
{
  "question": "What topics are taught in Primary 4 Mathematics?",
  "conversation_id": null
}
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `question` | string | yes | Curriculum question (1–4000 chars, non-blank) |
| `conversation_id` | uuid \| null | no | Continue an existing conversation |

### Response (Sprint 1)

```json
{
  "conversation_id": "uuid",
  "question": "What topics are taught in Primary 4 Mathematics?",
  "answer": null,
  "status": "received",
  "metadata": {
    "iterations": 0,
    "tool_calls": 0,
    "model": "stub-model",
    "provider": "stub"
  },
  "error": null
}
```

`answer` is intentionally `null` until retrieve/answer are implemented.

### Errors

| HTTP | Code | When |
| --- | --- | --- |
| 422 | `INVALID_REQUEST` | Validation failure / blank question |
| 500 | `CONFIGURATION_ERROR` | Misconfigured LLM provider |
| 500 | `AGENT_EXECUTION_FAILURE` | Agent turn failed |
| 500 | `UNEXPECTED_ERROR` | Unhandled failure (no stack traces to clients) |
| 502 | `LLM_PROVIDER_FAILURE` | Provider SDK/API failure |
| 502 | `TOOL_FAILURE` | Tool execution failure |
| 504 | `LLM_TIMEOUT` | Provider timeout |

Responses include `detail`, `code`, and `request_id`. The `X-Request-ID` response
header is always set.

### Example

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/agent/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What topics are taught in Primary 4 Mathematics?","conversation_id":null}'
```

## Configuration

See `.env.example`. Important keys:

| Variable | Purpose |
| --- | --- |
| `LLM_PROVIDER` | Provider id (`stub` in Sprint 1) |
| `LLM_MODEL` | Model name |
| `LLM_API_KEY` | Provider key (never logged) |
| `LLM_TIMEOUT_SECONDS` | Per-call LLM timeout |
| `AGENT_MAX_ITERATIONS` | Future loop limit (default 3) |
| `AGENT_MAX_TOOL_CALLS` | Future tool-call limit (default 10) |
| `AGENT_REQUEST_TIMEOUT_SECONDS` | Request budget |
| `CURRICULUM_API_BASE_URL` | Reserved for Sprint 2 read-only access |

## Architecture

```text
curriculum-agent/
├── app/
│   ├── api/v1/agent.py      # POST /agent/ask
│   ├── agent/               # state, conversation, CurriculumQAAgent
│   ├── llm/                 # LLMProvider + stub
│   ├── tools/               # Tool + ToolRegistry (+ echo mock)
│   ├── schemas/
│   ├── config.py
│   ├── exceptions.py
│   └── main.py
└── tests/
```

- **State** — `CurriculumQAState` (Pydantic) holds question, filters, plan, context,
  draft, verification, counters, status.
- **Orchestrator** — `CurriculumQAAgent` with `ask`, plus stub nodes
  `understand` / `retrieve` / `answer` / `verify` for Sprint 2 wiring.
- **LLM** — `LLMProvider.generate` / `generate_structured` / `generate_with_tools`.
- **Tools** — registry ready for `search_curriculum`, `get_topic`, etc. Sprint 1
  ships only an `echo` mock for tests.
- **Conversation** — in-memory `ConversationStore` keyed by `conversation_id`.
- **Observability** — structured logs with `request_id`, `conversation_id`, status,
  iteration, tool_calls, model, latency. Secrets are redacted.

## Tests

```bash
pytest
```

Tests mock/stub the LLM. No real API key is required.

## Out of scope (Sprint 1)

- Curriculum Structure API tool clients
- RAG / embeddings / vector DB
- Multi-agent systems
- Curriculum write operations
- Persistent conversation storage
