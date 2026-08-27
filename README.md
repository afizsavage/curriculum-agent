# Curriculum Q&A Agent

Independent FastAPI service for MBSSE Curriculum Q&A.

The **Curriculum Structure API** remains the sole source of truth. This agent is
read-only and does not duplicate or write curriculum data.

## Phases

| Phase | Status |
| --- | --- |
| 1 Foundation (state, LLM, tools, ask API) | Done |
| 2 Curriculum retrieval & tools | Done |
| 3 Grounded answer generation | Done |
| 4 Verification & bounded loops | Done |
| 5 LangGraph orchestration & checkpointing | Done |

```text
User Question
  → LangGraph: UNDERSTAND → RETRIEVE → GENERATE → VERIFY
      ├── PASS → END
      ├── RETRIEVE_MORE ↺
      ├── CLARIFY → END
      └── FALLBACK → END
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
- Graph inspection: http://127.0.0.1:8001/api/v1/agent/graph

## Ask

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/agent/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What topics are taught in Primary 4 Mathematics?",
    "conversation_id": null
  }'
```

## Curriculum tools

| Tool | Use when |
| --- | --- |
| `resolve_curriculum_context` | Preferred structured GradeCurriculum resolve (grade/subject/topic → units + LOs) |
| `search_curriculum` | Find content by concept/keyword |
| `get_curriculum_structure` | List topics (or subjects if subject omitted) for a grade |
| `get_subject` | Subject identity/metadata for a grade |
| `get_topic` | Canonical topic by `topic_id` or name |
| `get_learning_objectives` | Authoritative outcomes for a topic |

V2.1 note: `resolve_curriculum_context` calls `GET /api/v2/curriculum/context/resolve` on the Curriculum Structure API. It does not replace V1 tools; they remain as fallbacks. See `curriculum-structure/docs/V2_CONTEXT_RESOLVE.md`.

## Configuration

| Variable | Purpose |
| --- | --- |
| `CURRICULUM_API_URL` | Curriculum Structure API base URL |
| `CURRICULUM_API_TIMEOUT` | HTTP timeout seconds |
| `LLM_PROVIDER` | `stub`, `openai` (Chat Completions), or `deepseek` (Responses API) |
| `LLM_MODEL` / `LLM_API_KEY` / `LLM_BASE_URL` | Provider settings |
| `AGENT_MAX_ITERATIONS` | Retrieval loop iteration cap |
| `AGENT_MAX_TOOL_CALLS` | Hard tool-call cap |
| `AGENT_MAX_RETRIEVAL_ROUNDS` | Max retrieve→generate→verify cycles |
| `AGENT_CHECKPOINTING_ENABLED` | LangGraph short-term thread checkpoints |
| `AGENT_CHECKPOINT_BACKEND` | `sqlite` (default, survives restart) or `memory` |
| `AGENT_CHECKPOINT_SQLITE_PATH` | SQLite file path (default `data/checkpoints.sqlite`) |

## Tests

```bash
pytest
```

Uses mocked Curriculum API responses. No real LLM key required for `stub`.

## Architecture

```text
app/
├── api/v1/agent.py          # HTTP → graph invoke → response mapper
├── agent/
│   ├── graph.py             # build_curriculum_qa_graph
│   ├── graph_nodes.py       # thin adapters over domain services
│   ├── graph_routing.py     # constrained conditional edges
│   ├── graph_state.py       # GraphState envelope around CurriculumQAState
│   ├── memory.py            # checkpointer factory
│   ├── orchestrator.py      # CurriculumQAAgent facade
│   ├── retrieve / answer / verify  # domain nodes (unchanged)
│   └── context.py           # conversation message store
├── curriculum/              # API client, evidence, codes
├── llm/                     # providers
└── tools/                   # registry + curriculum tools
```

LangGraph **orchestrates**; tools, answer generation, and verification remain domain services.
