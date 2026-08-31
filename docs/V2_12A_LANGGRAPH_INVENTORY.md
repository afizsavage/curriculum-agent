# V2.12A LangGraph Architecture Inventory

Generated for the LangGraph → LangChain behavioral equivalence experiment.

## Purpose

Document the existing LangGraph execution path and distinguish **orchestration logic** from **domain/application logic** before implementing the parallel LangChain harness.

---

## Production Entry Point

```text
POST /api/v1/agent/ask
  → CurriculumQAAgent.ask()          [app/agent/orchestrator.py]
  → initial_graph_input()            [app/agent/graph_state.py]
  → compiled_graph.invoke()          [app/agent/graph.py]
  → attach_graph_metadata()          [app/agent/response_mapper.py]
  → map_graph_result_to_response()   [app/agent/response_mapper.py]
```

**Default orchestration:** LangGraph (`build_curriculum_qa_graph`). Unchanged by V2.12A.

---

## LangGraph Topology (Control)

**File:** `app/agent/graph.py`

| Node | Adapter | Domain service |
| --- | --- | --- |
| `understand` | `GraphNodes.understand` | Filter extraction |
| `prepare_cycle` | `GraphNodes.prepare_cycle` | Iteration/limit checks |
| `retrieve` | `GraphNodes.retrieve` | `RetrievalNode.run()` |
| `generate_answer` | `GraphNodes.generate_answer` | `AnswerGenerationNode.run()` |
| `verify_answer` | `GraphNodes.verify_answer` | `VerificationNode.run()` |
| `clarify` | `GraphNodes.clarify` | `apply_clarification()` |
| `fallback` | `GraphNodes.fallback` | `apply_fallback()` |
| `finish` | `GraphNodes.finish` | Mark completed |

### Conditional routing

**File:** `app/agent/graph_routing.py`

- `route_after_prepare` → `retrieve` | `fallback`
- `route_after_verification` → `finish` | `retrieve_more` | `regenerate` | `clarify` | `fallback`

---

## State Schema

| Layer | File | Type |
| --- | --- | --- |
| Domain | `app/agent/state.py` | `CurriculumQAState` |
| Graph envelope | `app/agent/graph_state.py` | `GraphState` |

Graph state wraps `qa: CurriculumQAState` plus routing metadata (`visited_nodes`, `route`, `fallback_reason`, etc.).

---

## Orchestration vs Domain Logic

### Orchestration (LangGraph-owned in production)

- Node registration and edge wiring (`graph.py`)
- Conditional routing (`graph_routing.py`)
- Checkpointing (`memory.py`)
- Visit tracing (`graph_state.py`, `trace.py`)
- Iteration / retrieval-round limits (`prepare_cycle`)

### Domain / Application (shared, framework-independent)

| Stage | Module | Key functions |
| --- | --- | --- |
| Retrieval | `retrieve.py` | `RetrievalNode.run()` |
| Normalization | `v29_evidence_normalization.py` | `normalize_evidence()` |
| Metadata integrity | `v211_metadata_integrity.py` | `validate_metadata_integrity()`, `apply_metadata_policy()` |
| Generation | `answer.py`, `answer_generator.py` | `AnswerGenerationNode.run()` |
| Verification | `verify.py`, `verifier.py` | `VerificationNode.run()`, `AnswerVerifier.verify()` |
| Recommendation mapping | `v28_recommendation_mapping.py` | `map_recommendation()` |
| Routing policy | `graph_routing.py` | `route_after_verification()` |

V2.9 normalization, V2.11 metadata guard, and V2.8 mapper are **harness-validated** but not yet wired into the production graph.

---

## V2.12A Experimental Architecture

V2.12A does **not** modify `graph.py`. It adds parallel harness orchestration over the validated pipeline:

```text
retrieve (fixture inject)
   ↓
normalize            ← v29
   ↓
metadata_integrity   ← v211
   ↓
generate (fixture answer)
   ↓
verify               ← existing verifier
   ↓
map_recommendation   ← v28
   ↓
route                ← graph_routing
```

| Implementation | Orchestration | File |
| --- | --- | --- |
| **CONTROL** | LangGraph mini-graph | `app/agent/v212_langchain.py` → `build_langgraph_harness_graph()` |
| **EXPERIMENT** | LangChain Runnable chain | `app/agent/v212_langchain.py` → `build_langchain_harness_chain()` |

Both call the **same domain functions** — no duplicated business logic.

---

## LLM Integration

Production uses custom `LLMProvider` (`app/llm/`), not LangChain LLM wrappers.

- `langgraph` — direct dependency (orchestration)
- `langchain-core` — direct dependency (V2.12A harness only)
- `langchain` (full package) — **not** required

---

## Observability

| Concern | Location |
| --- | --- |
| Agent traces | `app/agent/trace.py` |
| Metrics | `app/agent/metrics.py` |
| Phase timings | `trace.phase_timings_ms` (production) |
| Per-stage timings | `PipelineRunResult.timings` (V2.12A harness) |

---

## Persistence / Checkpointing

- SQLite or in-memory checkpointer via `app/agent/memory.py`
- Thread config keyed by `conversation_id`
- V2.12A harness: **stateless** (no checkpointing)

---

## Config Flags

| Flag | Default | Scope |
| --- | --- | --- |
| `v212_langchain_experiment` | `OFF` | Harness replay only |
| Production default | LangGraph | `default_implementation()` |

---

## Files Intentionally Untouched by V2.12A

- `app/agent/graph.py`
- `app/agent/graph_nodes.py`
- `app/agent/orchestrator.py` (default path)
- `app/agent/verifier.py`
- `app/agent/v28_recommendation_mapping.py`
- `app/agent/v29_evidence_normalization.py`
- `app/agent/v211_metadata_integrity.py` (rules unchanged; reused)
- `app/agent/retrieve.py`
- API contracts / database
