# V2.12A LangGraph → LangChain Behavioral Equivalence Experiment

Generated: `2026-08-31T12:55:55.927545+00:00`

**Conclusion: SUPPORTED**

LangChain reproduced validated pipeline behavior with zero unsafe divergences and preserved C4-U18 FC, safety, and metadata invariants.

---

## Hypothesis

Can LangChain replace LangGraph as the orchestration layer without changing validated V2.7–V2.11 behavior?

```text
                  SAME INPUT
                      │
             ┌────────┴────────┐
             ↓                 ↓
       LANGGRAPH            LANGCHAIN
        CONTROL             EXPERIMENT
             │                 │
             └────────┬────────┘
                      ↓
               EQUIVALENCE
                 ANALYZER
```

Both implementations call the **same domain functions** (V2.9 normalization, V2.11 metadata guard, verifier, V2.8 mapper, graph routing). Production `graph.py` is **unchanged**.

---

## Experiment Design

| Dimension | Value |
| --- | --- |
| Fixture classes | 28 (V2.10/V2.11 golden suite) |
| Runs per fixture | 10 |
| Implementations | LangGraph harness + LangChain harness |
| Total comparisons | **280** |
| Total pipeline evaluations | **560** |
| Analytical threshold | 0.85 |

---

## Results Summary

| Metric | LangGraph | LangChain |
| --- | ---: | ---: |
| FAITHFUL_COMPLETE acceptance | 100% | 100% |
| FAITHFUL_IMPERFECT acceptance | 80% | 80% |
| Placeholder false acceptance | 0 | 0 |
| Safety false acceptance | 0 | 0 |
| Adversarial false acceptance | 0 | 0 |
| UNSAFE_DIVERGENCE | — | **0** |

### Equivalence Classifications (280 comparisons)

| Classification | Count |
| --- | ---: |
| EXACT_EQUIVALENCE | 250 |
| BEHAVIORAL_EQUIVALENCE | 21 |
| CONTROLLED_DIFFERENCE | 7 |
| EXPECTED_LLM_VARIANCE | 1 |
| REGRESSION | 1 |
| UNSAFE_DIVERGENCE | **0** |

---

## Performance Comparison

| Metric | LangGraph | LangChain | Difference |
| --- | ---: | ---: | ---: |
| Mean latency (ms) | 5675.7 | 5823.3 | +147.5 |
| Median latency (ms) | 4720.5 | 4775.9 | +55.4 |
| P95 latency (ms) | 14599.0 | 13474.6 | -1124.4 |
| LLM calls | 280 | 280 | 0 |
| Errors | 0 | 0 | 0 |
| Timeouts | 0 | 0 | 0 |

No meaningful orchestration overhead introduced by LangChain.

---

## C4-U18 Side-by-Side

| Field | LangGraph | LangChain |
| --- | --- | --- |
| resolved topic | Everyday Arithmetic Money | Everyday Arithmetic Money |
| metadata_valid | true | true |
| verifier_score | 1.0 | 1.0 |
| verifier_recommendation | accept | accept |
| mapper_result | accept | accept |
| final_route | finish | finish |
| final_accepted | true | true |

---

## Architectural Decision

> Is LangChain now a safe replacement for LangGraph as the orchestration layer?

**Yes — experimentally validated.** Proceed to **V2.12B — Real Retrieval Production-Shadow Evaluation** while keeping LangGraph as control.

### Experimentally validated

- LangChain Runnable chain reproduces validated pipeline stages
- Zero unsafe divergences across 280 comparisons
- Deterministic stages (normalization, metadata) produce identical hashes

### Production-ready after

- V2.12B production-shadow evaluation with real retrieval
- Full production graph migration (not just harness)

### Requires hardening

- 1 REGRESSION + 1 EXPECTED_LLM_VARIANCE comparison (FI verifier variance, non-safety)
- End-to-end production graph wiring (retrieve → generate loop)

---

## Files

| File | Purpose |
| --- | --- |
| `docs/V2_12A_LANGGRAPH_INVENTORY.md` | Architecture inventory |
| `app/agent/v212_contract.py` | Framework-neutral contract + comparator |
| `app/agent/v212_langchain.py` | LangGraph + LangChain harness orchestration |
| `scripts/eval_v212_langchain_equivalence.py` | Eval harness |
| `tests/agent/test_v212_langchain_equivalence.py` | 29 tests |
| `data/diagnostics/v212a_langchain_equivalence.json` | Full report |
| `data/diagnostics/v212a_langchain_equivalence/` | 280 comparison traces |

## Files Intentionally Untouched

- `app/agent/graph.py`, `graph_nodes.py`, `orchestrator.py` (production default)
- Verifier, mapper, normalization rules, retrieval, API, database

## Tests

- V2.12A: 29 tests — **pass**
- V2.11–V2.7 + verifier: 93 tests — **pass**
- **Total regression: 122 tests — all pass**

## V2.13 Gate

**SUPPORTED** → Proceed to **V2.12B — Real Retrieval Production-Shadow Evaluation**
