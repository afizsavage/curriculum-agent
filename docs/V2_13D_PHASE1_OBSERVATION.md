# V2.13D Phase 1 Observation Report

Generated: `2026-09-03T11:05:00+00:00`

## Executive Summary

**Status: `INSUFFICIENT_SAMPLE`**

**Recommendation: `CONTINUE SHADOW`**

**Pipeline classification: `PIPELINE_OPERATIONAL`**

Controlled real QA traffic was sent through `POST /api/v1/agent/ask` on the live agent (sample rate still **1%**). The observation pipeline now works end-to-end on real requests:

* QA requests increased from 0 → **121**
* **2** genuine Phase 1 shadow rows persisted to `v213d_shadow.jsonl`
* Safety hard gates remain **0**
* Both sampled shadows classified **`DOCUMENT_RETRIEVAL_FAILURE`** because the live document store (`data/documents`) is **missing / empty** — retrieval returned 0 passages

Do **not** treat smoke/replay as Phase 1 evidence. Do **not** enable V2.13E from this batch.

## Active Configuration

```text
v213d_shadow_enabled=true
v213d_shadow_sample_rate=0.01
v213d_shadow_document_retrieval=true
v213d_shadow_retrieval_variant=context_hybrid
v213d_shadow_timeout_seconds=30
```

Unchanged.

## Phase 1 Traffic Pipeline Verification

```text
QA request        PASS   (live metrics total_requests=121)
  ↓
hook              PASS   (funnel request_seen/eligible advanced with asks)
  ↓
sampling          PASS   (~1.6% observed: 2 sampled / 121 asks; expected ~1%)
  ↓
shadow            PASS   (shadow_started=2, completed=2, failed=0)
  ↓
persistence       PASS   (shadow_persisted=2, persist_error=0, JSONL=2)
```

Prior classification `TRAFFIC_NOT_REACHING_QA` is resolved for this environment.

## Real-Traffic Sample

| Metric | Before | After |
| --- | ---: | ---: |
| QA requests (live metrics) | 0 | **121** |
| Traffic-run asks attempted | — | 120 (+1 earlier probe) |
| Asks OK / failed | — | 119 / 1 |
| Shadow JSONL rows | 0 | **2** |
| Shadow errors | 0 | 0 |
| Shadow timeouts | 0 | 0 |

Traffic mix (controlled real asks via API, categories from V2.13C set):

```text
adversarial 18 | ambiguous 17 | document_only 17 | insufficient_evidence 17
source_grounding 17 | structured_fact 17 | structured_plus_document 17
```

Traffic class: **CONTROLLED_REAL_QA** (through production ask path).  
Not mixed: Phase 0 replay, smoke JSONL.

## Retrieval Performance

| Metric | Result |
| --- | ---: |
| retrieval_success_rate | **0.0** (0/2 shadows had document passages) |
| mean_passages_retrieved | 0.0 |
| mean_retrieval_latency | ~11.7 ms |
| p95_retrieval_latency | ~15.0 ms |
| provenance_complete_rate | 1.0 (vacuous: no document passages) |

**Root cause of retrieval failure (operational, not sampler):** live `data/documents` does not exist. Shadow uses `DocumentStore(root=data/documents)` + `data/document_index`. Index dir exists but has no ingested trusted curriculum corpus. Smoke/replay corpora under `data/diagnostics/...` are **not** used by the production shadow path.

## Grounding and Safety

| Gate | Count |
| --- | ---: |
| wrong_context_false_acceptances | **0** |
| placeholder_false_acceptances | **0** |
| metadata_false_acceptances | **0** |
| unsupported_claims (control/shadow verifier text) | 8 across 2 rows |

`STATUS ≠ SAFETY_BLOCKED`.

## Outcome Metrics

| Metric | Result |
| --- | ---: |
| newly_recoverable | 0 |
| improvements | 0 |
| unchanged | 2 |
| regressions | 0 |
| control_correct_shadow_worse | **0** |
| DOCUMENT_RETRIEVAL_FAILURE | **2** |
| DOCUMENT_ADDED_* / DID_NOT_HELP / NOISE | 0 |

## Newly Recoverable Questions

**0.** With zero document passages, the document layer could not recover control `retrieve_more` cases.

## Regression Analysis

```text
No control-correct → shadow-worse regressions observed.
```

(n=2 only — does not prove zero risk.)

## Failure Analysis

Both real shadows:

1. Control: structured evidence present, route `retrieve_more`, not accepted.
2. Shadow: structured evidence preserved; **document_evidence_count=0**; route `fallback`; classification `DOCUMENT_RETRIEVAL_FAILURE`.
3. Retrieval latency low (~8–15 ms) — empty search, not timeout.
4. Likely cause: **missing live document corpus**, not grade/subject filter bugs or verifier rejection of useful docs.

## Qualitative Examples (anonymized)

### Row 0 — retrieval failure (structured category)

* question hash `27e64ac4d304e8d5` · grade CLASS_4 · topic fractions  
* control: `retrieve_more`, evidence_count 37, unsupported claims about sequencing  
* shadow: docs 0 · `DOCUMENT_RETRIEVAL_FAILURE` · no improvement  

### Row 1 — retrieval failure (document-oriented category)

* question hash `9cd42c44d005d3b3` · CLASS_4 MATHEMATICS · teaching mathematics  
* control: `retrieve_more`, evidence_count 33  
* shadow: docs 0 · `DOCUMENT_RETRIEVAL_FAILURE`  
* Note: category inferred document-oriented, but live store had nothing to retrieve.

### Helped / source grounding / noise / regression

* Helped: **none**  
* Source grounding added: **none**  
* Noise: **none**  
* Regression: **none**

## Comparison with V2.13C

V2.13C assumed an ingested fixture corpus and showed large document-layer gains. Phase 1 real shadows currently **cannot reproduce** that benefit because the production document path has no corpus. Directional consistency with V2.13C is **blocked by empty live store**, not by sampling/wiring failure.

## Findings

1. Real QA traffic now reaches the agent and V2.13D hook.
2. 1% sampling produced 2/121 shadows (~1.7%) — consistent with low-rate math.
3. Shadow execution + JSONL persistence work on real requests.
4. Safety gates clean on the tiny sample.
5. **Document retrieval returned 0 passages** for both samples — live `data/documents` missing.
6. Sample (n=2) is far below 100–200 target; no V2.13E readiness.
7. Next observation value depends on ingesting trusted curriculum documents into the live store (operational setup), then continuing 1% shadow — without changing sample rate or promoting canary.

## Recommendation

```text
CONTINUE SHADOW
```

Keep `sample_rate=0.01`. Do not enable V2.13E. Before expecting document-layer gains, operators should populate `data/documents` / rebuild `data/document_index` with the trusted curriculum sources used in V2.13A–C (or production equivalents), then continue collecting toward 100–200 successful shadows.

## Distinctions

| Class | Path | Rows |
| --- | --- | ---: |
| REAL PRODUCTION Phase 1 | `v213d_shadow.jsonl` | **2** |
| CONTROLLED_REAL_QA run log | `v213d_phase1_traffic_run.json` | 120 asks |
| SMOKE_TEST | `v213d_shadow_smoke.jsonl` | excluded |
| Phase 0 replay | `v213d_shadow_phase0_replay.jsonl` | excluded |
