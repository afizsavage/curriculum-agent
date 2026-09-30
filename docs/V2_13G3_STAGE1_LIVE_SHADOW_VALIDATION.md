# V2.13G.3 — Stage-1 Live Shadow Validation

Generated: `2026-09-15T17:42:24.027259+00:00`

**Decision: `INSUFFICIENT_LIVE_COVERAGE`**

Production arbitration remains **OFF**. This report is shadow-only.

## A. Runtime configuration

```text
V213G_PROVENANCE_ONLY_SEMANTICS=preserve_control_answer
v213d_shadow_enabled=True
v213d_shadow_sample_rate=0.01
v213f_document_arbitration_experiment=False
v213e_enabled=False
v213g_live_only=True
```

Verified at: `2026-09-15T17:26:59.971393+00:00`

## B. Infrastructure health

- valid rows: **1**
- infrastructure-invalid rows: **0**
- historical batch-1 rows (excluded from live n): **52**
- source JSONL rows: **53**

## C. Cohort composition (valid live only)

| Cohort | Count |
| --- | ---: |
| Structured sufficient + redundant | 1 |
| Structured sufficient + irrelevant | 0 |
| Structured sufficient + decisive | 0 |
| Structured insufficient + decisive | 0 |
| Structured insufficient + irrelevant | 0 |
| Structured insufficient + redundant | 0 |
| Document-only | 0 |
| Infrastructure-invalid | 0 |

## D. H1a classification

- evaluated: 1
- classification errors: **0**

## E. H1b enforcement

- non-generation cases: 1
- document leaks: **0**
- regeneration attempts: **0**

## F. H1c control preservation

- preserve cases: **1**
- hash matches: **1**
- hash mismatches: **0**
- regeneration attempts: **0**
- document leaks: **0**
- arbitration regressions: **0**
- hash match rate: **100.0%**

## G. H2 decisive recovery

- H2 cases: **0**
- regeneration attempted: 0
- outcome sources: `{}`
- arbitrated recoveries: 0
- baseline recoveries: 0
- arbitration regressions: 0

## H. Baseline contamination

- tracked ids: `['9cd42c', '27e64a']`
- observed in valid live: `[]`
- baseline document-merge regressions: 0
- arbitration regressions (non-contamination): 0

## I. Safety

- flag counts: `{}`
- total flags: **0** (expected 0)

## J. Latency (shadow only)

```json
{
  "structured_retrieval_ms": {
    "n": 1,
    "p50": 4.25,
    "p95": 4.25,
    "mean": 4.25
  },
  "document_retrieval_ms": {
    "n": 1,
    "p50": 8.621,
    "p95": 8.621,
    "mean": 8.62
  },
  "arbitration_ms": {
    "n": 1,
    "p50": 0.089,
    "p95": 0.089,
    "mean": 0.09
  },
  "total_shadow_ms": {
    "n": 1,
    "p50": 15094.535,
    "p95": 15094.535,
    "mean": 15094.53
  },
  "note": "Shadow latency only; not production user-facing latency."
}
```

## K. Production isolation

```json
{
  "production_arbitration": "OFF",
  "v213f_document_arbitration_experiment": false,
  "v213e_enabled": false,
  "sample_rate": 0.01,
  "api_v1_unchanged": true,
  "corpus_unchanged": true,
  "historical_artifacts_unchanged": true,
  "historical_hash_files": {
    "pre": "data/diagnostics/v213g3_historical_artifact_hashes_pre.txt",
    "post": "data/diagnostics/v213g3_historical_artifact_hashes_post.txt"
  }
}
```

## L. Decision

`INSUFFICIENT_LIVE_COVERAGE`

Recommendation: Continue Stage-1 live shadow at sample_rate=0.01 until sufficient+document, redundant/irrelevant, and insufficient+decisive cohorts are all represented; do not enable production arbitration.

Promotion rule: **DO NOT enable production arbitration** from this result alone.
