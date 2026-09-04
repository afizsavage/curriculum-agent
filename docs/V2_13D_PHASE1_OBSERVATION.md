# V2.13D Phase 1 Observation Report

Generated: `2026-09-04T09:16:03.910462+00:00`

## Executive Summary

**Status: `INSUFFICIENT_SAMPLE`**

**Recommendation: `CONTINUE SHADOW`**

**Pipeline classification: `PIPELINE_OPERATIONAL`**

Only 0 post-corpus successful real shadow evaluations (pre-corpus=2); target is 100–200 before a rollout recommendation.

Pre-corpus real shadows are infrastructure/corpus-availability failures (`DOCUMENT_CORPUS_UNAVAILABLE`), not retrieval-algorithm failures. Post-corpus observation sufficiency is measured separately.

## Phase 1 Timeline

```text
Phase 1A — pipeline verification
    0 production QA requests (TRAFFIC_NOT_REACHING_QA initially)

Phase 1B — first real traffic
    ~121 QA requests
    2 real shadows
    corpus unavailable (empty data/documents)
    classification: DOCUMENT_CORPUS_UNAVAILABLE (reclassified)

Phase 1C — corpus activation
    trusted V2.13A–C BENCHMARK_SOURCES activated
    documents=3
    passages=16
    index_entries=16
    activation_ok=True
    expected_hashes_matched=True

Phase 1D — resumed real observation
    post-corpus real shadows: 0
    sample_rate remains 0.01 (no forced sampling)
```

## Active Configuration

```text
v213d_shadow_enabled=True
v213d_shadow_sample_rate=0.01
v213d_shadow_document_retrieval=True
v213d_shadow_retrieval_variant=context_hybrid
v213d_shadow_timeout_seconds=30.0
```

No rollout escalation. No V2.13E. Document evidence does not enter the user-facing production answer path.

## Corpus Activation (Phase 1C)

```json
{
  "corpus_family": "V2.13A\u2013C BENCHMARK_SOURCES",
  "counts": {
    "documents": 3,
    "passages": 16,
    "index_entries": 16,
    "orphaned_document_dirs": 0,
    "passages_missing_provenance": 0
  },
  "document_hashes": {
    "bec-framework-2020": "26409f8e53267603f3b446ad19ef422cff73f7116b74661daa09c77459fb08a4",
    "math-primary-guidance": "3710ff81811f0fd2c436d89db7d43d9a03bdac5ed4e4e25712a0dde2c1733d70",
    "science-guidance": "feef53e14b590cbd983834cd0c144d2060f88ae68f493fe20b536350df3fea13"
  },
  "discrepancies": [],
  "orphaned": 0,
  "passages_missing_provenance": 0,
  "hierarchy": {
    "with_grade": 9,
    "with_subject": 16,
    "with_topic": 6
  },
  "activated_at": "2026-09-04T09:11:16.031986+00:00"
}
```

## Pre- vs Post-Corpus Real Shadows

```json
{
  "pre_corpus_shadow_evaluations": 2,
  "post_corpus_shadow_evaluations": 0,
  "post_corpus_successful_shadow_evaluations": 0,
  "corpus_unavailable_count": 2,
  "classifications": {
    "DOCUMENT_CORPUS_UNAVAILABLE": 2
  }
}
```

## Phase 1 Traffic Pipeline Verification

```json
{
  "classification": "PIPELINE_OPERATIONAL",
  "live_qa_metrics_total_requests": 0,
  "production_jsonl_rows": 2,
  "funnel_stages": {
    "request_seen": 124,
    "shadow_eligible": 124,
    "shadow_sampled": 4,
    "shadow_not_sampled": 120,
    "shadow_started": 2,
    "shadow_completed": 2,
    "shadow_failed": 0,
    "shadow_persisted": 2,
    "persist_error": 0
  },
  "config_enabled": true,
  "sample_rate": 0.01,
  "jsonl_path": "/home/afiz/Projects/side/curriculum-agent/data/diagnostics/v213d_shadow.jsonl",
  "stages_checklist": {
    "qa_request": "NOT OBSERVED",
    "hook": "PASS",
    "sampling": "PASS",
    "shadow": "PASS",
    "persistence": "PASS"
  }
}
```

## Real-Traffic Sample

```json
{
  "total_production_requests": 122,
  "live_qa_metrics_total_requests": 0,
  "sampled": 2,
  "completed": 2,
  "errors": 0,
  "timeouts": 0,
  "observation_target": [
    100,
    200
  ],
  "source": "production_shadow",
  "real_traffic_observed": true
}
```

## Retrieval Performance

```json
{
  "retrieval_success_rate": 0.0,
  "mean_retrieval_latency": 11.659051000151521,
  "p95_retrieval_latency": 15.040290699516845,
  "mean_passages_retrieved": 0.0,
  "provenance_complete_rate": 1.0,
  "note": "Pre-corpus rows had empty store (corpus unavailable). Do not interpret pre-corpus 0% retrieval as algorithm failure."
}
```

## Grounding and Safety

```json
{
  "wrong_context_false_acceptance": 0,
  "placeholder_false_acceptance": 0,
  "metadata_integrity_false_acceptance": 0,
  "metadata_false_acceptance": 0,
  "unsafe_adversarial_false_acceptance": 0,
  "shadow_errors_must_not_affect_production": true,
  "unsupported_claims": 8
}
```

## Outcome Metrics

```json
{
  "newly_recoverable": 0,
  "improvements": 0,
  "unchanged": 2,
  "regressions": 0,
  "control_correct_shadow_worse": 0
}
```

## Qualitative Examples (anonymized)

```json
{
  "document_improved": [],
  "structured_sufficient": [],
  "document_did_not_help": [],
  "document_noise": [],
  "regression": [],
  "safety": [],
  "shadow_failure": []
}
```

## Distinctions

- Production analysis uses only `v213d_shadow.jsonl` (no `replay_id`).
- Smoke/test records must live in `v213d_shadow_smoke.jsonl` only.
- Phase 0 replay is excluded from Phase 1 claims.
- Pre-corpus vs post-corpus shadows are analyzed separately.

V2.13C was a controlled harness (59.7%→90.3% grounded-correct). V2.13D Phase 1 real-traffic observations are not statistically equivalent.

## Recommendation

CONTINUE SHADOW

Keep `sample_rate=0.01`. Do not enable V2.13E until enough **post-corpus** real shadows exist to judge document-layer value in production.
