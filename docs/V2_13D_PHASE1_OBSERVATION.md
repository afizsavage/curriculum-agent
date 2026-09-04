# V2.13D Phase 1 Observation Report

Generated: `2026-09-04T14:46:08.722265+00:00`

## Executive Summary

**Status: `INSUFFICIENT_SAMPLE`**

**Recommendation: `CONTINUE SHADOW`**

**Pipeline classification: `PIPELINE_OPERATIONAL`**

Only 5 post-corpus successful real shadow evaluations (pre-corpus=2); target is 100–200 before a rollout recommendation.

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

Phase 1D — post-corpus observation
    post-corpus real shadows: 5
    sample_rate remains 0.01 (no forced sampling)
    metrics_scope: post_corpus
    retrieval_success (post-corpus): 1.0
    newly_recoverable: 1
    regressions (control_correct_shadow_worse): 1
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

## Phase 1D Traffic Batch

```json
{
  "traffic_class": "PHASE1D_POST_CORPUS",
  "requested": 600,
  "ok": 550,
  "failed": 50,
  "elapsed_s": 3923.4,
  "categories": {
    "adversarial": 86,
    "ambiguous": 86,
    "document_only": 86,
    "insufficient_evidence": 86,
    "source_grounding": 86,
    "structured_fact": 85,
    "structured_plus_document": 85
  },
  "shadow_rows_before": 2,
  "shadow_rows_after": 7,
  "funnel_before": {
    "request_seen": 140,
    "shadow_eligible": 140,
    "shadow_sampled": 4,
    "shadow_not_sampled": 136,
    "shadow_started": 2,
    "shadow_completed": 2,
    "shadow_failed": 0,
    "shadow_persisted": 2,
    "persist_error": 0
  },
  "funnel_after": {
    "request_seen": 690,
    "shadow_eligible": 690,
    "shadow_sampled": 9,
    "shadow_not_sampled": 681,
    "shadow_started": 7,
    "shadow_completed": 7,
    "shadow_failed": 0,
    "shadow_persisted": 7,
    "persist_error": 0
  },
  "traffic_before": {
    "total_production_requests": 138,
    "sampled_requests": 2
  },
  "traffic_after": {
    "total_production_requests": 688,
    "sampled_requests": 7
  },
  "mean_latency_ms": 36710.06735839821
}
```

## Pre- vs Post-Corpus Real Shadows

```json
{
  "pre_corpus_shadow_evaluations": 2,
  "post_corpus_shadow_evaluations": 5,
  "post_corpus_successful_shadow_evaluations": 5,
  "corpus_unavailable_count": 2,
  "metrics_scope": "post_corpus",
  "classifications": {
    "DOCUMENT_CORPUS_UNAVAILABLE": 2,
    "DOCUMENT_DID_NOT_HELP": 3,
    "DOCUMENT_NOISE": 1,
    "DOCUMENT_ADDED_MISSING_CONTEXT": 1
  },
  "post_corpus_classifications": {
    "DOCUMENT_DID_NOT_HELP": 3,
    "DOCUMENT_NOISE": 1,
    "DOCUMENT_ADDED_MISSING_CONTEXT": 1
  }
}
```

## Phase 1D Post-Corpus Performance (primary)

```json
{
  "retrieval_success_rate": 1.0,
  "no_match_rate": 0.0,
  "mean_passages_retrieved": 5.0,
  "provenance_complete_rate": 1.0,
  "metadata_valid_rate": 1.0,
  "newly_recoverable_count": 1,
  "newly_recoverable_rate": 0.2,
  "improvement_rate": 0.2,
  "regression_rate": 0.2,
  "control_correct_shadow_worse": 1,
  "document_added_missing_context": 1,
  "document_added_explanation": 0,
  "document_disambiguated_context": 0,
  "document_provided_source": 1,
  "document_did_not_help": 3,
  "document_noise": 1,
  "structured_data_already_sufficient": 0,
  "latency_metrics": {
    "shadow_mean_ms": 10677.568457800226,
    "shadow_p95_ms": 13521.566139400238,
    "retrieval_mean_ms": 116.09215839998797,
    "retrieval_p95_ms": 237.9934875978506
  }
}
```

Primary performance metrics above are scoped to **post-corpus** shadows. Pre-corpus `DOCUMENT_CORPUS_UNAVAILABLE` rows remain historical infrastructure failures and are excluded from retrieval-quality rates.

## Phase 1 Traffic Pipeline Verification

```json
{
  "classification": "PIPELINE_OPERATIONAL",
  "live_qa_metrics_total_requests": 0,
  "production_jsonl_rows": 7,
  "funnel_stages": {
    "request_seen": 690,
    "shadow_eligible": 690,
    "shadow_sampled": 9,
    "shadow_not_sampled": 681,
    "shadow_started": 7,
    "shadow_completed": 7,
    "shadow_failed": 0,
    "shadow_persisted": 7,
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
  "total_production_requests": 688,
  "live_qa_metrics_total_requests": 0,
  "sampled": 7,
  "completed": 7,
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
  "retrieval_success_rate": 1.0,
  "no_match_rate": 0.0,
  "mean_retrieval_latency": 116.09215839998797,
  "p95_retrieval_latency": 237.9934875978506,
  "mean_passages_retrieved": 5.0,
  "provenance_complete_rate": 1.0,
  "metrics_scope": "post_corpus"
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
  "unsupported_claims": 4
}
```

## Outcome Metrics

```json
{
  "newly_recoverable": 1,
  "newly_recoverable_rate": 0.2,
  "improvements": 1,
  "improvement_rate": 0.2,
  "unchanged": 3,
  "regressions": 1,
  "regression_rate": 0.2,
  "control_correct_shadow_worse": 1
}
```

## Qualitative Examples (anonymized)

```json
{
  "document_improved": [
    {
      "request_id": "ef053659d587ed24",
      "question_hash": "193dabeb42058a02",
      "category": "insufficient_evidence",
      "classification": "DOCUMENT_ADDED_MISSING_CONTEXT",
      "control_route": "fallback",
      "shadow_route": "finish",
      "document_evidence_count": 5
    }
  ],
  "structured_sufficient": [],
  "document_did_not_help": [
    {
      "request_id": "7c677c46e9df7f08",
      "question_hash": "9cd42c44d005d3b3",
      "category": "mixed",
      "classification": "DOCUMENT_DID_NOT_HELP",
      "control_route": "fallback",
      "shadow_route": "fallback",
      "document_evidence_count": 5
    },
    {
      "request_id": "bd79bb2210be9021",
      "question_hash": "0aeebea81ce9d3a4",
      "category": "mixed",
      "classification": "DOCUMENT_DID_NOT_HELP",
      "control_route": "fallback",
      "shadow_route": "fallback",
      "document_evidence_count": 5
    },
    {
      "request_id": "90e7b17ba1bb18b3",
      "question_hash": "453c2803de62dccf",
      "category": "mixed",
      "classification": "DOCUMENT_DID_NOT_HELP",
      "control_route": "retrieve_more",
      "shadow_route": "retrieve_more",
      "document_evidence_count": 5
    }
  ],
  "document_noise": [
    {
      "request_id": "63bbf28e5f0db6e1",
      "question_hash": "dd1c57ff32af875b",
      "category": "mixed",
      "classification": "DOCUMENT_NOISE",
      "control_route": "finish",
      "shadow_route": "fallback",
      "document_evidence_count": 5
    }
  ],
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
- Phase 1D primary rates are post-corpus only.

V2.13C was a controlled harness (59.7%→90.3% grounded-correct). V2.13D Phase 1 real-traffic observations are not statistically equivalent.

## Recommendation

CONTINUE SHADOW

Keep `sample_rate=0.01`. Do not enable V2.13E until enough **post-corpus** real shadows exist to judge document-layer value in production.
