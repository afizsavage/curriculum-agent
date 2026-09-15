# V2.13D Phase 1 Observation Report

Generated: `2026-09-14T14:04:33.571610+00:00`

## Executive Summary

**Status: `INSUFFICIENT_SAMPLE`**

**Recommendation: `CONTINUE SHADOW`**

**Pipeline classification: `PIPELINE_OPERATIONAL`**

Only 50 post-corpus successful real shadow evaluations (pre-corpus=2); target is 100–200 before a rollout recommendation.

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

Phase 1D — first post-corpus real observations
    initial post-corpus n≈5; retrieval operational; mixed document value

Phase 1E — diagnostic continuation
    post-corpus real shadows: 50
    sample_rate remains 0.01 (no forced sampling)
    metrics_scope: post_corpus
    document_helped/neutral/hurt: 20/25/5
    retrieval_success (post-corpus): 1.0
    newly_recoverable: 15
    regressions (control_correct_shadow_worse): 5
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

## Phase 1E Traffic Batch

```json
{
  "path": "data/diagnostics/v213d_phase1e_traffic_run3.json",
  "traffic_class": "PHASE1E_DIAGNOSTIC_STABLE",
  "requested": 800,
  "ok": 690,
  "failed": 110,
  "elapsed_s": 9191.4,
  "categories": {
    "adversarial": 115,
    "ambiguous": 115,
    "document_only": 114,
    "insufficient_evidence": 114,
    "source_grounding": 114,
    "structured_fact": 114,
    "structured_plus_document": 114
  },
  "shadow_rows_before": 21,
  "shadow_rows_after": 31,
  "funnel_before": {
    "request_seen": 1724,
    "shadow_eligible": 1724,
    "shadow_sampled": 23,
    "shadow_not_sampled": 1701,
    "shadow_started": 21,
    "shadow_completed": 20,
    "shadow_failed": 1,
    "shadow_persisted": 21,
    "persist_error": 0
  },
  "funnel_after": {
    "request_seen": 2414,
    "shadow_eligible": 2414,
    "shadow_sampled": 33,
    "shadow_not_sampled": 2381,
    "shadow_started": 31,
    "shadow_completed": 30,
    "shadow_failed": 1,
    "shadow_persisted": 31,
    "persist_error": 0
  },
  "traffic_before": {
    "total_production_requests": 1722,
    "sampled_requests": 21
  },
  "traffic_after": {
    "total_production_requests": 2412,
    "sampled_requests": 31
  },
  "mean_latency_ms": 25844.910882668148
}
```

Investigate note: none

## Pre- vs Post-Corpus Real Shadows

```json
{
  "pre_corpus_shadow_evaluations": 2,
  "post_corpus_shadow_evaluations": 51,
  "post_corpus_successful_shadow_evaluations": 50,
  "corpus_unavailable_count": 2,
  "metrics_scope": "post_corpus",
  "classifications": {
    "DOCUMENT_CORPUS_UNAVAILABLE": 2,
    "DOCUMENT_DID_NOT_HELP": 25,
    "DOCUMENT_NOISE": 5,
    "DOCUMENT_ADDED_MISSING_CONTEXT": 15,
    "VERIFIER_FAILURE": 1,
    "DOCUMENT_ADDED_EXPLANATION": 5
  },
  "post_corpus_classifications": {
    "DOCUMENT_DID_NOT_HELP": 25,
    "DOCUMENT_NOISE": 5,
    "DOCUMENT_ADDED_MISSING_CONTEXT": 15,
    "DOCUMENT_ADDED_EXPLANATION": 5
  }
}
```

## Phase 1D Post-Corpus Performance (primary)

```json
{
  "retrieval_success_rate": 1.0,
  "no_match_rate": 0.0,
  "mean_passages_retrieved": 4.94,
  "provenance_complete_rate": 1.0,
  "metadata_valid_rate": 0.94,
  "newly_recoverable_count": 15,
  "newly_recoverable_rate": 0.3,
  "improvement_rate": 0.4,
  "regression_rate": 0.1,
  "control_correct_shadow_worse": 5,
  "document_added_missing_context": 15,
  "document_added_explanation": 5,
  "document_disambiguated_context": 0,
  "document_provided_source": 20,
  "document_did_not_help": 25,
  "document_noise": 5,
  "structured_data_already_sufficient": 0,
  "latency_metrics": {
    "shadow_mean_ms": 12906.958862340134,
    "shadow_p95_ms": 26215.909381049947,
    "retrieval_mean_ms": 67.12216352005271,
    "retrieval_p95_ms": 236.47559134844863
  }
}
```

Primary performance metrics above are scoped to **post-corpus** shadows. Pre-corpus `DOCUMENT_CORPUS_UNAVAILABLE` rows remain historical infrastructure failures and are excluded from retrieval-quality rates.

## Phase 1E Diagnostic Attribution

```json
{
  "document_helped": 20,
  "document_neutral": 25,
  "document_hurt": 5,
  "document_effect_counts": {
    "neutral": 25,
    "hurt": 5,
    "helped": 20
  },
  "retrieval_quality_counts": {
    "retrieval_non_decisive": 24,
    "retrieval_noise": 5,
    "retrieval_decisive": 20,
    "retrieval_irrelevant": 1
  },
  "structured_sufficient_document_retrieved": 10,
  "transitions": {
    "route": {
      "fallback\u2192fallback": 9,
      "retrieve_more\u2192retrieve_more": 1,
      "finish\u2192fallback": 2,
      "fallback\u2192finish": 15,
      "fallback\u2192retrieve_more": 9,
      "finish\u2192retrieve_more": 2,
      "finish\u2192finish": 6,
      "retrieve_more\u2192finish": 1,
      "clarify\u2192fallback": 1,
      "clarify\u2192clarify": 1,
      "fallback\u2192clarify": 2,
      "retrieve_more\u2192clarify": 1
    },
    "verifier": {
      "retrieve_more\u2192retrieve_more": 13,
      "fallback\u2192fallback": 2,
      "accept\u2192retrieve_more": 4,
      "retrieve_more\u2192accept": 16,
      "retrieve_more\u2192fallback": 3,
      "accept\u2192accept": 6,
      "clarify\u2192fallback": 1,
      "clarify\u2192clarify": 1,
      "fallback\u2192retrieve_more": 1,
      "retrieve_more\u2192clarify": 3
    },
    "mapper": {
      "reject\u2192reject": 10,
      "reject\u2192retrieve_more": 2,
      "accept\u2192reject": 4,
      "retrieve_more\u2192accept": 14,
      "retrieve_more\u2192reject": 11,
      "retrieve_more\u2192retrieve_more": 1,
      "reject\u2192accept": 2,
      "accept\u2192accept": 6
    },
    "key_route_transitions": {
      "fallback\u2192finish": 15,
      "retrieve_more\u2192finish": 1,
      "finish\u2192fallback": 2,
      "finish\u2192retrieve_more": 2
    }
  },
  "regression_cause_counts": {
    "GENERATOR_DOCUMENT_DRIFT": 4,
    "METADATA_EFFECT": 1
  },
  "segmentation": {
    "by_category": {
      "mixed": {
        "DOCUMENT_DID_NOT_HELP": 13,
        "DOCUMENT_NOISE": 5,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 1,
        "DOCUMENT_ADDED_EXPLANATION": 4
      },
      "insufficient_evidence": {
        "DOCUMENT_ADDED_MISSING_CONTEXT": 7,
        "DOCUMENT_DID_NOT_HELP": 4
      },
      "document_oriented": {
        "DOCUMENT_DID_NOT_HELP": 5,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 4
      },
      "ambiguous": {
        "DOCUMENT_ADDED_MISSING_CONTEXT": 3,
        "DOCUMENT_DID_NOT_HELP": 3
      },
      "adversarial_suspicious": {
        "DOCUMENT_ADDED_EXPLANATION": 1
      }
    },
    "by_grade": {
      "CLASS_4": {
        "DOCUMENT_DID_NOT_HELP": 9,
        "DOCUMENT_NOISE": 2,
        "DOCUMENT_ADDED_EXPLANATION": 4,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 4
      },
      "unknown": {
        "DOCUMENT_DID_NOT_HELP": 14,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 10,
        "DOCUMENT_NOISE": 1,
        "DOCUMENT_ADDED_EXPLANATION": 1
      },
      "CLASS_1": {
        "DOCUMENT_DID_NOT_HELP": 1,
        "DOCUMENT_NOISE": 1
      },
      "CLASS_5": {
        "DOCUMENT_DID_NOT_HELP": 1,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 1
      },
      "JSS_3": {
        "DOCUMENT_NOISE": 1
      }
    },
    "by_subject": {
      "MATHEMATICS": {
        "DOCUMENT_DID_NOT_HELP": 3,
        "DOCUMENT_NOISE": 2,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 3
      },
      "unknown": {
        "DOCUMENT_DID_NOT_HELP": 16,
        "DOCUMENT_NOISE": 3,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 6,
        "DOCUMENT_ADDED_EXPLANATION": 5
      },
      "SCIENCE": {
        "DOCUMENT_ADDED_MISSING_CONTEXT": 6,
        "DOCUMENT_DID_NOT_HELP": 6
      }
    },
    "by_source": {
      "bec-framework-2020": {
        "DOCUMENT_DID_NOT_HELP": 25,
        "DOCUMENT_NOISE": 5,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 15,
        "DOCUMENT_ADDED_EXPLANATION": 5
      },
      "math-primary-guidance": {
        "DOCUMENT_DID_NOT_HELP": 16,
        "DOCUMENT_NOISE": 3,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 8,
        "DOCUMENT_ADDED_EXPLANATION": 5
      },
      "science-guidance": {
        "DOCUMENT_DID_NOT_HELP": 11,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 8
      }
    }
  }
}
```

## Phase 1 Traffic Pipeline Verification

```json
{
  "classification": "PIPELINE_OPERATIONAL",
  "live_qa_metrics_total_requests": null,
  "production_jsonl_rows": 53,
  "funnel_stages": {
    "request_seen": 4097,
    "shadow_eligible": 4097,
    "shadow_sampled": 55,
    "shadow_not_sampled": 4042,
    "shadow_started": 53,
    "shadow_completed": 52,
    "shadow_failed": 1,
    "shadow_persisted": 53,
    "persist_error": 0
  },
  "config_enabled": true,
  "sample_rate": 0.01,
  "jsonl_path": "/home/afiz/Projects/curriculumz/curriculum-agent/data/diagnostics/v213d_shadow.jsonl",
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
  "total_production_requests": 4095,
  "live_qa_metrics_total_requests": null,
  "sampled": 53,
  "completed": 52,
  "errors": 1,
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
  "mean_retrieval_latency": 67.12216352005271,
  "p95_retrieval_latency": 236.47559134844863,
  "mean_passages_retrieved": 4.94,
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
  "unsupported_claims": 34
}
```

## Outcome Metrics

```json
{
  "newly_recoverable": 15,
  "newly_recoverable_rate": 0.3,
  "improvements": 20,
  "improvement_rate": 0.4,
  "unchanged": 25,
  "regressions": 5,
  "regression_rate": 0.1,
  "control_correct_shadow_worse": 5
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
    },
    {
      "request_id": "38dd316e2b962efb",
      "question_hash": "66ac729ee5938a1e",
      "category": "mixed",
      "classification": "DOCUMENT_ADDED_MISSING_CONTEXT",
      "control_route": "retrieve_more",
      "shadow_route": "finish",
      "document_evidence_count": 5
    },
    {
      "request_id": "0d88157a03fcce6f",
      "question_hash": "94cad668f7c007b3",
      "category": "ambiguous",
      "classification": "DOCUMENT_ADDED_MISSING_CONTEXT",
      "control_route": "fallback",
      "shadow_route": "finish",
      "document_evidence_count": 5
    },
    {
      "request_id": "f4710eca04ab3b96",
      "question_hash": "4b5752b6d633379c",
      "category": "insufficient_evidence",
      "classification": "DOCUMENT_ADDED_MISSING_CONTEXT",
      "control_route": "fallback",
      "shadow_route": "finish",
      "document_evidence_count": 5
    },
    {
      "request_id": "82ef648847761207",
      "question_hash": "4b5752b6d633379c",
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
    },
    {
      "request_id": "af805ad7260e9ede",
      "question_hash": "4b5752b6d633379c",
      "category": "insufficient_evidence",
      "classification": "DOCUMENT_DID_NOT_HELP",
      "control_route": "fallback",
      "shadow_route": "retrieve_more",
      "document_evidence_count": 5
    },
    {
      "request_id": "ce621d8e4ea966b8",
      "question_hash": "395bb312246e9804",
      "category": "insufficient_evidence",
      "classification": "DOCUMENT_DID_NOT_HELP",
      "control_route": "fallback",
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
    },
    {
      "request_id": "36cd54bfc08b26d3",
      "question_hash": "b74fc0930166a222",
      "category": "mixed",
      "classification": "DOCUMENT_NOISE",
      "control_route": "finish",
      "shadow_route": "fallback",
      "document_evidence_count": 5
    },
    {
      "request_id": "1eb1effc10e36cb1",
      "question_hash": "b85bd73ca7146b91",
      "category": "mixed",
      "classification": "DOCUMENT_NOISE",
      "control_route": "finish",
      "shadow_route": "retrieve_more",
      "document_evidence_count": 5
    },
    {
      "request_id": "758e61ac1f22aba6",
      "question_hash": "99a9a0a7f2f718f6",
      "category": "mixed",
      "classification": "DOCUMENT_NOISE",
      "control_route": "finish",
      "shadow_route": "finish",
      "document_evidence_count": 5
    },
    {
      "request_id": "652b77a98af4d72f",
      "question_hash": "395bb312246e9804",
      "category": "mixed",
      "classification": "DOCUMENT_NOISE",
      "control_route": "finish",
      "shadow_route": "retrieve_more",
      "document_evidence_count": 5
    }
  ],
  "regression": [],
  "safety": [],
  "shadow_failure": [
    {
      "request_id": "c2dbfe3057da7ace",
      "question_hash": "7fb53cf210f937c5",
      "category": null,
      "classification": "VERIFIER_FAILURE",
      "control_route": "retrieve_more",
      "shadow_route": null,
      "document_evidence_count": 0
    }
  ]
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
