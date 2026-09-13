# V2.13D — Production Shadow for Context-Hybrid Curriculum Document Evidence

Generated: `2026-09-10T06:54:26.524724+00:00`

**Phase 1 status: INSUFFICIENT_SAMPLE**

**Recommendation: CONTINUE SHADOW**

Only 38 post-corpus successful real shadow evaluations (pre-corpus=2); target is 100–200 before a rollout recommendation.

## Metrics

```json
{
  "experiment": "v2.13d",
  "schema_version": "v213d.2",
  "source": "production_shadow",
  "total_production_requests": 3022,
  "sampled_requests": 41,
  "traffic_sampled": 41,
  "shadow_completed": 40,
  "successful_shadow_evaluations": 40,
  "pre_corpus_shadow_evaluations": 2,
  "post_corpus_shadow_evaluations": 39,
  "post_corpus_successful_shadow_evaluations": 38,
  "corpus_unavailable_count": 2,
  "shadow_errors": 1,
  "shadow_timeouts": 0,
  "shadow_error_rate": 0.024390243902439025,
  "retrieval_failures": 0,
  "retrieval_success_rate": 1.0,
  "retrieval_success": 1.0,
  "no_match_rate": 0.0,
  "mean_retrieval_latency": 83.3196082106425,
  "p95_retrieval_latency": 247.1640751974519,
  "mean_passages_retrieved": 4.921052631578948,
  "provenance_complete_rate": 1.0,
  "metadata_valid_rate": 0.9210526315789473,
  "wrong_context_false_acceptance_rate": 0.0,
  "placeholder_false_acceptance_rate": 0.0,
  "metadata_false_acceptance_rate": 0.0,
  "unsupported_claim_rate": 0.868421052631579,
  "newly_recoverable_count": 5,
  "newly_recoverable_rate": 0.13157894736842105,
  "improvements": 9,
  "improvement_rate": 0.23684210526315788,
  "regressions": 5,
  "regression_rate": 0.13157894736842105,
  "unchanged": 24,
  "control_correct_shadow_worse": 5,
  "document_added_explanation": 4,
  "document_disambiguated_context": 0,
  "document_provided_source": 9,
  "document_did_not_help": 24,
  "document_noise": 5,
  "document_added_missing_context": 5,
  "structured_data_already_sufficient": 0,
  "classifications": {
    "DOCUMENT_CORPUS_UNAVAILABLE": 2,
    "DOCUMENT_DID_NOT_HELP": 24,
    "DOCUMENT_NOISE": 5,
    "DOCUMENT_ADDED_MISSING_CONTEXT": 5,
    "VERIFIER_FAILURE": 1,
    "DOCUMENT_ADDED_EXPLANATION": 4
  },
  "post_corpus_classifications": {
    "DOCUMENT_DID_NOT_HELP": 24,
    "DOCUMENT_NOISE": 5,
    "DOCUMENT_ADDED_MISSING_CONTEXT": 5,
    "DOCUMENT_ADDED_EXPLANATION": 4
  },
  "metrics_scope": "post_corpus",
  "safety_metrics": {
    "wrong_context_false_acceptance": 0,
    "placeholder_false_acceptance": 0,
    "metadata_integrity_false_acceptance": 0,
    "metadata_false_acceptance": 0,
    "unsafe_adversarial_false_acceptance": 0,
    "shadow_errors_must_not_affect_production": true,
    "unsupported_claims": 33
  },
  "safety_blocked": false,
  "latency_metrics": {
    "shadow_mean_ms": 14180.368017526496,
    "shadow_p95_ms": 29815.052627250116,
    "retrieval_mean_ms": 83.3196082106425,
    "retrieval_p95_ms": 247.1640751974519
  },
  "observation_target_min": 100,
  "observation_target_max": 200,
  "v213c_comparison_note": "V2.13C was a controlled harness (59.7%\u219290.3% grounded-correct). V2.13D Phase 1 real-traffic observations are not statistically equivalent.",
  "phase1e": {
    "transitions": {
      "route": {
        "fallback\u2192fallback": 9,
        "retrieve_more\u2192retrieve_more": 1,
        "finish\u2192fallback": 2,
        "fallback\u2192finish": 5,
        "fallback\u2192retrieve_more": 9,
        "finish\u2192retrieve_more": 2,
        "finish\u2192finish": 5,
        "retrieve_more\u2192finish": 1,
        "clarify\u2192fallback": 1,
        "clarify\u2192clarify": 1,
        "fallback\u2192clarify": 1,
        "retrieve_more\u2192clarify": 1
      },
      "verifier": {
        "retrieve_more\u2192retrieve_more": 13,
        "fallback\u2192fallback": 2,
        "accept\u2192retrieve_more": 4,
        "retrieve_more\u2192accept": 6,
        "retrieve_more\u2192fallback": 3,
        "accept\u2192accept": 5,
        "clarify\u2192fallback": 1,
        "clarify\u2192clarify": 1,
        "fallback\u2192retrieve_more": 1,
        "retrieve_more\u2192clarify": 2
      },
      "mapper": {
        "reject\u2192reject": 10,
        "reject\u2192retrieve_more": 2,
        "accept\u2192reject": 4,
        "retrieve_more\u2192accept": 4,
        "retrieve_more\u2192reject": 10,
        "retrieve_more\u2192retrieve_more": 1,
        "reject\u2192accept": 2,
        "accept\u2192accept": 5
      },
      "key_route_transitions": {
        "fallback\u2192finish": 5,
        "retrieve_more\u2192finish": 1,
        "finish\u2192fallback": 2,
        "finish\u2192retrieve_more": 2
      }
    },
    "document_effect_counts": {
      "neutral": 24,
      "hurt": 5,
      "helped": 9
    },
    "retrieval_quality_counts": {
      "retrieval_non_decisive": 23,
      "retrieval_noise": 5,
      "retrieval_decisive": 9,
      "retrieval_irrelevant": 1
    },
    "regression_cause_counts": {
      "GENERATOR_DOCUMENT_DRIFT": 4,
      "METADATA_EFFECT": 1
    },
    "structured_sufficient_document_retrieved": 9,
    "control_correct_shadow_same": 4,
    "control_correct_shadow_better": 5,
    "control_correct_shadow_worse": 5,
    "segmentation": {
      "by_category": {
        "mixed": {
          "DOCUMENT_DID_NOT_HELP": 13,
          "DOCUMENT_NOISE": 5,
          "DOCUMENT_ADDED_MISSING_CONTEXT": 1,
          "DOCUMENT_ADDED_EXPLANATION": 3
        },
        "insufficient_evidence": {
          "DOCUMENT_ADDED_MISSING_CONTEXT": 3,
          "DOCUMENT_DID_NOT_HELP": 4
        },
        "document_oriented": {
          "DOCUMENT_DID_NOT_HELP": 5
        },
        "ambiguous": {
          "DOCUMENT_ADDED_MISSING_CONTEXT": 1,
          "DOCUMENT_DID_NOT_HELP": 2
        },
        "adversarial_suspicious": {
          "DOCUMENT_ADDED_EXPLANATION": 1
        }
      },
      "by_grade": {
        "CLASS_4": {
          "DOCUMENT_DID_NOT_HELP": 9,
          "DOCUMENT_NOISE": 2,
          "DOCUMENT_ADDED_EXPLANATION": 3
        },
        "unknown": {
          "DOCUMENT_DID_NOT_HELP": 13,
          "DOCUMENT_ADDED_MISSING_CONTEXT": 5,
          "DOCUMENT_NOISE": 1,
          "DOCUMENT_ADDED_EXPLANATION": 1
        },
        "CLASS_1": {
          "DOCUMENT_DID_NOT_HELP": 1,
          "DOCUMENT_NOISE": 1
        },
        "CLASS_5": {
          "DOCUMENT_DID_NOT_HELP": 1
        },
        "JSS_3": {
          "DOCUMENT_NOISE": 1
        }
      },
      "by_subject": {
        "MATHEMATICS": {
          "DOCUMENT_DID_NOT_HELP": 3,
          "DOCUMENT_NOISE": 2,
          "DOCUMENT_ADDED_MISSING_CONTEXT": 1
        },
        "unknown": {
          "DOCUMENT_DID_NOT_HELP": 15,
          "DOCUMENT_NOISE": 3,
          "DOCUMENT_ADDED_MISSING_CONTEXT": 1,
          "DOCUMENT_ADDED_EXPLANATION": 4
        },
        "SCIENCE": {
          "DOCUMENT_ADDED_MISSING_CONTEXT": 3,
          "DOCUMENT_DID_NOT_HELP": 6
        }
      },
      "by_source": {
        "bec-framework-2020": {
          "DOCUMENT_DID_NOT_HELP": 24,
          "DOCUMENT_NOISE": 5,
          "DOCUMENT_ADDED_MISSING_CONTEXT": 5,
          "DOCUMENT_ADDED_EXPLANATION": 4
        },
        "math-primary-guidance": {
          "DOCUMENT_DID_NOT_HELP": 15,
          "DOCUMENT_NOISE": 3,
          "DOCUMENT_ADDED_MISSING_CONTEXT": 2,
          "DOCUMENT_ADDED_EXPLANATION": 4
        },
        "science-guidance": {
          "DOCUMENT_DID_NOT_HELP": 10,
          "DOCUMENT_ADDED_MISSING_CONTEXT": 4
        }
      }
    },
    "document_helped": 9,
    "document_neutral": 24,
    "document_hurt": 5
  },
  "document_helped": 9,
  "document_neutral": 24,
  "document_hurt": 5,
  "structured_sufficient_document_retrieved": 9,
  "transitions": {
    "route": {
      "fallback\u2192fallback": 9,
      "retrieve_more\u2192retrieve_more": 1,
      "finish\u2192fallback": 2,
      "fallback\u2192finish": 5,
      "fallback\u2192retrieve_more": 9,
      "finish\u2192retrieve_more": 2,
      "finish\u2192finish": 5,
      "retrieve_more\u2192finish": 1,
      "clarify\u2192fallback": 1,
      "clarify\u2192clarify": 1,
      "fallback\u2192clarify": 1,
      "retrieve_more\u2192clarify": 1
    },
    "verifier": {
      "retrieve_more\u2192retrieve_more": 13,
      "fallback\u2192fallback": 2,
      "accept\u2192retrieve_more": 4,
      "retrieve_more\u2192accept": 6,
      "retrieve_more\u2192fallback": 3,
      "accept\u2192accept": 5,
      "clarify\u2192fallback": 1,
      "clarify\u2192clarify": 1,
      "fallback\u2192retrieve_more": 1,
      "retrieve_more\u2192clarify": 2
    },
    "mapper": {
      "reject\u2192reject": 10,
      "reject\u2192retrieve_more": 2,
      "accept\u2192reject": 4,
      "retrieve_more\u2192accept": 4,
      "retrieve_more\u2192reject": 10,
      "retrieve_more\u2192retrieve_more": 1,
      "reject\u2192accept": 2,
      "accept\u2192accept": 5
    },
    "key_route_transitions": {
      "fallback\u2192finish": 5,
      "retrieve_more\u2192finish": 1,
      "finish\u2192fallback": 2,
      "finish\u2192retrieve_more": 2
    }
  },
  "segmentation": {
    "by_category": {
      "mixed": {
        "DOCUMENT_DID_NOT_HELP": 13,
        "DOCUMENT_NOISE": 5,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 1,
        "DOCUMENT_ADDED_EXPLANATION": 3
      },
      "insufficient_evidence": {
        "DOCUMENT_ADDED_MISSING_CONTEXT": 3,
        "DOCUMENT_DID_NOT_HELP": 4
      },
      "document_oriented": {
        "DOCUMENT_DID_NOT_HELP": 5
      },
      "ambiguous": {
        "DOCUMENT_ADDED_MISSING_CONTEXT": 1,
        "DOCUMENT_DID_NOT_HELP": 2
      },
      "adversarial_suspicious": {
        "DOCUMENT_ADDED_EXPLANATION": 1
      }
    },
    "by_grade": {
      "CLASS_4": {
        "DOCUMENT_DID_NOT_HELP": 9,
        "DOCUMENT_NOISE": 2,
        "DOCUMENT_ADDED_EXPLANATION": 3
      },
      "unknown": {
        "DOCUMENT_DID_NOT_HELP": 13,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 5,
        "DOCUMENT_NOISE": 1,
        "DOCUMENT_ADDED_EXPLANATION": 1
      },
      "CLASS_1": {
        "DOCUMENT_DID_NOT_HELP": 1,
        "DOCUMENT_NOISE": 1
      },
      "CLASS_5": {
        "DOCUMENT_DID_NOT_HELP": 1
      },
      "JSS_3": {
        "DOCUMENT_NOISE": 1
      }
    },
    "by_subject": {
      "MATHEMATICS": {
        "DOCUMENT_DID_NOT_HELP": 3,
        "DOCUMENT_NOISE": 2,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 1
      },
      "unknown": {
        "DOCUMENT_DID_NOT_HELP": 15,
        "DOCUMENT_NOISE": 3,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 1,
        "DOCUMENT_ADDED_EXPLANATION": 4
      },
      "SCIENCE": {
        "DOCUMENT_ADDED_MISSING_CONTEXT": 3,
        "DOCUMENT_DID_NOT_HELP": 6
      }
    },
    "by_source": {
      "bec-framework-2020": {
        "DOCUMENT_DID_NOT_HELP": 24,
        "DOCUMENT_NOISE": 5,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 5,
        "DOCUMENT_ADDED_EXPLANATION": 4
      },
      "math-primary-guidance": {
        "DOCUMENT_DID_NOT_HELP": 15,
        "DOCUMENT_NOISE": 3,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 2,
        "DOCUMENT_ADDED_EXPLANATION": 4
      },
      "science-guidance": {
        "DOCUMENT_DID_NOT_HELP": 10,
        "DOCUMENT_ADDED_MISSING_CONTEXT": 4
      }
    }
  },
  "regression_cause_counts": {
    "GENERATOR_DOCUMENT_DRIFT": 4,
    "METADATA_EFFECT": 1
  },
  "retrieval_quality_counts": {
    "retrieval_non_decisive": 23,
    "retrieval_noise": 5,
    "retrieval_decisive": 9,
    "retrieval_irrelevant": 1
  },
  "document_effect_counts": {
    "neutral": 24,
    "hurt": 5,
    "helped": 9
  },
  "phase1e_investigate_signal": false,
  "phase1_status": "INSUFFICIENT_SAMPLE",
  "phase1_recommendation": "CONTINUE SHADOW",
  "canary_recommendation": "CANARY_NOT_READY",
  "canary_note": "Only 38 post-corpus successful real shadow evaluations (pre-corpus=2); target is 100\u2013200 before a rollout recommendation.",
  "pipeline_verification": {
    "classification": "PIPELINE_OPERATIONAL",
    "live_qa_metrics_total_requests": 1,
    "production_jsonl_rows": 41,
    "funnel_stages": {
      "request_seen": 3024,
      "shadow_eligible": 3024,
      "shadow_sampled": 43,
      "shadow_not_sampled": 2981,
      "shadow_started": 41,
      "shadow_completed": 40,
      "shadow_failed": 1,
      "shadow_persisted": 41,
      "persist_error": 0
    },
    "config_enabled": true,
    "sample_rate": 0.01,
    "jsonl_path": "/home/afiz/Projects/side/curriculum-agent/data/diagnostics/v213d_shadow.jsonl",
    "stages_checklist": {
      "qa_request": "PASS",
      "hook": "PASS",
      "sampling": "PASS",
      "shadow": "PASS",
      "persistence": "PASS"
    }
  },
  "generated_at": "2026-09-10T06:54:26.524724+00:00",
  "active_configuration": {
    "shadow_enabled": true,
    "sample_rate": 0.01,
    "document_retrieval": true,
    "retrieval_variant": "context_hybrid",
    "timeout_seconds": 30.0,
    "v213f_document_arbitration_experiment": false,
    "v213f_arbitration_policy": "C_ARBITRATED"
  },
  "real_traffic_observed": true
}
```
