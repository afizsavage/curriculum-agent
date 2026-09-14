# V2.13F Document Evidence Arbitration

Generated: `2026-09-14T13:27:23.823862+00:00`

**Decision: `ARBITRATION_SUPPORTED`**

On a frozen real-traffic replay set (n=50), structured-first / arbitrated
policies eliminate all 5 document-induced regressions (including
`GENERATOR_DOCUMENT_DRIFT` cases such as `dd1c57…`) while preserving all 15
document recoveries. Hard safety gates remain zero. Production arbitration
remains disabled.

## Phase 1E freeze (V2.13D baseline, post_corpus_n=50)

Before arbitration replay, Phase 1E was stopped at n≈50 and frozen:

| Metric | Value |
| --- | ---: |
| post_corpus_successful | 50 |
| document_helped | 20 |
| document_neutral | 25 |
| document_hurt | 5 |
| control_correct_shadow_worse | 5 |
| wrong-context / placeholder / metadata false accepts | 0 / 0 / 0 |
| retrieval_success | 1.0 |
| corpus | bec-framework-2020, math-primary-guidance, science-guidance (unchanged) |
| `v213f_document_arbitration_experiment` | false |
| V2.13E | disabled |

JSONL SHA256: `4dcbca7a46ca11e603552eb520cb0ac7703f4582247e9ecb995e43005167ca83`

## 1. Hypothesis

Document retrieval is operational, but unconditional document-conditioned
regeneration can introduce answer drift when structured evidence is already
sufficient. Deterministic evidence arbitration can withhold redundant documents
from generation while still allowing decisive documents to recover insufficient
structured answers.

## 2. Frozen dataset definition

```json
{
  "path": "/home/afiz/Projects/curriculumz/curriculum-agent/data/diagnostics/v213f_replay/dataset.json",
  "n": 50,
  "created_at": "2026-09-14T13:27:23.330235+00:00",
  "group_pool_sizes": {
    "D_regression": 5,
    "B_structured_insufficient_recovery": 15,
    "A_structured_sufficient": 5,
    "C_neutral": 25
  },
  "selected_groups": {
    "A_structured_sufficient": 5,
    "B_structured_insufficient_recovery": 15,
    "C_neutral": 25,
    "D_regression": 5
  }
}
```

## 3. V2.13D baseline

Variant A (`A_BASELINE_MERGE`) always merges retrieved documents into generation
(frozen V2.13D shadow outcomes).

## 4. Arbitration variants

- **A_BASELINE_MERGE** — current V2.13D behavior
- **B_STRUCTURED_FIRST** — if control accepted, documents are provenance-only
- **C_ARBITRATED** — deterministic role → use mapping (DECISIVE/SUPPORTING/...)

Arbitration is non-LLM and shadow/replay-only.
`v213f_document_arbitration_experiment=False`

## 5. Structured-sufficiency classification

```json
{
  "SUFFICIENT": 10,
  "INSUFFICIENT": 40
}
```

## 6. Document-role classification

```json
{
  "REDUNDANT": 7,
  "IRRELEVANT": 3,
  "DECISIVE": 40
}
```

## 7. Recovery results

- Baseline recoveries: **15**
- Structured-first recoveries: **15**
- Arbitrated recoveries: **15**

## 8. Regression results

- Baseline regressions: **5**
- Structured-first regressions: **0**
- Arbitrated regressions: **0**
- Generator-drift cases prevented (arbitrated): **5**

## 9. Generator-drift results

Offline counterfactual: when documents are withheld from generation,
the accepted control outcome is retained (no document-conditioned regeneration).

## 10. `dd1c57…` analysis

```json
{
  "question_hash": "dd1c57ff32af875b",
  "baseline": {
    "decision": {
      "document_role": "REDUNDANT",
      "structured_sufficiency": "SUFFICIENT",
      "document_use": "INCLUDE_IN_GENERATION",
      "policy": "A_BASELINE_MERGE",
      "reasons": [
        "structured_already_sufficient_docs_redundant"
      ],
      "relevant_passage_count": 1,
      "conflicting_passage_count": 0,
      "document_count": 5,
      "structured_count": 40
    },
    "counterfactual": {
      "policy": "A_BASELINE_MERGE",
      "document_use": "INCLUDE_IN_GENERATION",
      "document_role": "REDUNDANT",
      "structured_sufficiency": "SUFFICIENT",
      "outcome_source": "shadow",
      "final_accepted": false,
      "final_route": "fallback",
      "verifier_decision": "retrieve_more",
      "verifier_score": 0.6,
      "mapper_decision": "reject",
      "unsupported_claim_count": 1,
      "answer_length": null,
      "answer_hash": "fce7464c37160aa2",
      "control_correct_shadow_worse": true,
      "control_insufficient_shadow_accepted": false,
      "document_helped": false,
      "document_hurt": true,
      "document_neutral": false,
      "generator_drift_prevented": false
    }
  },
  "structured_first": {
    "decision": {
      "document_role": "REDUNDANT",
      "structured_sufficiency": "SUFFICIENT",
      "document_use": "PROVENANCE_ONLY",
      "policy": "B_STRUCTURED_FIRST",
      "reasons": [
        "structured_already_sufficient_docs_redundant"
      ],
      "relevant_passage_count": 1,
      "conflicting_passage_count": 0,
      "document_count": 5,
      "structured_count": 40
    },
    "counterfactual": {
      "policy": "B_STRUCTURED_FIRST",
      "document_use": "PROVENANCE_ONLY",
      "document_role": "REDUNDANT",
      "structured_sufficiency": "SUFFICIENT",
      "outcome_source": "control",
      "final_accepted": true,
      "final_route": "finish",
      "verifier_decision": "accept",
      "verifier_score": 1.0,
      "mapper_decision": "accept",
      "unsupported_claim_count": 0,
      "answer_length": null,
      "answer_hash": "16088fa24c2d597f",
      "control_correct_shadow_worse": false,
      "control_insufficient_shadow_accepted": false,
      "document_helped": false,
      "document_hurt": false,
      "document_neutral": true,
      "generator_drift_prevented": true
    }
  },
  "arbitrated": {
    "decision": {
      "document_role": "REDUNDANT",
      "structured_sufficiency": "SUFFICIENT",
      "document_use": "PROVENANCE_ONLY",
      "policy": "C_ARBITRATED",
      "reasons": [
        "structured_already_sufficient_docs_redundant"
      ],
      "relevant_passage_count": 1,
      "conflicting_passage_count": 0,
      "document_count": 5,
      "structured_count": 40
    },
    "counterfactual": {
      "policy": "C_ARBITRATED",
      "document_use": "PROVENANCE_ONLY",
      "document_role": "REDUNDANT",
      "structured_sufficiency": "SUFFICIENT",
      "outcome_source": "control",
      "final_accepted": true,
      "final_route": "finish",
      "verifier_decision": "accept",
      "verifier_score": 1.0,
      "mapper_decision": "accept",
      "unsupported_claim_count": 0,
      "answer_length": null,
      "answer_hash": "16088fa24c2d597f",
      "control_correct_shadow_worse": false,
      "control_insufficient_shadow_accepted": false,
      "document_helped": false,
      "document_hurt": false,
      "document_neutral": true,
      "generator_drift_prevented": true
    }
  },
  "interpretation": "If structured_sufficiency=SUFFICIENT and document_use=PROVENANCE_ONLY, generation stays structured-first and the accepted control outcome is retained."
}
```

## 11. Safety results

```json
{
  "baseline": {},
  "structured_first": {},
  "arbitrated": {}
}
```

## 12. Latency

Arbitration is deterministic metadata/score logic only (no extra LLM).
Mean arbitration overhead on replay set: `0.138 ms`.

## 13. Per-category results

```json
{
  "D_regression": {
    "helped": 0,
    "hurt": 0,
    "neutral": 5
  },
  "B_structured_insufficient_recovery": {
    "helped": 14,
    "hurt": 0,
    "neutral": 1
  },
  "A_structured_sufficient": {
    "helped": 0,
    "hurt": 0,
    "neutral": 5
  },
  "C_neutral": {
    "helped": 7,
    "hurt": 0,
    "neutral": 18
  }
}
```

## 14. Subject/grade segmentation

```json
{
  "by_subject": {
    "unknown": {
      "helped": 7,
      "hurt": 0,
      "neutral": 23
    },
    "MATHEMATICS": {
      "helped": 4,
      "hurt": 0,
      "neutral": 4
    },
    "SCIENCE": {
      "helped": 10,
      "hurt": 0,
      "neutral": 2
    }
  },
  "by_grade": {
    "CLASS_4": {
      "helped": 3,
      "hurt": 0,
      "neutral": 16
    },
    "unknown": {
      "helped": 16,
      "hurt": 0,
      "neutral": 10
    },
    "JSS_3": {
      "helped": 0,
      "hurt": 0,
      "neutral": 1
    },
    "CLASS_1": {
      "helped": 0,
      "hurt": 0,
      "neutral": 2
    },
    "CLASS_5": {
      "helped": 2,
      "hurt": 0,
      "neutral": 0
    }
  }
}
```

## Comparison table

Primary (required):

| Metric                      | V2.13D | V2.13F |
| --------------------------- | -----: | -----: |
| Document helped             |     15 |     15 |
| Document neutral            |     30 |     35 |
| Document hurt               |      5 |      0 |
| Recoveries                  |     15 |     15 |
| Regressions                 |      5 |      0 |
| Generator drift             |      5 |      0 |
| Unsupported claims          |     34 |     28 |
| Wrong-context false accepts |      0 |      0 |
| Placeholder false accepts   |      0 |      0 |
| Metadata false accepts      |      0 |      0 |

Detail (includes structured-first):

| Metric                      | V2.13D | B-SF   | V2.13F |
| --------------------------- | -----: | -----: | -----: |
| Document helped             |     15 |     15 |     15 |
| Document neutral            |     30 |     35 |     35 |
| Document hurt               |      5 |      0 |      0 |
| Recoveries                  |     15 |     15 |     15 |
| Regressions                 |      5 |      0 |      0 |
| Generator drift prevented   |      0 |      5 |      5 |
| Unsupported claims          |     34 |     28 |     28 |
| Wrong-context false accepts |      0 |      0 |      0 |
| Placeholder false accepts   |      0 |      0 |      0 |
| Metadata false accepts      |      0 |      0 |      0 |

Method note: offline counterfactual on frozen shadows — `INCLUDE_IN_GENERATION`
keeps baseline shadow outcomes; `PROVENANCE_ONLY` / `DO_NOT_USE` / `REQUIRE_REVIEW`
retain control outcomes. No additional LLM is used for arbitration.

## 15. Recommendation

**Decision: `ARBITRATION_SUPPORTED`**

Regressions eliminated while recoveries preserved; safety gates remain zero. Do not enable in production yet — promote only after larger confirmation.

V2.13E remains disabled. Arbitration is not enabled in production.

## Production isolation

```text
{
  "v213d_shadow_enabled": true,
  "v213d_shadow_sample_rate": 0.01,
  "v213f_document_arbitration_experiment": false,
  "v213f_arbitration_policy": "C_ARBITRATED",
  "v213e": "disabled"
}
```

