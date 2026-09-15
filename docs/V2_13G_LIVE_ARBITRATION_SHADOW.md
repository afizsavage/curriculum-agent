# V2.13G Live Document Arbitration Shadow

Generated: `2026-09-15T12:09:59.097590+00:00`

**Overall status: `INVESTIGATE_BEFORE_PROMOTION`**

**Recommendation: `PROCEED_TO_TARGETED_LIVE_VALIDATION`**

**Classification determination: `SAMPLING_BIAS`** (`CLASSIFICATION_CORRECT`)

All live dual-arm rows have structured_count=0 and control_accepted=false. Classification is consistent with that evidence. The live sample did not include any structured-sufficient/document-redundant cases, so the primary V2.13F regression hypothesis was untested.

## Hard isolation (unchanged)

```text
{
  "v213d_shadow_enabled": true,
  "v213d_shadow_sample_rate": 0.01,
  "v213d_shadow_document_retrieval": true,
  "v213d_shadow_retrieval_variant": "context_hybrid",
  "v213f_document_arbitration_experiment": false,
  "v213f_arbitration_policy": "C_ARBITRATED",
  "v213g_live_arbitration_shadow": true,
  "v213e": "disabled",
  "curriculum_api_url": "http://127.0.0.1:8000"
}
```

## A. Root cause of 52/52 INSUFFICIENT + DECISIVE

```json
{
  "live_n": 52,
  "control_accepted": {
    "false": 52
  },
  "structured_count_hist": {
    "0": 52
  },
  "sufficiency": {
    "INSUFFICIENT": 52
  },
  "document_role": {
    "DECISIVE": 52
  },
  "document_use": {
    "INCLUDE_IN_GENERATION": 52
  },
  "categories": {
    "document_oriented": 17,
    "insufficient_evidence": 17,
    "ambiguous": 15,
    "adversarial_suspicious": 3
  },
  "curriculum_api_reachable_at_audit": true
}
```

Every live dual-arm row had `structured_count=0` and `control.final_accepted=false`.
With documents retrieved (n=5), the existing V2.13F rules correctly emit
`INSUFFICIENT → DECISIVE → INCLUDE_IN_GENERATION`.

This is **not** evidence that arbitration is ineffective. The primary V2.13F
regression cohort (`structured_sufficient_with_docs`) had live coverage **0**.

**Primary regression hypothesis (live): `PRIMARY_REGRESSION_HYPOTHESIS_UNTESTED`**

## B. Classification validity

`CLASSIFICATION_CORRECT` — SAMPLING_BIAS

Checked: no silent default to INSUFFICIENT when control is accepted;
sufficiency is decided from control acceptance / structured count **before**
document role mapping; live control snapshots match zero structured evidence
(not shadow evidence-loss).

## C. V2.13F replay compatibility (TARGETED_REPLAY)

Frozen V2.13F replay cases were passed through the current classifier
without modifying the freeze.

```json
{
  "n": 50,
  "classifier_distribution": {
    "SUFFICIENT|REDUNDANT|PROVENANCE_ONLY": 7,
    "SUFFICIENT|IRRELEVANT|DO_NOT_USE": 3,
    "INSUFFICIENT|DECISIVE|INCLUDE_IN_GENERATION": 40
  },
  "group_compatibility": {
    "D_regression": {
      "n": 5,
      "compatible": 5,
      "expected": "SUFFICIENT + REDUNDANT/IRRELEVANT + withhold docs"
    },
    "B_structured_insufficient_recovery": {
      "n": 15,
      "compatible": 15,
      "expected": "INSUFFICIENT + DECISIVE + INCLUDE"
    },
    "A_structured_sufficient": {
      "n": 5,
      "compatible": 5,
      "expected": "SUFFICIENT + REDUNDANT/IRRELEVANT + PROVENANCE_ONLY/DO_NOT_USE"
    },
    "C_neutral": {
      "n": 25,
      "compatible": 25,
      "expected": "mixed; often INSUFFICIENT if control not accepted"
    }
  },
  "cohort_counts": {
    "structured_sufficient_with_docs": 10,
    "structured_sufficient_redundant_docs": 7,
    "structured_sufficient_irrelevant_docs": 3,
    "structured_insufficient_decisive_docs": 40,
    "structured_insufficient_irrelevant_docs": 0,
    "neutral_document_cases": 35
  }
}
```

### dd1c57 permanent fixture

```json
{
  "question_hash": "dd1c57ff32af875b",
  "request_id": "63bbf28e5f0db6e1",
  "v213f_group": "D_regression",
  "expected": {
    "structured_sufficiency": "SUFFICIENT",
    "document_role": "REDUNDANT",
    "document_use": "PROVENANCE_ONLY"
  },
  "observed": {
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
  },
  "pass": true
}
```

## D. Cohort coverage

### LIVE_TRAFFIC

```json
{
  "structured_sufficient_with_docs": 0,
  "structured_sufficient_redundant_docs": 0,
  "structured_sufficient_irrelevant_docs": 0,
  "structured_insufficient_decisive_docs": 52,
  "structured_insufficient_irrelevant_docs": 0,
  "neutral_document_cases": 22
}
```

### TARGETED_REPLAY (from frozen V2.13F cases)

```json
{
  "structured_sufficient_with_docs": 10,
  "structured_sufficient_redundant_docs": 7,
  "structured_sufficient_irrelevant_docs": 3,
  "structured_insufficient_decisive_docs": 40,
  "structured_insufficient_irrelevant_docs": 0,
  "neutral_document_cases": 35
}
```

## E. Hypothesis results

```json
{
  "H1": {
    "result": "UNTESTED",
    "note": "PRIMARY_REGRESSION_HYPOTHESIS_UNTESTED on LIVE_TRAFFIC (structured_sufficient_with_docs=0). TARGETED_REPLAY reproduces V2.13F sufficient/redundant withholding including dd1c57."
  },
  "H2": {
    "result": "CONFIRMED",
    "note": "Arbitrated recoveries (36) preserved vs baseline (30)."
  },
  "H3": {
    "result": "PARTIALLY_CONFIRMED",
    "note": "Unsupported claims fell 30\u219217 (live sufficient+doc cohort=0)."
  },
  "safety": "PASS"
}
```

## F. Safety

```json
{
  "baseline": {
    "wrong_context_false_acceptance": 0,
    "placeholder_false_acceptance": 0,
    "metadata_false_acceptance": 0
  },
  "arbitrated": {
    "wrong_context_false_acceptance": 0,
    "placeholder_false_acceptance": 0,
    "metadata_false_acceptance": 0
  },
  "safety_blocked": false
}
```

## G. Recommendation

`PROCEED_TO_TARGETED_LIVE_VALIDATION`

Do **not** promote arbitration to production.
Do **not** raise `sample_rate` above 0.01.
Do **not** overwrite V2.13F freeze/replay artifacts.

## H. Next experiment

1. Keep Curriculum Structure API available so live traffic can resolve structured evidence.
2. Continue V2.13G dual-arm shadow at sample_rate=0.01 until live
   `structured_sufficient_with_docs >= 20` (and retain insufficient+decisive recovery coverage).
3. Only then expand toward n=100 / n=200 dual-arm comparisons.
4. Keep TARGETED_REPLAY cohort diagnostics additive alongside LIVE_TRAFFIC.

## Live comparison table (unchanged n=52 snapshot)

```json
{
  "recoveries": {
    "baseline": 30,
    "arbitrated": 36
  },
  "neutral": {
    "baseline": 22,
    "arbitrated": 16
  },
  "hurt": {
    "baseline": 0,
    "arbitrated": 0
  },
  "regressions": {
    "baseline": 0,
    "arbitrated": 0
  },
  "generator_drift": {
    "baseline": 0,
    "arbitrated": 0
  },
  "unsupported_claims": {
    "baseline": 30,
    "arbitrated": 17
  },
  "wrong_context_false_accepts": {
    "baseline": 0,
    "arbitrated": 0
  },
  "placeholder_false_accepts": {
    "baseline": 0,
    "arbitrated": 0
  },
  "metadata_false_accepts": {
    "baseline": 0,
    "arbitrated": 0
  }
}
```

## Latency (live)

```json
{
  "mean_retrieval_ms": 5.561,
  "mean_arbitration_ms": 0.071,
  "mean_baseline_shadow_ms": 7079.447,
  "mean_arbitrated_shadow_ms": 7003.629
}
```

