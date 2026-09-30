# V2.13G Live Document Arbitration Shadow

Generated: `2026-09-29T23:45:20.092219+00:00`

**Overall status: `ARBITRATION_CONFIRMED`**

**Recommendation: `ARBITRATION_CONFIRMED`**

**Classification determination: `INSUFFICIENT_INFORMATION`** (`INSUFFICIENT_INFORMATION`)

Live distribution is mixed; further inspection required.

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
  "live_n": 85,
  "control_accepted": {
    "false": 65,
    "true": 20
  },
  "structured_count_hist": {
    "0": 65,
    "5": 6,
    "9": 1,
    "13": 2,
    "14": 4,
    "15": 1,
    "22": 1,
    "27": 1,
    "31": 2,
    "37": 2
  },
  "sufficiency": {
    "INSUFFICIENT": 65,
    "SUFFICIENT": 20
  },
  "document_role": {
    "DECISIVE": 65,
    "REDUNDANT": 16,
    "IRRELEVANT": 4
  },
  "document_use": {
    "INCLUDE_IN_GENERATION": 65,
    "PROVENANCE_ONLY": 16,
    "DO_NOT_USE": 4
  },
  "categories": {
    "document_oriented": 19,
    "insufficient_evidence": 22,
    "ambiguous": 21,
    "adversarial_suspicious": 3,
    "mixed": 20
  },
  "curriculum_api_reachable_at_audit": true
}
```

Every live dual-arm row had `structured_count=0` and `control.final_accepted=false`.
With documents retrieved (n=5), the existing V2.13F rules correctly emit
`INSUFFICIENT → DECISIVE → INCLUDE_IN_GENERATION`.

This is **not** evidence that arbitration is ineffective. The primary V2.13F
regression cohort (`structured_sufficient_with_docs`) had live coverage **0**.

**Primary regression hypothesis (live): `TESTED`**

## B. Classification validity

`INSUFFICIENT_INFORMATION` — INSUFFICIENT_INFORMATION

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
  "structured_sufficient_with_docs": 20,
  "structured_sufficient_redundant_docs": 16,
  "structured_sufficient_irrelevant_docs": 4,
  "structured_insufficient_decisive_docs": 65,
  "structured_insufficient_irrelevant_docs": 0,
  "neutral_document_cases": 42
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
    "result": "CONFIRMED",
    "note": "Live arbitrated regressions below baseline and at zero."
  },
  "H2": {
    "result": "CONFIRMED",
    "note": "Arbitrated recoveries (9) preserved vs baseline (9)."
  },
  "H3": {
    "result": "CONFIRMED",
    "note": "Unsupported claims fell 9\u21922 (live sufficient+doc cohort=20)."
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

`ARBITRATION_CONFIRMED`

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
    "baseline": 9,
    "arbitrated": 9
  },
  "neutral": {
    "baseline": 3,
    "arbitrated": 23
  },
  "hurt": {
    "baseline": 4,
    "arbitrated": 0
  },
  "regressions": {
    "baseline": 4,
    "arbitrated": 0
  },
  "generator_drift": {
    "baseline": 4,
    "arbitrated": 0
  },
  "unsupported_claims": {
    "baseline": 9,
    "arbitrated": 2
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
  "mean_retrieval_ms": 6.552,
  "mean_arbitration_ms": 0.084,
  "mean_baseline_shadow_ms": 12541.51,
  "mean_arbitrated_shadow_ms": 2918.319
}
```

