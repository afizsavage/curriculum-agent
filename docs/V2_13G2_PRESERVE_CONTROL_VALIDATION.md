# V2.13G.2 Preserve-Control-Answer Validation

Generated: `2026-09-15T15:31:40.564376+00:00`

**Decision: `PRESERVE_CONTROL_CONFIRMED`**

**Recommendation: `resume Stage-1 live shadow at sample_rate=0.01`**

Production arbitration remains **disabled**. Default semantics remain
`structured_only_regeneration` until explicitly enabled for shadow.

## A. Implementation

- Config: `V213G_PROVENANCE_ONLY_SEMANTICS` (`structured_only_regeneration` | `preserve_control_answer`)
- Shadow path: `app/agent/v213g_live_shadow.py` + `app/agent/v213g1_provenance_only.py`
- When `preserve_control_answer` and `document_use != INCLUDE_IN_GENERATION` and control accepted: skip `agent.answer()`, set `outcome_source=control`, `generation_attempted=false`, copy control hash/verifier/mapper/route
- Documents stay in `provenance_evidence` only (`generation_document_count=0`)
- Validation: paired Mode A/B on frozen V2.13G.1 targeted JSONL (no resample; historical files not overwritten)

## B. Semantics

```text
PROVENANCE_ONLY
  → documents excluded from generation_evidence
  → retained as provenance_evidence

PRESERVE_CONTROL_ANSWER
  → if control accepted and document_use != INCLUDE_IN_GENERATION:
       do not regenerate; final answer == control answer

INCLUDE_IN_GENERATION
  → documents available to generation; regeneration permitted
  → preserve-control gate does NOT apply
```

## C. Targeted sufficient+redundant (n=17)

| Metric | Regeneration (A) | Preserve Control (B) |
| --- | ---: | ---: |
| PROVENANCE_ONLY cases | 17 | 17 |
| generation document leaks | 0 | 0 |
| regeneration attempted | 17 | 0 |
| control hash matches | 0 | 17 |
| answer regressions | 2 | 0 |
| verifier regressions | 2 | 0 |
| mapper regressions | 2 | 0 |
| control-preservation violations | None | 0 |

## D. H2 decisive recovery

```json
{
  "n": 5,
  "baseline_recovery": 5,
  "arbitrated_recovery_mode_a": 5,
  "arbitrated_recovery_mode_b": 5,
  "generation_document_count_sum_mode_a": 25,
  "generation_document_count_sum_mode_b": 25,
  "preserve_wrongly_applied": 0
}
```

## E. H3 / drift

Mode A (regeneration): control-regeneration drift produced arbitrated regressions.
Mode B (preserve): answer difference vs control is **0** for all preserved cases;
document-induced drift is not introduced because generation is not re-run.
Baseline document-merge contamination (2 cases) is unchanged — baseline arm still merges.

## F. Five prior regression cases

```json
{
  "question_hash": "9cd42c44d005d3b3",
  "role": "REDUNDANT",
  "use": "PROVENANCE_ONLY",
  "mode_a": {
    "semantics": "structured_only_regeneration",
    "generation_attempted": true,
    "outcome_source": "regenerated",
    "generation_document_count": 0,
    "answer_hash": "60d940d9f643ff1f",
    "hash_match_control": false,
    "final_accepted": true,
    "verifier": "accept",
    "mapper": "accept",
    "route": "finish",
    "regression": false,
    "evidence_snapshot": "552eff9a79f09c3d"
  },
  "mode_b": {
    "semantics": "preserve_control_answer",
    "preserve_applied": true,
    "generation_attempted": false,
    "outcome_source": "control",
    "generation_document_count": 0,
    "answer_hash": "a751fce50ec280c1",
    "hash_match_control": true,
    "final_accepted": true,
    "verifier": "accept",
    "mapper": "accept",
    "route": "finish",
    "regression": false,
    "evidence_snapshot": "552eff9a79f09c3d",
    "generation_evidence_fingerprint": "552eff9a79f09c3d",
    "provenance_evidence_fingerprint": "0b3183083069b4da"
  }
}
```

```json
{
  "question_hash": "fd894182926a5f8b",
  "role": "REDUNDANT",
  "use": "PROVENANCE_ONLY",
  "mode_a": {
    "semantics": "structured_only_regeneration",
    "generation_attempted": true,
    "outcome_source": "regenerated",
    "generation_document_count": 0,
    "answer_hash": "38b84a8ebf5aff77",
    "hash_match_control": false,
    "final_accepted": false,
    "verifier": "retrieve_more",
    "mapper": "reject",
    "route": "retrieve_more",
    "regression": true,
    "evidence_snapshot": "552eff9a79f09c3d"
  },
  "mode_b": {
    "semantics": "preserve_control_answer",
    "preserve_applied": true,
    "generation_attempted": false,
    "outcome_source": "control",
    "generation_document_count": 0,
    "answer_hash": "6dcb51ba957698ee",
    "hash_match_control": true,
    "final_accepted": true,
    "verifier": "accept",
    "mapper": "accept",
    "route": "finish",
    "regression": false,
    "evidence_snapshot": "552eff9a79f09c3d",
    "generation_evidence_fingerprint": "552eff9a79f09c3d",
    "provenance_evidence_fingerprint": "79e3576191546faa"
  }
}
```

```json
{
  "question_hash": "7fb53cf210f937c5",
  "role": "IRRELEVANT",
  "use": "DO_NOT_USE",
  "mode_a": {
    "semantics": "structured_only_regeneration",
    "generation_attempted": true,
    "outcome_source": "regenerated",
    "generation_document_count": 0,
    "answer_hash": "ffc60bc78f45784e",
    "hash_match_control": false,
    "final_accepted": false,
    "verifier": "retrieve_more",
    "mapper": "retrieve_more",
    "route": "retrieve_more",
    "regression": true,
    "evidence_snapshot": "fe288367a544b900"
  },
  "mode_b": {
    "semantics": "preserve_control_answer",
    "preserve_applied": true,
    "generation_attempted": false,
    "outcome_source": "control",
    "generation_document_count": 0,
    "answer_hash": "3923be6df399e6dd",
    "hash_match_control": true,
    "final_accepted": true,
    "verifier": "accept",
    "mapper": "accept",
    "route": "finish",
    "regression": false,
    "evidence_snapshot": "fe288367a544b900",
    "generation_evidence_fingerprint": "fe288367a544b900",
    "provenance_evidence_fingerprint": "346c6a15c52def21"
  }
}
```

```json
{
  "question_hash": "27e64ac4d304e8d5",
  "role": "REDUNDANT",
  "use": "PROVENANCE_ONLY",
  "mode_a": {
    "semantics": "structured_only_regeneration",
    "generation_attempted": true,
    "outcome_source": "regenerated",
    "generation_document_count": 0,
    "answer_hash": "1449389263c9617c",
    "hash_match_control": false,
    "final_accepted": true,
    "verifier": "accept",
    "mapper": "accept",
    "route": "finish",
    "regression": false,
    "evidence_snapshot": "977b259fcfb4b282"
  },
  "mode_b": {
    "semantics": "preserve_control_answer",
    "preserve_applied": true,
    "generation_attempted": false,
    "outcome_source": "control",
    "generation_document_count": 0,
    "answer_hash": "a4d35997ce52c0a4",
    "hash_match_control": true,
    "final_accepted": true,
    "verifier": "accept",
    "mapper": "accept",
    "route": "finish",
    "regression": false,
    "evidence_snapshot": "977b259fcfb4b282",
    "generation_evidence_fingerprint": "977b259fcfb4b282",
    "provenance_evidence_fingerprint": "22500cea894ad45a"
  }
}
```

```json
{
  "question_hash": "3445d54939ac9ed9",
  "role": "REDUNDANT",
  "use": "PROVENANCE_ONLY",
  "mode_a": {
    "semantics": "structured_only_regeneration",
    "generation_attempted": true,
    "outcome_source": "regenerated",
    "generation_document_count": 0,
    "answer_hash": "1f071b2504ae5cff",
    "hash_match_control": false,
    "final_accepted": false,
    "verifier": "retrieve_more",
    "mapper": "retrieve_more",
    "route": "retrieve_more",
    "regression": true,
    "evidence_snapshot": "552eff9a79f09c3d"
  },
  "mode_b": {
    "semantics": "preserve_control_answer",
    "preserve_applied": true,
    "generation_attempted": false,
    "outcome_source": "control",
    "generation_document_count": 0,
    "answer_hash": "6c5922ec6f1bcabb",
    "hash_match_control": true,
    "final_accepted": true,
    "verifier": "accept",
    "mapper": "accept",
    "route": "finish",
    "regression": false,
    "evidence_snapshot": "552eff9a79f09c3d",
    "generation_evidence_fingerprint": "552eff9a79f09c3d",
    "provenance_evidence_fingerprint": "e950613915016b07"
  }
}
```

## G. Safety

Hard gates remain zero on the frozen cohort (no new adversarial generation in G.2).
Unit/integration safety suites remain green.

## H. Production isolation

```json
{
  "v213f_document_arbitration_experiment": false,
  "v213d_shadow_sample_rate": 0.01,
  "v213g_provenance_only_semantics_default": "structured_only_regeneration",
  "production_arbitration": "OFF",
  "note": "Preserve semantics are shadow-only via V213G_PROVENANCE_ONLY_SEMANTICS; default remains regeneration until explicitly enabled. Historical JSONL not overwritten."
}
```

## I. Decision

`PRESERVE_CONTROL_CONFIRMED`

```json
{
  "S1_control_preservation": true,
  "S2_no_generation_leakage": true,
  "S3_no_unnecessary_regeneration": true,
  "S4_regression_elimination": true,
  "S5_h2_preservation": true,
  "S6_safety": true,
  "S7_isolation": true
}
```

## J. Recommendation

resume Stage-1 live shadow at sample_rate=0.01

Do **not** enable production arbitration.
Do **not** expand to n=100/200 until Stage-1 live coverage is re-established
under preserve-control shadow semantics.
