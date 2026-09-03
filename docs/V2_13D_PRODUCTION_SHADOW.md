# V2.13D — Production Shadow for Context-Hybrid Curriculum Document Evidence

Generated: `2026-09-03T07:31:35.685816+00:00`

**Canary recommendation: CANARY_NOT_READY**

Local replay is operationally stable and shows document grounding gains, but real production traffic has not been collected. Enable shadow at a low sample rate; do not promote document retrieval.

## Executive summary

V2.13D is a **shadow-only** evaluation. Production LangGraph responses are unchanged. Document retrieval runs after the user response is determined, in a failure-isolated thread.

V2.13C was a controlled harness (59.7%→90.3% grounded-correct). V2.13D replay is not statistically equivalent to that dataset.

## Replay vs production traffic

This report is from **controlled local replay** (Phase 0), not observed production traffic. Do not treat these rates as equivalent to V2.13C's 72-question harness.

## Metrics

```json
{
  "experiment": "v2.13d",
  "schema_version": "v213d.1",
  "traffic_sampled": 5,
  "successful_shadow_evaluations": 4,
  "shadow_errors": 1,
  "shadow_error_rate": 0.2,
  "retrieval_success": 1.0,
  "provenance_complete_rate": 1.0,
  "metadata_valid_rate": 1.0,
  "newly_recoverable_count": 1,
  "newly_recoverable_rate": 0.2,
  "improvements": 1,
  "regressions": 0,
  "unchanged": 3,
  "classifications": {
    "DOCUMENT_ADDED_GROUNDING": 1,
    "BOTH_ACCEPTED": 2,
    "SHADOW_ERROR": 1,
    "BOTH_REJECTED": 1
  },
  "safety_metrics": {
    "wrong_context_false_acceptance": 0,
    "placeholder_false_acceptance": 0,
    "metadata_integrity_false_acceptance": 0,
    "unsafe_adversarial_false_acceptance": 0,
    "shadow_errors_must_not_affect_production": true
  },
  "latency_metrics": {
    "shadow_mean_ms": 37.402367750473786,
    "retrieval_mean_ms": 30.220322749300976
  },
  "canary_recommendation": "CANARY_NOT_READY",
  "canary_note": "Local replay is operationally stable and shows document grounding gains, but real production traffic has not been collected. Enable shadow at a low sample rate; do not promote document retrieval.",
  "v213c_comparison_note": "V2.13C was a controlled harness (59.7%\u219290.3% grounded-correct). V2.13D replay is not statistically equivalent to that dataset.",
  "mode": "controlled_replay"
}
```

## Safety gates

```json
{
  "wrong_context_false_acceptance": 0,
  "placeholder_false_acceptance": 0,
  "metadata_integrity_false_acceptance": 0,
  "unsafe_adversarial_false_acceptance": 0,
  "shadow_errors_must_not_affect_production": true
}
```

## Production integrity

- LangGraph production path unchanged (shadow scheduled after response)
- Answer generator / verifier / mapper / V2.9 / V2.11 unchanged
- `/api/v1` unchanged
- `v213d_shadow_enabled=false`, `v213d_shadow_sample_rate=0` by default
- Shadow exceptions cannot propagate to the production caller

## Qualitative analysis (Phase 0 replay)

1. **Control insufficient → shadow grounded:** `V213C-A01` (purpose of mathematics education). Control `retrieve_more` / 0 structured evidence; shadow retrieved 5 document passages, verifier `accept`, route `finish`.
2. **Control correct → shadow unchanged:** `V213C-D01` structured money LOs — both accepted (`BOTH_ACCEPTED`).
3. **Document evidence added without changing a sufficient structured answer:** `V213C-B01` mixed — both accepted; document count 5, structured count 3.
4. **Document retrieval did not flip an adversarial reject:** `V213C-G03` placeholder LOs — mapper `reject` on both arms (`BOTH_REJECTED`); placeholder present, not accepted.
5. **Regressions:** none in this replay set.
6. **Isolated shadow failure:** `V213C-F01` forced retrieval `RuntimeError` at `document_retrieval` → `SHADOW_ERROR`; production control snapshot still recorded.

## Recommendation

CANARY_NOT_READY

Do **not** automatically enable document retrieval or increase sample rate.

Initial production shadow configuration if operators choose to collect traffic:

```text
v213d_shadow_enabled=true
v213d_shadow_sample_rate=0.01
v213d_shadow_document_retrieval=true
v213d_shadow_retrieval_variant=context_hybrid
```