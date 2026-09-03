# V2.13D Phase 1 — Production Shadow Observation

Generated: `2026-09-03T09:56:07.497727+00:00`

## Executive status

**INSUFFICIENT_SAMPLE**

Only 0 successful real shadow evaluations; target is 100–200 before a rollout recommendation.

## Configuration

```text
v213d_shadow_enabled=True
v213d_shadow_sample_rate=0.01
v213d_shadow_document_retrieval=True
v213d_shadow_retrieval_variant=context_hybrid
v213d_shadow_timeout_seconds=30.0
```

## Sample

```json
{
  "total_production_requests": 0,
  "sampled": 0,
  "completed": 0,
  "errors": 0,
  "timeouts": 0,
  "observation_target": [
    100,
    200
  ],
  "source": "production_shadow"
}
```

## Retrieval performance

```json
{
  "retrieval_success_rate": 0.0,
  "mean_retrieval_latency": 0.0,
  "p95_retrieval_latency": 0.0,
  "mean_passages_retrieved": 0.0,
  "provenance_complete_rate": 0.0
}
```

## Grounding safety

```json
{
  "wrong_context_false_acceptance": 0,
  "placeholder_false_acceptance": 0,
  "metadata_integrity_false_acceptance": 0,
  "metadata_false_acceptance": 0,
  "unsafe_adversarial_false_acceptance": 0,
  "shadow_errors_must_not_affect_production": true,
  "unsupported_claims": 0
}
```

## Product impact

```json
{
  "newly_recoverable": 0,
  "newly_recoverable_rate": 0.0,
  "improvements": 0,
  "unchanged": 0,
  "regressions": 0,
  "control_correct_shadow_worse": 0,
  "document_added_explanation": 0,
  "document_disambiguated_context": 0,
  "document_did_not_help": 0,
  "document_noise": 0
}
```

## Qualitative examples (anonymized)

1. Document retrieval improving an answer:

```json
[]
```

2. Structured data already sufficient:

```json
[]
```

3. Document retrieval did not help:

```json
[]
```

4. Document noise:

```json
[]
```

5. Regressions:

```json
[]
```

6. Safety violations / shadow failures:

```json
{
  "safety": [],
  "shadow_failure": []
}
```

## Distinctions

- This report counts **real production shadow** records only when `source=production_shadow`.
- Controlled V2.13C / Phase 0 replay is **not** mixed into Phase 1 success claims.

V2.13C was a controlled harness (59.7%→90.3% grounded-correct). V2.13D Phase 1 real-traffic observations are not statistically equivalent.

## Recommendation

CONTINUE SHADOW
