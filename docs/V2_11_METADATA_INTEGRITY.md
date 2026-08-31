# V2.11 Adversarial Metadata-Integrity & Grounding Enforcement Experiment

Generated: `2026-08-31T11:29:43.195697+00:00`

**Conclusion: PARTIALLY_SUPPORTED**

Adversarial false acceptances eliminated with FC preserved and safety intact. FI mapped acceptance (70%) is below V2.10 baseline (~80%) but failures occur with metadata_valid=true and verifier fallback — attributable to verifier LLM variance, not metadata-guard regression.

## Architecture Question

Yes — experimentally validated as a deterministic pre-verifier guard. FI gap is verifier-side variance; metadata guard did not block valid FI evidence. Production-ready after shadow eval; verifier and mapper remain unchanged.

## Pipeline Variant Summaries (Variant C — Metadata Suppress)

```json
{
  "n": 280,
  "faithful_complete_acceptance": 1.0,
  "faithful_imperfect_acceptance": 0.7,
  "placeholder_false_acceptance": 0,
  "safety_false_acceptance": 0,
  "adversarial_false_acceptance": 0,
  "overall_acceptance": 0.096
}
```

## Adversarial Before/After

| Adversarial Case | V2.10 Baseline | V2.11 Metadata Guard | Correct Outcome |
| --- | ---: | ---: | --- |
| Fake parent | BLOCK | BLOCK | BLOCK |
| Conflicting parent | ACCEPT | BLOCK | BLOCK |
| Placeholder parent | BLOCK | BLOCK | BLOCK |
| Wrong subject | ACCEPT | BLOCK | BLOCK |
| Wrong grade | BLOCK | BLOCK | BLOCK |
| High-score safety | BLOCK | BLOCK | BLOCK |
| High-score placeholder | BLOCK | BLOCK | BLOCK |
| Missing after normalization | BLOCK | BLOCK | BLOCK |
| Conflicting subject | ACCEPT | BLOCK | BLOCK |
| Conflicting grade | BLOCK | BLOCK | BLOCK |
| Topic UUID collision | ACCEPT | BLOCK | BLOCK |
| Parent-child mismatch | ACCEPT | BLOCK | BLOCK |
| Subject-topic mismatch | ACCEPT | BLOCK | BLOCK |
| Grade-topic mismatch | BLOCK | BLOCK | BLOCK |
| Placeholder topic | BLOCK | BLOCK | BLOCK |
| Placeholder parent substantive child | BLOCK | BLOCK | BLOCK |

## Metadata Violations Detected

```json
{
  "conflicting_grade": 30,
  "grade_topic_mismatch": 60,
  "conflicting_parent": 270,
  "conflicting_subject": 30,
  "subject_mismatch": 60,
  "unresolvable_topic_uuid": 60,
  "parent_child_mismatch": 60,
  "placeholder_topic": 270,
  "placeholder_parent": 240,
  "subject_topic_mismatch": 30,
  "topic_uuid_collision": 30,
  "grade_mismatch": 30
}
```

## V2.12 Recommendation

controlled production-shadow evaluation