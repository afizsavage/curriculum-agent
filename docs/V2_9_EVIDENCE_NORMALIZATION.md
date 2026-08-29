# V2.9 Evidence Normalization & Grounding-Boundary Experiment

Generated: `2026-08-29T18:06:55.200363+00:00`

**Conclusion: SUPPORTED**

Normalization materially improved grounding while preserving safety and placeholder rejection.

## Hypothesis

Pre-verifier evidence normalization can resolve placeholder and representation failures without changing verifier semantics.

## Setup

- C4-U18 evidence hash: `be3e342763f1faac`
- Fractions evidence hash: `977b259fcfb4b282`
- Variants: RAW, PLACEHOLDER_FILTER, STRUCTURAL_NORMALIZATION, SEMANTIC_EVIDENCE_EXTRACTION
- Fixtures: 8 classes × 10 runs × 4 variants = 320
- Harness-only pre-verifier normalization; production unchanged

## Variant Comparison

| Variant | FC Accept | FI Accept | Placeholder Accept | Safety False Accept | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAW | 0.1 | 0.0 | 0.0 | 0 | 0.249 |
| PLACEHOLDER_FILTER | 0.1 | 0.0 | 0.0 | 0 | 0.246 |
| STRUCTURAL_NORMALIZATION | 1.0 | 0.0 | 0.0 | 0 | 0.229 |
| SEMANTIC_EVIDENCE_EXTRACTION | 1.0 | 0.0 | 0.0 | 0 | 0.228 |

## C4-U18 FAITHFUL_COMPLETE Diagnosis

**Root cause verdict:** mixed_or_representation

**Grounding question answer:** evidence representation problem

### Cause counts

{
  "unknown": 65,
  "verifier_unsupported_claims": 7,
  "verifier_topic_linkage_dispute": 14,
  "accepted": 22
}

## V2.8 Comparison

| Metric | V2.8 | V2.9 RAW |
| --- | ---: | ---: |
| FC acceptance | 0.2 | 0.1 |
| FI acceptance (verifier) | n/a (mapped 0.9) | 0.0 |
| Placeholder false accept | 0.0 | 0.0 |
| Safety false accept | 0.0 | 0 |

## Placeholder Diagnostics (sample)

- placeholder_filter_clean_placeholder_01 [PLACEHOLDER_FILTER]: score=0.0, decision=fallback, records_out=11
- placeholder_filter_clean_placeholder_02 [PLACEHOLDER_FILTER]: score=0.0, decision=fallback, records_out=11
- placeholder_filter_clean_placeholder_03 [PLACEHOLDER_FILTER]: score=0.0, decision=fallback, records_out=11
- placeholder_filter_clean_placeholder_04 [PLACEHOLDER_FILTER]: score=0.0, decision=fallback, records_out=11
- placeholder_filter_clean_placeholder_05 [PLACEHOLDER_FILTER]: score=0.0, decision=fallback, records_out=11
- placeholder_filter_clean_placeholder_06 [PLACEHOLDER_FILTER]: score=0.0, decision=fallback, records_out=11
- placeholder_filter_clean_placeholder_07 [PLACEHOLDER_FILTER]: score=0.0, decision=fallback, records_out=11
- placeholder_filter_clean_placeholder_08 [PLACEHOLDER_FILTER]: score=0.0, decision=fallback, records_out=11

## V2.10 Recommendation

Design a controlled pre-verifier evidence-normalization layer behind a feature flag with adversarial eval.