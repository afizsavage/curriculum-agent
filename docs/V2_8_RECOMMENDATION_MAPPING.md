# V2.8 Recommendation-Mapping Experiment

Generated: `2026-08-29T17:26:40.107116+00:00`

**Conclusion: PARTIALLY_SUPPORTED**

Faithful-imperfect handling improved with zero safety false-acceptance, but placeholder grounding and/or faithful-complete acceptance remain unresolved.

## Hypothesis

An independent post-verifier recommendation-mapping layer can safely translate verifier results into operational actions without modifying verifier scoring.

## Setup

- Primary unit: C4-U18 Everyday Arithmetic Money
- C4-U18 evidence hash: `be3e342763f1faac`
- Fractions evidence hash (imperfect/placeholder): `977b259fcfb4b282`
- Fixtures: 8 classes × 10 runs = 80
- Harness-only post-verifier recommendation mapper; production unchanged

## Policy

1. Safety failures always reject
2. Missing evidence → insufficient/retrieve_more
3. Faithful imperfect + score threshold + retrieve_more → accept
4. Placeholder evidence never accepted via score alone

## Threshold Sweep

| Threshold | FI Accept | FI Retrieve | FC Accept | Placeholder Accept | Safety Rejections |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7 | 0.9 | 0.1 | 0.2 | 0.0 | 1.0 |
| 0.75 | 0.9 | 0.1 | 0.2 | 0.0 | 1.0 |
| 0.8 | 0.9 | 0.1 | 0.2 | 0.0 | 1.0 |
| 0.85 | 0.9 | 0.1 | 0.2 | 0.0 | 1.0 |
| 0.9 | 0.9 | 0.1 | 0.2 | 0.0 | 1.0 |
| 0.95 | 0.0 | 1.0 | 0.2 | 0.0 | 1.0 |

## V2.7 Comparison

| Metric | V2.7 | V2.8 |
| --- | ---: | ---: |
| FI acceptance | 0.8 | 0.9 |
| FI false retrieval residual | 0.2 | 0.1 |
| Overall false retrieval | 0.133 | 0.113 |
| Safety false acceptance | 0.0 | 0 |
| FC acceptance | failed (placeholders) | 0.2 |

## Placeholder Diagnostics

- clean_placeholder_01: score=0.0, mapped=reject, class=sentinel
- clean_placeholder_02: score=0.0, mapped=reject, class=sentinel
- clean_placeholder_03: score=0.0, mapped=reject, class=sentinel
- clean_placeholder_04: score=0.1, mapped=reject, class=sentinel
- clean_placeholder_05: score=0.0, mapped=reject, class=sentinel

## V2.9 Recommendation

Run V2.9 placeholder/evidence normalization experiment before production mapping.