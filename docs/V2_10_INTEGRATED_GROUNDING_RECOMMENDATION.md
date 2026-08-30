# V2.10 Integrated Grounding + Recommendation Safety Experiment

Generated: `2026-08-30T13:26:10.406840+00:00`

**Conclusion: PARTIALLY_SUPPORTED**

Core integration fixtures meet FC/FI/safety/placeholder targets, but 5 adversarial metadata-corruption cases were verifier-accepted (verifier does not enforce subject/grade/parent integrity).

## Architecture Question

Partially — harness validates Normalization → Verifier → Recommendation Mapping for primary fixtures (C4-U18 FC, FI mapping, safety, placeholders). Production-hardening required for metadata-integrity adversarial cases (wrong subject/grade, conflicting parent, placeholder parent).

## Integration Comparison

| Fixture | RAW+Verifier | Normalized+Verifier | RAW+Mapper | Normalized+Mapper |
| --- | ---: | ---: | ---: | ---: |
| FAITHFUL_COMPLETE | 0.3 | 1.0 | 0.4 | 1.0 |
| FAITHFUL_IMPERFECT | 0.0 | 0.0 | 0.9 | 0.8 |
| CLEAN_PLACEHOLDER | 0.0 | 0.0 | 0.0 | 0.0 |
| UNSUPPORTED_CLAIM | 0.0 | 0.0 | 0.0 | 0.0 |
| UNSUPPORTED_ABSENCE | 0.0 | 0.0 | 0.0 | 0.0 |
| SPECULATIVE | 0.0 | 0.0 | 0.0 | 0.0 |
| RECONSTRUCTION | 0.0 | 0.0 | 0.0 | 0.0 |
| MISSING_EVIDENCE | 0.0 | 0.0 | 0.0 | 0.0 |
| HIGH_SCORE_UNSUPPORTED | 0.0 | 0.0 | 0.0 | 0.0 |
| NORMALIZATION_ONLY_GROUNDING | 0.2 | 1.0 | 0.2 | 1.0 |
| NORMALIZATION_MUST_NOT_INVENT | 0.0 | 0.0 | 0.0 | 0.0 |
| PLACEHOLDER_PLUS_HIGH_SCORE | 0.0 | 0.0 | 0.0 | 0.0 |

## Pipeline D Threshold Sweep

| Threshold | FC Accept | FI Accept | Placeholder Accept | Safety False Accept | Missing-Evidence Accept |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7 | 1.0 | 0.8 | 0.0 | 0 | 0 |
| 0.75 | 1.0 | 0.8 | 0.0 | 0 | 0 |
| 0.8 | 1.0 | 0.8 | 0.0 | 0 | 0 |
| 0.85 | 1.0 | 0.8 | 0.0 | 0 | 0 |
| 0.9 | 1.0 | 0.8 | 0.0 | 0 | 0 |
| 0.95 | 1.0 | 0.0 | 0.0 | 0 | 0 |

## V2.11 Recommendation

adversarial grounding stress test