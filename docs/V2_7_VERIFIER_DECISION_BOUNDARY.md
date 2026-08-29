# V2.7 Verifier Decision-Boundary Experiment

Generated: `2026-08-29T16:39:59.281766+00:00`

## Executive Summary

**Conclusion: PARTIALLY SUPPORTED**

Faithful-imperfect acceptance improved materially at score-based thresholds with safety preserved, but faithful-complete placeholder handling remains unresolved.

## Hypothesis

When evidence is present and answers faithfully report imperfect source text, a post-verifier decision boundary can accept instead of retrieve_more without weakening grounding safety.

## Experimental Design

- Frozen deterministic answer fixtures (no stochastic generation)
- 10 verifier evaluations per fixture class
- Baseline evidence hash: `977b259fcfb4b282`
- Arm A: existing verifier decision (control)
- Arm B: harness-only post-verifier decision boundary + threshold sweep
- Production verifier unchanged; flag defaults OFF

## Control (Arm A)

- Acceptance: 0.0
- Retrieve-more: 0.517
- Insufficient evidence: 0.483
- Avg score: 0.333 (min 0.0, max 0.9)

## Experimental Decision Boundary (Arm B)

- Analytical threshold: 0.85
- Acceptance: 0.133
- Retrieve-more (post-policy residual): 0.383
- Insufficient evidence: 0.483
- Avg score (unchanged): 0.333

## Threshold Sweep

| Threshold | Faithful Imperfect Accept | False Retrieval | Unsupported Rejected | Reconstruction Rejected |
| ---: | ---: | ---: | ---: | ---: |
| 0.7 | 0.8 | 0.8 | 1.0 | 1.0 |
| 0.75 | 0.8 | 0.8 | 1.0 | 1.0 |
| 0.8 | 0.8 | 0.8 | 1.0 | 1.0 |
| 0.85 | 0.8 | 0.8 | 1.0 | 1.0 |
| 0.9 | 0.8 | 0.8 | 1.0 | 1.0 |
| 0.95 | 0.0 | 0.8 | 1.0 | 1.0 |

## Critical Metrics

- Faithful imperfect acceptance (control): 0.0
- Faithful imperfect acceptance (experimental @ 0.85): 0.8
- Faithful imperfect false retrieval (control): 0.8
- Faithful imperfect false retrieval (experimental): 0.8

## Grounding Safety (Arm B @ analytical threshold)

- **unsupported_claims_rejected**: rejected=True
- **absence_claims_rejected**: rejected=True
- **reconstruction_claims_rejected**: rejected=True
- **missing_evidence_rejected**: rejected=True
- **speculative_claims_rejected**: rejected=True

## Comparison with V2.6

- V2.6 faithful imperfect acceptance: 0.0
- V2.6 retrieve_more: 0.9
- V2.6 avg score: 0.825
- V2.7 faithful imperfect acceptance (experimental): 0.8
- V2.7 retrieve-more residual (experimental): 0.383

## Next Recommendation

Refine decision-boundary guards and re-test with larger sample.