# V2.6 Verifier Evidence-State Experiment

Generated: `2026-08-29T15:42:08.566029+00:00`

## Executive Summary

**Conclusion: NOT SUPPORTED**

Explicit evidence-state semantics did not materially change verifier acceptance.

Hypothesis: explicit `EVIDENCE_PRESENT_IMPERFECT` semantics reduce inappropriate `retrieve_more` / `insufficient_evidence` for present-but-imperfect evidence without weakening grounding safety.

## Experimental Design

- Frozen-answer methodology: generate once per cycle, verify under A/B/C/D
- Primary runs: 10 cycles × 4 arms = 40 verifier evaluations
- Baseline evidence hash: `977b259fcfb4b282`
- Golden question: fractions learning objectives, Primary 4, MBSSE-BEC 2020
- Arm A: existing verifier + original imperfect evidence
- Arm B: explicit `EVIDENCE_PRESENT_IMPERFECT` semantics
- Arm C: `EVIDENCE_PRESENT_COMPLETE` on V2.5 clean evidence
- Arm D: `EVIDENCE_MISSING` (imperfect LOs removed)
- Production verifier unchanged; experiment isolated behind `v26_verifier_replay` metadata

## Results

| Metric | A Existing | B Present-Imperfect | C Present-Complete | D Missing |
| --- | ---: | ---: | ---: | ---: |
| Acceptance | 0.0 | 0.0 | 0.0 | 0.0 |
| Avg verifier score | 0.84 | 0.825 | 0.655 | 0.68 |
| Rejection | 1.0 | 1.0 | 1.0 | 1.0 |
| Retrieve-more | 1.0 | 0.9 | 1.0 | 1.0 |
| Insufficient evidence | 0.0 | 0.1 | 0.0 | 0.0 |
| Unsupported claims | 4 | 10 | 20 | 21 |

## Key Comparisons

- Arm B vs A acceptance delta: 0
- Arm B vs C acceptance delta: 0
- Arm B vs D acceptance delta: 0

## Claim-Level Analysis

Representative imperfect-evidence failures still cite `C4U06-LO02` and `C4U04-LO04` as corrupted/truncated even when the answer quotes source text faithfully.

## Retrieval Analysis

Frozen-answer replay does not invoke post-verify retrieval; `retrieve_more` reflects verifier recommendation only. New evidence after retrieval is not applicable in replay mode.

## Grounding Safety

- **truncation_faithful**: accepted=False (expected=True), score=0.4
- **truncation_reconstruction**: accepted=False (expected=False), score=0.3
- **unsupported_claim**: accepted=False (expected=False), score=0.0
- **absence_claim**: accepted=False (expected=False), score=0.0

## Historical Context

- V2.3 constrained generation: ~0.6
- V2.3 productionization: ~0.1
- V2.4 arms: {'A': 0.1, 'B': 0.2, 'C': 0.3, 'D': 0.2}
- V2.5 clean vs imperfect: {'clean': 0.7, 'imperfect': 0.2}

These prior experiments are not directly equivalent; use only as context.

## Interpretation

**NOT SUPPORTED** — Explicit evidence-state semantics did not materially change verifier acceptance.

## Next Recommendation

Investigate generation claim patterns before changing verifier semantics.