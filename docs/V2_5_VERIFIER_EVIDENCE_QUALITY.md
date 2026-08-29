# V2.5 Verifier Evidence-Quality Experiment

Generated: `2026-08-29T01:55:40.712408+00:00`

## Experiment Setup

The verifier's low acceptance is substantially caused by imperfect source text being treated as insufficient/ungrounded even when the curriculum record is present.

- Runs per arm: 10
- Arms: A clean, B original imperfect, C clean+annotation, D original+annotation
- Baseline evidence hash: `977b259fcfb4b282`

## Evidence Inventory

| LO code | quality | length | issue |
| --- | --- | ---: | --- |
| C4U06-LO02 | GARBLED | 197 | truncated or garbled source wording |
| C4U04-LO04 | GARBLED | 81 | truncated or garbled source wording |

## Results

| Metric | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| Acceptance | 0.7 | 0.2 | 0.7 | 0.2 |
| Success | 0.7 | 0.2 | 0.7 | 0.2 |
| Avg verifier score | 0.94 | 0.837 | 0.94 | 0.82 |
| Grounding failures | 2 | 4 | 3 | 2 |
| Imperfect-evidence failures | 3 | 8 | 3 | 8 |
| Retrieve-more | 0.3 | 0.8 | 0.3 | 0.8 |
| Insufficient evidence | 0.3 | 0.8 | 0.3 | 0.8 |

## Claim-Level Failures

Representative imperfect-evidence rejections reference `C4U06-LO02` and `C4U04-LO04` with classifications `TRUNCATED_SOURCE`, `GROUNDING_FAILURE`, and `UNSUPPORTED`.

## Counterfactual Results

- Same-answer original evidence acceptance: 0.0
- Same-answer clean evidence acceptance: 0.0
- Average acceptance delta (clean - original): 0.0
- Average score delta: 0

Note: counterfactual replay used insufficient_evidence fallback answers from Arm B; both evidence conditions rejected those conservative answers.

## Interpretation: **SUPPORTED**

Clean evidence arms (A/C) materially outperform original imperfect arms (B/D); counterfactual replay on insufficient_evidence fallback answers showed no delta.

## Next Recommendation

Design a verifier follow-up that treats EVIDENCE_PRESENT_BUT_IMPERFECT separately from EVIDENCE_MISSING.