# V2.4 Routing / Verifier Isolation Experiment

Generated: `2026-08-29T00:01:07.985331+00:00`

## Executive Summary

**V2.4 EXPERIMENT COMPLETE**

Verifier rejects conservative answers about already-present but garbled LO text across all arms; routing shows mixed effect (C>D modestly, A<B contradicts frozen routing regression).

**Recommendation:** VERIFIER FOLLOW-UP

## Experimental Arms

| Arm | Configuration |
| --- | --- |
| A | Frozen evidence + single pass |
| B | Frozen evidence + production graph |
| C | Live resolve + single pass |
| D | Live resolve + production graph |

## Evidence Equivalence

- Frozen hashes (A/B): `['977b259fcfb4b282']`
- Live hashes (C/D): `['977b259fcfb4b282'] / ['977b259fcfb4b282']`
- Evidence materially different: **False**

## Results

| Metric | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| Success | 0.1 | 0.2 | 0.3 | 0.2 |
| Verifier acceptance | 0.1 | 0.2 | 0.3 | 0.2 |
| Avg verifier score | 0.765 | 0.78 | 0.79 | 0.844 |
| retrieve_more rate | 0.9 | 0.7 | 0.7 | 0.8 |
| Regeneration | 0.0 | 0.4 | 0.0 | 0.2 |
| insufficient_evidence | 0.9 | 0.8 | 0.7 | 0.8 |
| Avg tools | 1 | 1 | 1 | 1 |
| Avg latency | 18325.8 | 25717.3 | 26575.5 | 36364.0 |
| Evidence count | 13 | 13 | 13 | 13 |
| New evidence after retrieval | 0 | 0 | 0 | 0 |
| Already-present rejection | 10 | 10 | 10 | 10 |
| Routing intervention rate | 0.9 | 0.7 | 0.7 | 0.8 |

## Failure Matrix

| Arm | Tag | Verifier issue | Evidence present? | New evidence | Routing transition | Terminal |
| --- | --- | --- | --- | --- | --- | --- |
| A | arm_a_01 | GROUNDING_FAILURE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → fallback (v23_single_pass_complete) → insufficient_evidence (v23_single_pass) | insufficient_evidence |
| A | arm_a_02 | GROUNDING_FAILURE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → fallback (v23_single_pass_complete) → insufficient_evidence (v23_single_pass) | insufficient_evidence |
| A | arm_a_03 | TRUNCATED_SOURCE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → fallback (v23_single_pass_complete) → insufficient_evidence (v23_single_pass) | insufficient_evidence |
| A | arm_a_05 | GROUNDING_FAILURE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → fallback (v23_single_pass_complete) → insufficient_evidence (v23_single_pass) | insufficient_evidence |
| A | arm_a_06 | TRUNCATED_SOURCE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → fallback (v23_single_pass_complete) → insufficient_evidence (v23_single_pass) | insufficient_evidence |
| A | arm_a_07 | GROUNDING_FAILURE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → fallback (v23_single_pass_complete) → insufficient_evidence (v23_single_pass) | insufficient_evidence |
| A | arm_a_08 | TRUNCATED_SOURCE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → fallback (v23_single_pass_complete) → insufficient_evidence (v23_single_pass) | insufficient_evidence |
| A | arm_a_09 | TRUNCATED_SOURCE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → fallback (v23_single_pass_complete) → insufficient_evidence (v23_single_pass) | insufficient_evidence |
| A | arm_a_10 | GROUNDING_FAILURE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → fallback (v23_single_pass_complete) → insufficient_evidence (v23_single_pass) | insufficient_evidence |
| B | arm_b_01 | TRUNCATED_SOURCE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier fallback → fallback (verification_fallback) → insufficient_evidence (verification_fallback) | insufficient_evidence |
| B | arm_b_02 | GROUNDING_FAILURE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → fallback (no_retrieval_progress_incomplete_source) → insufficient_evidence (no_retrieval_progress) | insufficient_evidence |
| B | arm_b_03 | TRUNCATED_SOURCE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → regenerate (evidence_already_present) → regenerate (evidence_already_present) → fallback (no_retrieval_progress_incomplete_source) → insufficient_evidence (no_retrieval_progress) | insufficient_evidence |
| B | arm_b_05 | TRUNCATED_SOURCE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → regenerate (evidence_already_present) → fallback (no_retrieval_progress_incomplete_source) → insufficient_evidence (no_retrieval_progress) | insufficient_evidence |
| B | arm_b_06 | TRUNCATED_SOURCE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → fallback (no_retrieval_progress_incomplete_source) → insufficient_evidence (no_retrieval_progress) | insufficient_evidence |
| B | arm_b_07 | GROUNDING_FAILURE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → fallback (no_retrieval_progress_incomplete_source) → insufficient_evidence (no_retrieval_progress) | insufficient_evidence |
| B | arm_b_08 | TRUNCATED_SOURCE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → regenerate (evidence_already_present) → fallback (no_retrieval_progress_incomplete_source) → insufficient_evidence (no_retrieval_progress) | insufficient_evidence |
| B | arm_b_10 | TRUNCATED_SOURCE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → regenerate (evidence_already_present) → fallback (no_retrieval_progress_incomplete_source) → insufficient_evidence (no_retrieval_progress) | insufficient_evidence |
| C | arm_c_01 | TRUNCATED_SOURCE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → fallback (v23_single_pass_complete) → insufficient_evidence (v23_single_pass) | insufficient_evidence |
| C | arm_c_02 | GROUNDING_FAILURE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → fallback (v23_single_pass_complete) → insufficient_evidence (v23_single_pass) | insufficient_evidence |
| C | arm_c_03 | GROUNDING_FAILURE | EVIDENCE_PRESENT_BUT_IMPERFECT | — | verifier retrieve_more → fallback (v23_single_pass_complete) → insufficient_evidence (v23_single_pass) | insufficient_evidence |
| … | (12 more rows in JSON) | | | | | |

## Causal Interpretation

Verifier rejects conservative answers about already-present but garbled LO text across all arms; routing shows mixed effect (C>D modestly, A<B contradicts frozen routing regression).

1. **Frozen evidence under production routing?** Arm A (10%) vs B (20%): production routing does not clearly worsen frozen evidence outcomes.
2. **Live vs frozen evidence?** Live and frozen evidence hashes are identical; evidence construction is not the primary differentiator.
3. **retrieve_more new evidence?** Post-verify retrieve_more cycles add negligible new evidence (avg 0.0 items per retrieve_more run).
4. **Already-present evidence rejections?** Rejections overwhelmingly reference already-present LOs (10/10 per arm with rejections).
5. **Routing → insufficient_evidence?** insufficient_evidence rates remain high even without production graph (A=90%); production graph adds regeneration (B=40%) but does not uniquely cause terminal failure.

**Recommendation:** VERIFIER FOLLOW-UP

## Representative Traces

### Representative success (`arm_a_04`)

**arm_a_04** — status `completed`

```text
initial_generation
  → verifier_score=0.95
  → verifier_decision=accept
  → retrieve_more_requested=False
  → evidence_already_present=True
  → evidence_presence_class=EVIDENCE_PRESENT_AND_SUFFICIENT
  → verifier_issue_class=TRUNCATED_SOURCE
  → finish (verification_passed)
  → final_decision=accept
  → final_failure_reason=verification_passed
```

### Representative routing failure (`arm_a_01`)

**arm_a_01** — status `insufficient_evidence`

```text
initial_generation
  → verifier_score=0.7
  → verifier_decision=retrieve_more
  → retrieve_more_requested=True
  → evidence_already_present=True
  → evidence_presence_class=EVIDENCE_PRESENT_BUT_IMPERFECT
  → verifier_issue_class=GROUNDING_FAILURE
  → fallback (v23_single_pass_complete)
  → final_decision=retrieve_more
  → final_failure_reason=v23_single_pass
```

### Representative garbled LO failure (`arm_a_01`)

**arm_a_01** — status `insufficient_evidence`

```text
initial_generation
  → verifier_score=0.7
  → verifier_decision=retrieve_more
  → retrieve_more_requested=True
  → evidence_already_present=True
  → evidence_presence_class=EVIDENCE_PRESENT_BUT_IMPERFECT
  → verifier_issue_class=GROUNDING_FAILURE
  → fallback (v23_single_pass_complete)
  → final_decision=retrieve_more
  → final_failure_reason=v23_single_pass
```

### Representative conservative success (`arm_a_04`)

**arm_a_04** — status `completed`

```text
initial_generation
  → verifier_score=0.95
  → verifier_decision=accept
  → retrieve_more_requested=False
  → evidence_already_present=True
  → evidence_presence_class=EVIDENCE_PRESENT_AND_SUFFICIENT
  → verifier_issue_class=TRUNCATED_SOURCE
  → finish (verification_passed)
  → final_decision=accept
  → final_failure_reason=verification_passed
```
