# V2.3 Generation / Verifier Diagnostic Experiment

Generated: `2026-08-28T11:12:55.243587+00:00`

## 1. Hypothesis

After V2.2 frozen retrieval (`resolve_curriculum_context` only, context boundary enabled), remaining end-to-end failures are caused primarily by **answer generation** grounding/wording or **verifier** criteria/calibration — not retrieval.

## 2. Experimental Design

| Control | Treatment |
| --- | --- |
| Arm A — current `AnswerGenerator` | Arm B — constrained diagnostic generation |
| V2.2 context boundary ON | V2.2 context boundary ON |
| Frozen resolve-only retrieval | Frozen resolve-only retrieval |
| Verifier unchanged | Same verifier |

- **Golden question:** _What are the learning objectives for fractions in Primary 4?_
- **Runs:** 10 per arm (20 total), order `ABBAABABABBAABABBAAB`
- **Model / temperature / curriculum / verifier:** identical to V2.2

## 3. Control Configuration (Arm A)

- `v23_diagnostic_experiment: true`
- `context_boundary_experiment: true`
- `generation_mode: current`
- Single-pass: no legacy retrieval after resolve; `retrieve_more` → fallback

## 4. Treatment Configuration (Arm B)

- Same as Arm A except `generation_mode: constrained`
- Constrained rules: evidence-only wording, no speculation, no truncated-text repair, flag incomplete LO source text

## 5. Evidence Snapshot Definition

After `resolve_curriculum_context` returns `status=resolved`, the experiment records:

```text
curriculum, grade, subject, topic, units, learning_outcomes
```

- **Snapshot hash (both arms):** `977b259fcfb4b282`
- **Resolver:** 10 LOs, 3 units, 13 evidence items (10 LO + 3 unit)
- **Legacy retrieval after resolve:** 0

## 6. Results Summary (20 runs)

| Metric | Current Generator | Constrained Generator | Δ |
| --- | ---: | ---: | ---: |
| success_rate | 0.2 | 0.6 | +0.400 |
| verifier_acceptance_rate | 0.2 | 0.6 | +0.400 |
| verifier_rejection_rate | 0.8 | 0.4 | -0.400 |
| avg_verifier_score | 0.91 | 0.95 | +0.040 |
| min_verifier_score | 0.9 | 0.9 | +0.000 |
| avg_latency_ms | 20352.9 | 20446.5 | +93.600 |
| avg_generation_latency_ms | 10920.6 | 13127.4 | +2206.800 |
| avg_verification_latency_ms | 9082.7 | 6982.0 | -2100.700 |
| unsupported_claims_total | 6 | 3 | -3.000 |
| speculative_claims_runs | 0 | 0 | +0.000 |
| truncation_mishandling_runs | 0 | 0 | +0.000 |
| legacy_calls_total | 0 | 0 | +0.000 |

## 7. Per-Run Results

See `data/diagnostics/v23_generation_verifier_experiment.json` (`current_runs`, `constrained_runs`) and per-run `*_trace.json` artifacts.

### Current (Arm A)

| Run | Order | Verifier | Score | Failure class | Hash |
| --- | ---: | --- | ---: | --- | --- |
| current_01 | 1 | retrieve_more | 0.9 | GENERATION_UNSUPPORTED_CLAIM | 977b259fcfb4b282 |
| current_02 | 4 | retrieve_more | 0.9 | GENERATION_UNSUPPORTED_CLAIM | 977b259fcfb4b282 |
| current_03 | 5 | retrieve_more | 0.9 | GENERATION_UNSUPPORTED_CLAIM | 977b259fcfb4b282 |
| current_04 | 7 | accept | 0.95 | — | 977b259fcfb4b282 |
| current_05 | 9 | retrieve_more | 0.9 | GENERATION_UNSUPPORTED_CLAIM | 977b259fcfb4b282 |
| current_06 | 12 | accept | 0.95 | — | 977b259fcfb4b282 |
| current_07 | 13 | retrieve_more | 0.9 | GENERATION_UNSUPPORTED_CLAIM | 977b259fcfb4b282 |
| current_08 | 15 | retrieve_more | 0.9 | GENERATION_UNSUPPORTED_CLAIM | 977b259fcfb4b282 |
| current_09 | 18 | retrieve_more | 0.9 | GENERATION_UNSUPPORTED_CLAIM | 977b259fcfb4b282 |
| current_10 | 19 | retrieve_more | 0.9 | GENERATION_UNSUPPORTED_CLAIM | 977b259fcfb4b282 |

### Constrained (Arm B)

| Run | Order | Verifier | Score | Failure class | Hash |
| --- | ---: | --- | ---: | --- | --- |
| constrained_01 | 2 | accept | 1.0 | — | 977b259fcfb4b282 |
| constrained_02 | 3 | retrieve_more | 0.9 | GENERATION_UNSUPPORTED_CLAIM | 977b259fcfb4b282 |
| constrained_03 | 6 | accept | 1.0 | — | 977b259fcfb4b282 |
| constrained_04 | 8 | accept | 0.95 | — | 977b259fcfb4b282 |
| constrained_05 | 10 | accept | 0.95 | — | 977b259fcfb4b282 |
| constrained_06 | 11 | retrieve_more | 0.9 | VERIFIER_GROUNDING_FAILURE | 977b259fcfb4b282 |
| constrained_07 | 14 | retrieve_more | 0.9 | GENERATION_UNSUPPORTED_CLAIM | 977b259fcfb4b282 |
| constrained_08 | 16 | accept | 1.0 | — | 977b259fcfb4b282 |
| constrained_09 | 17 | accept | 1.0 | — | 977b259fcfb4b282 |
| constrained_10 | 20 | retrieve_more | 0.9 | GENERATION_UNSUPPORTED_CLAIM | 977b259fcfb4b282 |

## 8. Generation Comparison

- Avg generation latency: 10920.6 ms (current) vs 13127.4 ms (constrained)
- Speculative wording runs: 0 vs 0
- Unsupported claims (verifier-reported): 6 vs 3
- Truncation mishandling runs: 0 vs 0

## 9. Verification Comparison

- Verifier acceptance: **0.2** vs **0.6** (primary metric)
- Avg verifier score: 0.91 vs 0.95
- Avg verification latency: 9082.7 ms vs 6982.0 ms

## 10. Failure Classification

- **Current:** `{'GENERATION_UNSUPPORTED_CLAIM': 8}`
- **Constrained:** `{'GENERATION_UNSUPPORTED_CLAIM': 3, 'VERIFIER_GROUNDING_FAILURE': 1}`

Dominant current-generator failure: **GENERATION_UNSUPPORTED_CLAIM** — negative absence claims (e.g. "no division LOs") and over-interpretation of truncated LO text.

## 11. Claim-Grounding Analysis

Verifier claim verdicts were used where available. Current generator often adds unsupported negative claims and paraphrases truncated LO wording; constrained generator reports LO text verbatim and flags incomplete source records.

## 12. Truncation Analysis

Known garbled LOs (e.g. C4U04-LO04, C4U06-LO02) appear in resolved evidence. Current generator sometimes paraphrases or completes them; constrained generator quotes available text and states incompleteness without repair.

## 13. Representative Accepted Answer

Run: `constrained_01` (constrained)

```
## Learning Objectives for Fractions in Primary 4

The Primary 4 Mathematics curriculum includes three units related to fractions, each with specific learning outcomes. Below are the objectives as stated in the curriculum evidence.

### Unit C4-U04: Number and Numeration – FRACTION
- **C4U04-LO01**: Simplify like fraction with common denominators.
- **C4U04-LO02**: Compare and order like fraction.
- **C4U04-LO03**: Identify Equivalent fractions.
- **C4U04-LO04**: Relate fractions with denominators up to compare equivalent fraction greater than

### Unit C4-U05: Number and Numeration – OPERATION ON FRACTIONS
- **C4U05-LO01**: Add Equivalent fractions.
- **C4U05-LO02**: Subtract Equivalent fractions.
- **C4U05-LO03**: Solve both Addition and Subtraction of Equivalent fractions.
- **C4U05-LO04**: Solve word problems involving Addition and Subtraction of Equivalent fractions.

### Unit C4-U06: Number and Numeration – Operation on Fraction (Multiplication)
- **C4U06-LO01**: Multiply equivalent fractions.
- **C4U06-LO02**: Multiply like fractions with denominators up to multiply like fractions with denominators up to multiply related fractions with denominators up to multiply related fractions with denominators up to

**Note**: The text for C4U06-LO02 appears truncated/repetitive in the source; the full intended wording is not available in the evidence.
```

## 14. Representative Rejected Answer (Current)

Run: `current_01`

```
## Learning Objectives for Fractions in Primary 4

The Primary 4 Mathematics curriculum covers fractions across three units. The learning outcomes are organized as follows:

### Unit C4-U04: Number and Numeration – FRACTION
- **C4U04-LO01**: Simplify like fractions with common denominators.
- **C4U04-LO02**: Compare and order like fractions.
- **C4U04-LO03**: Identify equivalent fractions.
- **C4U04-LO04**: Relate fractions with denominators up to compare equivalent fractions greater than (the objective text is incomplete in the evidence).

### Unit C4-U05: Number and Numeration – OPERATION ON FRACTIONS
- **C4U05-LO01**: Add equivalent fractions.
- **C4U05-LO02**: Subtract equivalent fractions.
- **C4U05-LO03**: Solve both addition and subtraction of equivalent fractions.
- **C4U05-LO04**: Solve word problems involving addition and subtraction of equivalent fractions.

### Unit C4-U06: Number and Numeration – Operation on Fraction (Multiplication)
- **C4U06-LO01**: Multiply equivalent fractions.
- **C4U06-LO02**: Multiply like fractions with denominators up to multiply like fractions with denominators up to multiply related fractions with denominators up to multiply related fractions with denominators up to (the objective text is incomplete in the evidence).

**Note**: The evidence does not include any learning outcomes for division of fractions in Primary 4.
```

Verifier issues: ['The generated answer includes a claim that there are no learning outcomes for division of fractions in Primary 4, but the evidence does not explicitly confirm this absence; it may be missing evidence rather than a definitive statement.', 'The generated answer notes that some objective texts are incomplete in the evidence, which is accurate but reflects the evidence quality rather than a curriculum fact.']
Failure class: GENERATION_UNSUPPORTED_CLAIM

## 15. Representative Constrained Answer

Run: `constrained_01` (accept)

```
## Learning Objectives for Fractions in Primary 4

The Primary 4 Mathematics curriculum includes three units related to fractions, each with specific learning outcomes. Below are the objectives as stated in the curriculum evidence.

### Unit C4-U04: Number and Numeration – FRACTION
- **C4U04-LO01**: Simplify like fraction with common denominators.
- **C4U04-LO02**: Compare and order like fraction.
- **C4U04-LO03**: Identify Equivalent fractions.
- **C4U04-LO04**: Relate fractions with denominators up to compare equivalent fraction greater than

### Unit C4-U05: Number and Numeration – OPERATION ON FRACTIONS
- **C4U05-LO01**: Add Equivalent fractions.
- **C4U05-LO02**: Subtract Equivalent fractions.
- **C4U05-LO03**: Solve both Addition and Subtraction of Equivalent fractions.
- **C4U05-LO04**: Solve word problems involving Addition and Subtraction of Equivalent fractions.

### Unit C4-U06: Number and Numeration – Operation on Fraction (Multiplication)
- **C4U06-LO01**: Multiply equivalent fractions.
- **C4U06-LO02**: Multiply like fractions with denominators up to multiply like fractions with denominators up to multiply related fractions with denominators up to multiply related fractions with denominators up to

**Note**: The text for C4U06-LO02 appears truncated/repetitive in the source; the full intended wording is not available in the evidence.
```

## 16. Interpretation

**Result A** — Constrained generation materially improves verifier acceptance (2/10 → 6/10) with identical frozen evidence. Generation behavior (unsupported negative claims, LO paraphrase/truncation handling) is the primary remaining failure mode. Retrieval is not implicated.

## 17. Recommendation

**GENERATION** — Constrained generation materially improves verifier acceptance.

Do not modify the verifier yet. Next production step: integrate constrained-generation principles into `AnswerGenerator` (evidence-faithful LO wording, no negative absence claims, explicit truncation handling).
