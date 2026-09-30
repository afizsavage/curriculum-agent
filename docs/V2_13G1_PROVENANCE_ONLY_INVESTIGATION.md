# V2.13G.1 PROVENANCE_ONLY Generation Boundary Investigation

Generated: `2026-09-15T15:18:42.512820+00:00`

**Enforcement: `PARTIALLY_ENFORCED`**

**Recommended architecture: `PRESERVE_CONTROL_ANSWER`**

**Promotion: `INVESTIGATE_BEFORE_PROMOTION`**

Production arbitration remains disabled. Sample rate remains 0.01.

## A. Root cause

Documents are withheld from generation_evidence (generation_document_count=0) but the arbitrated arm still regenerates instead of preserving control (V2.13F outcome_source=control).

For all 17 `SUFFICIENT + REDUNDANT + PROVENANCE_ONLY` targeted cases:

- `generation_document_count = 0` (documents not passed to the LLM)
- control evidence fingerprint **equals** arbitrated evidence fingerprint
- arbitrated answer hash **never** equals control answer hash (0/17)

V2.13F offline semantics use `outcome_source=control` whenever
`document_use != INCLUDE_IN_GENERATION`. V2.13G dual-arm instead cleared
the control answer and called `agent.answer()` again (structured-only
regeneration).

## B. Enforcement status

`PARTIALLY_ENFORCED`

- Document withhold: **ENFORCED**
- Control-answer preservation: **NOT ENFORCED** (current default)

## C. Questions A–E

```json
{
  "A_documents_in_generation": {
    "leaked_cases": 0,
    "result": "NO"
  },
  "B_structured_evidence_identical": {
    "identical_cases": 17,
    "n": 17,
    "result": "YES"
  },
  "C_prompt_identical": {
    "result": "LIKELY_YES",
    "note": "Arbitrated arm uses agent.answer() with structured-only evidence; no alternate system prompt. Control ran earlier in the same ask with the same generator (temperature=0.0)."
  },
  "D_generation_parameters": {
    "result": "LIKELY_YES",
    "temperature": 0.0,
    "note": "AnswerGenerator.generate_structured uses temperature=0.0."
  },
  "E_intended_semantics": {
    "v213f_documented": "PRESERVE_CONTROL_ANSWER (outcome_source=control)",
    "v213g_current": "STRUCTURED_ONLY_REGENERATION",
    "result": "CURRENT_DIVERGES_FROM_V213F"
  }
}
```


## C2. Per-regression root causes (all targeted rows)

### Baseline regressions (2)

1. `9cd42c44d005d3b3` — REDUNDANT/PROVENANCE_ONLY — **baseline DOCUMENT contamination**:
   baseline merges docs into generation (`generation_evidence_count` > structured);
   verifier `retrieve_more` while control accepted. Arbitrated arm withheld docs and accepted.

2. `27e64ac4d304e8d5` — REDUNDANT/PROVENANCE_ONLY — **baseline DOCUMENT contamination**:
   same pattern: baseline merge hurt; arbitrated structured-only accepted.

These are **not** general nondeterminism alone: the baseline arm uniquely injects documents.

### Arbitrated regressions (3)

1. `fd894182926a5f8b` — REDUNDANT/PROVENANCE_ONLY — **CONTROL_REGENERATION**:
   docs NOT in generation; evidence fingerprint identical to control; new answer failed verifier (`retrieve_more`). Not DOCUMENT_LEAK.

2. `3445d54939ac9ed9` — REDUNDANT/PROVENANCE_ONLY — **CONTROL_REGENERATION**:
   same pattern as (1).

3. `7fb53cf210f937c5` — IRRELEVANT/DO_NOT_USE — **CONTROL_REGENERATION** (or STRUCTURED_ONLY regen):
   docs withheld; regenerating structured-only answer diverged and failed verifier.

Under V2.13F preserve-control counterfactual, all three arbitrated regressions become non-regressions (`outcome_source=control`).

### dd1c57

Appears in the PROVENANCE_ONLY redundant set (`dd1c57ff32af875b`). Classification PASS.
Live dual-arm regenerated a different hash but still accepted. V2.13F fixture counterfactual still preserves control hash `16088fa24c2d597f`.

## D. Answer preservation

```json
{
  "arbitrated_hash_match_control": 0,
  "n": 17,
  "rate": 0.0,
  "preserve_counterfactual_hash_match": 17
}
```

## E. Regressions on PROVENANCE_ONLY cohort

```json
{
  "baseline_regression": 2,
  "arbitrated_regression": 2,
  "preserve_counterfactual_arbitrated_regression": 0
}
```

Root-cause counts:

```json
{
  "CONTROL_REGENERATION": 17
}
```

## F. Determinism

```json
{
  "generation_determinism_rate": 0.0,
  "identical_evidence_but_divergent_answer": 17,
  "interpretation": "Control is generated once (production path) and reused as the comparison baseline \u2014 control is not regenerated for scoring. Baseline and arbitrated arms each regenerate independently. With identical evidence fingerprints, answer-hash divergence is CONTROL_REGENERATION / LLM variance, not document leak."
}
```

Control is generated once on the production path and used as the fixed
comparison baseline. Baseline and arbitrated arms regenerate independently.

## G. Recommended architecture

**`PRESERVE_CONTROL_ANSWER`**

V2.13F counterfactual uses outcome_source=control for PROVENANCE_ONLY. Current structured-only regeneration yields 0/17 hash matches despite identical evidence fingerprints and zero document leak. Preserve-control would yield 17/17 hash matches and 0 preserve-path regressions on this set.

H2 path must stay intact: `INSUFFICIENT + DECISIVE → INCLUDE_IN_GENERATION`
continues to regenerate with documents.

## H. Hypotheses

```json
{
  "H1a_classification": "PASS",
  "H1b_enforcement": "PARTIALLY_ENFORCED",
  "H1c_control_preservation": "NOT_CONFIRMED",
  "H2_recovery": "PASS",
  "H3_generator_behavior": "NOT_CONFIRMED",
  "safety": "PASS"
}
```

## I. Safety

All hard safety gates remain zero on the preserved targeted set.

## J. Next experiment

1. Enable shadow-only `v213g_provenance_only_semantics=preserve_control_answer`.
2. Re-run targeted dual-arm on the same sufficient+redundant cohort.
3. Require `arbitrated_answer_hash == control_answer_hash` for PROVENANCE_ONLY
   when control is accepted.
4. Re-confirm H2 insufficient+decisive recoveries unchanged.
5. Only then resume healthy LIVE Stage-1 toward n≥20 sufficient+docs
   (still at sample_rate=0.01; no n=100/200 expansion yet).

## Isolation

```json
{
  "v213f_document_arbitration_experiment": false,
  "v213d_shadow_sample_rate": 0.01,
  "note": "No production enablement; sample rate unchanged."
}
```
