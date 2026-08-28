# V2.3 Productionization — Evidence-Conservative Generation

Generated: `2026-08-28T11:38:21.133029+00:00`

## Change Summary

Production `AnswerGenerator` now applies **`generation_policy=evidence_conservative`** by default via `app/agent/generation_policy.py`.

- Verifier: **unchanged**
- Resolver / retrieval / curriculum data: **unchanged**
- V2.2 context boundary flag: **compatible** (unchanged)

## Policy Rules (production)

1. Evidence is the authoritative boundary — no model-knowledge gap filling
2. Preserve official LO code + source wording
3. Do not reconstruct truncated/garbled source text
4. No unsupported absence claims (`not observed ≠ does not exist`)
5. No speculative curriculum inference
6. Answer only the question asked
7. Evidence-first structure when useful
8. Preserve LO/unit/topic codes and entity IDs

## Before / After Comparison

| Metric | V2.2 treatment | V2.3 constrained (experiment) | Productionized (10 runs) |
| --- | ---: | ---: | ---: |
| Verifier acceptance | 30% (3/10) | 60% (6/10) | **10% (1/10)** |
| End-to-end success | 30% | 60% | **10%** |
| Unsupported claims (verifier) | — | 3 total | **3 total** |
| Speculative wording runs | — | 0 | **0** |
| Unsupported absence runs | — | — | **0** |
| Truncation warning runs | — | — | **10** |
| Avg latency (ms) | ~20,700 | ~20,700 | **37,366** |
| Avg tool calls | 1.0 | 1.0 | **1.0** |

Golden question: _What are the learning objectives for fractions in Primary 4?_

Configuration: `context_boundary_experiment: true`, no V2.3 diagnostic flags, production graph.

## Grounding Improvements

The productionized generator **eliminated** the dominant V2.3 failure modes in observability:

- **0** unsupported absence claims across 10 runs (previously a primary current-generator failure)
- **0** speculative wording runs
- **10/10** runs flagged truncated/garbled source text via `truncation_warning_count`

## Acceptance Gap Analysis

Production acceptance (10%) did **not** reproduce the V2.3 constrained experiment (60%). Likely causes:

1. **Graph routing difference:** V2.3 used `v23_single_pass` (frozen resolve + single verify). Production uses the full graph; `retrieve_more` at score 0.7 often terminates as `insufficient_evidence` fallback.
2. **Verifier sensitivity to garbled LO text:** Several rejections cite verbatim C4U06-LO02 garbled wording as an "unsupported claim" even when limitations are stated — verifier unchanged per experiment charter.
3. **LLM non-determinism:** Single-run variance is high on this golden question.

## Tests

| Suite | Result |
| --- | --- |
| `curriculum-agent` | All tests pass (including 14 new evidence-conservative regression tests) |
| `curriculum-structure` | All tests pass |

New tests: `tests/agent/test_evidence_conservative_generation.py` (6 required scenarios).

## Artifacts

- `data/diagnostics/v23_productionization.json`
- `data/diagnostics/v23_productionization/prod_*_{response,trace}.json`

## Conclusion: **REGRESSION** (acceptance metric)

Grounding policy objectives are met (no absence/speculation; truncation flagged), but **verifier acceptance did not reproduce** the V2.3 constrained arm under production graph routing.

## Next Recommendation (not implemented)

Investigate verifier interaction with verbatim garbled LO records under production routing, or run a follow-up eval with conservative regeneration enabled on first pass — without modifying verifier thresholds.
