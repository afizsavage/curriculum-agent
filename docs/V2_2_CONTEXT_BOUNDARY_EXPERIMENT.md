# V2.2 Context Boundary Experiment

Generated: `2026-08-28T02:41:35+00:00`

## 1. Hypothesis

Once `resolve_curriculum_context` returns `resolution.status=resolved` with authoritative learning outcomes, the QA agent should treat that snapshot as the **evidence boundary** — blocking redundant legacy retrieval and regenerating conservatively when the verifier asks for evidence already present.

## 2. Experimental design

| Arm | Configuration | Runs |
| --- | --- | ---: |
| **Control** | V2.1 (default) — reused from `v21_evaluation.json` | 10 |
| **Treatment** | `context_boundary_experiment=true` per request (or `CURRICULUM_V2_CONTEXT_BOUNDARY_EXPERIMENT=true`) | 10 |

- **Question:** What are the learning objectives for fractions in Primary 4?
- **Model / limits / verifier:** unchanged between arms
- **Run order (alternating):** `C T T C C T C T C T T C C T C T T C C T`
- **No post-hoc tuning** of limits, verifier, or resolver

## 3. Treatment behavior (isolated)

- Captures `ContextBoundarySnapshot` after successful resolve
- Skips redundant `search_curriculum`, `get_curriculum_structure`, `get_topic`, `get_learning_objectives` when boundary already covers the request (`context_boundary_covered`)
- On verifier `retrieve_more` with evidence already present → **`regenerate`** route (conservative generation prompt) instead of another retrieval round
- Legacy tools remain available when resolver fails or genuinely new entities are requested

## 4. Primary metrics

| Metric | Control (V2.1) | Treatment (V2.2) | Δ |
| --- | ---: | ---: | ---: |
| End-to-end success | 3/10 (30%) | 3/10 (30%) | 0 pp |
| Resolver success | 10/10 resolved (artifact) | 10/10 (100%) | — |
| Verifier acceptance | 30% | 30% | 0 pp |
| `no_retrieval_progress` (classified) | 7/10 | 7/10 | 0 |
| `retrieve_more` rate | 50% | **0%** | **−50 pp** |
| Avg tool calls | 4.9 | **1.0** | **−3.9 (−80%)** |
| Avg latency | 39.1s | 43.5s | +4.4s |
| Median latency | 38.6s | 43.7s | +5.1s |
| Legacy calls after resolve | 3.5 | **0** | **−3.5** |
| Regeneration without retrieval | 0 | **0.7 / run** | +0.7 |
| Duplicate evidence additions | 14.7 | **0** | −14.7 |
| Boundary-blocked tool skips | 0 | **4.8 / run** | +4.8 |

## 5. Failure classification

| Class | Control | Treatment |
| --- | ---: | ---: |
| `NO_RETRIEVAL_PROGRESS` | 7 | 7 |
| `OTHER` (includes successes) | 3 | 3 |

Treatment **eliminated redundant retrieval** but did **not** shift the dominant failure mode: verifier still rejects or requests more, and runs still terminate without credible new retrieval paths after conservative regeneration.

## 6. Evidence integrity

- Treatment runs that succeeded used resolver authoritative LOs (10 outcomes from boundary)
- No increase in speculative wording (`likely` / `probably`) vs control in the harness checks
- Duplicate evidence additions dropped to zero (no legacy re-fetch of same LOs)
- **Risk:** blocking all legacy calls after resolve may prevent recovery when verifier needs a *different* evidence shape (not just clearer LO wording)

## 7. Representative traces

- **Treatment success:** `run-*` from `treatment_03` / `treatment_04` / `treatment_09` — 1 resolve call, 0 legacy, completed
- **Treatment failure:** `treatment_01` — 1 resolve, 6 boundary skips, `insufficient_evidence` / `no_retrieval_progress`

Full artifacts: `data/diagnostics/v22_context_boundary_experiment.json` and `data/diagnostics/v22_context_boundary_experiment/`.

## 8. Subject=null (recorded, not fixed)

All golden runs still show `understand.subject=null`. Resolver/tool path often infers Mathematics; recorded separately as `understand_subject` / `final_context_subject` per run.

## 9. Interpretation

**Confirmed:** The hypothesis is **partially supported** — redundant retrieval after successful resolve is real and reducible (80% fewer tool calls, zero legacy-after-resolve, zero duplicate evidence).

**Not confirmed:** End-to-end success and verifier acceptance **did not improve** (still 3/10). Conservative regeneration without retrieval does not yet satisfy the verifier; failures remain `NO_RETRIEVAL_PROGRESS`-shaped.

**Latency:** Slightly worse on average despite fewer tools — likely from extra generate/verify cycles on the `regenerate` path without new evidence.

## 10. Recommendation

**ITERATE** — Keep the experiment **isolated** (flag off by default). Do not promote to default behavior yet.

Next steps (out of scope for this sprint):

1. Tune regenerate/verifier interaction when boundary is satisfied (without changing verifier thresholds)
2. Allow **targeted** legacy retrieval only for genuinely missing entity types (e.g. lesson plans)
3. Address generation grounding on truncated source LO text
4. `subject=null` follow-up

---

```
EXPERIMENT COMPLETE

Control:     3/10 success · 4.9 tools · 50% retrieve_more · 3.5 legacy-after-resolve
Treatment:   3/10 success · 1.0 tools ·  0% retrieve_more · 0 legacy-after-resolve

Observed improvement:  −80% tool calls, −100% legacy-after-resolve, −100% duplicate evidence
Observed regressions:  +70% explicit no_retrieval_progress routing; latency +4.4s avg

Recommendation: ITERATE
```
