# V2.12B Real Retrieval Production-Shadow Evaluation

Generated: `2026-09-01T09:31:35.628163+00:00`

**Conclusion: SUPPORTED**

Real-retrieval shadow evaluation preserved safety invariants with zero unsafe LangChain divergences and reproducible evidence snapshots.

## 1. Objective

Determine whether LangChain can safely operate the Curriculum QA pipeline under real retrieval while LangGraph remains production control.

## 2. Shadow architecture

One shared retrieval → evidence snapshot → LangGraph control + LangChain experiment → shadow analyzer. Production responses unaffected.

## 3. Sampling methodology

- Representative curriculum questions: `24`
- Evaluations completed: `24`
- Reproducible question set in `app/agent/v212b_shadow.py` (`REAL_QUESTIONS`)

## 4. Number of real questions

24

## 5. Evidence statistics

```json
{
  "mean_evidence_count": 27.375,
  "no_evidence": 0,
  "weak_evidence": 0,
  "evidence_found": 24
}
```

## 6. Normalization statistics

```json
{
  "normalization_success_rate": 1.0,
  "uuid_resolution_total": 0
}
```

## 7. Metadata-integrity statistics

```json
{
  "valid_evidence_pct": 0.958,
  "metadata_blocked_count": 1
}
```

## 8. Verifier statistics

```json
{
  "langgraph": {
    "accept_pct": 0.292,
    "retrieve_more_pct": 0.708,
    "fallback_pct": 0.0,
    "mean_score": 0.658,
    "mapped_accept_pct": 0.333
  },
  "langchain": {
    "accept_pct": 0.292,
    "retrieve_more_pct": 0.708,
    "fallback_pct": 0.0,
    "mean_score": 0.656,
    "mapped_accept_pct": 0.333
  }
}
```

## 9. Recommendation-mapping statistics

```json
{
  "mapped_acceptance_pct": 0.333,
  "safety_blocks": 16
}
```

## 10. Routing statistics

```json
{
  "finish_pct": 0.292,
  "retrieve_more_pct": 0.667,
  "fallback_pct": 0.042
}
```

## 11. LangGraph vs LangChain comparison

```json
{
  "langgraph": {
    "n": 24,
    "faithful_complete_acceptance": 0.0,
    "faithful_imperfect_acceptance": 0.0,
    "placeholder_false_acceptance": 0,
    "safety_false_acceptance": 0,
    "adversarial_false_acceptance": 0,
    "mean_latency_ms": 7548.668,
    "median_latency_ms": 7678.809,
    "p95_latency_ms": 12368.79,
    "llm_calls": 24,
    "tool_calls": 0,
    "errors": 0,
    "timeouts": 0
  },
  "langchain": {
    "n": 24,
    "faithful_complete_acceptance": 0.0,
    "faithful_imperfect_acceptance": 0.0,
    "placeholder_false_acceptance": 0,
    "safety_false_acceptance": 0,
    "adversarial_false_acceptance": 0,
    "mean_latency_ms": 7840.029,
    "median_latency_ms": 7170.998,
    "p95_latency_ms": 11680.129,
    "llm_calls": 24,
    "tool_calls": 0,
    "errors": 0,
    "timeouts": 0
  }
}
```

## 12. Equivalence classifications

```json
{
  "total_comparisons": 24,
  "classification_counts": {
    "EXACT_EQUIVALENCE": 22,
    "BEHAVIORAL_EQUIVALENCE": 1,
    "CONTROLLED_DIFFERENCE": 1
  },
  "unsafe_divergence_count": 0
}
```

## 13. FC behavior

LangGraph FC proxy accept: `0.0`

## 14. FI behavior

```json
{
  "retrieve_more_rows": 6,
  "langgraph_accept_after_retrieve_more": 0,
  "langchain_accept_after_retrieve_more": 0
}
```

## 15. Safety behavior

```json
{
  "unsafe_divergence_count": 0,
  "metadata_false_acceptance": 0,
  "placeholder_false_acceptance": 0
}
```

## 16. Metadata adversarial behavior

Metadata false acceptance (LangChain): `0`

## 17. Placeholder behavior

Placeholder false acceptance: `0`

## 18. Divergence analysis

Unsafe divergences: `0`

## 19. Latency

| Metric | LangGraph | LangChain | Overhead |
| --- | ---: | ---: | ---: |
| Mean (ms) | 7548.668 | 7840.029 | 291.361 |

## 20. Error/timeout rates

Shadow errors: `0`

## 21. Replay results

```json
{
  "sample_evaluation_id": "v212b_c42c747ddbc4",
  "evidence_hash_match": true,
  "classification": "EXACT_EQUIVALENCE"
}
```

## 22. Regression results

144 passed (V2.12B + V2.12A + V2.7–V2.11 + verifier)

## 23. Production-readiness assessment

V2.13 — Controlled LangChain Production Canary (LangGraph rollback retained)

**Production remains on LangGraph throughout V2.12B.**