# V2.13C — Controlled Hybrid Retrieval + Real Curriculum QA Evaluation

Generated: `2026-09-04T09:12:58.735211+00:00`

**Conclusion: SUPPORTED**

Document evidence substantially increases correctly grounded answers on narrative questions without safety failures or structured regressions.

## Executive Summary

Hypothesis: adding grounded MBSSE document evidence via V2.13B context-hybrid retrieval increases correctly answerable narrative curriculum questions without increasing unsupported or wrong-context answers.

- Dataset: **72** questions (`v213c.1`)
- Dataset hash: `de6b387d3c2588df96dbe543eea05332c814044306123852ff7aa916c2841b4f`
- Corpus: V2.13A fixtures, hashes `{"bec-framework-2020": "26409f8e53267603f3b446ad19ef422cff73f7116b74661daa09c77459fb08a4", "math-primary-guidance": "3710ff81811f0fd2c436d89db7d43d9a03bdac5ed4e4e25712a0dde2c1733d70", "science-guidance": "feef53e14b590cbd983834cd0c144d2060f88ae68f493fe20b536350df3fea13"}`
- Control: structured Curriculum API evidence only (frozen catalog)
- Experiment: structured + V2.13B `context_hybrid` document retrieval
- Newly answerable: **22** (0.306); document-only: **13**

## Dataset Breakdown

| Category | Count |
|----------|------:|
| adversarial | 12 |
| ambiguous | 8 |
| document_only | 14 |
| insufficient_evidence | 10 |
| source_grounding | 8 |
| structured_fact | 12 |
| structured_plus_document | 8 |

## Retrieval Results

```json
{
  "control_evidence_found_rate": 0.4305555555555556,
  "experiment_evidence_found_rate": 0.9583333333333334,
  "document_only_control_gold": 0.0,
  "document_only_experiment_gold": 0.9285714285714286,
  "experiment_provenance_complete": 1.0,
  "wrong_context_rate_experiment": 0.06944444444444445,
  "recall_proxy_document_only": 0.9285714285714286
}
```

## Answer Quality

| Metric | Control | Experiment |
|--------|--------:|-----------:|
| Grounded correct | 0.597 | 0.903 |
| Verifier/mapper accept | 0.264 | 0.597 |
| Structured-fact delta | | 0.000 |

Paired: improved 22, unchanged 50, regressed 0.

```json
{
  "statistic": 20.045454545454547,
  "n_discordant": 22,
  "improved": 22,
  "regressed": 0,
  "note": "continuity-corrected McNemar chi-square; not a p-value claim"
}
```

McNemar statistic is reported as an observed paired discordance measure, not a significance claim.

## Safety Results

```json
{
  "wrong_context_false_acceptance": 0,
  "placeholder_false_acceptance": 0,
  "metadata_integrity_false_acceptance": 0,
  "unsafe_adversarial_false_acceptance": 0
}
```

## Latency

```json
{
  "control_mean_ms": 1.6206384166821408,
  "experiment_mean_ms": 37.932264319433926,
  "control_p50_ms": 0.7809849998920981,
  "control_p95_ms": 2.871455999866157,
  "experiment_p50_ms": 35.77778299995771,
  "experiment_p95_ms": 42.260884000370424,
  "added_mean_ms": 36.311625902751786,
  "added_pct": 2240.5754133047717,
  "retrieval_mean_ms": 33.11490455551949
}
```

## Biggest Improvements

```json
[
  {
    "id": "V213C-A01",
    "question": "What does the MBSSE curriculum say about the purpose of mathematics education?",
    "difference": "DOCUMENT_ADDED_MISSING_CONTEXT"
  },
  {
    "id": "V213C-A02",
    "question": "What does the curriculum say about teaching and learning mathematics?",
    "difference": "DOCUMENT_ADDED_MISSING_CONTEXT"
  },
  {
    "id": "V213C-A03",
    "question": "What principles guide mathematics teaching according to the curriculum?",
    "difference": "DOCUMENT_ADDED_MISSING_CONTEXT"
  },
  {
    "id": "V213C-A04",
    "question": "What does the curriculum say about mathematics at the primary level?",
    "difference": "DOCUMENT_ADDED_MISSING_CONTEXT"
  },
  {
    "id": "V213C-A05",
    "question": "Why is mathematics education considered important according to the curriculum?",
    "difference": "DOCUMENT_ADDED_MISSING_CONTEXT"
  },
  {
    "id": "V213C-A06",
    "question": "What does the curriculum say about science inquiry at primary level?",
    "difference": "DOCUMENT_ADDED_MISSING_CONTEXT"
  },
  {
    "id": "V213C-A07",
    "question": "What does the curriculum emphasize about science education?",
    "difference": "DOCUMENT_ADDED_MISSING_CONTEXT"
  },
  {
    "id": "V213C-A08",
    "question": "What does the curriculum say about classroom investigations in science?",
    "difference": "DOCUMENT_ADDED_MISSING_CONTEXT"
  }
]
```

## Regressions

```json
[]
```

## Failure Analysis

```json
{
  "DOCUMENT_ADDED_MISSING_CONTEXT": 22,
  "DOCUMENT_DID_NOT_HELP": 7,
  "STRUCTURED_ALREADY_SUFFICIENT": 43
}
```

Document-only questions lack structured LOs, so control is typically insufficient. Experiment succeeds when context-hybrid retrieves gold fragments and the evidence-quoting synthesizer stays within retrieved text. Adversarial structured poison is present in both arms; V2.11 + V2.8 mapper must keep false acceptance at zero.

## Production integrity

- LangGraph path unchanged
- Verifier unchanged
- V2.8 mapper unchanged
- V2.11 guard unchanged
- `/api/v1` unchanged
- Production document retrieval disabled
- `v213c_experiment=false`, `v213c_document_retrieval=false`

## Recommendation

V2.13D — controlled production-shadow / canary of context-hybrid document evidence

Do **not** automatically promote document retrieval to production.