#!/usr/bin/env python3
"""V2.12B real-retrieval production-shadow evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "diagnostics" / "v212b_production_shadow"
DOC = ROOT / "docs" / "V2_12B_PRODUCTION_SHADOW.md"
OUT_JSON = ROOT / "data" / "diagnostics" / "v212b_production_shadow.json"
MAX_LLM_ATTEMPTS = 5


def with_retry(fn, *, attempts: int = MAX_LLM_ATTEMPTS, base_delay: float = 5.0):
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            retryable = any(
                t in str(exc).lower()
                for t in (
                    "timeout",
                    "timed out",
                    "connection",
                    "temporarily",
                    "name resolution",
                    "llmprovidererror",
                )
            )
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(base_delay * (attempt + 1))
    raise last_exc  # pragma: no cover


def trace_path(evaluation_id: str) -> Path:
    return OUT / f"{evaluation_id}.json"


def write_doc(report: dict[str, Any]) -> None:
    metrics = report.get("metrics", {})
    latency = metrics.get("latency", {})
    lines = [
        "# V2.12B Real Retrieval Production-Shadow Evaluation",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        f"**Conclusion: {report['conclusion']}**",
        "",
        report.get("interpretation_note", ""),
        "",
        "## 1. Objective",
        "",
        "Determine whether LangChain can safely operate the Curriculum QA pipeline under "
        "real retrieval while LangGraph remains production control.",
        "",
        "## 2. Shadow architecture",
        "",
        "One shared retrieval → evidence snapshot → LangGraph control + LangChain experiment "
        "→ shadow analyzer. Production responses unaffected.",
        "",
        "## 3. Sampling methodology",
        "",
        f"- Representative curriculum questions: `{report.get('n_questions', 0)}`",
        f"- Evaluations completed: `{report.get('n_evaluations', 0)}`",
        "- Reproducible question set in `app/agent/v212b_shadow.py` (`REAL_QUESTIONS`)",
        "",
        "## 4. Number of real questions",
        "",
        str(report.get("n_evaluations", 0)),
        "",
        "## 5. Evidence statistics",
        "",
        "```json",
        json.dumps(metrics.get("retrieval_statistics", {}), indent=2),
        "```",
        "",
        "## 6. Normalization statistics",
        "",
        "```json",
        json.dumps(metrics.get("normalization_statistics", {}), indent=2),
        "```",
        "",
        "## 7. Metadata-integrity statistics",
        "",
        "```json",
        json.dumps(metrics.get("metadata_statistics", {}), indent=2),
        "```",
        "",
        "## 8. Verifier statistics",
        "",
        "```json",
        json.dumps(metrics.get("verifier_statistics", {}), indent=2),
        "```",
        "",
        "## 9. Recommendation-mapping statistics",
        "",
        "```json",
        json.dumps(metrics.get("mapper_statistics", {}), indent=2),
        "```",
        "",
        "## 10. Routing statistics",
        "",
        "```json",
        json.dumps(metrics.get("routing_statistics", {}), indent=2),
        "```",
        "",
        "## 11. LangGraph vs LangChain comparison",
        "",
        "```json",
        json.dumps(
            {
                "langgraph": metrics.get("langgraph_summary", {}),
                "langchain": metrics.get("langchain_summary", {}),
            },
            indent=2,
        ),
        "```",
        "",
        "## 12. Equivalence classifications",
        "",
        "```json",
        json.dumps(metrics.get("comparison_summary", {}), indent=2),
        "```",
        "",
        "## 13. FC behavior",
        "",
        f"LangGraph FC proxy accept: `{metrics.get('langgraph_summary', {}).get('faithful_complete_acceptance', 'n/a')}`",
        "",
        "## 14. FI behavior",
        "",
        "```json",
        json.dumps(metrics.get("fi_monitoring", {}), indent=2),
        "```",
        "",
        "## 15. Safety behavior",
        "",
        "```json",
        json.dumps(metrics.get("safety", {}), indent=2),
        "```",
        "",
        "## 16. Metadata adversarial behavior",
        "",
        f"Metadata false acceptance (LangChain): `{metrics.get('safety', {}).get('metadata_false_acceptance', 0)}`",
        "",
        "## 17. Placeholder behavior",
        "",
        f"Placeholder false acceptance: `{metrics.get('safety', {}).get('placeholder_false_acceptance', 0)}`",
        "",
        "## 18. Divergence analysis",
        "",
        f"Unsafe divergences: `{metrics.get('safety', {}).get('unsafe_divergence_count', 0)}`",
        "",
        "## 19. Latency",
        "",
        "| Metric | LangGraph | LangChain | Overhead |",
        "| --- | ---: | ---: | ---: |",
        f"| Mean (ms) | {latency.get('langgraph_mean_ms', 0)} | {latency.get('langchain_mean_ms', 0)} | {latency.get('overhead_mean_ms', 0)} |",
        "",
        "## 20. Error/timeout rates",
        "",
        f"Shadow errors: `{metrics.get('shadow_errors', 0)}`",
        "",
        "## 21. Replay results",
        "",
        "```json",
        json.dumps(report.get("replay_validation", {}), indent=2),
        "```",
        "",
        "## 22. Regression results",
        "",
        report.get("regression_summary", "See test suite output."),
        "",
        "## 23. Production-readiness assessment",
        "",
        report.get("v213_recommendation", ""),
        "",
        "**Production remains on LangGraph throughout V2.12B.**",
    ]
    DOC.write_text("\n".join(lines))


def build_report(
    rows: list[dict[str, Any]],
    *,
    replay_validation: dict[str, Any],
    regression_summary: str,
) -> dict[str, Any]:
    from app.agent.v212b_shadow import aggregate_metrics, interpret_v212b

    metrics = aggregate_metrics(rows)
    conclusion, note, v213 = interpret_v212b(metrics)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "v2.12b_production_shadow",
        "n_questions": len({r.get("question") for r in rows}),
        "n_evaluations": len(rows),
        "metrics": metrics,
        "conclusion": conclusion,
        "interpretation_note": note,
        "v213_recommendation": v213,
        "replay_validation": replay_validation,
        "regression_summary": regression_summary,
        "production_default": "langgraph",
        "langchain_mode": "shadow_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V2.12B production-shadow eval")
    parser.add_argument("--limit", type=int, default=0, help="Max questions (0=all)")
    parser.add_argument("--resume", action="store_true", help="Skip completed traces")
    parser.add_argument("--replay", type=str, default="", help="Replay evaluation_id")
    parser.add_argument("--skip-regression", action="store_true")
    args = parser.parse_args()

    from app.agent.orchestrator import CurriculumQAAgent
    from app.agent.v212b_shadow import REAL_QUESTIONS, persist_trace, replay_evaluation, run_shadow_evaluation

    if args.replay:
        result = replay_evaluation(evaluation_id=args.replay)
        print(json.dumps(result, indent=2))
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    agent = CurriculumQAAgent()
    questions = REAL_QUESTIONS
    if args.limit > 0:
        questions = questions[: args.limit]

    rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(questions, start=1):
        tag = f"q{idx:02d}_{hash(spec['question']) & 0xFFFF:04x}"
        if args.resume:
            existing = list(OUT.glob("v212b_*.json"))
            for path in existing:
                try:
                    data = json.loads(path.read_text())
                    if data.get("question") == spec["question"]:
                        rows.append(data)
                        print(f"SKIP resume {spec['question'][:50]}...")
                        break
                except Exception:
                    continue
            else:
                pass
            if any(r.get("question") == spec["question"] for r in rows):
                continue

        print(f"[{idx}/{len(questions)}] {spec['question'][:60]}...")
        try:
            trace = with_retry(
                lambda: run_shadow_evaluation(
                    agent,
                    question_spec=spec,
                    run_index=idx,
                    request_id=tag,
                )
            )
            persist_trace(trace)
            rows.append(trace)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            rows.append(
                {
                    "question": spec["question"],
                    "category": spec.get("category"),
                    "shadow_error": str(exc),
                    "langgraph": {},
                    "langchain": {},
                    "comparison": {},
                }
            )

    replay_validation: dict[str, Any] = {}
    completed = [r for r in rows if r.get("evaluation_id")]
    if completed:
        sample_id = completed[0]["evaluation_id"]
        try:
            replay = replay_evaluation(evaluation_id=sample_id, verifier=agent.verification_node.verifier)
            replay_validation = {
                "sample_evaluation_id": sample_id,
                "evidence_hash_match": replay.get("evidence_hash_match"),
                "classification": replay.get("comparison", {}).get("classification"),
            }
        except Exception as exc:
            replay_validation = {"error": str(exc)}

    regression_summary = "skipped"
    if not args.skip_regression:
        import subprocess

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/agent/test_v212b_production_shadow.py",
                "tests/agent/test_v212_langchain_equivalence.py",
                "tests/agent/test_v211_metadata_integrity.py",
                "tests/agent/test_v210_integrated_experiment.py",
                "tests/agent/test_v29_evidence_normalization.py",
                "tests/agent/test_v28_recommendation_mapping.py",
                "tests/agent/test_v27_experiment.py",
                "tests/verifier/",
                "-q",
                "--tb=no",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        regression_summary = proc.stdout.strip() or proc.stderr.strip()

    report = build_report(
        rows,
        replay_validation=replay_validation,
        regression_summary=regression_summary,
    )
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    write_doc(report)
    print(f"\nConclusion: {report['conclusion']}")
    print(f"Report: {DOC}")
    print(f"JSON: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
