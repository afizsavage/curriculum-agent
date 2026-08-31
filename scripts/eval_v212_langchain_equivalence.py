#!/usr/bin/env python3
"""V2.12A LangGraph vs LangChain behavioral equivalence experiment."""

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

OUT = ROOT / "data" / "diagnostics" / "v212a_langchain_equivalence"
DOC = ROOT / "docs/V2_12A_LANGCHAIN_EQUIVALENCE.md"
OUT_JSON = ROOT / "data" / "diagnostics/v212a_langchain_equivalence.json"
RUNS_PER_FIXTURE = 10
MAX_LLM_ATTEMPTS = 5
ANALYTICAL_THRESHOLD = 0.85


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


def trace_path(fixture: str, run_index: int) -> Path:
    return OUT / f"{fixture.lower()}_{run_index:02d}.json"


def write_doc(report: dict[str, Any]) -> None:
    perf = report.get("performance_comparison", {})
    lines = [
        "# V2.12A LangGraph → LangChain Behavioral Equivalence Experiment",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        f"**Conclusion: {report['conclusion']}**",
        "",
        report.get("interpretation_note", ""),
        "",
        "## Architecture Question",
        "",
        report.get("architecture_answer", ""),
        "",
        "## Implementation Summaries",
        "",
        "```json",
        json.dumps(
            {
                "langgraph": report.get("langgraph_summary", {}),
                "langchain": report.get("langchain_summary", {}),
            },
            indent=2,
        ),
        "```",
        "",
        "## Equivalence Classifications",
        "",
        "```json",
        json.dumps(report.get("comparison_summary", {}), indent=2),
        "```",
        "",
        "## Performance Comparison",
        "",
        "| Metric | LangGraph | LangChain | Difference |",
        "| --- | ---: | ---: | ---: |",
        f"| Mean latency (ms) | {perf.get('langgraph_mean_ms', 0)} | {perf.get('langchain_mean_ms', 0)} | {perf.get('mean_delta_ms', 0)} |",
        f"| Median latency (ms) | {perf.get('langgraph_median_ms', 0)} | {perf.get('langchain_median_ms', 0)} | {perf.get('median_delta_ms', 0)} |",
        f"| P95 latency (ms) | {perf.get('langgraph_p95_ms', 0)} | {perf.get('langchain_p95_ms', 0)} | {perf.get('p95_delta_ms', 0)} |",
        f"| LLM calls | {perf.get('langgraph_llm_calls', 0)} | {perf.get('langchain_llm_calls', 0)} | {perf.get('llm_calls_delta', 0)} |",
        f"| Errors | {perf.get('langgraph_errors', 0)} | {perf.get('langchain_errors', 0)} | {perf.get('errors_delta', 0)} |",
        "",
        "## C4-U18 Regression",
        "",
        "```json",
        json.dumps(report.get("c4u18_comparison", {}), indent=2),
        "```",
        "",
        "## V2.13 Recommendation",
        "",
        report.get("v213_recommendation", ""),
    ]
    DOC.write_text("\n".join(lines))


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from app.agent.v212_langchain import (
        FIXTURE_CLASSES,
        build_threshold_sweep,
        interpret_v212,
        summarize_comparisons,
        summarize_implementation_rows,
    )

    langgraph_summary = summarize_implementation_rows(rows, key="langgraph")
    langchain_summary = summarize_implementation_rows(rows, key="langchain")
    comparison_summary = summarize_comparisons(rows)
    threshold_langgraph = build_threshold_sweep(rows, key="langgraph")
    threshold_langchain = build_threshold_sweep(rows, key="langchain")
    conclusion, note, v213, arch = interpret_v212(
        langgraph_summary=langgraph_summary,
        langchain_summary=langchain_summary,
        comparison_summary=comparison_summary,
    )

    c4_graph = next(
        (
            r["langgraph"].get("c4u18_regression")
            for r in rows
            if r.get("fixture_class") == "FAITHFUL_COMPLETE" and r.get("langgraph")
        ),
        None,
    )
    c4_chain = next(
        (
            r["langchain"].get("c4u18_regression")
            for r in rows
            if r.get("fixture_class") == "FAITHFUL_COMPLETE" and r.get("langchain")
        ),
        None,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "LangChain can orchestrate the validated curriculum pipeline with behavioral "
            "equivalence to the LangGraph control path."
        ),
        "fixture_classes": list(FIXTURE_CLASSES),
        "total_evaluations": len(rows) * 2,
        "total_comparisons": len(rows),
        "analytical_threshold": ANALYTICAL_THRESHOLD,
        "langgraph_summary": langgraph_summary,
        "langchain_summary": langchain_summary,
        "comparison_summary": comparison_summary,
        "threshold_sweep": {
            "langgraph": threshold_langgraph,
            "langchain": threshold_langchain,
        },
        "performance_comparison": {
            "langgraph_mean_ms": langgraph_summary.get("mean_latency_ms", 0),
            "langchain_mean_ms": langchain_summary.get("mean_latency_ms", 0),
            "mean_delta_ms": round(
                langchain_summary.get("mean_latency_ms", 0)
                - langgraph_summary.get("mean_latency_ms", 0),
                3,
            ),
            "langgraph_median_ms": langgraph_summary.get("median_latency_ms", 0),
            "langchain_median_ms": langchain_summary.get("median_latency_ms", 0),
            "median_delta_ms": round(
                langchain_summary.get("median_latency_ms", 0)
                - langgraph_summary.get("median_latency_ms", 0),
                3,
            ),
            "langgraph_p95_ms": langgraph_summary.get("p95_latency_ms", 0),
            "langchain_p95_ms": langchain_summary.get("p95_latency_ms", 0),
            "p95_delta_ms": round(
                langchain_summary.get("p95_latency_ms", 0)
                - langgraph_summary.get("p95_latency_ms", 0),
                3,
            ),
            "langgraph_llm_calls": langgraph_summary.get("llm_calls", 0),
            "langchain_llm_calls": langchain_summary.get("llm_calls", 0),
            "llm_calls_delta": langchain_summary.get("llm_calls", 0)
            - langgraph_summary.get("llm_calls", 0),
            "langgraph_errors": langgraph_summary.get("errors", 0),
            "langchain_errors": langchain_summary.get("errors", 0),
            "errors_delta": langchain_summary.get("errors", 0)
            - langgraph_summary.get("errors", 0),
        },
        "c4u18_comparison": {"langgraph": c4_graph, "langchain": c4_chain},
        "conclusion": conclusion,
        "interpretation_note": note,
        "architecture_answer": arch,
        "v213_recommendation": v213,
        "comparisons": rows,
        "production_unchanged": {
            "langgraph_production_graph": True,
            "verifier": True,
            "recommendation_mapper": True,
            "normalization": True,
            "metadata_integrity_rules": True,
            "generator": True,
            "retrieval": True,
            "routing": True,
            "database": True,
            "api_contracts": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=RUNS_PER_FIXTURE)
    parser.add_argument(
        "--implementation",
        choices=("langgraph", "langchain", "both"),
        default="both",
    )
    parser.add_argument("--from-controls", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.from_controls:
        rows = [
            json.loads(p.read_text())
            for p in sorted(OUT.glob("*.json"))
            if p.name != "variant_comparison.json"
        ]
        rows = [r for r in rows if r.get("experiment") == "v2.12a_langchain_equivalence"]
        report = build_report(rows)
        OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
        write_doc(report)
        print(json.dumps({"conclusion": report["conclusion"]}, indent=2))
        return

    from app.agent.orchestrator import CurriculumQAAgent
    from app.agent.v27_experiment import load_baseline_evidence
    from app.agent.v28_recommendation_mapping import bootstrap_c4u18_baseline
    from app.agent.v212_langchain import (
        FIXTURE_CLASSES,
        Implementation,
        run_equivalence_pair,
        run_implementation,
    )
    from app.config import get_settings

    agent = CurriculumQAAgent()
    verifier = agent.verification_node.verifier
    settings = get_settings()
    c4u18 = bootstrap_c4u18_baseline(agent)
    fractions = load_baseline_evidence()

    rows: list[dict[str, Any]] = []
    total = len(FIXTURE_CLASSES) * args.runs
    done = 0

    for fixture_class in FIXTURE_CLASSES:
        for i in range(1, args.runs + 1):
            out_path = trace_path(fixture_class, i)
            if args.resume and out_path.exists():
                rows.append(json.loads(out_path.read_text()))
                done += 1
                continue
            done += 1
            tag = f"{fixture_class.lower()}_{i:02d}"
            print(f"=== [{done}/{total}] {fixture_class} {i}/{args.runs} ===", flush=True)

            if args.implementation == "both":
                row = with_retry(
                    lambda tag=tag: run_equivalence_pair(
                        fixture_class=fixture_class,
                        c4u18_baseline=c4u18,
                        fractions_baseline=fractions,
                        verifier=verifier,
                        settings=settings,
                        threshold=ANALYTICAL_THRESHOLD,
                        run_index=i,
                        request_id=tag,
                    )
                )
            else:
                impl = Implementation(args.implementation)
                result = with_retry(
                    lambda tag=tag, impl=impl: run_implementation(
                        implementation=impl,
                        fixture_class=fixture_class,
                        c4u18_baseline=c4u18,
                        fractions_baseline=fractions,
                        verifier=verifier,
                        settings=settings,
                        threshold=ANALYTICAL_THRESHOLD,
                        run_index=i,
                        request_id=tag,
                    )
                )
                row = {
                    "experiment": "v2.12a_langchain_equivalence",
                    "fixture_class": fixture_class,
                    "run_index": i,
                    "threshold": ANALYTICAL_THRESHOLD,
                    args.implementation: result.to_dict(),
                }

            rows.append(row)
            out_path.write_text(json.dumps(row, indent=2, default=str))
            if args.implementation == "both":
                print(
                    json.dumps(
                        {
                            "fixture": fixture_class,
                            "classification": row["comparison"]["classification"],
                            "lg_accepted": row["langgraph"]["final_accepted"],
                            "lc_accepted": row["langchain"]["final_accepted"],
                            "lg_hash": row["langgraph"]["normalized_evidence_hash"],
                            "lc_hash": row["langchain"]["normalized_evidence_hash"],
                        }
                    ),
                    flush=True,
                )
            time.sleep(0.1)

    report = build_report(rows)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    write_doc(report)
    print("--- V2.12A EXPERIMENT COMPLETE ---")
    print(json.dumps({"conclusion": report["conclusion"]}, indent=2))


if __name__ == "__main__":
    main()
