#!/usr/bin/env python3
"""V2.10 integrated grounding + recommendation safety experiment."""

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

OUT = ROOT / "data" / "diagnostics" / "v210_integrated_experiment"
DOC = ROOT / "docs/V2_10_INTEGRATED_GROUNDING_RECOMMENDATION.md"
OUT_JSON = ROOT / "data" / "diagnostics/v210_integrated_experiment.json"
RUNS_PER_FIXTURE = 10
ADV_RUNS = 1
MAX_LLM_ATTEMPTS = 3
ANALYTICAL_THRESHOLD = 0.85

V28_BASELINE = {"fi_mapped_acceptance": 0.9, "safety_false_acceptance": 0}
V29_BASELINE = {"fc_structural_acceptance": 1.0, "safety_false_acceptance": 0}


def with_retry(fn, *, attempts: int = MAX_LLM_ATTEMPTS, base_delay: float = 3.0):
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            retryable = any(
                t in str(exc).lower()
                for t in ("timeout", "timed out", "connection", "temporarily")
            )
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(base_delay * (attempt + 1))
    raise last_exc  # pragma: no cover


def trace_path(pipeline: str, fixture: str, run_index: int) -> Path:
    return OUT / f"{pipeline.lower()}_{fixture.lower()}_{run_index:02d}.json"


def write_doc(report: dict[str, Any]) -> None:
    comparison = report.get("integration_comparison", [])
    sweep = report.get("threshold_sweep_pipeline_d", [])
    lines = [
        "# V2.10 Integrated Grounding + Recommendation Safety Experiment",
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
        "## Integration Comparison",
        "",
        "| Fixture | RAW+Verifier | Normalized+Verifier | RAW+Mapper | Normalized+Mapper |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison:
        lines.append(
            f"| {row['fixture']} | {row.get('raw_verifier')} | {row.get('normalized_verifier')} | "
            f"{row.get('raw_mapper')} | {row.get('normalized_mapper')} |"
        )
    lines.extend(
        [
            "",
            "## Pipeline D Threshold Sweep",
            "",
            "| Threshold | FC Accept | FI Accept | Placeholder Accept | Safety False Accept | Missing-Evidence Accept |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sweep:
        lines.append(
            f"| {row['threshold']} | {row['faithful_complete_acceptance']} | "
            f"{row['faithful_imperfect_acceptance']} | {row['placeholder_acceptance']} | "
            f"{row['safety_false_acceptance']} | {row['missing_evidence_false_acceptance']} |"
        )
    lines.extend(
        [
            "",
            "## V2.11 Recommendation",
            "",
            report.get("v211_recommendation", ""),
        ]
    )
    DOC.write_text("\n".join(lines))


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from app.agent.v210_integrated_experiment import (
        ADVERSARIAL_FIXTURE_CLASSES,
        INTEGRATION_FIXTURE_CLASSES,
        PIPELINES,
        Pipeline,
        build_adversarial_report,
        build_integration_comparison,
        build_safety_report,
        get_threshold_sweep,
        interpret_v210,
        summarize_pipeline_d_threshold,
        summarize_pipeline_rows,
    )

    pipeline_summaries = {
        p.value: summarize_pipeline_rows([r for r in rows if r["pipeline"] == p.value])
        for p in PIPELINES
    }
    integration_rows = [r for r in rows if r["fixture_class"] in INTEGRATION_FIXTURE_CLASSES]
    d_rows = [r for r in rows if r["pipeline"] == Pipeline.D_NORMALIZED_VERIFIER_MAPPER.value]
    sweep = [
        summarize_pipeline_d_threshold(d_rows, threshold=t) for t in get_threshold_sweep()
    ]
    safety = build_safety_report(integration_rows)
    adversarial = build_adversarial_report(rows)
    comparison = build_integration_comparison(integration_rows)
    conclusion, note, v211, arch = interpret_v210(
        integration_comparison=comparison,
        pipeline_d_summary=pipeline_summaries.get(Pipeline.D_NORMALIZED_VERIFIER_MAPPER.value, {}),
        safety=safety,
        adversarial=adversarial,
        threshold_sweep_d=sweep,
    )
    c4_traces = {
        p.value: next(
            (
                r.get("c4u18_trace")
                for r in rows
                if r.get("pipeline") == p.value
                and r.get("fixture_class") == "FAITHFUL_COMPLETE"
                and r.get("c4u18_trace")
            ),
            None,
        )
        for p in PIPELINES
    }
    attributions: dict[str, int] = {}
    for r in integration_rows:
        attr = r.get("attribution", "OTHER")
        attributions[attr] = attributions.get(attr, 0) + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "Structural normalization repairs grounding before verification; recommendation "
            "mapping safely converts faithful-imperfect retrieve_more without bypassing safety."
        ),
        "c4u18_evidence_hash": "be3e342763f1faac",
        "fractions_evidence_hash": "977b259fcfb4b282",
        "pipelines": [p.value for p in PIPELINES],
        "integration_fixture_classes": list(INTEGRATION_FIXTURE_CLASSES),
        "adversarial_fixture_classes": list(ADVERSARIAL_FIXTURE_CLASSES),
        "total_evaluations": len(rows),
        "analytical_threshold": ANALYTICAL_THRESHOLD,
        "pipeline_summaries": pipeline_summaries,
        "integration_comparison": comparison,
        "threshold_sweep_pipeline_d": sweep,
        "safety": safety,
        "adversarial": adversarial,
        "causal_attribution_counts": attributions,
        "c4u18_pipeline_traces": c4_traces,
        "v28_comparison": V28_BASELINE,
        "v29_comparison": V29_BASELINE,
        "conclusion": conclusion,
        "interpretation_note": note,
        "architecture_answer": arch,
        "v211_recommendation": v211,
        "control_runs": rows,
        "production_unchanged": {
            "verifier": True,
            "recommendation_mapper": True,
            "generator": True,
            "retrieval": True,
            "routing": True,
            "resolver": True,
            "database": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=RUNS_PER_FIXTURE)
    parser.add_argument("--from-controls", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.from_controls:
        rows = [
            json.loads(p.read_text())
            for p in sorted(OUT.glob("*.json"))
            if p.name not in {"variant_comparison.json"}
        ]
        rows = [r for r in rows if r.get("experiment") == "v2.10_integrated_experiment"]
        report = build_report(rows)
        OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
        write_doc(report)
        print(json.dumps({"conclusion": report["conclusion"]}, indent=2))
        return

    from app.agent.orchestrator import CurriculumQAAgent
    from app.agent.v27_experiment import load_baseline_evidence
    from app.agent.v28_recommendation_mapping import bootstrap_c4u18_baseline
    from app.agent.v210_integrated_experiment import (
        ADVERSARIAL_FIXTURE_CLASSES,
        INTEGRATION_FIXTURE_CLASSES,
        PIPELINES,
        Pipeline,
        run_pipeline,
    )

    agent = CurriculumQAAgent()
    verifier = agent.verification_node.verifier
    c4u18 = bootstrap_c4u18_baseline(agent)
    fractions = load_baseline_evidence()

    rows: list[dict[str, Any]] = []
    fixtures = [(f, args.runs) for f in INTEGRATION_FIXTURE_CLASSES] + [
        (f, ADV_RUNS) for f in ADVERSARIAL_FIXTURE_CLASSES
    ]
    total = sum(runs for _, runs in fixtures) * len(PIPELINES)
    done = 0

    for fixture_class, runs in fixtures:
        for i in range(1, runs + 1):
            raw_row: dict[str, Any] | None = None
            norm_row: dict[str, Any] | None = None
            for pipeline in PIPELINES:
                out_path = trace_path(pipeline.value, fixture_class, i)
                if args.resume and out_path.exists():
                    row = json.loads(out_path.read_text())
                    rows.append(row)
                    if pipeline == Pipeline.A_RAW_VERIFIER:
                        raw_row = row
                    if pipeline == Pipeline.B_NORMALIZED_VERIFIER:
                        norm_row = row
                    done += 1
                    continue
                tag = f"{pipeline.value.lower()}_{fixture_class.lower()}_{i:02d}"
                done += 1
                print(
                    f"=== [{done}/{total}] {pipeline.value} {fixture_class} {i}/{runs} ===",
                    flush=True,
                )
                row = with_retry(
                    lambda pipeline=pipeline, tag=tag: run_pipeline(
                        fixture_class=fixture_class,
                        pipeline=pipeline,
                        c4u18_baseline=c4u18,
                        fractions_baseline=fractions,
                        verifier=verifier,
                        threshold=ANALYTICAL_THRESHOLD,
                        request_id=tag,
                        raw_baseline_row=raw_row,
                        normalized_baseline_row=norm_row,
                    )
                )
                row["tag"] = tag
                row["run_index"] = i
                rows.append(row)
                out_path.write_text(json.dumps(row, indent=2, default=str))
                if pipeline == Pipeline.A_RAW_VERIFIER:
                    raw_row = row
                if pipeline == Pipeline.B_NORMALIZED_VERIFIER:
                    norm_row = row
                print(
                    json.dumps(
                        {
                            "pipeline": pipeline.value,
                            "fixture": fixture_class,
                            "score": row["verifier_score"],
                            "verifier": row["verifier_decision"],
                            "final": row["final_recommendation"],
                            "accepted": row["final_accepted"],
                        }
                    ),
                    flush=True,
                )
                time.sleep(0.1)

    report = build_report(rows)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    write_doc(report)
    print("--- V2.10 EXPERIMENT COMPLETE ---")
    print(json.dumps({"conclusion": report["conclusion"]}, indent=2))


if __name__ == "__main__":
    main()
