#!/usr/bin/env python3
"""V2.9 evidence normalization & grounding-boundary experiment."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "diagnostics" / "v29_evidence_normalization"
DOC = ROOT / "docs/V2_9_EVIDENCE_NORMALIZATION.md"
OUT_JSON = ROOT / "data" / "diagnostics/v29_evidence_normalization.json"
RUNS_PER_FIXTURE = 10
MAX_LLM_ATTEMPTS = 3

V28_BASELINE = {
    "faithful_imperfect_mapped_acceptance": 0.9,
    "faithful_complete_verifier_acceptance": 0.2,
    "placeholder_false_acceptance": 0.0,
    "safety_false_acceptance": 0.0,
}


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


def trace_path(variant: str, fixture: str, run_index: int) -> Path:
    return OUT / f"{variant.lower()}_{fixture.lower()}_{run_index:02d}.json"


def write_doc(report: dict[str, Any]) -> None:
    comparison = report.get("variant_comparison", [])
    fc_analysis = report.get("fc_root_cause_analysis", {})
    lines = [
        "# V2.9 Evidence Normalization & Grounding-Boundary Experiment",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        f"**Conclusion: {report['conclusion']}**",
        "",
        report.get("interpretation_note", ""),
        "",
        "## Hypothesis",
        "",
        report["hypothesis"],
        "",
        "## Setup",
        "",
        f"- C4-U18 evidence hash: `{report['c4u18_evidence_hash']}`",
        f"- Fractions evidence hash: `{report['fractions_evidence_hash']}`",
        f"- Variants: {', '.join(report['normalization_variants'])}",
        f"- Fixtures: {len(report['fixture_classes'])} classes × {RUNS_PER_FIXTURE} runs × "
        f"{len(report['normalization_variants'])} variants = {report['total_evaluations']}",
        "- Harness-only pre-verifier normalization; production unchanged",
        "",
        "## Variant Comparison",
        "",
        "| Variant | FC Accept | FI Accept | Placeholder Accept | Safety False Accept | Avg Score |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison:
        lines.append(
            f"| {row['variant']} | {row['faithful_complete_acceptance']} | "
            f"{row['faithful_imperfect_acceptance']} | {row['placeholder_acceptance']} | "
            f"{row['safety_false_acceptance']} | {row['avg_verifier_score']} |"
        )
    lines.extend(
        [
            "",
            "## C4-U18 FAITHFUL_COMPLETE Diagnosis",
            "",
            f"**Root cause verdict:** {fc_analysis.get('verdict', 'unknown')}",
            "",
            f"**Grounding question answer:** {report.get('fc_grounding_answer', '')}",
            "",
            "### Cause counts",
            "",
            json.dumps(fc_analysis.get("cause_counts", {}), indent=2),
            "",
            "## V2.8 Comparison",
            "",
            "| Metric | V2.8 | V2.9 RAW |",
            "| --- | ---: | ---: |",
        ]
    )
    raw = report.get("variant_summaries", {}).get("RAW", {})
    v28 = report.get("v28_comparison", {})
    lines.append(
        f"| FC acceptance | {v28.get('faithful_complete_verifier_acceptance')} | "
        f"{raw.get('faithful_complete_acceptance')} |"
    )
    lines.append(
        f"| FI acceptance (verifier) | n/a (mapped 0.9) | {raw.get('faithful_imperfect_acceptance')} |"
    )
    lines.append(
        f"| Placeholder false accept | {v28.get('placeholder_false_acceptance')} | "
        f"{raw.get('placeholder_acceptance')} |"
    )
    lines.append(
        f"| Safety false accept | {v28.get('safety_false_acceptance')} | "
        f"{raw.get('safety_false_acceptance', 0)} |"
    )
    lines.extend(
        [
            "",
            "## Placeholder Diagnostics (sample)",
            "",
        ]
    )
    for diag in report.get("placeholder_diagnostics", [])[:8]:
        lines.append(
            f"- {diag.get('fixture')} [{diag.get('variant')}]: "
            f"score={diag.get('verifier_score')}, decision={diag.get('verifier_decision')}, "
            f"records_out={diag.get('normalized_records_out')}"
        )
    lines.extend(
        [
            "",
            "## V2.10 Recommendation",
            "",
            report.get("v210_recommendation", ""),
        ]
    )
    DOC.write_text("\n".join(lines))


def build_report(control_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from app.agent.v28_recommendation_mapping import FIXTURES
    from app.agent.v29_evidence_normalization import (
        NORMALIZATION_VARIANTS,
        analyze_fc_root_cause,
        build_fc_diagnostics,
        build_placeholder_diagnostics,
        build_safety_report,
        interpret_v29,
        summarize_variant_rows,
    )

    variant_summaries: dict[str, dict[str, Any]] = {}
    variant_comparison: list[dict[str, Any]] = []
    raw_summary: dict[str, Any] = {}

    for variant in NORMALIZATION_VARIANTS:
        subset = [r for r in control_rows if r.get("normalization_variant") == variant.value]
        summary = summarize_variant_rows(subset)
        variant_summaries[variant.value] = summary
        if variant.value == "RAW":
            raw_summary = summary
        variant_comparison.append(
            {
                "variant": variant.value,
                "faithful_complete_acceptance": summary["faithful_complete_acceptance"],
                "faithful_imperfect_acceptance": summary["faithful_imperfect_acceptance"],
                "placeholder_acceptance": summary["placeholder_acceptance"],
                "safety_false_acceptance": summary["safety_false_acceptance"],
                "avg_verifier_score": summary["avg_verifier_score"],
                "delta_fc_vs_raw": round(
                    summary["faithful_complete_acceptance"]
                    - variant_summaries.get("RAW", {}).get("faithful_complete_acceptance", 0),
                    3,
                ),
            }
        )

    safety = build_safety_report(
        [r for r in control_rows if r.get("normalization_variant") == "RAW"]
    )
    fc_rows = [r for r in control_rows if r["fixture_class"] == "FAITHFUL_COMPLETE"]
    fc_analysis = analyze_fc_root_cause(fc_rows)
    conclusion, note, v210, grounding_answer = interpret_v29(
        variant_summaries,
        fc_analysis=fc_analysis,
        safety=safety,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "Pre-verifier evidence normalization can resolve placeholder and representation "
            "failures without changing verifier semantics."
        ),
        "c4u18_evidence_hash": next(
            (r.get("evidence_hash") for r in control_rows if r.get("evidence_source") == "c4u18"),
            None,
        ),
        "fractions_evidence_hash": next(
            (r.get("evidence_hash") for r in control_rows if r.get("evidence_source") == "fractions"),
            None,
        ),
        "normalization_variants": [v.value for v in NORMALIZATION_VARIANTS],
        "fixture_classes": list(FIXTURES.keys()),
        "total_evaluations": len(control_rows),
        "variant_summaries": variant_summaries,
        "variant_comparison": variant_comparison,
        "safety": safety,
        "placeholder_diagnostics": build_placeholder_diagnostics(control_rows),
        "fc_diagnostics": build_fc_diagnostics(control_rows),
        "fc_root_cause_analysis": fc_analysis,
        "v28_comparison": V28_BASELINE,
        "conclusion": conclusion,
        "interpretation_note": note,
        "fc_grounding_answer": grounding_answer,
        "v210_recommendation": v210,
        "control_runs": control_rows,
        "production_unchanged": {
            "verifier": True,
            "recommendation_mapper": True,
            "generator": True,
            "retrieval": True,
            "routing": True,
            "resolver": True,
            "database": True,
            "semantic_search": True,
            "v2_api": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=RUNS_PER_FIXTURE)
    parser.add_argument("--from-controls", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--variant", type=str, default="")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.from_controls:
        control_rows = [json.loads(path.read_text()) for path in sorted(OUT.glob("*.json")) if path.name != "variant_comparison.json"]
        control_rows = [r for r in control_rows if r.get("experiment") == "v2.9_evidence_normalization"]
        report = build_report(control_rows)
        OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
        (OUT / "variant_comparison.json").write_text(
            json.dumps(report["variant_comparison"], indent=2)
        )
        write_doc(report)
        print("--- V2.9 EXPERIMENT COMPLETE (from controls) ---")
        print(
            json.dumps(
                {
                    "conclusion": report["conclusion"],
                    "variant_comparison": report["variant_comparison"],
                },
                indent=2,
            )
        )
        return

    from app.agent.orchestrator import CurriculumQAAgent
    from app.agent.v27_experiment import load_baseline_evidence
    from app.agent.v28_recommendation_mapping import FIXTURES, bootstrap_c4u18_baseline
    from app.agent.v29_evidence_normalization import NORMALIZATION_VARIANTS, replay_fixture

    agent = CurriculumQAAgent()
    verifier = agent.verification_node.verifier
    c4u18 = bootstrap_c4u18_baseline(agent)
    fractions = load_baseline_evidence()

    variants = NORMALIZATION_VARIANTS
    if args.variant:
        variants = [v for v in NORMALIZATION_VARIANTS if v.value == args.variant.upper()]

    control_rows: list[dict[str, Any]] = []
    total = len(variants) * len(FIXTURES) * args.runs
    done = 0

    for variant in variants:
        for fixture_class in FIXTURES:
            for i in range(1, args.runs + 1):
                out_path = trace_path(variant.value, fixture_class, i)
                if args.resume and out_path.exists():
                    control_rows.append(json.loads(out_path.read_text()))
                    done += 1
                    continue
                tag = f"{variant.value.lower()}_{fixture_class.lower()}_{i:02d}"
                done += 1
                print(
                    f"=== [{done}/{total}] {variant.value} {fixture_class} {i}/{args.runs} ===",
                    flush=True,
                )
                row = with_retry(
                    lambda fixture_class=fixture_class, variant=variant, tag=tag: replay_fixture(
                        fixture_class=fixture_class,  # type: ignore[arg-type]
                        variant=variant,
                        c4u18_baseline=c4u18,
                        fractions_baseline=fractions,
                        verifier=verifier,
                        request_id=tag,
                    )
                )
                row["tag"] = tag
                row["run_index"] = i
                control_rows.append(row)
                out_path.write_text(json.dumps(row, indent=2, default=str))
                print(
                    json.dumps(
                        {
                            "variant": variant.value,
                            "fixture": fixture_class,
                            "score": row["verifier_score"],
                            "decision": row["verifier_decision"],
                            "accepted": row["verifier_accepted"],
                        }
                    ),
                    flush=True,
                )
                time.sleep(0.1)

    report = build_report(control_rows)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    (OUT / "variant_comparison.json").write_text(
        json.dumps(report["variant_comparison"], indent=2)
    )
    write_doc(report)
    print("--- V2.9 EXPERIMENT COMPLETE ---")
    print(
        json.dumps(
            {
                "conclusion": report["conclusion"],
                "variant_comparison": report["variant_comparison"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
