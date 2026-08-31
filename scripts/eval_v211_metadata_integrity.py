#!/usr/bin/env python3
"""V2.11 adversarial metadata-integrity & grounding enforcement experiment."""

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

OUT = ROOT / "data" / "diagnostics" / "v211_metadata_integrity"
DOC = ROOT / "docs/V2_11_METADATA_INTEGRITY.md"
OUT_JSON = ROOT / "data" / "diagnostics/v211_metadata_integrity.json"
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


def trace_path(variant: str, fixture: str, run_index: int) -> Path:
    return OUT / f"{variant.lower()}_{fixture.lower()}_{run_index:02d}.json"


def write_doc(report: dict[str, Any]) -> None:
    adv = report.get("adversarial_comparison", [])
    lines = [
        "# V2.11 Adversarial Metadata-Integrity & Grounding Enforcement Experiment",
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
        "## Pipeline Variant Summaries (Variant C — Metadata Suppress)",
        "",
        "```json",
        json.dumps(report.get("variant_summaries", {}).get("C_METADATA_SUPPRESS", {}), indent=2),
        "```",
        "",
        "## Adversarial Before/After",
        "",
        "| Adversarial Case | V2.10 Baseline | V2.11 Metadata Guard | Correct Outcome |",
        "| --- | ---: | ---: | --- |",
    ]
    label_map = {
        "ADV_FAKE_PARENT": "Fake parent",
        "ADV_CONFLICTING_PARENT": "Conflicting parent",
        "ADV_PLACEHOLDER_PARENT": "Placeholder parent",
        "ADV_WRONG_SUBJECT": "Wrong subject",
        "ADV_WRONG_GRADE": "Wrong grade",
        "ADV_CONFLICTING_SUBJECT": "Conflicting subject",
        "ADV_CONFLICTING_GRADE": "Conflicting grade",
        "ADV_TOPIC_UUID_COLLISION": "Topic UUID collision",
        "ADV_PARENT_CHILD_MISMATCH": "Parent-child mismatch",
        "ADV_SUBJECT_TOPIC_MISMATCH": "Subject-topic mismatch",
        "ADV_GRADE_TOPIC_MISMATCH": "Grade-topic mismatch",
        "ADV_PLACEHOLDER_TOPIC": "Placeholder topic",
        "ADV_PLACEHOLDER_PARENT_SUBSTANTIVE_CHILD": "Placeholder parent substantive child",
        "ADV_HIGH_SCORE_SAFETY": "High-score safety",
        "ADV_HIGH_SCORE_PLACEHOLDER": "High-score placeholder",
        "ADV_MISSING_AFTER_NORM": "Missing after normalization",
    }
    for row in adv:
        label = label_map.get(row["fixture"], row["fixture"])
        baseline = row.get("v210_baseline_accept_rate")
        guard = row.get("v211_metadata_guard_accept_rate")
        baseline_s = "ACCEPT" if baseline and baseline > 0 else "BLOCK"
        guard_s = "ACCEPT" if guard and guard > 0 else "BLOCK"
        lines.append(
            f"| {label} | {baseline_s} | {guard_s} | {row.get('correct_outcome', 'BLOCK')} |"
        )
    lines.extend(
        [
            "",
            "## Metadata Violations Detected",
            "",
            "```json",
            json.dumps(report.get("violation_summary", {}), indent=2),
            "```",
            "",
            "## V2.12 Recommendation",
            "",
            report.get("v212_recommendation", ""),
        ]
    )
    DOC.write_text("\n".join(lines))


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from app.agent.v211_metadata_integrity import (
        ADVERSARIAL_FIXTURE_CLASSES,
        FIXTURE_CLASSES,
        PIPELINE_VARIANTS,
        PipelineVariant,
        build_adversarial_comparison,
        build_violation_summary,
        interpret_v211,
        summarize_rows,
    )

    variant_summaries = {
        v.value: summarize_rows([r for r in rows if r["pipeline_variant"] == v.value])
        for v in PIPELINE_VARIANTS
    }
    adversarial = build_adversarial_comparison(rows)
    violations = build_violation_summary(rows)
    c_summary = variant_summaries.get(PipelineVariant.C_METADATA_SUPPRESS.value, {})
    conclusion, note, v212, arch = interpret_v211(
        variant_summaries=variant_summaries,
        adversarial_comparison=adversarial,
        c_variant=c_summary,
    )
    c4_trace = next(
        (
            r.get("c4u18_regression")
            for r in rows
            if r.get("fixture_class") == "FAITHFUL_COMPLETE"
            and r.get("pipeline_variant") == PipelineVariant.C_METADATA_SUPPRESS.value
            and r.get("c4u18_regression")
        ),
        None,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "A deterministic metadata-integrity guard before verification can eliminate "
            "adversarial false acceptances without rewriting the verifier or mapper."
        ),
        "c4u18_evidence_hash": "be3e342763f1faac",
        "fractions_evidence_hash": "977b259fcfb4b282",
        "pipeline_variants": [v.value for v in PIPELINE_VARIANTS],
        "fixture_classes": list(FIXTURE_CLASSES),
        "primary_fixture_classes": list(FIXTURE_CLASSES[:12]),
        "adversarial_fixture_classes": list(ADVERSARIAL_FIXTURE_CLASSES),
        "total_evaluations": len(rows),
        "analytical_threshold": ANALYTICAL_THRESHOLD,
        "variant_summaries": variant_summaries,
        "adversarial_comparison": adversarial,
        "violation_summary": violations,
        "c4u18_regression_trace": c4_trace,
        "v210_comparison": {
            "adversarial_false_acceptances": 5,
            "fc_acceptance": 1.0,
            "fi_acceptance": 0.8,
        },
        "conclusion": conclusion,
        "interpretation_note": note,
        "architecture_answer": arch,
        "v212_recommendation": v212,
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
        rows = [r for r in rows if r.get("experiment") == "v2.11_metadata_integrity"]
        report = build_report(rows)
        OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
        write_doc(report)
        print(json.dumps({"conclusion": report["conclusion"]}, indent=2))
        return

    from app.agent.orchestrator import CurriculumQAAgent
    from app.agent.v27_experiment import load_baseline_evidence
    from app.agent.v28_recommendation_mapping import bootstrap_c4u18_baseline
    from app.agent.v211_metadata_integrity import (
        FIXTURE_CLASSES,
        PIPELINE_VARIANTS,
        run_pipeline,
    )

    agent = CurriculumQAAgent()
    verifier = agent.verification_node.verifier
    c4u18 = bootstrap_c4u18_baseline(agent)
    fractions = load_baseline_evidence()

    rows: list[dict[str, Any]] = []
    total = len(FIXTURE_CLASSES) * args.runs * len(PIPELINE_VARIANTS)
    done = 0

    for fixture_class in FIXTURE_CLASSES:
        for i in range(1, args.runs + 1):
            for variant in PIPELINE_VARIANTS:
                out_path = trace_path(variant.value, fixture_class, i)
                if args.resume and out_path.exists():
                    rows.append(json.loads(out_path.read_text()))
                    done += 1
                    continue
                tag = f"{variant.value.lower()}_{fixture_class.lower()}_{i:02d}"
                done += 1
                print(
                    f"=== [{done}/{total}] {variant.value} {fixture_class} {i}/{args.runs} ===",
                    flush=True,
                )
                row = with_retry(
                    lambda variant=variant, tag=tag: run_pipeline(
                        fixture_class=fixture_class,
                        variant=variant,
                        c4u18_baseline=c4u18,
                        fractions_baseline=fractions,
                        verifier=verifier,
                        threshold=ANALYTICAL_THRESHOLD,
                        request_id=tag,
                    )
                )
                row["tag"] = tag
                row["run_index"] = i
                rows.append(row)
                out_path.write_text(json.dumps(row, indent=2, default=str))
                print(
                    json.dumps(
                        {
                            "variant": variant.value,
                            "fixture": fixture_class,
                            "metadata_valid": row["metadata_integrity"]["valid"],
                            "metadata_blocked": row["metadata_blocked"],
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
    print("--- V2.11 EXPERIMENT COMPLETE ---")
    print(json.dumps({"conclusion": report["conclusion"]}, indent=2))


if __name__ == "__main__":
    main()
