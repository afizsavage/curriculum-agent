#!/usr/bin/env python3
"""V2.8 recommendation-mapping experiment."""

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

OUT = ROOT / "data" / "diagnostics" / "v28_recommendation_mapping"
DOC = ROOT / "docs/V2_8_RECOMMENDATION_MAPPING.md"
OUT_JSON = ROOT / "data" / "diagnostics" / "v28_recommendation_mapping.json"
RUNS_PER_FIXTURE = 10
MAX_LLM_ATTEMPTS = 3
ANALYTICAL_THRESHOLD = 0.85
V27_BASELINE = {
    "faithful_imperfect_acceptance": 0.8,
    "faithful_imperfect_false_retrieval_residual": 0.2,
    "overall_false_retrieval": 0.133,
    "safety_false_acceptance": 0.0,
    "faithful_complete_acceptance": 0.0,
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


def summarize_control(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    scores = [float(r.get("verifier_score") or 0) for r in rows]
    fi = [r for r in rows if r["fixture_class"] == "FAITHFUL_IMPERFECT"]
    fi_n = len(fi) or 1
    return {
        "n": len(rows),
        "verifier_acceptance_rate": round(sum(1 for r in rows if r.get("verifier_accepted")) / n, 3),
        "retrieve_more_rate": round(sum(1 for r in rows if r.get("retrieve_more_requested")) / n, 3),
        "insufficient_evidence_rate": round(sum(1 for r in rows if r.get("insufficient_evidence")) / n, 3),
        "rejection_rate": round(
            sum(1 for r in rows if r.get("verifier_decision") in {"fallback", "reject"}) / n, 3
        ),
        "avg_verifier_score": round(statistics.mean(scores), 3) if scores else None,
        "faithful_imperfect_verifier_acceptance": round(
            sum(1 for r in fi if r.get("verifier_accepted")) / fi_n, 3
        ),
        "faithful_imperfect_false_retrieval_rate": round(
            sum(1 for r in fi if r.get("false_retrieval")) / fi_n, 3
        ),
        "overall_false_retrieval_rate": round(sum(1 for r in rows if r.get("false_retrieval")) / n, 3),
    }


def compute_safety(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    from app.agent.v28_recommendation_mapping import remap_row_for_threshold

    groups = {
        "unsupported_claim": "UNSUPPORTED_CLAIM",
        "unsupported_absence": "UNSUPPORTED_ABSENCE",
        "speculative": "SPECULATIVE",
        "reconstruction": "RECONSTRUCTION",
        "missing_evidence": "MISSING_EVIDENCE",
        "clean_placeholder": "CLEAN_PLACEHOLDER",
    }
    out: dict[str, Any] = {}
    false_accept = 0
    for key, fixture in groups.items():
        subset = [r for r in rows if r["fixture_class"] == fixture]
        mapped = [remap_row_for_threshold(r, threshold) for r in subset]
        rejected = sum(1 for r in mapped if not r["mapped_accepted"])
        accepted = sum(1 for r in mapped if r["mapped_accepted"])
        false_accept += accepted if fixture != "FAITHFUL_COMPLETE" else 0
        out[key] = {
            "cases": len(subset),
            "rejected": rejected,
            "incorrectly_accepted": accepted if fixture != "FAITHFUL_COMPLETE" else 0,
        }
    out["false_acceptance_total"] = false_accept
    return out


def write_doc(report: dict[str, Any]) -> None:
    c = report["control_summary"]
    m = report["mapped_summary"]
    sweep = report["threshold_sweep"]
    lines = [
        "# V2.8 Recommendation-Mapping Experiment",
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
        f"- Primary unit: C4-U18 Everyday Arithmetic Money",
        f"- C4-U18 evidence hash: `{report['c4u18_evidence_hash']}`",
        f"- Fractions evidence hash (imperfect/placeholder): `{report['fractions_evidence_hash']}`",
        f"- Fixtures: {len(report['fixture_classes'])} classes × {RUNS_PER_FIXTURE} runs = {report['total_evaluations']}",
        "- Harness-only post-verifier recommendation mapper; production unchanged",
        "",
        "## Policy",
        "",
        "1. Safety failures always reject",
        "2. Missing evidence → insufficient/retrieve_more",
        "3. Faithful imperfect + score threshold + retrieve_more → accept",
        "4. Placeholder evidence never accepted via score alone",
        "",
        "## Threshold Sweep",
        "",
        "| Threshold | FI Accept | FI Retrieve | FC Accept | Placeholder Accept | Safety Rejections |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sweep:
        lines.append(
            f"| {row['threshold']} | {row['faithful_imperfect_acceptance']} | "
            f"{row['faithful_imperfect_retrieve_more']} | {row['faithful_complete_acceptance']} | "
            f"{row['placeholder_acceptance']} | {row['safety_rejections']} |"
        )
    lines.extend(
        [
            "",
            "## V2.7 Comparison",
            "",
            "| Metric | V2.7 | V2.8 |",
            "| --- | ---: | ---: |",
            f"| FI acceptance | {V27_BASELINE['faithful_imperfect_acceptance']} | {m.get('faithful_imperfect_acceptance')} |",
            f"| FI false retrieval residual | {V27_BASELINE['faithful_imperfect_false_retrieval_residual']} | {m.get('faithful_imperfect_retrieve_more')} |",
            f"| Overall false retrieval | {V27_BASELINE['overall_false_retrieval']} | {c.get('overall_false_retrieval_rate')} |",
            f"| Safety false acceptance | {V27_BASELINE['safety_false_acceptance']} | {report['safety'].get('false_acceptance_total', 0)} |",
            f"| FC acceptance | failed (placeholders) | {m.get('faithful_complete_acceptance')} |",
            "",
            "## Placeholder Diagnostics",
            "",
        ]
    )
    for diag in report.get("placeholder_diagnostics", [])[:5]:
        lines.append(
            f"- {diag['fixture']}: score={diag.get('verifier_score')}, "
            f"mapped={diag.get('mapped_recommendation')}, class={diag.get('placeholder_classification')}"
        )
    lines.extend(
        [
            "",
            "## V2.9 Recommendation",
            "",
            report.get("v29_recommendation", ""),
        ]
    )
    DOC.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=RUNS_PER_FIXTURE)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--from-controls", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.replay:
        report = json.loads(OUT_JSON.read_text())
        write_doc(report)
        print(json.dumps({"conclusion": report["conclusion"]}, indent=2))
        return

    if args.from_controls:
        from app.agent.v28_recommendation_mapping import (
            FIXTURES,
            build_placeholder_diagnostics,
            interpret_v28,
            summarize_threshold_sweep,
            threshold_sweep,
        )

        control_rows = [
            json.loads(path.read_text())
            for path in sorted(OUT.glob("*_control.json"))
        ]
        sweep = [summarize_threshold_sweep(control_rows, threshold=t) for t in threshold_sweep()]
        safety = compute_safety(control_rows, ANALYTICAL_THRESHOLD)
        control_summary = summarize_control(control_rows)
        mapped_summary = next(r for r in sweep if r["threshold"] == ANALYTICAL_THRESHOLD)
        conclusion, note, v29 = interpret_v28(
            control_summary, sweep, safety, analytical_threshold=ANALYTICAL_THRESHOLD
        )
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "hypothesis": (
                "An independent post-verifier recommendation-mapping layer can safely translate "
                "verifier results into operational actions without modifying verifier scoring."
            ),
            "c4u18_evidence_hash": next(
                (r.get("evidence_hash") for r in control_rows if r.get("evidence_source") == "c4u18"),
                None,
            ),
            "fractions_evidence_hash": next(
                (r.get("evidence_hash") for r in control_rows if r.get("evidence_source") == "fractions"),
                None,
            ),
            "fixture_classes": list(FIXTURES.keys()),
            "total_evaluations": len(control_rows),
            "analytical_threshold": ANALYTICAL_THRESHOLD,
            "control_summary": control_summary,
            "mapped_summary": mapped_summary,
            "threshold_sweep": sweep,
            "control_runs": control_rows,
            "safety": safety,
            "placeholder_diagnostics": build_placeholder_diagnostics(control_rows),
            "v27_comparison": V27_BASELINE,
            "conclusion": conclusion,
            "interpretation_note": note,
            "v29_recommendation": v29,
            "production_unchanged": {
                "verifier": True,
                "generator": True,
                "retrieval": True,
                "routing": True,
                "resolver": True,
                "database": True,
                "semantic_search": True,
                "v2_api": True,
            },
        }
        OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
        (OUT / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2))
        write_doc(report)
        print("--- V2.8 EXPERIMENT COMPLETE (from controls) ---")
        print(json.dumps({"conclusion": conclusion, "sweep": sweep}, indent=2))
        return

    from app.agent.evidence_snapshot import evidence_snapshot_hash
    from app.agent.orchestrator import CurriculumQAAgent
    from app.agent.v27_experiment import load_baseline_evidence
    from app.agent.v28_recommendation_mapping import (
        FIXTURES,
        bootstrap_c4u18_baseline,
        build_placeholder_diagnostics,
        interpret_v28,
        replay_fixture,
        summarize_threshold_sweep,
        threshold_sweep,
    )
    from app.agent.v25_experiment import _serialize_evidence

    agent = CurriculumQAAgent()
    verifier = agent.verification_node.verifier
    c4u18 = bootstrap_c4u18_baseline(agent)
    fractions = load_baseline_evidence()
    (OUT / "c4u18_baseline_evidence.json").write_text(
        json.dumps(_serialize_evidence(c4u18), indent=2, default=str)
    )
    c4_hash = evidence_snapshot_hash(c4u18)
    fr_hash = evidence_snapshot_hash(fractions)

    control_rows: list[dict[str, Any]] = []
    for fixture_class in FIXTURES:
        for i in range(1, args.runs + 1):
            tag = f"{fixture_class.lower()}_{i:02d}"
            print(f"=== {fixture_class} {i}/{args.runs} ===", flush=True)
            row = with_retry(
                lambda fixture_class=fixture_class, tag=tag: replay_fixture(
                    fixture_class=fixture_class,  # type: ignore[arg-type]
                    c4u18_baseline=c4u18,
                    fractions_baseline=fractions,
                    verifier=verifier,
                    threshold=ANALYTICAL_THRESHOLD,
                    request_id=tag,
                )
            )
            row["tag"] = tag
            row["run_index"] = i
            control_rows.append(row)
            (OUT / f"{tag}_control.json").write_text(json.dumps(row, indent=2, default=str))
            print(
                json.dumps(
                    {
                        "fixture": fixture_class,
                        "score": row["verifier_score"],
                        "verifier": row["verifier_decision"],
                        "mapped": row["mapped_recommendation"],
                    }
                ),
                flush=True,
            )
            time.sleep(0.15)

    sweep = [summarize_threshold_sweep(control_rows, threshold=t) for t in threshold_sweep()]
    safety = compute_safety(control_rows, ANALYTICAL_THRESHOLD)
    control_summary = summarize_control(control_rows)
    mapped_summary = next(r for r in sweep if r["threshold"] == ANALYTICAL_THRESHOLD)
    conclusion, note, v29 = interpret_v28(
        control_summary, sweep, safety, analytical_threshold=ANALYTICAL_THRESHOLD
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "An independent post-verifier recommendation-mapping layer can safely translate "
            "verifier results into operational actions without modifying verifier scoring."
        ),
        "c4u18_evidence_hash": c4_hash,
        "fractions_evidence_hash": fr_hash,
        "fixture_classes": list(FIXTURES.keys()),
        "total_evaluations": len(control_rows),
        "analytical_threshold": ANALYTICAL_THRESHOLD,
        "control_summary": control_summary,
        "mapped_summary": mapped_summary,
        "threshold_sweep": sweep,
        "control_runs": control_rows,
        "safety": safety,
        "placeholder_diagnostics": build_placeholder_diagnostics(control_rows),
        "v27_comparison": V27_BASELINE,
        "conclusion": conclusion,
        "interpretation_note": note,
        "v29_recommendation": v29,
        "production_unchanged": {
            "verifier": True,
            "generator": True,
            "retrieval": True,
            "routing": True,
            "resolver": True,
            "database": True,
            "semantic_search": True,
            "v2_api": True,
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    (OUT / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2))
    write_doc(report)
    print("--- V2.8 EXPERIMENT COMPLETE ---")
    print(json.dumps({"conclusion": conclusion, "sweep": sweep}, indent=2))


if __name__ == "__main__":
    main()
