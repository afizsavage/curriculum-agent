#!/usr/bin/env python3
"""V2.7 verifier decision-boundary experiment (frozen fixtures + threshold sweep)."""

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

GOLDEN = "What are the learning objectives for fractions in Primary 4?"
OUT = ROOT / "data" / "diagnostics" / "v27_verifier_decision_boundary"
DOC = ROOT / "docs/V2_7_VERIFIER_DECISION_BOUNDARY.md"
OUT_JSON = ROOT / "data" / "diagnostics" / "v27_verifier_decision_boundary.json"
EXPECTED_HASH = "977b259fcfb4b282"
RUNS_PER_FIXTURE = 10
MAX_LLM_ATTEMPTS = 3

V26_BASELINE = {
    "faithful_imperfect_acceptance": 0.0,
    "retrieve_more_rate": 0.9,
    "avg_score": 0.825,
}


def with_retry(fn, *, attempts: int = MAX_LLM_ATTEMPTS, base_delay: float = 3.0):
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            retryable = any(
                token in str(exc).lower()
                for token in ("timeout", "timed out", "connection", "temporarily")
            )
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(base_delay * (attempt + 1))
    raise last_exc  # pragma: no cover


def summarize_control(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    scores = [r["verifier_score"] for r in rows if r.get("verifier_score") is not None]
    accepted = sum(1 for r in rows if r.get("verifier_accepted"))
    retrieve_more = sum(1 for r in rows if r.get("retrieve_more_requested"))
    insufficient = sum(1 for r in rows if r.get("insufficient_evidence"))
    fi = [r for r in rows if r.get("fixture_class") == "FAITHFUL_IMPERFECT"]
    fi_n = len(fi) or 1
    return {
        "n": len(rows),
        "verifier_acceptance_rate": round(accepted / n, 3),
        "retrieve_more_rate": round(retrieve_more / n, 3),
        "insufficient_evidence_rate": round(insufficient / n, 3),
        "avg_verifier_score": round(statistics.mean(scores), 3) if scores else None,
        "score_min": round(min(scores), 3) if scores else None,
        "score_max": round(max(scores), 3) if scores else None,
        "faithful_imperfect_acceptance_rate": round(
            sum(1 for r in fi if r.get("verifier_accepted")) / fi_n, 3
        ),
        "faithful_imperfect_retrieve_more_rate": round(
            sum(1 for r in fi if r.get("retrieve_more_requested")) / fi_n, 3
        ),
        "faithful_imperfect_false_retrieval_rate": round(
            sum(1 for r in fi if r.get("false_retrieval")) / fi_n, 3
        ),
        "overall_false_retrieval_rate": round(
            sum(1 for r in rows if r.get("false_retrieval")) / n, 3
        ),
    }


def summarize_experimental(
    control_rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    from app.agent.v27_experiment import apply_experimental_decision_boundary

    experimental = [
        apply_experimental_decision_boundary(
            row, fixture_class=row["fixture_class"], threshold=threshold
        )
        for row in control_rows
    ]
    n = len(experimental) or 1
    scores = [r["verifier_score"] for r in experimental if r.get("verifier_score") is not None]
    accepted = sum(1 for r in experimental if r.get("experimental_accepted"))
    retrieve_more = sum(
        1
        for r in experimental
        if not r.get("experimental_accepted")
        and r.get("verifier_decision") == "retrieve_more"
    )
    insufficient = sum(1 for r in experimental if r.get("insufficient_evidence"))
    fi = [r for r in experimental if r.get("fixture_class") == "FAITHFUL_IMPERFECT"]
    fi_n = len(fi) or 1
    return {
        "threshold": threshold,
        "n": len(experimental),
        "experimental_acceptance_rate": round(accepted / n, 3),
        "retrieve_more_rate": round(retrieve_more / n, 3),
        "insufficient_evidence_rate": round(insufficient / n, 3),
        "avg_verifier_score": round(statistics.mean(scores), 3) if scores else None,
        "faithful_imperfect_acceptance_rate": round(
            sum(1 for r in fi if r.get("experimental_accepted")) / fi_n, 3
        ),
        "faithful_imperfect_false_retrieval_rate": round(
            sum(1 for r in fi if r.get("false_retrieval")) / fi_n, 3
        ),
        "overall_false_retrieval_rate": round(
            sum(1 for r in experimental if r.get("false_retrieval")) / n, 3
        ),
    }


def write_doc(report: dict[str, Any]) -> None:
    c = report["control_summary"]
    e = report["experimental_summary"]
    sweep = report["threshold_sweep"]
    lines = [
        "# V2.7 Verifier Decision-Boundary Experiment",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Executive Summary",
        "",
        f"**Conclusion: {report['conclusion']}**",
        "",
        report.get("interpretation_note", ""),
        "",
        "## Hypothesis",
        "",
        report.get("hypothesis", ""),
        "",
        "## Experimental Design",
        "",
        "- Frozen deterministic answer fixtures (no stochastic generation)",
        f"- {RUNS_PER_FIXTURE} verifier evaluations per fixture class",
        f"- Baseline evidence hash: `{EXPECTED_HASH}`",
        "- Arm A: existing verifier decision (control)",
        "- Arm B: harness-only post-verifier decision boundary + threshold sweep",
        "- Production verifier unchanged; flag defaults OFF",
        "",
        "## Control (Arm A)",
        "",
        f"- Acceptance: {c['verifier_acceptance_rate']}",
        f"- Retrieve-more: {c['retrieve_more_rate']}",
        f"- Insufficient evidence: {c['insufficient_evidence_rate']}",
        f"- Avg score: {c['avg_verifier_score']} (min {c['score_min']}, max {c['score_max']})",
        "",
        "## Experimental Decision Boundary (Arm B)",
        "",
        f"- Analytical threshold: {report.get('analytical_threshold')}",
        f"- Acceptance: {e['experimental_acceptance_rate']}",
        f"- Retrieve-more (post-policy residual): {e['retrieve_more_rate']}",
        f"- Insufficient evidence: {e['insufficient_evidence_rate']}",
        f"- Avg score (unchanged): {e['avg_verifier_score']}",
        "",
        "## Threshold Sweep",
        "",
        "| Threshold | Faithful Imperfect Accept | False Retrieval | Unsupported Rejected | Reconstruction Rejected |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sweep:
        lines.append(
            f"| {row['threshold']} | {row['faithful_imperfect_acceptance']} | "
            f"{row['faithful_imperfect_false_retrieval_rate']} | {row['unsupported_rejection']} | "
            f"{row['reconstruction_rejection']} |"
        )
    lines.extend(
        [
            "",
            "## Critical Metrics",
            "",
            f"- Faithful imperfect acceptance (control): {c['faithful_imperfect_acceptance_rate']}",
            f"- Faithful imperfect acceptance (experimental @ {report.get('analytical_threshold')}): "
            f"{e['faithful_imperfect_acceptance_rate']}",
            f"- Faithful imperfect false retrieval (control): {c['faithful_imperfect_false_retrieval_rate']}",
            f"- Faithful imperfect false retrieval (experimental): "
            f"{e['faithful_imperfect_false_retrieval_rate']}",
            "",
            "## Grounding Safety (Arm B @ analytical threshold)",
            "",
        ]
    )
    for case, ok in (report.get("safety") or {}).items():
        lines.append(f"- **{case}**: rejected={ok}")
    lines.extend(
        [
            "",
            "## Comparison with V2.6",
            "",
            f"- V2.6 faithful imperfect acceptance: {V26_BASELINE['faithful_imperfect_acceptance']}",
            f"- V2.6 retrieve_more: {V26_BASELINE['retrieve_more_rate']}",
            f"- V2.6 avg score: {V26_BASELINE['avg_score']}",
            f"- V2.7 faithful imperfect acceptance (experimental): {e['faithful_imperfect_acceptance_rate']}",
            f"- V2.7 retrieve-more residual (experimental): {e['retrieve_more_rate']}",
            "",
            "## Next Recommendation",
            "",
            report.get("next_recommendation", ""),
        ]
    )
    DOC.write_text("\n".join(lines))


def build_report(
    control_rows: list[dict[str, Any]],
    sweep: list[dict[str, Any]],
    safety: dict[str, bool],
    *,
    analytical_threshold: float,
) -> dict[str, Any]:
    from app.agent.v27_experiment import interpret_experiment

    control_summary = summarize_control(control_rows)
    experimental_summary = summarize_experimental(
        control_rows, threshold=analytical_threshold
    )
    conclusion, note, best_threshold = interpret_experiment(
        control_summary, sweep, safety, analytical_threshold=analytical_threshold
    )
    next_rec = {
        "SUPPORTED": "Design a production follow-up on verifier recommendation mapping (not prompt relabeling).",
        "PARTIALLY SUPPORTED": "Refine decision-boundary guards and re-test with larger sample.",
        "NOT SUPPORTED": "Do not change production verifier decision logic; investigate scoring vs routing mismatch.",
        "INCONCLUSIVE": "Increase runs or reduce verifier variance before selecting any boundary.",
    }[conclusion]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "When evidence is present and answers faithfully report imperfect source text, "
            "a post-verifier decision boundary can accept instead of retrieve_more without "
            "weakening grounding safety."
        ),
        "golden_question": GOLDEN,
        "baseline_evidence_hash": EXPECTED_HASH,
        "authoritative_los": 10,
        "imperfect_los": ["C4U06-LO02", "C4U04-LO04"],
        "configuration": {
            "runs_per_fixture": RUNS_PER_FIXTURE,
            "fixture_classes": list(
                __import__("app.agent.v27_experiment", fromlist=["ANSWER_FIXTURES"]).ANSWER_FIXTURES
            ),
            "threshold_sweep": list(
                __import__("app.agent.v27_experiment", fromlist=["threshold_sweep"]).threshold_sweep()
            ),
        },
        "control_summary": control_summary,
        "experimental_summary": experimental_summary,
        "analytical_threshold": analytical_threshold,
        "threshold_sweep": sweep,
        "control_runs": control_rows,
        "safety": safety,
        "v26_baseline": V26_BASELINE,
        "conclusion": conclusion,
        "interpretation_note": note,
        "best_threshold_analysis": best_threshold,
        "next_recommendation": next_rec,
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


def run_safety_checks(
    control_rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, bool]:
    from app.agent.v27_experiment import apply_experimental_decision_boundary

    mapping = {
        "unsupported": "UNSUPPORTED",
        "absence": "UNSUPPORTED_ABSENCE",
        "reconstruction": "RECONSTRUCTED_IMPERFECT",
        "missing_evidence": "MISSING_EVIDENCE",
        "speculative": "UNSUPPORTED_ABSENCE",
    }
    out: dict[str, bool] = {}
    for key, fixture in mapping.items():
        subset = [r for r in control_rows if r["fixture_class"] == fixture]
        if not subset:
            out[key] = True
            continue
        rejected = all(
            not apply_experimental_decision_boundary(
                row, fixture_class=row["fixture_class"], threshold=threshold
            )["experimental_accepted"]
            for row in subset
        )
        out[f"{key}_claims_rejected" if key != "missing_evidence" else "missing_evidence_rejected"] = (
            rejected
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=RUNS_PER_FIXTURE)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.replay:
        data = json.loads(OUT_JSON.read_text())
        write_doc(data)
        print(json.dumps({"conclusion": data["conclusion"]}, indent=2))
        return

    from app.agent.orchestrator import CurriculumQAAgent
    from app.agent.v27_experiment import (
        ANSWER_FIXTURES,
        apply_experimental_decision_boundary,
        load_baseline_evidence,
        replay_verifier_control,
        summarize_threshold_sweep,
        threshold_sweep,
    )

    agent = CurriculumQAAgent()
    verifier = agent.verification_node.verifier
    baseline = load_baseline_evidence()
    from app.agent.evidence_snapshot import evidence_snapshot_hash

    if evidence_snapshot_hash(baseline) != EXPECTED_HASH:
        raise SystemExit(f"Baseline hash mismatch: expected {EXPECTED_HASH}")

    control_rows: list[dict[str, Any]] = []
    for fixture_class, spec in ANSWER_FIXTURES.items():
        answer = spec["answer"]
        for i in range(1, args.runs + 1):
            tag = f"{fixture_class.lower()}_{i:02d}"
            print(f"=== {fixture_class} run {i}/{args.runs} ===", flush=True)
            row = with_retry(
                lambda fixture_class=fixture_class, answer=answer, tag=tag: replay_verifier_control(
                    question=GOLDEN,
                    answer=answer,
                    baseline_evidence=baseline,
                    fixture_class=fixture_class,  # type: ignore[arg-type]
                    verifier=verifier,
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
                        "run": i,
                        "score": row["verifier_score"],
                        "decision": row["verifier_decision"],
                        "false_retrieval": row["false_retrieval"],
                    }
                ),
                flush=True,
            )
            time.sleep(0.2)

    sweep = [summarize_threshold_sweep(control_rows, threshold=t) for t in threshold_sweep()]
    analytical_threshold = 0.85
    safety = run_safety_checks(control_rows, threshold=analytical_threshold)
    report = build_report(
        control_rows,
        sweep,
        safety,
        analytical_threshold=analytical_threshold,
    )
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    (OUT / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2))
    write_doc(report)

    c = report["control_summary"]
    e = report["experimental_summary"]
    print("--- V2.7 EXPERIMENT COMPLETE ---")
    print(json.dumps({"conclusion": report["conclusion"], "sweep": sweep}, indent=2))


if __name__ == "__main__":
    main()
