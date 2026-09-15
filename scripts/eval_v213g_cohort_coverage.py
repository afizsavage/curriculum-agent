#!/usr/bin/env python3
"""V2.13G cohort-coverage investigation (does not overwrite live G summary JSONL)."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def curriculum_reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def write_markdown(report: dict, path: Path) -> None:
    live = report.get("live_traffic") or {}
    root = report.get("root_cause") or {}
    replay = report.get("targeted_replay") or {}
    hyp = report.get("hypotheses") or {}
    dd = replay.get("dd1c57") or {}
    lines = [
        "# V2.13G Live Document Arbitration Shadow",
        "",
        f"Generated: `{report.get('generated_at')}`",
        "",
        f"**Overall status: `{hyp.get('overall_status')}`**",
        "",
        f"**Recommendation: `{hyp.get('recommendation')}`**",
        "",
        f"**Classification determination: `{root.get('determination')}`** "
        f"(`{root.get('classification_validity')}`)",
        "",
        root.get("detail", ""),
        "",
        "## Hard isolation (unchanged)",
        "",
        "```text",
        json.dumps(report.get("config") or {}, indent=2),
        "```",
        "",
        "## A. Root cause of 52/52 INSUFFICIENT + DECISIVE",
        "",
        "```json",
        json.dumps(
            {
                "live_n": live.get("n"),
                "control_accepted": live.get("control_accepted"),
                "structured_count_hist": live.get("structured_count_hist"),
                "sufficiency": live.get("sufficiency"),
                "document_role": live.get("document_role"),
                "document_use": live.get("document_use"),
                "categories": live.get("categories"),
                "curriculum_api_reachable_at_audit": root.get("curriculum_api_reachable"),
            },
            indent=2,
        ),
        "```",
        "",
        "Every live dual-arm row had `structured_count=0` and `control.final_accepted=false`.",
        "With documents retrieved (n=5), the existing V2.13F rules correctly emit",
        "`INSUFFICIENT → DECISIVE → INCLUDE_IN_GENERATION`.",
        "",
        "This is **not** evidence that arbitration is ineffective. The primary V2.13F",
        "regression cohort (`structured_sufficient_with_docs`) had live coverage **0**.",
        "",
        f"**Primary regression hypothesis (live): `{hyp.get('primary_regression_hypothesis_live')}`**",
        "",
        "## B. Classification validity",
        "",
        f"`{root.get('classification_validity')}` — {root.get('determination')}",
        "",
        "Checked: no silent default to INSUFFICIENT when control is accepted;",
        "sufficiency is decided from control acceptance / structured count **before**",
        "document role mapping; live control snapshots match zero structured evidence",
        "(not shadow evidence-loss).",
        "",
        "## C. V2.13F replay compatibility (TARGETED_REPLAY)",
        "",
        "Frozen V2.13F replay cases were passed through the current classifier",
        "without modifying the freeze.",
        "",
        "```json",
        json.dumps(
            {
                "n": replay.get("n"),
                "classifier_distribution": replay.get("classifier_distribution"),
                "group_compatibility": replay.get("group_compatibility"),
                "cohort_counts": replay.get("cohort_counts"),
            },
            indent=2,
        ),
        "```",
        "",
        "### dd1c57 permanent fixture",
        "",
        "```json",
        json.dumps(dd, indent=2),
        "```",
        "",
        "## D. Cohort coverage",
        "",
        "### LIVE_TRAFFIC",
        "",
        "```json",
        json.dumps(report.get("live_cohort_counts") or {}, indent=2),
        "```",
        "",
        "### TARGETED_REPLAY (from frozen V2.13F cases)",
        "",
        "```json",
        json.dumps((replay.get("cohort_counts") or {}), indent=2),
        "```",
        "",
        "## E. Hypothesis results",
        "",
        "```json",
        json.dumps(
            {
                "H1": hyp.get("H1_regression_prevention"),
                "H2": hyp.get("H2_recovery_preservation"),
                "H3": hyp.get("H3_generator_drift_reduction"),
                "safety": hyp.get("safety"),
            },
            indent=2,
        ),
        "```",
        "",
        "## F. Safety",
        "",
        "```json",
        json.dumps(report.get("safety") or {}, indent=2),
        "```",
        "",
        "## G. Recommendation",
        "",
        f"`{hyp.get('recommendation')}`",
        "",
        "Do **not** promote arbitration to production.",
        "Do **not** raise `sample_rate` above 0.01.",
        "Do **not** overwrite V2.13F freeze/replay artifacts.",
        "",
        "## H. Next experiment",
        "",
        "1. Keep Curriculum Structure API available so live traffic can resolve structured evidence.",
        "2. Continue V2.13G dual-arm shadow at sample_rate=0.01 until live",
        "   `structured_sufficient_with_docs >= 20` (and retain insufficient+decisive recovery coverage).",
        "3. Only then expand toward n=100 / n=200 dual-arm comparisons.",
        "4. Keep TARGETED_REPLAY cohort diagnostics additive alongside LIVE_TRAFFIC.",
        "",
        "## Live comparison table (unchanged n=52 snapshot)",
        "",
        "```json",
        json.dumps(report.get("live_comparison_table") or {}, indent=2),
        "```",
        "",
        "## Latency (live)",
        "",
        "```json",
        json.dumps(report.get("live_latency") or {}, indent=2),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live-jsonl",
        type=Path,
        default=ROOT / "data" / "diagnostics" / "v213g_arbitration_shadow.jsonl",
    )
    parser.add_argument(
        "--f-replay",
        type=Path,
        default=ROOT / "data" / "diagnostics" / "v213f_replay" / "dataset.json",
    )
    parser.add_argument(
        "--live-summary",
        type=Path,
        default=ROOT / "data" / "diagnostics" / "v213g_arbitration_summary.json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "data" / "diagnostics" / "v213g_cohort_coverage.json",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=ROOT / "data" / "diagnostics" / "v213g_cohort_coverage.jsonl",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "docs" / "V2_13G_LIVE_ARBITRATION_SHADOW.md",
    )
    args = parser.parse_args()

    from app.agent.v213g_cohort_coverage import (
        analyze_live_g_rows,
        determine_root_cause,
        enrich_cohort_counts_from_live,
        hypothesis_results,
        load_jsonl,
        replay_v213f_through_classifier,
    )
    from app.config import Settings

    settings = Settings()
    api_ok = curriculum_reachable(settings.resolved_curriculum_api_url())
    live_rows = load_jsonl(args.live_jsonl)
    live = analyze_live_g_rows(live_rows)
    root = determine_root_cause(live, curriculum_api_reachable=api_ok)

    f_cases: list[dict] = []
    if args.f_replay.is_file():
        f_cases = list(json.loads(args.f_replay.read_text()).get("cases") or [])
    replay = replay_v213f_through_classifier(f_cases)

    live_summary = {}
    if args.live_summary.is_file() and args.live_summary.stat().st_size > 0:
        live_summary = json.loads(args.live_summary.read_text())

    live_cohort = enrich_cohort_counts_from_live(live_rows)
    safety = (live_summary.get("safety") or {}) if live_summary else {
        "baseline": {},
        "arbitrated": {},
        "safety_blocked": False,
    }
    reg = live_summary.get("regression") or {}
    rec = live_summary.get("recovery") or {}
    drift = live_summary.get("generator_drift") or {}
    hyp = hypothesis_results(
        live_sufficient_with_docs=live_cohort["structured_sufficient_with_docs"],
        live_baseline_reg=int(reg.get("baseline_regression") or 0),
        live_arb_reg=int(reg.get("arbitrated_regression") or 0),
        live_baseline_rec=int(rec.get("baseline_recovery") or 0),
        live_arb_rec=int(rec.get("arbitrated_recovery") or 0),
        live_baseline_unsup=int(drift.get("baseline_unsupported_claims") or 0),
        live_arb_unsup=int(drift.get("arbitrated_unsupported_claims") or 0),
        targeted_sufficient_with_docs=int(
            (replay.get("cohort_counts") or {}).get("structured_sufficient_with_docs") or 0
        ),
        targeted_dd1c57_pass=(replay.get("dd1c57") or {}).get("pass"),
        safety_blocked=bool(safety.get("safety_blocked")),
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "v2.13g_cohort_coverage",
        "config": {
            "v213d_shadow_enabled": settings.v213d_shadow_enabled,
            "v213d_shadow_sample_rate": settings.v213d_shadow_sample_rate,
            "v213d_shadow_document_retrieval": settings.v213d_shadow_document_retrieval,
            "v213d_shadow_retrieval_variant": settings.v213d_shadow_retrieval_variant,
            "v213f_document_arbitration_experiment": settings.v213f_document_arbitration_experiment,
            "v213f_arbitration_policy": settings.v213f_arbitration_policy,
            "v213g_live_arbitration_shadow": settings.v213g_live_arbitration_shadow,
            "v213e": "disabled",
            "curriculum_api_url": settings.resolved_curriculum_api_url(),
        },
        "live_traffic": live,
        "live_cohort_counts": live_cohort,
        "live_comparison_table": live_summary.get("comparison_table"),
        "live_latency": live_summary.get("latency"),
        "safety": safety,
        "root_cause": root,
        "targeted_replay": {
            k: v for k, v in replay.items() if k != "rows"
        },
        "hypotheses": hyp,
        "preserved_artifacts": {
            "v213g_arbitration_shadow_jsonl": str(args.live_jsonl),
            "v213g_arbitration_summary_json": str(args.live_summary),
            "v213f_replay_dataset": str(args.f_replay),
            "note": "Live G JSONL/summary not overwritten by this investigation.",
        },
        "next_experiment": {
            "keep_sample_rate": 0.01,
            "require_live_structured_sufficient_with_docs": 20,
            "then_continue_toward": [100, 200],
            "ensure_curriculum_api": True,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in replay.get("rows") or []:
            handle.write(json.dumps(row) + "\n")
        # Also emit one LIVE_TRAFFIC audit marker per live row (classification only).
        for r in live_rows:
            arb = r.get("arbitration") or {}
            handle.write(
                json.dumps(
                    {
                        "source": "LIVE_TRAFFIC",
                        "request_id": r.get("request_id"),
                        "question_hash": (r.get("question") or {}).get("hash"),
                        "category": (r.get("question") or {}).get("category"),
                        "control_final_accepted": bool(
                            (r.get("control") or {}).get("final_accepted")
                        ),
                        "structured_count": arb.get("structured_count"),
                        "document_count": arb.get("document_count"),
                        "decision": {
                            "structured_sufficiency": arb.get("structured_sufficiency"),
                            "document_role": arb.get("document_role"),
                            "document_use": arb.get("document_use"),
                            "reasons": arb.get("reasons"),
                        },
                    }
                )
                + "\n"
            )

    write_markdown(report, args.out_md)
    print(
        json.dumps(
            {
                "determination": root.get("determination"),
                "classification_validity": root.get("classification_validity"),
                "overall_status": hyp.get("overall_status"),
                "recommendation": hyp.get("recommendation"),
                "live_sufficient_with_docs": live_cohort[
                    "structured_sufficient_with_docs"
                ],
                "targeted_sufficient_with_docs": (replay.get("cohort_counts") or {}).get(
                    "structured_sufficient_with_docs"
                ),
                "dd1c57_pass": (replay.get("dd1c57") or {}).get("pass"),
                "curriculum_api_reachable": api_ok,
                "out_json": str(args.out_json),
                "out_jsonl": str(args.out_jsonl),
                "out_md": str(args.out_md),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
