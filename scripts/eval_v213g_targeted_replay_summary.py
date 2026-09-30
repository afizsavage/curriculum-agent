#!/usr/bin/env python3
"""Aggregate V2.13G TARGETED_REPLAY JSONL separately from LIVE_TRAFFIC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=ROOT / "data" / "diagnostics" / "v213g_arbitration_targeted_replay.jsonl",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "data" / "diagnostics" / "v213g_targeted_replay_summary.json",
    )
    args = parser.parse_args()

    from app.agent.v213g_live_shadow import (
        aggregate_v213g_records,
        count_sufficient_with_docs,
        load_v213g_records,
    )
    from app.config import Settings

    records = load_v213g_records(args.jsonl)
    # Treat targeted batch as the only "healthy" batch for this file.
    for r in records:
        r.setdefault("batch_id", "batch_2_targeted_replay")
        r.setdefault("source", "TARGETED_REPLAY")
    summary = aggregate_v213g_records(records)
    summary["source"] = "TARGETED_REPLAY"
    summary["healthy_batch_coverage"] = count_sufficient_with_docs(records)
    settings = Settings()
    summary["config"] = {
        "v213d_shadow_enabled": settings.v213d_shadow_enabled,
        "v213d_shadow_sample_rate": settings.v213d_shadow_sample_rate,
        "v213f_document_arbitration_experiment": settings.v213f_document_arbitration_experiment,
        "v213g_live_arbitration_shadow": settings.v213g_live_arbitration_shadow,
        "v213e": "disabled",
        "note": "TARGETED_REPLAY forced dual-arm; sample_rate unchanged for live path",
    }
    # Hypothesis slice for sufficient+docs
    h1 = {
        "n": 0,
        "control_accepted": 0,
        "baseline_regression": 0,
        "arbitrated_regression": 0,
        "baseline_generator_drift": 0,
        "arbitrated_generator_drift": 0,
        "baseline_unsupported_claims": 0,
        "arbitrated_unsupported_claims": 0,
        "redundant": 0,
        "irrelevant": 0,
        "document_use": {},
    }
    from collections import Counter

    use_c: Counter[str] = Counter()
    for r in records:
        arb = r.get("arbitration") or {}
        if str(arb.get("structured_sufficiency")) != "SUFFICIENT":
            continue
        if int(arb.get("document_count") or 0) <= 0:
            continue
        if (r.get("infrastructure") or {}).get("validity") == "INFRASTRUCTURE_INVALID":
            continue
        h1["n"] += 1
        hyp = r.get("hypothesis") or {}
        h1["control_accepted"] += int(bool((r.get("control") or {}).get("final_accepted")))
        h1["baseline_regression"] += int(bool(hyp.get("control_correct_baseline_worse")))
        h1["arbitrated_regression"] += int(bool(hyp.get("control_correct_arbitrated_worse")))
        h1["baseline_generator_drift"] += int(bool(hyp.get("generator_document_drift_baseline")))
        h1["arbitrated_generator_drift"] += int(
            bool(hyp.get("generator_document_drift_arbitrated"))
        )
        h1["baseline_unsupported_claims"] += int(hyp.get("baseline_unsupported_claims") or 0)
        h1["arbitrated_unsupported_claims"] += int(
            hyp.get("arbitrated_unsupported_claims") or 0
        )
        role = str(arb.get("document_role") or "")
        if role == "REDUNDANT":
            h1["redundant"] += 1
        if role == "IRRELEVANT":
            h1["irrelevant"] += 1
        use_c[str(arb.get("document_use") or "UNKNOWN")] += 1
    h1["document_use"] = dict(use_c)
    if h1["n"] >= 20:
        if h1["arbitrated_regression"] < h1["baseline_regression"] or (
            h1["baseline_regression"] == 0 and h1["arbitrated_regression"] == 0
        ):
            h1["result"] = (
                "PARTIALLY_CONFIRMED"
                if h1["baseline_regression"] == 0
                else "CONFIRMED"
            )
        else:
            h1["result"] = "NOT_CONFIRMED"
    else:
        h1["result"] = "UNTESTED"
    summary["H1_structured_sufficient_with_docs"] = h1
    summary["never_mixed_with_live"] = True
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "source": "TARGETED_REPLAY",
                "n": summary.get("n_successful"),
                "sufficient_with_docs": h1["n"],
                "H1": h1["result"],
                "baseline_reg": h1["baseline_regression"],
                "arbitrated_reg": h1["arbitrated_regression"],
                "path": str(args.out_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
