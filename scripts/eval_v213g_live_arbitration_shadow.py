#!/usr/bin/env python3
"""Aggregate V2.13G live dual-arm arbitration shadow JSONL into summary + report."""

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
        default=ROOT / "data" / "diagnostics" / "v213g_arbitration_shadow.jsonl",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "data" / "diagnostics" / "v213g_arbitration_summary.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "docs" / "V2_13G_LIVE_ARBITRATION_SHADOW.md",
    )
    args = parser.parse_args()

    from app.agent.v213g_live_shadow import (
        aggregate_v213g_records,
        load_v213g_records,
        write_v213g_report,
    )
    from app.config import Settings

    records = load_v213g_records(args.jsonl)
    summary = aggregate_v213g_records(records)
    settings = Settings()
    summary["config"] = {
        "v213d_shadow_enabled": settings.v213d_shadow_enabled,
        "v213d_shadow_sample_rate": settings.v213d_shadow_sample_rate,
        "v213d_shadow_document_retrieval": settings.v213d_shadow_document_retrieval,
        "v213d_shadow_retrieval_variant": settings.v213d_shadow_retrieval_variant,
        "v213f_document_arbitration_experiment": settings.v213f_document_arbitration_experiment,
        "v213f_arbitration_policy": settings.v213f_arbitration_policy,
        "v213g_live_arbitration_shadow": settings.v213g_live_arbitration_shadow,
        "v213e": "disabled",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_v213g_report(summary, out_md=args.out_md)
    print(
        json.dumps(
            {
                "status": summary.get("status"),
                "n_successful": summary.get("n_successful"),
                "baseline_regressions": (summary.get("regression") or {}).get(
                    "baseline_regression"
                ),
                "arbitrated_regressions": (summary.get("regression") or {}).get(
                    "arbitrated_regression"
                ),
                "baseline_recoveries": (summary.get("recovery") or {}).get(
                    "baseline_recovery"
                ),
                "arbitrated_recoveries": (summary.get("recovery") or {}).get(
                    "arbitrated_recovery"
                ),
                "json": str(args.out_json),
                "md": str(args.out_md),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
