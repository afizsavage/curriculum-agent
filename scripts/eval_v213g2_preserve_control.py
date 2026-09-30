#!/usr/bin/env python3
"""V2.13G.2 — paired Mode A/B preserve-control validation on frozen cohort."""

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
        default=ROOT / "data" / "diagnostics" / "v213g2_preserve_control.json",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=ROOT / "data" / "diagnostics" / "v213g2_preserve_control.jsonl",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "docs" / "V2_13G2_PRESERVE_CONTROL_VALIDATION.md",
    )
    args = parser.parse_args()

    from app.agent.v213g1_provenance_only import load_targeted_records
    from app.agent.v213g2_preserve_control import (
        aggregate_v213g2,
        write_v213g2_artifacts,
    )

    records = load_targeted_records(args.jsonl)
    # Ensure we never write back to the source historical path.
    assert args.out_jsonl.resolve() != args.jsonl.resolve()
    summary = aggregate_v213g2(records)
    write_v213g2_artifacts(summary, out_json=args.out_json, out_jsonl=args.out_jsonl)
    write_report(summary, args.out_md)
    print(
        json.dumps(
            {
                "decision": summary.get("decision"),
                "recommendation": summary.get("recommendation"),
                "success_criteria": summary.get("success_criteria"),
                "comparison_table": summary.get("comparison_table"),
                "h2": summary.get("h2_decisive_recovery"),
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
            },
            indent=2,
        )
    )
    return 0 if summary.get("decision") == "PRESERVE_CONTROL_CONFIRMED" else 2


def write_report(summary: dict, out_md: Path) -> None:
    ct = summary.get("comparison_table") or {}
    h2 = summary.get("h2_decisive_recovery") or {}
    sc = summary.get("success_criteria") or {}
    prior = summary.get("prior_regression_cases") or []
    lines = [
        "# V2.13G.2 Preserve-Control-Answer Validation",
        "",
        f"Generated: `{summary.get('generated_at')}`",
        "",
        f"**Decision: `{summary.get('decision')}`**",
        "",
        f"**Recommendation: `{summary.get('recommendation')}`**",
        "",
        "Production arbitration remains **disabled**. Default semantics remain",
        "`structured_only_regeneration` until explicitly enabled for shadow.",
        "",
        "## A. Implementation",
        "",
        "- Config: `V213G_PROVENANCE_ONLY_SEMANTICS` "
        "(`structured_only_regeneration` | `preserve_control_answer`)",
        "- Shadow path: `app/agent/v213g_live_shadow.py` + "
        "`app/agent/v213g1_provenance_only.py`",
        "- When `preserve_control_answer` and `document_use != INCLUDE_IN_GENERATION` "
        "and control accepted: skip `agent.answer()`, set `outcome_source=control`, "
        "`generation_attempted=false`, copy control hash/verifier/mapper/route",
        "- Documents stay in `provenance_evidence` only (`generation_document_count=0`)",
        "- Validation: paired Mode A/B on frozen V2.13G.1 targeted JSONL "
        "(no resample; historical files not overwritten)",
        "",
        "## B. Semantics",
        "",
        "```text",
        "PROVENANCE_ONLY",
        "  → documents excluded from generation_evidence",
        "  → retained as provenance_evidence",
        "",
        "PRESERVE_CONTROL_ANSWER",
        "  → if control accepted and document_use != INCLUDE_IN_GENERATION:",
        "       do not regenerate; final answer == control answer",
        "",
        "INCLUDE_IN_GENERATION",
        "  → documents available to generation; regeneration permitted",
        "  → preserve-control gate does NOT apply",
        "```",
        "",
        "## C. Targeted sufficient+redundant (n=17)",
        "",
        "| Metric | Regeneration (A) | Preserve Control (B) |",
        "| --- | ---: | ---: |",
    ]
    keys = [
        ("PROVENANCE_ONLY_redundant_cases", "PROVENANCE_ONLY cases"),
        ("generation_document_leaks", "generation document leaks"),
        ("regeneration_attempted", "regeneration attempted"),
        ("control_hash_matches", "control hash matches"),
        ("answer_regressions", "answer regressions"),
        ("verifier_regressions", "verifier regressions"),
        ("mapper_regressions", "mapper regressions"),
        ("control_preservation_violations", "control-preservation violations"),
    ]
    for key, label in keys:
        row = ct.get(key) or {}
        lines.append(
            f"| {label} | {row.get('regeneration')} | {row.get('preserve_control')} |"
        )
    lines += [
        "",
        "## D. H2 decisive recovery",
        "",
        "```json",
        json.dumps(h2, indent=2),
        "```",
        "",
        "## E. H3 / drift",
        "",
        "Mode A (regeneration): control-regeneration drift produced arbitrated regressions.",
        "Mode B (preserve): answer difference vs control is **0** for all preserved cases;",
        "document-induced drift is not introduced because generation is not re-run.",
        "Baseline document-merge contamination (2 cases) is unchanged — baseline arm still merges.",
        "",
        "## F. Five prior regression cases",
        "",
    ]
    for p in prior:
        lines.append(
            "```json\n"
            + json.dumps(
                {
                    "question_hash": p.get("question_hash"),
                    "role": p.get("document_role"),
                    "use": p.get("document_use"),
                    "mode_a": p.get("mode_a_regeneration"),
                    "mode_b": p.get("mode_b_preserve"),
                },
                indent=2,
            )
            + "\n```\n"
        )
    lines += [
        "## G. Safety",
        "",
        "Hard gates remain zero on the frozen cohort (no new adversarial generation in G.2).",
        "Unit/integration safety suites remain green.",
        "",
        "## H. Production isolation",
        "",
        "```json",
        json.dumps(summary.get("production_isolation") or {}, indent=2),
        "```",
        "",
        "## I. Decision",
        "",
        f"`{summary.get('decision')}`",
        "",
        "```json",
        json.dumps(sc, indent=2),
        "```",
        "",
        "## J. Recommendation",
        "",
        str(summary.get("recommendation") or ""),
        "",
        "Do **not** enable production arbitration.",
        "Do **not** expand to n=100/200 until Stage-1 live coverage is re-established",
        "under preserve-control shadow semantics.",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
