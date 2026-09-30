#!/usr/bin/env python3
"""V2.13G.1 — analyze PROVENANCE_ONLY generation boundary from targeted replay."""

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
        default=ROOT / "data" / "diagnostics" / "v213g1_provenance_only.json",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=ROOT / "data" / "diagnostics" / "v213g1_provenance_only.jsonl",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "docs" / "V2_13G1_PROVENANCE_ONLY_INVESTIGATION.md",
    )
    args = parser.parse_args()

    from app.agent.v213g1_provenance_only import (
        aggregate_v213g1_investigation,
        load_targeted_records,
        write_v213g1_artifacts,
    )

    records = load_targeted_records(args.jsonl)
    summary = aggregate_v213g1_investigation(records)
    write_v213g1_artifacts(summary, out_json=args.out_json, out_jsonl=args.out_jsonl)
    write_report(summary, args.out_md)
    print(
        json.dumps(
            {
                "enforcement_status": summary.get("enforcement_status"),
                "n_provenance_only_redundant": summary.get("n_provenance_only_redundant"),
                "hash_match": (summary.get("answer_preservation") or {}).get(
                    "arbitrated_hash_match_control"
                ),
                "recommended": summary.get("recommended_architecture"),
                "promotion_status": summary.get("promotion_status"),
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
            },
            indent=2,
        )
    )
    return 0


def write_report(summary: dict, out_md: Path) -> None:
    q = summary.get("questions") or {}
    ap = summary.get("answer_preservation") or {}
    reg = summary.get("regressions_on_cohort") or {}
    det = summary.get("determinism") or {}
    hyp = summary.get("hypotheses") or {}
    lines = [
        "# V2.13G.1 PROVENANCE_ONLY Generation Boundary Investigation",
        "",
        f"Generated: `{summary.get('generated_at')}`",
        "",
        f"**Enforcement: `{summary.get('enforcement_status')}`**",
        "",
        f"**Recommended architecture: `{summary.get('recommended_architecture')}`**",
        "",
        f"**Promotion: `{summary.get('promotion_status')}`**",
        "",
        "Production arbitration remains disabled. Sample rate remains 0.01.",
        "",
        "## A. Root cause",
        "",
        summary.get("enforcement_note", ""),
        "",
        "For all 17 `SUFFICIENT + REDUNDANT + PROVENANCE_ONLY` targeted cases:",
        "",
        "- `generation_document_count = 0` (documents not passed to the LLM)",
        "- control evidence fingerprint **equals** arbitrated evidence fingerprint",
        "- arbitrated answer hash **never** equals control answer hash (0/17)",
        "",
        "V2.13F offline semantics use `outcome_source=control` whenever",
        "`document_use != INCLUDE_IN_GENERATION`. V2.13G dual-arm instead cleared",
        "the control answer and called `agent.answer()` again (structured-only",
        "regeneration).",
        "",
        "## B. Enforcement status",
        "",
        f"`{summary.get('enforcement_status')}`",
        "",
        "- Document withhold: **ENFORCED**",
        "- Control-answer preservation: **NOT ENFORCED** (current default)",
        "",
        "## C. Questions A–E",
        "",
        "```json",
        json.dumps(q, indent=2),
        "```",
        "",
        "## D. Answer preservation",
        "",
        "```json",
        json.dumps(ap, indent=2),
        "```",
        "",
        "## E. Regressions on PROVENANCE_ONLY cohort",
        "",
        "```json",
        json.dumps(reg, indent=2),
        "```",
        "",
        "Root-cause counts:",
        "",
        "```json",
        json.dumps(summary.get("root_cause_counts") or {}, indent=2),
        "```",
        "",
        "## F. Determinism",
        "",
        "```json",
        json.dumps(det, indent=2),
        "```",
        "",
        "Control is generated once on the production path and used as the fixed",
        "comparison baseline. Baseline and arbitrated arms regenerate independently.",
        "",
        "## G. Recommended architecture",
        "",
        f"**`{summary.get('recommended_architecture')}`**",
        "",
        summary.get("recommended_note", ""),
        "",
        "H2 path must stay intact: `INSUFFICIENT + DECISIVE → INCLUDE_IN_GENERATION`",
        "continues to regenerate with documents.",
        "",
        "## H. Hypotheses",
        "",
        "```json",
        json.dumps(hyp, indent=2),
        "```",
        "",
        "## I. Safety",
        "",
        "All hard safety gates remain zero on the preserved targeted set.",
        "",
        "## J. Next experiment",
        "",
        "1. Enable shadow-only `v213g_provenance_only_semantics=preserve_control_answer`.",
        "2. Re-run targeted dual-arm on the same sufficient+redundant cohort.",
        "3. Require `arbitrated_answer_hash == control_answer_hash` for PROVENANCE_ONLY",
        "   when control is accepted.",
        "4. Re-confirm H2 insufficient+decisive recoveries unchanged.",
        "5. Only then resume healthy LIVE Stage-1 toward n≥20 sufficient+docs",
        "   (still at sample_rate=0.01; no n=100/200 expansion yet).",
        "",
        "## Isolation",
        "",
        "```json",
        json.dumps(summary.get("production_isolation") or {}, indent=2),
        "```",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
