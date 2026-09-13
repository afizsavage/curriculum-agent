#!/usr/bin/env python3
"""Evaluate V2.13F document arbitration on a frozen replay dataset (offline)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_report(summary: dict, *, out_md: Path) -> None:
    policies = summary.get("policies") or {}
    baseline = policies.get("A_BASELINE_MERGE") or {}
    arb = policies.get("C_ARBITRATED") or {}
    sf = policies.get("B_STRUCTURED_FIRST") or {}
    dd = summary.get("dd1c57") or {}
    b_safety = baseline.get("safety") or {}
    sf_safety = sf.get("safety") or {}
    a_safety = arb.get("safety") or {}

    def row(metric: str, b, s, a) -> str:
        return f"| {metric:<27} | {b:>6} | {s:>6} | {a:>6} |"

    lines = [
        "# V2.13F Document Evidence Arbitration",
        "",
        f"Generated: `{summary.get('generated_at')}`",
        "",
        "## 1. Hypothesis",
        "",
        "Document retrieval is operational, but unconditional document-conditioned",
        "regeneration can introduce answer drift when structured evidence is already",
        "sufficient. Deterministic evidence arbitration can withhold redundant documents",
        "from generation while still allowing decisive documents to recover insufficient",
        "structured answers.",
        "",
        "## 2. Frozen dataset definition",
        "",
        "```json",
        json.dumps(summary.get("dataset_meta") or {}, indent=2),
        "```",
        "",
        "## 3. V2.13D baseline",
        "",
        "Variant A (`A_BASELINE_MERGE`) always merges retrieved documents into generation",
        "(frozen V2.13D shadow outcomes).",
        "",
        "## 4. Arbitration variants",
        "",
        "- **A_BASELINE_MERGE** — current V2.13D behavior",
        "- **B_STRUCTURED_FIRST** — if control accepted, documents are provenance-only",
        "- **C_ARBITRATED** — deterministic role → use mapping (DECISIVE/SUPPORTING/...)",
        "",
        "Arbitration is non-LLM and shadow/replay-only.",
        f"`v213f_document_arbitration_experiment={summary.get('config', {}).get('v213f_document_arbitration_experiment')}`",
        "",
        "## 5. Structured-sufficiency classification",
        "",
        "```json",
        json.dumps(summary.get("sufficiency_counts") or {}, indent=2),
        "```",
        "",
        "## 6. Document-role classification",
        "",
        "```json",
        json.dumps((arb.get("role_counts") or {}), indent=2),
        "```",
        "",
        "## 7. Recovery results",
        "",
        f"- Baseline recoveries: **{baseline.get('recoveries')}**",
        f"- Structured-first recoveries: **{sf.get('recoveries')}**",
        f"- Arbitrated recoveries: **{arb.get('recoveries')}**",
        "",
        "## 8. Regression results",
        "",
        f"- Baseline regressions: **{baseline.get('regressions')}**",
        f"- Structured-first regressions: **{sf.get('regressions')}**",
        f"- Arbitrated regressions: **{arb.get('regressions')}**",
        f"- Generator-drift cases prevented (arbitrated): **{arb.get('generator_drift_prevented')}**",
        "",
        "## 9. Generator-drift results",
        "",
        "Offline counterfactual: when documents are withheld from generation,",
        "the accepted control outcome is retained (no document-conditioned regeneration).",
        "",
        "## 10. `dd1c57…` analysis",
        "",
        "```json",
        json.dumps(dd, indent=2),
        "```",
        "",
        "## 11. Safety results",
        "",
        "```json",
        json.dumps(
            {
                "baseline": b_safety,
                "structured_first": sf_safety,
                "arbitrated": a_safety,
            },
            indent=2,
        ),
        "```",
        "",
        "## 12. Latency",
        "",
        "Arbitration is deterministic metadata/score logic only (no extra LLM).",
        f"Mean arbitration overhead on replay set: `{summary.get('arbitration_mean_ms')} ms`.",
        "",
        "## 13. Per-category results",
        "",
        "```json",
        json.dumps(summary.get("by_category") or {}, indent=2),
        "```",
        "",
        "## 14. Subject/grade segmentation",
        "",
        "```json",
        json.dumps(
            {
                "by_subject": summary.get("by_subject"),
                "by_grade": summary.get("by_grade"),
            },
            indent=2,
        ),
        "```",
        "",
        "## Comparison table",
        "",
        "| Metric                      | V2.13D | B-SF   | V2.13F |",
        "| --------------------------- | -----: | -----: | -----: |",
        row(
            "Document helped",
            baseline.get("document_helped", 0),
            sf.get("document_helped", 0),
            arb.get("document_helped", 0),
        ),
        row(
            "Document neutral",
            baseline.get("document_neutral", 0),
            sf.get("document_neutral", 0),
            arb.get("document_neutral", 0),
        ),
        row(
            "Document hurt",
            baseline.get("document_hurt", 0),
            sf.get("document_hurt", 0),
            arb.get("document_hurt", 0),
        ),
        row(
            "Recoveries",
            baseline.get("recoveries", 0),
            sf.get("recoveries", 0),
            arb.get("recoveries", 0),
        ),
        row(
            "Regressions",
            baseline.get("regressions", 0),
            sf.get("regressions", 0),
            arb.get("regressions", 0),
        ),
        row(
            "Generator drift prevented",
            baseline.get("generator_drift_prevented", 0),
            sf.get("generator_drift_prevented", 0),
            arb.get("generator_drift_prevented", 0),
        ),
        row(
            "Unsupported claims",
            baseline.get("unsupported_claims", 0),
            sf.get("unsupported_claims", 0),
            arb.get("unsupported_claims", 0),
        ),
        row(
            "Wrong-context false accepts",
            b_safety.get("wrong_context_false_acceptance", 0),
            sf_safety.get("wrong_context_false_acceptance", 0),
            a_safety.get("wrong_context_false_acceptance", 0),
        ),
        row(
            "Placeholder false accepts",
            b_safety.get("placeholder_false_acceptance", 0),
            sf_safety.get("placeholder_false_acceptance", 0),
            a_safety.get("placeholder_false_acceptance", 0),
        ),
        row(
            "Metadata false accepts",
            b_safety.get("metadata_false_acceptance", 0),
            sf_safety.get("metadata_false_acceptance", 0),
            a_safety.get("metadata_false_acceptance", 0),
        ),
        "",
        "## 15. Recommendation",
        "",
        f"**Decision: `{summary.get('decision')}`**",
        "",
        summary.get("recommendation_text", ""),
        "",
        "V2.13E remains disabled. Arbitration is not enabled in production.",
        "",
        "## Production isolation",
        "",
        "```text",
        json.dumps(summary.get("config") or {}, indent=2),
        "```",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def decide(summary_policies: dict) -> tuple[str, str]:
    baseline = summary_policies.get("A_BASELINE_MERGE") or {}
    arb = summary_policies.get("C_ARBITRATED") or {}
    b_reg = int(baseline.get("regressions") or 0)
    a_reg = int(arb.get("regressions") or 0)
    b_rec = int(baseline.get("recoveries") or 0)
    a_rec = int(arb.get("recoveries") or 0)
    safety = arb.get("safety") or {}
    if any(int(safety.get(k) or 0) > 0 for k in safety):
        return (
            "ARBITRATION_NOT_SUPPORTED",
            "Safety gate non-zero under arbitration counterfactual.",
        )
    if b_reg == 0 and b_rec == 0:
        return (
            "INSUFFICIENT_EVIDENCE",
            "Frozen set lacks both regressions and recoveries.",
        )
    reg_improved = a_reg < b_reg
    rec_preserved = a_rec >= b_rec
    if reg_improved and rec_preserved and a_reg == 0:
        return (
            "ARBITRATION_SUPPORTED",
            "Regressions eliminated while recoveries preserved; safety gates remain zero. "
            "Do not enable in production yet — promote only after larger confirmation.",
        )
    if reg_improved and rec_preserved:
        return (
            "ARBITRATION_PARTIALLY_SUPPORTED",
            "Regressions decreased and recoveries preserved, but residual harm remains.",
        )
    if reg_improved and not rec_preserved:
        return (
            "ARBITRATION_NOT_SUPPORTED",
            "Regressions fell but recoveries were reduced — fails reduce-harm-without-reducing-recovery.",
        )
    return (
        "ARBITRATION_NOT_SUPPORTED",
        "No meaningful reduction in document-induced regressions.",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "diagnostics" / "v213f_replay" / "dataset.json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "data" / "diagnostics" / "v213f_arbitration_summary.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "docs" / "V2_13F_DOCUMENT_ARBITRATION.md",
    )
    args = parser.parse_args()

    from app.agent.v213f_arbitration import (
        ArbitrationPolicy,
        aggregate_policy_comparison,
        arbitrate_from_shadow_record,
    )
    from app.config import Settings
    import time

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases = list(data.get("cases") or [])
    t0 = time.perf_counter()
    comparison = aggregate_policy_comparison(cases)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # dd1c57 special case
    dd = next(
        (
            c
            for c in cases
            if str((c.get("question") or {}).get("hash") or "").startswith("dd1c57")
        ),
        None,
    )
    dd_analysis = None
    if dd is not None:
        d_base = arbitrate_from_shadow_record(dd, policy=ArbitrationPolicy.BASELINE_MERGE)
        d_sf = arbitrate_from_shadow_record(dd, policy=ArbitrationPolicy.STRUCTURED_FIRST)
        d_arb = arbitrate_from_shadow_record(dd, policy=ArbitrationPolicy.ARBITRATED)
        from app.agent.v213f_arbitration import counterfactual_outcome

        dd_analysis = {
            "question_hash": (dd.get("question") or {}).get("hash"),
            "baseline": {
                "decision": d_base.to_dict(),
                "counterfactual": counterfactual_outcome(dd, d_base),
            },
            "structured_first": {
                "decision": d_sf.to_dict(),
                "counterfactual": counterfactual_outcome(dd, d_sf),
            },
            "arbitrated": {
                "decision": d_arb.to_dict(),
                "counterfactual": counterfactual_outcome(dd, d_arb),
            },
            "interpretation": (
                "If structured_sufficiency=SUFFICIENT and document_use=PROVENANCE_ONLY, "
                "generation stays structured-first and the accepted control outcome is retained."
            ),
        }

    settings = Settings()
    by_category: dict[str, dict[str, int]] = {}
    by_subject: dict[str, dict[str, int]] = {}
    by_grade: dict[str, dict[str, int]] = {}
    sufficiency_counts: dict[str, int] = {}
    for case in cases:
        q = case.get("question") or {}
        dec = arbitrate_from_shadow_record(case, policy=ArbitrationPolicy.ARBITRATED)
        sufficiency_counts[dec.structured_sufficiency.value] = (
            sufficiency_counts.get(dec.structured_sufficiency.value, 0) + 1
        )
        for bag, key in (
            (by_category, str(case.get("v213f_group") or q.get("category") or "unknown")),
            (by_subject, str(q.get("subject") or "unknown")),
            (by_grade, str(q.get("grade") or "unknown")),
        ):
            bag.setdefault(key, {"helped": 0, "hurt": 0, "neutral": 0})
            cf = (comparison["policies"]["C_ARBITRATED"]["cases"])
            # find matching cf
            match = next(
                (
                    x
                    for x in cf
                    if x.get("question_hash") == q.get("hash")
                ),
                None,
            )
            if not match:
                continue
            cfo = match["counterfactual"]
            if cfo.get("document_helped"):
                bag[key]["helped"] += 1
            elif cfo.get("document_hurt"):
                bag[key]["hurt"] += 1
            else:
                bag[key]["neutral"] += 1

    decision, rec_text = decide(comparison["policies"])
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "v2.13f",
        "dataset_meta": {
            "path": str(args.dataset),
            "n": len(cases),
            "created_at": data.get("created_at"),
            "group_pool_sizes": data.get("group_pool_sizes"),
            "selected_groups": {
                g: sum(1 for c in cases if c.get("v213f_group") == g)
                for g in (
                    "A_structured_sufficient",
                    "B_structured_insufficient_recovery",
                    "C_neutral",
                    "D_regression",
                )
            },
        },
        "config": {
            "v213d_shadow_enabled": settings.v213d_shadow_enabled,
            "v213d_shadow_sample_rate": settings.v213d_shadow_sample_rate,
            "v213f_document_arbitration_experiment": settings.v213f_document_arbitration_experiment,
            "v213f_arbitration_policy": settings.v213f_arbitration_policy,
            "v213e": "disabled",
        },
        "policies": {
            k: {kk: vv for kk, vv in v.items() if kk != "cases"}
            for k, v in comparison["policies"].items()
        },
        "policy_cases": {
            k: v.get("cases") for k, v in comparison["policies"].items()
        },
        "dd1c57": dd_analysis,
        "sufficiency_counts": sufficiency_counts,
        "by_category": by_category,
        "by_subject": by_subject,
        "by_grade": by_grade,
        "arbitration_mean_ms": round(elapsed_ms / max(len(cases), 1), 3),
        "decision": decision,
        "recommendation_text": rec_text,
        "method_note": (
            "Offline counterfactual on frozen real-traffic shadows: "
            "INCLUDE_IN_GENERATION keeps baseline shadow outcomes; "
            "PROVENANCE_ONLY/DO_NOT_USE/REQUIRE_REVIEW retain control outcomes. "
            "No additional LLM used for arbitration."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))
    write_report(summary, out_md=args.out_md)
    print(
        json.dumps(
            {
                "decision": decision,
                "n": len(cases),
                "baseline_regressions": (summary["policies"]["A_BASELINE_MERGE"]).get(
                    "regressions"
                ),
                "arbitrated_regressions": (summary["policies"]["C_ARBITRATED"]).get(
                    "regressions"
                ),
                "baseline_recoveries": (summary["policies"]["A_BASELINE_MERGE"]).get(
                    "recoveries"
                ),
                "arbitrated_recoveries": (summary["policies"]["C_ARBITRATED"]).get(
                    "recoveries"
                ),
                "md": str(args.out_md),
                "json": str(args.out_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
