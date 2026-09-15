"""V2.13G cohort-coverage investigation helpers (shadow/replay diagnostics only)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.v213f_arbitration import (
    ArbitrationPolicy,
    DocumentRole,
    DocumentUse,
    StructuredSufficiency,
    arbitrate_from_shadow_record,
    counterfactual_outcome,
)

ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def analyze_live_g_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c_acc = Counter()
    c_route = Counter()
    cats = Counter()
    grades = Counter()
    subjects = Counter()
    struct_n = Counter()
    doc_n = Counter()
    suff = Counter()
    role = Counter()
    use = Counter()
    for r in rows:
        c = r.get("control") or {}
        arb = r.get("arbitration") or {}
        q = r.get("question") or {}
        b = r.get("baseline_shadow") or {}
        c_acc[bool(c.get("final_accepted"))] += 1
        c_route[str(c.get("final_route"))] += 1
        cats[str(q.get("category"))] += 1
        grades[str(q.get("grade"))] += 1
        subjects[str(q.get("subject"))] += 1
        struct_n[
            int(
                arb.get("structured_count")
                or b.get("structured_evidence_count")
                or c.get("evidence_count")
                or 0
            )
        ] += 1
        doc_n[int(arb.get("document_count") or b.get("document_evidence_count") or 0)] += 1
        suff[str(arb.get("structured_sufficiency"))] += 1
        role[str(arb.get("document_role"))] += 1
        use[str(arb.get("document_use"))] += 1

    zero_struct = int(struct_n.get(0, 0))
    return {
        "n": len(rows),
        "control_accepted": dict(c_acc),
        "control_route": dict(c_route),
        "categories": dict(cats),
        "grades": dict(grades),
        "subjects": dict(subjects),
        "structured_count_hist": {str(k): v for k, v in sorted(struct_n.items())},
        "document_count_hist": {str(k): v for k, v in sorted(doc_n.items())},
        "sufficiency": dict(suff),
        "document_role": dict(role),
        "document_use": dict(use),
        "zero_structured_evidence": zero_struct,
        "all_zero_structured": zero_struct == len(rows) and len(rows) > 0,
        "all_insufficient_decisive_include": (
            suff.get("INSUFFICIENT", 0) == len(rows)
            and role.get("DECISIVE", 0) == len(rows)
            and use.get("INCLUDE_IN_GENERATION", 0) == len(rows)
            and len(rows) > 0
        ),
    }


def determine_root_cause(live: dict[str, Any], *, curriculum_api_reachable: bool | None) -> dict[str, Any]:
    """Map audit evidence onto the required determination vocabulary."""
    if live.get("all_zero_structured") and live.get("all_insufficient_decisive_include"):
        if curriculum_api_reachable is False:
            determination = "SAMPLING_BIAS"
            detail = (
                "Live shadows captured only questions whose production control path "
                "resolved zero structured evidence (control never accepted). "
                "Curriculum Structure API was unreachable during investigation, which "
                "explains the missing structured-sufficient cohort. Given empty "
                "structured evidence, the classifier correctly emits "
                "INSUFFICIENT→DECISIVE→INCLUDE_IN_GENERATION when documents exist."
            )
        else:
            determination = "SAMPLING_BIAS"
            detail = (
                "All live dual-arm rows have structured_count=0 and control_accepted=false. "
                "Classification is consistent with that evidence. The live sample did not "
                "include any structured-sufficient/document-redundant cases, so the "
                "primary V2.13F regression hypothesis was untested."
            )
        classification_validity = "CLASSIFICATION_CORRECT"
    else:
        determination = "INSUFFICIENT_INFORMATION"
        classification_validity = "INSUFFICIENT_INFORMATION"
        detail = "Live distribution is mixed; further inspection required."

    return {
        "determination": determination,
        "classification_validity": classification_validity,
        "detail": detail,
        "primary_regression_hypothesis": "UNTESTED"
        if live.get("all_zero_structured")
        else "TESTABLE",
        "curriculum_api_reachable": curriculum_api_reachable,
    }


def replay_v213f_through_classifier(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    class_counts: Counter[str] = Counter()
    dd1c57: dict[str, Any] | None = None
    cohort_counts = {
        "structured_sufficient_with_docs": 0,
        "structured_sufficient_redundant_docs": 0,
        "structured_sufficient_irrelevant_docs": 0,
        "structured_insufficient_decisive_docs": 0,
        "structured_insufficient_irrelevant_docs": 0,
        "neutral_document_cases": 0,
    }
    rows_out: list[dict[str, Any]] = []

    for case in cases:
        decision = arbitrate_from_shadow_record(
            case, policy=ArbitrationPolicy.ARBITRATED
        )
        cf = counterfactual_outcome(case, decision)
        group = str(case.get("v213f_group") or "unknown")
        qh = str((case.get("question") or {}).get("hash") or "")
        key = (
            f"{decision.structured_sufficiency.value}|"
            f"{decision.document_role.value}|"
            f"{decision.document_use.value}"
        )
        class_counts[key] += 1
        control = case.get("control") or {}
        shadow = case.get("shadow") or {}
        docs = int(shadow.get("document_evidence_count") or decision.document_count or 0)
        row = {
            "source": "TARGETED_REPLAY",
            "v213f_group": group,
            "request_id": case.get("request_id"),
            "question_hash": qh,
            "control_final_accepted": bool(control.get("final_accepted")),
            "control_evidence_count": int(control.get("evidence_count") or 0),
            "document_evidence_count": docs,
            "decision": decision.to_dict(),
            "counterfactual": cf,
        }
        rows_out.append(row)
        by_group.setdefault(group, []).append(row)

        if (
            decision.structured_sufficiency == StructuredSufficiency.SUFFICIENT
            and docs > 0
        ):
            cohort_counts["structured_sufficient_with_docs"] += 1
            if decision.document_role == DocumentRole.REDUNDANT:
                cohort_counts["structured_sufficient_redundant_docs"] += 1
            if decision.document_role == DocumentRole.IRRELEVANT:
                cohort_counts["structured_sufficient_irrelevant_docs"] += 1
        if decision.structured_sufficiency == StructuredSufficiency.INSUFFICIENT:
            if decision.document_role == DocumentRole.DECISIVE:
                cohort_counts["structured_insufficient_decisive_docs"] += 1
            if decision.document_role == DocumentRole.IRRELEVANT:
                cohort_counts["structured_insufficient_irrelevant_docs"] += 1
        if group == "C_neutral" or (
            cf.get("document_neutral") and docs > 0
        ):
            cohort_counts["neutral_document_cases"] += 1

        if qh.startswith("dd1c57") and bool(control.get("final_accepted")):
            dd1c57 = {
                "question_hash": qh,
                "request_id": case.get("request_id"),
                "v213f_group": group,
                "expected": {
                    "structured_sufficiency": "SUFFICIENT",
                    "document_role": "REDUNDANT",
                    "document_use": "PROVENANCE_ONLY",
                },
                "observed": decision.to_dict(),
                "counterfactual": cf,
                "pass": (
                    decision.structured_sufficiency == StructuredSufficiency.SUFFICIENT
                    and decision.document_role == DocumentRole.REDUNDANT
                    and decision.document_use == DocumentUse.PROVENANCE_ONLY
                    and cf.get("final_accepted") is True
                    and cf.get("control_correct_shadow_worse") is False
                    and cf.get("generator_drift_prevented") is True
                ),
            }

    # Group compatibility summary
    group_compat: dict[str, Any] = {}
    for group, items in by_group.items():
        expected_notes = {
            "A_structured_sufficient": "SUFFICIENT + REDUNDANT/IRRELEVANT + PROVENANCE_ONLY/DO_NOT_USE",
            "B_structured_insufficient_recovery": "INSUFFICIENT + DECISIVE + INCLUDE",
            "C_neutral": "mixed; often INSUFFICIENT if control not accepted",
            "D_regression": "SUFFICIENT + REDUNDANT/IRRELEVANT + withhold docs",
        }.get(group, "")
        ok = 0
        for item in items:
            d = item["decision"]
            if group in {"A_structured_sufficient", "D_regression"}:
                if (
                    d["structured_sufficiency"] == "SUFFICIENT"
                    and d["document_use"]
                    in {"PROVENANCE_ONLY", "DO_NOT_USE", "REQUIRE_REVIEW"}
                ):
                    ok += 1
            elif group == "B_structured_insufficient_recovery":
                if (
                    d["structured_sufficiency"] == "INSUFFICIENT"
                    and d["document_use"] == "INCLUDE_IN_GENERATION"
                ):
                    ok += 1
            else:
                ok += 1  # informational
        group_compat[group] = {
            "n": len(items),
            "compatible": ok,
            "expected": expected_notes,
        }

    return {
        "n": len(cases),
        "classifier_distribution": dict(class_counts),
        "group_compatibility": group_compat,
        "cohort_counts": cohort_counts,
        "dd1c57": dd1c57,
        "rows": rows_out,
    }


def hypothesis_results(
    *,
    live_sufficient_with_docs: int,
    live_baseline_reg: int,
    live_arb_reg: int,
    live_baseline_rec: int,
    live_arb_rec: int,
    live_baseline_unsup: int,
    live_arb_unsup: int,
    targeted_sufficient_with_docs: int,
    targeted_dd1c57_pass: bool | None,
    safety_blocked: bool,
) -> dict[str, Any]:
    if safety_blocked:
        safety = "BLOCKED"
    else:
        safety = "PASS"

    if live_sufficient_with_docs == 0:
        h1 = "UNTESTED"
        h1_note = "PRIMARY_REGRESSION_HYPOTHESIS_UNTESTED on LIVE_TRAFFIC (structured_sufficient_with_docs=0)."
        if targeted_sufficient_with_docs > 0 and targeted_dd1c57_pass:
            h1_note += " TARGETED_REPLAY reproduces V2.13F sufficient/redundant withholding including dd1c57."
    elif live_arb_reg < live_baseline_reg and live_arb_reg == 0:
        h1 = "CONFIRMED"
        h1_note = "Live arbitrated regressions below baseline and at zero."
    elif live_arb_reg < live_baseline_reg:
        h1 = "PARTIALLY_CONFIRMED"
        h1_note = "Live regressions decreased but residual remain."
    else:
        h1 = "NOT_CONFIRMED"
        h1_note = "No live regression reduction observed."

    if live_baseline_rec == 0 and live_arb_rec == 0:
        h2 = "UNTESTED"
        h2_note = "No recovery opportunities in live sample."
    elif live_arb_rec >= live_baseline_rec:
        h2 = "CONFIRMED"
        h2_note = (
            f"Arbitrated recoveries ({live_arb_rec}) preserved vs baseline "
            f"({live_baseline_rec})."
        )
    elif live_arb_rec >= max(0, live_baseline_rec - max(1, live_baseline_rec // 10)):
        h2 = "PARTIALLY_CONFIRMED"
        h2_note = "Minor recovery loss within tolerance."
    else:
        h2 = "NOT_CONFIRMED"
        h2_note = "Material recovery loss under arbitration."

    if live_baseline_unsup == 0 and live_arb_unsup == 0 and live_sufficient_with_docs == 0:
        h3 = "UNTESTED"
        h3_note = "Generator-drift reduction not testable without sufficient+doc cohort; unsupported-claim delta still informative."
    elif live_arb_unsup < live_baseline_unsup:
        h3 = "PARTIALLY_CONFIRMED" if live_sufficient_with_docs == 0 else "CONFIRMED"
        h3_note = (
            f"Unsupported claims fell {live_baseline_unsup}→{live_arb_unsup} "
            f"(live sufficient+doc cohort={live_sufficient_with_docs})."
        )
    elif live_arb_unsup == live_baseline_unsup:
        h3 = "PARTIALLY_CONFIRMED"
        h3_note = "Unsupported claims unchanged."
    else:
        h3 = "NOT_CONFIRMED"
        h3_note = "Unsupported claims increased under arbitration."

    if safety == "BLOCKED":
        overall = "SAFETY_BLOCKED"
        recommendation = "SAFETY_BLOCKED"
    elif h1 == "UNTESTED":
        overall = "INVESTIGATE_BEFORE_PROMOTION"
        recommendation = "PROCEED_TO_TARGETED_LIVE_VALIDATION"
        if targeted_sufficient_with_docs > 0 and targeted_dd1c57_pass and h2 == "CONFIRMED":
            recommendation = "PROCEED_TO_TARGETED_LIVE_VALIDATION"
    elif h1 == "CONFIRMED" and h2 in {"CONFIRMED", "PARTIALLY_CONFIRMED"} and safety == "PASS":
        overall = "ARBITRATION_CONFIRMED"
        recommendation = "ARBITRATION_CONFIRMED"
    elif h1 in {"CONFIRMED", "PARTIALLY_CONFIRMED"} or h2 == "CONFIRMED":
        overall = "ARBITRATION_PARTIALLY_CONFIRMED"
        recommendation = "CONTINUE_LIVE_SHADOW"
    else:
        overall = "ARBITRATION_NOT_CONFIRMED"
        recommendation = "ARBITRATION_NOT_CONFIRMED"

    return {
        "H1_regression_prevention": {"result": h1, "note": h1_note},
        "H2_recovery_preservation": {"result": h2, "note": h2_note},
        "H3_generator_drift_reduction": {"result": h3, "note": h3_note},
        "safety": safety,
        "overall_status": overall,
        "recommendation": recommendation,
        "primary_regression_hypothesis_live": (
            "PRIMARY_REGRESSION_HYPOTHESIS_UNTESTED"
            if live_sufficient_with_docs == 0
            else "TESTED"
        ),
    }


def enrich_cohort_counts_from_live(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "structured_sufficient_with_docs": 0,
        "structured_sufficient_redundant_docs": 0,
        "structured_sufficient_irrelevant_docs": 0,
        "structured_insufficient_decisive_docs": 0,
        "structured_insufficient_irrelevant_docs": 0,
        "neutral_document_cases": 0,
    }
    for r in rows:
        arb = r.get("arbitration") or {}
        docs = int(arb.get("document_count") or 0)
        suff = str(arb.get("structured_sufficiency") or "")
        role = str(arb.get("document_role") or "")
        hyp = r.get("hypothesis") or {}
        if suff == "SUFFICIENT" and docs > 0:
            counts["structured_sufficient_with_docs"] += 1
            if role == "REDUNDANT":
                counts["structured_sufficient_redundant_docs"] += 1
            if role == "IRRELEVANT":
                counts["structured_sufficient_irrelevant_docs"] += 1
        if suff == "INSUFFICIENT":
            if role == "DECISIVE":
                counts["structured_insufficient_decisive_docs"] += 1
            if role == "IRRELEVANT":
                counts["structured_insufficient_irrelevant_docs"] += 1
        if (not hyp.get("control_correct_baseline_worse")) and (
            not hyp.get("control_insufficient_baseline_accepted")
        ):
            counts["neutral_document_cases"] += 1
    return counts


__all__ = [
    "analyze_live_g_rows",
    "determine_root_cause",
    "enrich_cohort_counts_from_live",
    "hypothesis_results",
    "load_jsonl",
    "replay_v213f_through_classifier",
]
