#!/usr/bin/env python3
"""V2.13G.3 Stage-1 live shadow validation under preserve_control_answer.

Reads the append-only live JSONL; writes ONLY:
  data/diagnostics/v213g3_stage1_live.json
  data/diagnostics/v213g3_stage1_live.jsonl
  docs/V2_13G3_STAGE1_LIVE_SHADOW_VALIDATION.md

Does not overwrite V2.13F / V2.13G / G.1 / G.2 artifacts.
Does not enable production arbitration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_JSONL = ROOT / "data" / "diagnostics" / "v213g_arbitration_shadow.jsonl"
OUT_JSON = ROOT / "data" / "diagnostics" / "v213g3_stage1_live.json"
OUT_JSONL = ROOT / "data" / "diagnostics" / "v213g3_stage1_live.jsonl"
OUT_MD = ROOT / "docs" / "V2_13G3_STAGE1_LIVE_SHADOW_VALIDATION.md"
RUNTIME_CFG = ROOT / "data" / "diagnostics" / "v213g3_runtime_config.json"
PRE_HASHES = ROOT / "data" / "diagnostics" / "v213g3_historical_artifact_hashes_pre.txt"
POST_HASHES = ROOT / "data" / "diagnostics" / "v213g3_historical_artifact_hashes_post.txt"

BATCH_HEALTHY = "batch_2_healthy_structured_api"
BATCH_HISTORICAL = "batch_1_api_unavailable"
BASELINE_CONTAMINATION_PREFIXES = ("9cd42c", "27e64a")
NON_GENERATION_USES = {"PROVENANCE_ONLY", "DO_NOT_USE", "REQUIRE_REVIEW"}


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _pct(n: int, d: int) -> float | None:
    if d <= 0:
        return None
    return round(100.0 * n / d, 2)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return float(ordered[f])
    return float(ordered[f] + (ordered[c] - ordered[f]) * (k - f))


def _is_historical(row: dict[str, Any]) -> bool:
    bid = row.get("batch_id")
    if bid == BATCH_HISTORICAL:
        return True
    if not bid:
        return True
    return False


def _infra_invalid(row: dict[str, Any]) -> bool:
    infra = row.get("infrastructure") or {}
    if infra.get("validity") == "INFRASTRUCTURE_INVALID":
        return True
    if _is_historical(row) and not infra:
        # Untagged batch-1 confounded rows.
        return True
    return False


def _valid_healthy(row: dict[str, Any]) -> bool:
    if _is_historical(row):
        return False
    if row.get("batch_id") != BATCH_HEALTHY:
        return False
    infra = row.get("infrastructure") or {}
    if infra.get("validity") == "INFRASTRUCTURE_INVALID":
        return False
    if (row.get("baseline_shadow") or {}).get("error"):
        return False
    if (row.get("arbitrated_shadow") or {}).get("error"):
        return False
    return True


def _control_accepted(row: dict[str, Any]) -> bool:
    control = row.get("control") or {}
    if control.get("final_accepted") is True:
        return True
    if control.get("verifier_accepted") is True:
        return True
    if control.get("accepted") is True:
        return True
    verifier = str(control.get("verifier_decision") or "").lower()
    if verifier in {"accept", "accepted", "pass", "passed"}:
        return True
    return False


def _latency_view(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("latency") or row.get("latencies") or {}
    if not isinstance(raw, dict):
        return {}
    return {
        "structured_retrieval_ms": raw.get("structured_api_ms") or raw.get("structured_retrieval_ms"),
        "document_retrieval_ms": raw.get("retrieval_ms") or raw.get("document_retrieval_ms"),
        "arbitration_ms": raw.get("arbitration_ms"),
        "baseline_shadow_ms": raw.get("baseline_shadow_ms"),
        "arbitrated_shadow_ms": raw.get("arbitrated_shadow_ms"),
        "total_shadow_ms": raw.get("total_ms") or raw.get("total_shadow_ms"),
    }


def _expected_document_use(sufficiency: str, role: str) -> str | None:
    if sufficiency == "SUFFICIENT" and role == "REDUNDANT":
        return "PROVENANCE_ONLY"
    if sufficiency == "SUFFICIENT" and role == "IRRELEVANT":
        return "DO_NOT_USE"
    if sufficiency == "INSUFFICIENT" and role == "DECISIVE":
        return "INCLUDE_IN_GENERATION"
    return None


def _cohort_bucket(row: dict[str, Any]) -> str:
    arb = row.get("arbitration") or {}
    suff = str(arb.get("structured_sufficiency") or "")
    role = str(arb.get("document_role") or "")
    docs = int(arb.get("document_count") or 0)
    structured = int(arb.get("structured_count") or (row.get("infrastructure") or {}).get("structured_count") or 0)
    if structured <= 0 and docs > 0:
        return "document_only"
    if suff == "SUFFICIENT" and role == "REDUNDANT":
        return "structured_sufficient_redundant"
    if suff == "SUFFICIENT" and role == "IRRELEVANT":
        return "structured_sufficient_irrelevant"
    if suff == "SUFFICIENT" and role == "DECISIVE":
        return "structured_sufficient_decisive"
    if suff == "INSUFFICIENT" and role == "DECISIVE":
        return "structured_insufficient_decisive"
    if suff == "INSUFFICIENT" and role == "IRRELEVANT":
        return "structured_insufficient_irrelevant"
    if suff == "INSUFFICIENT" and role == "REDUNDANT":
        return "structured_insufficient_redundant"
    return "other"


def analyze_case(row: dict[str, Any]) -> dict[str, Any]:
    arb = row.get("arbitration") or {}
    control = row.get("control") or {}
    arm = row.get("arbitrated_shadow") or {}
    baseline = row.get("baseline_shadow") or {}
    q = row.get("question") or {}
    latency = _latency_view(row)

    sufficiency = str(arb.get("structured_sufficiency") or "")
    role = str(arb.get("document_role") or "")
    use = str(arb.get("document_use") or "")
    expected_use = _expected_document_use(sufficiency, role)
    classification_ok = expected_use is None or expected_use == use

    gen_docs = int(arb.get("generation_document_count") or arm.get("generation_document_count") or 0)
    regen = bool(arm.get("generation_attempted"))
    outcome = str(arm.get("outcome_source") or "")
    control_hash = str(control.get("answer_hash") or "")
    arb_hash = str(arm.get("answer_hash") or "")
    hash_match = bool(control_hash) and control_hash == arb_hash
    semantics = str(
        arb.get("provenance_only_semantics")
        or arm.get("provenance_only_semantics")
        or ""
    )
    control_ok = _control_accepted(row)
    preserve_eligible = (
        semantics == "preserve_control_answer"
        and sufficiency == "SUFFICIENT"
        and control_ok
        and use in NON_GENERATION_USES
    )
    preserve_ok = (
        preserve_eligible
        and gen_docs == 0
        and regen is False
        and outcome == "control"
        and hash_match
    )
    doc_leak = use in NON_GENERATION_USES and gen_docs > 0
    regen_violation = use in NON_GENERATION_USES and regen is True
    hash_violation = preserve_eligible and not hash_match

    hyp = row.get("hypothesis") or {}
    baseline_reg = bool(hyp.get("control_correct_baseline_worse"))
    arb_reg = bool(hyp.get("control_correct_arbitrated_worse"))
    qhash = str(q.get("hash") or "")
    baseline_contamination = any(qhash.startswith(p) for p in BASELINE_CONTAMINATION_PREFIXES)

    h2 = sufficiency == "INSUFFICIENT" and role == "DECISIVE" and use == "INCLUDE_IN_GENERATION"

    return {
        "timestamp": row.get("timestamp"),
        "request_id": row.get("request_id"),
        "question_hash": qhash,
        "question": q.get("text") or q.get("question"),
        "batch_id": row.get("batch_id"),
        "infrastructure": row.get("infrastructure"),
        "cohort_bucket": _cohort_bucket(row),
        "structured_sufficiency": sufficiency,
        "document_role": role,
        "document_use": use,
        "expected_document_use": expected_use,
        "classification_ok": classification_ok,
        "structured_count": arb.get("structured_count")
        or (row.get("infrastructure") or {}).get("structured_count"),
        "document_count": arb.get("document_count"),
        "generation_document_count": gen_docs,
        "provenance_document_count": arb.get("provenance_document_count"),
        "provenance_only_semantics": semantics,
        "control_accepted": control_ok,
        "control_answer_hash": control_hash,
        "control_verifier": control.get("verifier_decision"),
        "control_mapper": control.get("mapper_recommendation"),
        "control_route": control.get("final_route"),
        "arbitrated_answer_hash": arb_hash,
        "arbitrated_verifier": arm.get("verifier_decision"),
        "arbitrated_mapper": arm.get("mapper_recommendation") or arm.get("mapper_decision"),
        "arbitrated_route": arm.get("final_route") or arm.get("route"),
        "outcome_source": outcome,
        "generation_attempted": regen,
        "preserve_eligible": preserve_eligible,
        "preserve_ok": preserve_ok,
        "document_leak": doc_leak,
        "regeneration_violation": regen_violation,
        "control_hash_violation": hash_violation,
        "h2_case": h2,
        "baseline_recovery": bool(hyp.get("control_incorrect_baseline_better")),
        "arbitrated_recovery": bool(hyp.get("control_incorrect_arbitrated_better")),
        "baseline_regression": baseline_reg,
        "arbitration_regression": arb_reg and not baseline_contamination,
        "baseline_document_merge_regression": baseline_reg and baseline_contamination,
        "baseline_contamination_id": baseline_contamination,
        "safety": {
            "baseline": baseline.get("safety") or baseline.get("safety_flags"),
            "arbitrated": arm.get("safety") or arm.get("safety_flags"),
        },
        "latency": latency,
        "shadow_stage": arm.get("shadow_stage"),
        "normalization_status": arm.get("normalization_status"),
    }


def decide(summary: dict[str, Any]) -> str:
    if int(summary["infrastructure_invalid_n"]) > 0 and int(summary["valid_n"]) == 0:
        return "INFRASTRUCTURE_INVALID"
    preserve = summary["preserve_control"]
    h2 = summary["h2"]
    composition = summary["cohort_composition"]
    sufficient_non_gen = int(preserve["preserve_control_cases"])
    has_redundant = int(composition.get("structured_sufficient_redundant") or 0) > 0
    has_irrelevant = int(composition.get("structured_sufficient_irrelevant") or 0) > 0
    has_h2 = int(h2["n"]) > 0
    meaningful = sufficient_non_gen >= 5 and (has_redundant or has_irrelevant) and has_h2

    hard_fail = (
        int(preserve["preserve_control_document_leaks"]) > 0
        or int(preserve["preserve_control_regeneration_attempts"]) > 0
        or int(preserve["preserve_control_hash_mismatches"]) > 0
        or int(summary["safety"]["total_flags"]) > 0
    )
    if hard_fail and sufficient_non_gen > 0:
        return "STAGE1_LIVE_PRESERVE_NOT_CONFIRMED"
    if not meaningful:
        # Partial confirmation only with a small but clean preserve set.
        if (
            sufficient_non_gen >= 3
            and int(preserve["preserve_control_hash_matches"]) == sufficient_non_gen
            and int(preserve["preserve_control_document_leaks"]) == 0
            and int(preserve["preserve_control_regeneration_attempts"]) == 0
        ):
            return "STAGE1_LIVE_PRESERVE_PARTIALLY_CONFIRMED"
        if (
            sufficient_non_gen > 0
            and int(preserve["preserve_control_hash_matches"]) == sufficient_non_gen
            and int(preserve["preserve_control_document_leaks"]) == 0
            and int(preserve["preserve_control_regeneration_attempts"]) == 0
        ):
            # Clean but too sparse for even partial Stage-1 confirmation.
            return "INSUFFICIENT_LIVE_COVERAGE"
        return "INSUFFICIENT_LIVE_COVERAGE"
    if (
        int(preserve["preserve_control_hash_matches"]) == sufficient_non_gen
        and int(preserve["preserve_control_document_leaks"]) == 0
        and int(preserve["preserve_control_regeneration_attempts"]) == 0
        and int(preserve["preserve_control_regressions"]) == 0
        and int(summary["h1a"]["classification_errors"]) == 0
        and int(summary["h1b"]["document_leaks"]) == 0
    ):
        return "STAGE1_LIVE_PRESERVE_CONFIRMED"
    if int(preserve["preserve_control_hash_matches"]) == sufficient_non_gen:
        return "STAGE1_LIVE_PRESERVE_PARTIALLY_CONFIRMED"
    return "STAGE1_LIVE_PRESERVE_NOT_CONFIRMED"


def build_summary(
    rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    *,
    runtime: dict[str, Any],
    historical_unchanged: bool,
) -> dict[str, Any]:
    tagged_infra_invalid = [
        r
        for r in rows
        if (r.get("infrastructure") or {}).get("validity") == "INFRASTRUCTURE_INVALID"
    ]
    historical = [r for r in rows if _is_historical(r)]
    valid = [c for c in cases]
    composition = Counter(c["cohort_bucket"] for c in valid)

    preserve_cases = [c for c in valid if c["preserve_eligible"]]
    non_gen = [c for c in valid if c["document_use"] in NON_GENERATION_USES]
    h2_cases = [c for c in valid if c["h2_case"]]

    class_errors = [c for c in valid if c["expected_document_use"] and not c["classification_ok"]]
    doc_leaks = [c for c in non_gen if c["document_leak"]]
    regen_viol = [c for c in non_gen if c["regeneration_violation"]]

    lat_struct = [
        float(c["latency"]["structured_retrieval_ms"])
        for c in valid
        if isinstance(c.get("latency"), dict) and c["latency"].get("structured_retrieval_ms") is not None
    ]
    lat_docs = [
        float(c["latency"]["document_retrieval_ms"])
        for c in valid
        if isinstance(c.get("latency"), dict) and c["latency"].get("document_retrieval_ms") is not None
    ]
    lat_arb = [
        float(c["latency"]["arbitration_ms"])
        for c in valid
        if isinstance(c.get("latency"), dict) and c["latency"].get("arbitration_ms") is not None
    ]
    lat_total = [
        float(c["latency"]["total_shadow_ms"])
        for c in valid
        if isinstance(c.get("latency"), dict) and c["latency"].get("total_shadow_ms") is not None
    ]

    safety_flags = Counter()
    for c in valid:
        for arm_name in ("baseline", "arbitrated"):
            flags = c.get("safety", {}).get(arm_name)
            if isinstance(flags, dict):
                for k, v in flags.items():
                    if v:
                        safety_flags[k] += 1
            elif isinstance(flags, list):
                for item in flags:
                    safety_flags[str(item)] += 1

    summary: dict[str, Any] = {
        "schema_version": "v213g3.1",
        "experiment": "v2.13g.3_stage1_live_shadow_validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_jsonl": str(SRC_JSONL.relative_to(ROOT)),
        "runtime": runtime,
        "production_isolation": {
            "production_arbitration": "OFF",
            "v213f_document_arbitration_experiment": False,
            "v213e_enabled": False,
            "sample_rate": 0.01,
            "api_v1_unchanged": True,
            "corpus_unchanged": True,
            "historical_artifacts_unchanged": historical_unchanged,
            "historical_hash_files": {
                "pre": str(PRE_HASHES.relative_to(ROOT)) if PRE_HASHES.exists() else None,
                "post": str(POST_HASHES.relative_to(ROOT)) if POST_HASHES.exists() else None,
            },
        },
        "valid_n": len(valid),
        "infrastructure_invalid_n": len(tagged_infra_invalid),
        "historical_batch1_n": len(historical),
        "n_source_rows": len(rows),
        "cohort_composition": {
            "structured_sufficient_redundant": composition.get("structured_sufficient_redundant", 0),
            "structured_sufficient_irrelevant": composition.get("structured_sufficient_irrelevant", 0),
            "structured_sufficient_decisive": composition.get("structured_sufficient_decisive", 0),
            "structured_insufficient_decisive": composition.get("structured_insufficient_decisive", 0),
            "structured_insufficient_irrelevant": composition.get("structured_insufficient_irrelevant", 0),
            "structured_insufficient_redundant": composition.get("structured_insufficient_redundant", 0),
            "document_only": composition.get("document_only", 0),
            "other": composition.get("other", 0),
            "infrastructure_invalid": len(tagged_infra_invalid),
        },
        "h1a": {
            "evaluated": sum(1 for c in valid if c["expected_document_use"]),
            "classification_errors": len(class_errors),
            "error_cases": [
                {
                    "question_hash": c["question_hash"],
                    "expected": c["expected_document_use"],
                    "actual": c["document_use"],
                }
                for c in class_errors
            ],
        },
        "h1b": {
            "non_generation_cases": len(non_gen),
            "document_leaks": len(doc_leaks),
            "regeneration_attempts": len(regen_viol),
            "leak_cases": [c["question_hash"] for c in doc_leaks],
            "regen_cases": [c["question_hash"] for c in regen_viol],
        },
        "preserve_control": {
            "preserve_control_cases": len(preserve_cases),
            "preserve_control_hash_matches": sum(
                1
                for c in preserve_cases
                if c["control_answer_hash"]
                and c["control_answer_hash"] == c["arbitrated_answer_hash"]
            ),
            "preserve_control_hash_mismatches": sum(
                1 for c in preserve_cases if c["control_hash_violation"]
            ),
            "preserve_control_regeneration_attempts": sum(
                1 for c in preserve_cases if c["regeneration_violation"]
            ),
            "preserve_control_document_leaks": sum(
                1 for c in preserve_cases if c["document_leak"]
            ),
            "preserve_control_regressions": sum(
                1 for c in preserve_cases if c["arbitration_regression"]
            ),
            "hash_match_rate_pct": _pct(
                sum(
                    1
                    for c in preserve_cases
                    if c["control_answer_hash"]
                    and c["control_answer_hash"] == c["arbitrated_answer_hash"]
                ),
                len(preserve_cases),
            ),
            "violation_cases": [
                {
                    "question_hash": c["question_hash"],
                    "hash_violation": c["control_hash_violation"],
                    "document_leak": c["document_leak"],
                    "regeneration_violation": c["regeneration_violation"],
                    "arbitration_regression": c["arbitration_regression"],
                }
                for c in preserve_cases
                if c["control_hash_violation"]
                or c["document_leak"]
                or c["regeneration_violation"]
                or c["arbitration_regression"]
            ],
        },
        "h2": {
            "n": len(h2_cases),
            "generation_document_counts": [c["generation_document_count"] for c in h2_cases],
            "regeneration_attempted": sum(1 for c in h2_cases if c["generation_attempted"]),
            "outcome_sources": dict(Counter(c["outcome_source"] for c in h2_cases)),
            "arbitrated_recoveries": sum(1 for c in h2_cases if c["arbitrated_recovery"]),
            "baseline_recoveries": sum(1 for c in h2_cases if c["baseline_recovery"]),
            "arbitration_regressions": sum(1 for c in h2_cases if c["arbitration_regression"]),
            "cases": [c["question_hash"] for c in h2_cases],
        },
        "baseline_contamination": {
            "ids": list(BASELINE_CONTAMINATION_PREFIXES),
            "observed_live_valid": [
                c["question_hash"] for c in valid if c["baseline_contamination_id"]
            ],
            "baseline_document_merge_regressions": sum(
                1 for c in valid if c["baseline_document_merge_regression"]
            ),
            "arbitration_regressions": sum(1 for c in valid if c["arbitration_regression"]),
        },
        "safety": {
            "flag_counts": dict(safety_flags),
            "total_flags": sum(safety_flags.values()),
            "expected_all_zero": True,
        },
        "latency": {
            "structured_retrieval_ms": {
                "n": len(lat_struct),
                "p50": _percentile(lat_struct, 50),
                "p95": _percentile(lat_struct, 95),
                "mean": round(statistics.fmean(lat_struct), 2) if lat_struct else None,
            },
            "document_retrieval_ms": {
                "n": len(lat_docs),
                "p50": _percentile(lat_docs, 50),
                "p95": _percentile(lat_docs, 95),
                "mean": round(statistics.fmean(lat_docs), 2) if lat_docs else None,
            },
            "arbitration_ms": {
                "n": len(lat_arb),
                "p50": _percentile(lat_arb, 50),
                "p95": _percentile(lat_arb, 95),
                "mean": round(statistics.fmean(lat_arb), 2) if lat_arb else None,
            },
            "total_shadow_ms": {
                "n": len(lat_total),
                "p50": _percentile(lat_total, 50),
                "p95": _percentile(lat_total, 95),
                "mean": round(statistics.fmean(lat_total), 2) if lat_total else None,
            },
            "note": "Shadow latency only; not production user-facing latency.",
        },
    }
    summary["decision"] = decide(summary)
    summary["recommendation"] = {
        "enable_production_arbitration": False,
        "continue_live_at_0_01": summary["decision"]
        in {
            "INSUFFICIENT_LIVE_COVERAGE",
            "STAGE1_LIVE_PRESERVE_PARTIALLY_CONFIRMED",
        },
        "next": (
            "Continue Stage-1 live shadow at sample_rate=0.01 until sufficient+document, "
            "redundant/irrelevant, and insufficient+decisive cohorts are all represented; "
            "do not enable production arbitration."
            if summary["decision"] != "STAGE1_LIVE_PRESERVE_CONFIRMED"
            else "Stage-1 live preserve confirmed on observed evidence; keep production arbitration OFF and schedule a separate promotion review."
        ),
    }
    summary["config_and_cohort_hashes"] = {
        "runtime_config_hash": _sha16(json.dumps(runtime, sort_keys=True)),
        "valid_case_hashes": sorted(c["question_hash"] for c in valid if c.get("question_hash")),
        "cohort_fingerprint": _sha16(json.dumps(summary["cohort_composition"], sort_keys=True)),
    }
    return summary


def write_report(summary: dict[str, Any]) -> str:
    pc = summary["preserve_control"]
    comp = summary["cohort_composition"]
    lines = [
        "# V2.13G.3 — Stage-1 Live Shadow Validation",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        f"**Decision: `{summary['decision']}`**",
        "",
        "Production arbitration remains **OFF**. This report is shadow-only.",
        "",
        "## A. Runtime configuration",
        "",
        "```text",
        f"V213G_PROVENANCE_ONLY_SEMANTICS={summary['runtime'].get('V213G_PROVENANCE_ONLY_SEMANTICS')}",
        f"v213d_shadow_enabled={summary['runtime'].get('v213d_shadow_enabled')}",
        f"v213d_shadow_sample_rate={summary['runtime'].get('v213d_shadow_sample_rate')}",
        f"v213f_document_arbitration_experiment={summary['runtime'].get('v213f_document_arbitration_experiment')}",
        f"v213e_enabled={summary['runtime'].get('v213e_enabled')}",
        f"v213g_live_only={summary['runtime'].get('v213g_live_only')}",
        "```",
        "",
        f"Verified at: `{summary['runtime'].get('verified_at')}`",
        "",
        "## B. Infrastructure health",
        "",
        f"- valid rows: **{summary['valid_n']}**",
        f"- infrastructure-invalid rows: **{summary['infrastructure_invalid_n']}**",
        f"- historical batch-1 rows (excluded from live n): **{summary['historical_batch1_n']}**",
        f"- source JSONL rows: **{summary['n_source_rows']}**",
        "",
        "## C. Cohort composition (valid live only)",
        "",
        "| Cohort | Count |",
        "| --- | ---: |",
        f"| Structured sufficient + redundant | {comp['structured_sufficient_redundant']} |",
        f"| Structured sufficient + irrelevant | {comp['structured_sufficient_irrelevant']} |",
        f"| Structured sufficient + decisive | {comp['structured_sufficient_decisive']} |",
        f"| Structured insufficient + decisive | {comp['structured_insufficient_decisive']} |",
        f"| Structured insufficient + irrelevant | {comp['structured_insufficient_irrelevant']} |",
        f"| Structured insufficient + redundant | {comp['structured_insufficient_redundant']} |",
        f"| Document-only | {comp['document_only']} |",
        f"| Infrastructure-invalid | {comp['infrastructure_invalid']} |",
        "",
        "## D. H1a classification",
        "",
        f"- evaluated: {summary['h1a']['evaluated']}",
        f"- classification errors: **{summary['h1a']['classification_errors']}**",
        "",
        "## E. H1b enforcement",
        "",
        f"- non-generation cases: {summary['h1b']['non_generation_cases']}",
        f"- document leaks: **{summary['h1b']['document_leaks']}**",
        f"- regeneration attempts: **{summary['h1b']['regeneration_attempts']}**",
        "",
        "## F. H1c control preservation",
        "",
        f"- preserve cases: **{pc['preserve_control_cases']}**",
        f"- hash matches: **{pc['preserve_control_hash_matches']}**",
        f"- hash mismatches: **{pc['preserve_control_hash_mismatches']}**",
        f"- regeneration attempts: **{pc['preserve_control_regeneration_attempts']}**",
        f"- document leaks: **{pc['preserve_control_document_leaks']}**",
        f"- arbitration regressions: **{pc['preserve_control_regressions']}**",
        f"- hash match rate: **{pc['hash_match_rate_pct']}%**",
        "",
        "## G. H2 decisive recovery",
        "",
        f"- H2 cases: **{summary['h2']['n']}**",
        f"- regeneration attempted: {summary['h2']['regeneration_attempted']}",
        f"- outcome sources: `{summary['h2']['outcome_sources']}`",
        f"- arbitrated recoveries: {summary['h2']['arbitrated_recoveries']}",
        f"- baseline recoveries: {summary['h2']['baseline_recoveries']}",
        f"- arbitration regressions: {summary['h2']['arbitration_regressions']}",
        "",
        "## H. Baseline contamination",
        "",
        f"- tracked ids: `{summary['baseline_contamination']['ids']}`",
        f"- observed in valid live: `{summary['baseline_contamination']['observed_live_valid']}`",
        f"- baseline document-merge regressions: {summary['baseline_contamination']['baseline_document_merge_regressions']}",
        f"- arbitration regressions (non-contamination): {summary['baseline_contamination']['arbitration_regressions']}",
        "",
        "## I. Safety",
        "",
        f"- flag counts: `{summary['safety']['flag_counts']}`",
        f"- total flags: **{summary['safety']['total_flags']}** (expected 0)",
        "",
        "## J. Latency (shadow only)",
        "",
        "```json",
        json.dumps(summary["latency"], indent=2),
        "```",
        "",
        "## K. Production isolation",
        "",
        "```json",
        json.dumps(summary["production_isolation"], indent=2),
        "```",
        "",
        "## L. Decision",
        "",
        f"`{summary['decision']}`",
        "",
        f"Recommendation: {summary['recommendation']['next']}",
        "",
        "Promotion rule: **DO NOT enable production arbitration** from this result alone.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=SRC_JSONL)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-jsonl", type=Path, default=OUT_JSONL)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    from app.agent.v213g_live_shadow import load_v213g_records

    rows = load_v213g_records(args.jsonl)
    valid_rows = [r for r in rows if _valid_healthy(r)]
    cases = [analyze_case(r) for r in valid_rows]

    runtime = {
        "V213G_PROVENANCE_ONLY_SEMANTICS": "preserve_control_answer",
        "v213d_shadow_enabled": True,
        "v213d_shadow_sample_rate": 0.01,
        "v213f_document_arbitration_experiment": False,
        "v213e_enabled": False,
        "v213g_live_only": True,
        "verified_at": None,
    }
    if RUNTIME_CFG.is_file():
        loaded = json.loads(RUNTIME_CFG.read_text())
        runtime.update(loaded.get("checks") or {})
        runtime["verified_at"] = loaded.get("verified_at")
        runtime["agent_restarted_at"] = loaded.get("agent_restarted_at")

    historical_unchanged = False
    if PRE_HASHES.is_file() and POST_HASHES.is_file():
        historical_unchanged = PRE_HASHES.read_text() == POST_HASHES.read_text()

    summary = build_summary(
        rows,
        cases,
        runtime=runtime,
        historical_unchanged=historical_unchanged,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.write_text(
        "\n".join(json.dumps(c) for c in cases) + ("\n" if cases else ""),
        encoding="utf-8",
    )
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.out_md.write_text(write_report(summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "valid_n": summary["valid_n"],
                "infrastructure_invalid_n": summary["infrastructure_invalid_n"],
                "preserve_control_cases": summary["preserve_control"]["preserve_control_cases"],
                "hash_match_rate_pct": summary["preserve_control"]["hash_match_rate_pct"],
                "document_leaks": summary["preserve_control"]["preserve_control_document_leaks"],
                "regeneration_attempts": summary["preserve_control"][
                    "preserve_control_regeneration_attempts"
                ],
                "h2_n": summary["h2"]["n"],
                "json": str(args.out_json),
                "jsonl": str(args.out_jsonl),
                "md": str(args.out_md),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
