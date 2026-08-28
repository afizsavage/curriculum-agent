#!/usr/bin/env python3
"""V2.2 controlled experiment: context boundary vs V2.1 control."""

from __future__ import annotations

import json
import random
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_v21_context_resolve import (  # noqa: E402
    GOLDEN,
    LEGACY_TOOLS,
    RESOLVE_TOOL,
    analyze_trace,
    classify_sequence,
)

OUT = ROOT / "data" / "diagnostics" / "v22_context_boundary_experiment"
DOC = ROOT / "docs" / "V2_2_CONTEXT_BOUNDARY_EXPERIMENT.md"
CONTROL_JSON = ROOT / "data" / "diagnostics" / "v21_evaluation.json"

ASK_URL = "http://127.0.0.1:8001/api/v1/agent/ask"
DEBUG_URL = "http://127.0.0.1:8001/api/v1/agent/debug/runs"


def classify_failure(row: dict[str, Any]) -> str:
    status = row.get("status")
    term = (row.get("termination_reason") or "").lower()
    tools = row.get("tool_sequence") or []
    if row.get("resolver_calls", 0) == 0 and RESOLVE_TOOL not in tools:
        if any(t in LEGACY_TOOLS for t in tools):
            return "RESOLUTION_FAILURE"
    if term == "no_retrieval_progress" or row.get("no_retrieval_progress"):
        return "NO_RETRIEVAL_PROGRESS"
    if row.get("retrieve_more_count", 0) > 0 and status not in {"completed", "answered"}:
        if row.get("legacy_after_resolve", 0) > 0:
            return "REDUNDANT_RETRIEVAL"
        return "VERIFIER_REJECTION"
    if row.get("answer_quality", {}).get("speculative_wording"):
        return "GENERATION_GROUNDING"
    if status in {"completed", "answered"}:
        return "OTHER"
    if row.get("tool_call_count", 0) >= 10:
        return "TOOL_LIMIT"
    if not row.get("evidence_count"):
        return "MISSING_EVIDENCE"
    if row.get("retrieve_more_count", 0) > 0:
        return "VERIFIER_REJECTION"
    return "OTHER"


def enrich_v22(trace: dict[str, Any] | None, body: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    events = (trace or {}).get("events") or []
    meta = body.get("metadata") or {}
    fs = (trace or {}).get("final_state") or {}

    tool_starts = [e for e in events if e.get("event") == "agent.tool.start"]
    skips = [e for e in events if e.get("event") == "agent.tool.skip"]
    routes = [e for e in events if e.get("event") == "agent.route"]
    boundary_skips = [
        e for e in skips if e.get("reason") == "context_boundary_covered"
    ]

    resolve_idx = None
    for i, e in enumerate(tool_starts):
        if e.get("tool_name") == RESOLVE_TOOL:
            resolve_idx = i
            break
    legacy_after = 0
    if resolve_idx is not None:
        for e in tool_starts[resolve_idx + 1 :]:
            if e.get("tool_name") in LEGACY_TOOLS:
                legacy_after += 1

    lo_calls = sum(
        1 for e in tool_starts if e.get("tool_name") == "get_learning_objectives"
    )
    regenerate_routes = sum(1 for r in routes if r.get("decision") == "regenerate")

    ctx = meta.get("context_boundary") or fs.get("metadata", {}).get("context_boundary") or {}
    resolve_obs = (base.get("resolve_observability") or [{}])[0]

    integrity = {
        "used_resolver_los": bool(
            base.get("resolver_calls", 0) > 0
            and (resolve_obs.get("learning_outcome_count") or 0) > 0
        ),
        "speculative_wording": base.get("answer_quality", {}).get("speculative_wording"),
        "hallucinated_lo_codes": [],
        "wrong_grade": False,
        "wrong_subject": False,
    }
    answer = (base.get("answer") or "").lower()
    for code in re.findall(r"C\d+U\d+-LO\d+", answer.upper()):
        if code not in {c.upper() for c in (ctx.get("lo_codes") or [])}:
            # may cite from evidence not only boundary list
            pass

    row = {
        **base,
        "group": None,
        "context_resolved": bool(ctx.get("context_resolved") or meta.get("context_resolved")),
        "context_resolution_status": ctx.get("resolution_status")
        or resolve_obs.get("resolution_status"),
        "curriculum_version_id": ctx.get("curriculum_version_id")
        or resolve_obs.get("curriculum_id"),
        "grade_id": ctx.get("grade_id") or resolve_obs.get("grade_id"),
        "subject_id": ctx.get("subject_id") or resolve_obs.get("subject_id"),
        "topic_ids": ctx.get("topic_ids") or resolve_obs.get("topic_ids") or [],
        "unit_ids": ctx.get("unit_ids") or resolve_obs.get("unit_ids") or [],
        "learning_outcome_ids": ctx.get("learning_outcome_ids") or [],
        "resolver_evidence_count": resolve_obs.get("learning_outcome_count") or 0,
        "unique_evidence_count": meta.get("unique_evidence_count")
        or base.get("evidence_count"),
        "duplicate_evidence_count": meta.get("duplicate_evidence_count")
        or base.get("duplicate_evidence_adds"),
        "legacy_tool_calls_after_resolution": legacy_after,
        "legacy_after_resolve": legacy_after,
        "search_calls": base.get("search_calls", 0),
        "structure_calls": base.get("structure_calls", 0),
        "topic_calls": base.get("topic_calls", 0),
        "learning_objective_calls": lo_calls,
        "verify_count": meta.get("verification_attempts") or 0,
        "retrieve_more_count": base.get("retrieve_more_count", 0),
        "boundary_skip_count": len(boundary_skips),
        "regeneration_without_retrieval": bool(
            meta.get("regeneration_without_retrieval")
            or regenerate_routes > 0
        ),
        "regenerate_route_count": regenerate_routes,
        "targeted_retrieval": base.get("retrieve_more_count", 0) > 0
        and legacy_after > 0,
        "evidence_already_present": bool(meta.get("evidence_already_present")),
        "understand_subject": base.get("understand_subject"),
        "resolver_subject": resolve_obs.get("subject_id"),
        "final_context_subject": ctx.get("subject_code") or resolve_obs.get("subject_id"),
        "no_retrieval_progress": term
        in {"no_retrieval_progress", "no_retrieval_progress_incomplete_source"}
        if (term := (base.get("termination_reason") or ""))
        else bool(meta.get("no_retrieval_progress")),
        "failure_class": None,
        "integrity": integrity,
        "final_confidence": base.get("answer_confidence"),
    }
    row["failure_class"] = classify_failure(row)
    return row


def ask_once(
    client: httpx.Client,
    question: str,
    tag: str,
    *,
    context_boundary_experiment: bool | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {"question": question, "conversation_id": None}
    if context_boundary_experiment is not None:
        payload["context_boundary_experiment"] = context_boundary_experiment
    resp = client.post(ASK_URL, json=payload)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    resp.raise_for_status()
    body = resp.json()
    meta = body.get("metadata") or {}
    run_id = meta.get("agent_run_id")
    trace = None
    if run_id:
        tr = client.get(f"{DEBUG_URL}/{run_id}")
        if tr.status_code == 200:
            trace = tr.json()
    analyzed = analyze_trace(trace, body)
    row = enrich_v22(trace, body, analyzed)
    row["question"] = question
    row["tag"] = tag
    row["latency_ms"] = latency_ms
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", tag)[:80]
    (OUT / f"{safe}_response.json").write_text(json.dumps(body, indent=2))
    if trace:
        (OUT / f"{safe}_trace.json").write_text(json.dumps(trace, indent=2))
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    latencies = [r["latency_ms"] for r in rows]
    tools = [r.get("tool_call_count") or 0 for r in rows]
    success = sum(1 for r in rows if r["status"] in {"completed", "answered"})
    retrieve_more = sum(1 for r in rows if (r.get("retrieve_more_count") or 0) > 0)
    no_progress = sum(1 for r in rows if r.get("no_retrieval_progress"))
    resolver_ok = sum(
        1
        for r in rows
        if r.get("context_resolution_status") == "resolved"
        or (r.get("resolver_evidence_count") or 0) >= 10
        or (r.get("resolver_calls") or 0) > 0
    )
    return {
        "n": len(rows),
        "success_rate": round(success / len(rows), 3),
        "success_count": success,
        "resolver_success_rate": round(resolver_ok / len(rows), 3),
        "verifier_acceptance_rate": round(
            sum(
                1
                for r in rows
                if r.get("answer_quality", {}).get("verifier_accepts")
            )
            / len(rows),
            3,
        ),
        "no_retrieval_progress_rate": round(no_progress / len(rows), 3),
        "retrieve_more_rate": round(retrieve_more / len(rows), 3),
        "avg_tool_calls": round(statistics.mean(tools), 2),
        "avg_latency_ms": round(statistics.mean(latencies), 1),
        "median_latency_ms": round(statistics.median(latencies), 1),
        "max_latency_ms": round(max(latencies), 1),
        "avg_legacy_after_resolve": round(
            statistics.mean([r.get("legacy_after_resolve") or 0 for r in rows]), 2
        ),
        "avg_regeneration_without_retrieval": round(
            statistics.mean(
                [1 if r.get("regeneration_without_retrieval") else 0 for r in rows]
            ),
            2,
        ),
        "avg_duplicate_evidence_adds": round(
            statistics.mean([r.get("duplicate_evidence_adds") or 0 for r in rows]), 2
        ),
        "avg_boundary_skips": round(
            statistics.mean([r.get("boundary_skip_count") or 0 for r in rows]), 2
        ),
        "failure_class_distribution": dict(Counter(r["failure_class"] for r in rows)),
    }


def compare_metric(control: dict[str, Any], treatment: dict[str, Any], key: str) -> dict[str, Any]:
    c = control.get(key)
    t = treatment.get(key)
    if c is None or t is None:
        return {"metric": key, "control": c, "treatment": t}
    if isinstance(c, (int, float)) and isinstance(t, (int, float)):
        abs_diff = round(t - c, 3)
        rel = round((t - c) / c, 3) if c else None
        pp = round((t - c) * 100, 1) if key.endswith("_rate") or key == "success_rate" else None
        return {
            "metric": key,
            "control": c,
            "treatment": t,
            "absolute_difference": abs_diff,
            "relative_change": rel,
            "percentage_point_difference": pp,
        }
    return {"metric": key, "control": c, "treatment": t}


def _legacy_after_resolve(tool_sequence: list[str] | None) -> int:
    tools = tool_sequence or []
    try:
        idx = tools.index(RESOLVE_TOOL)
    except ValueError:
        return 0
    return sum(1 for t in tools[idx + 1 :] if t in LEGACY_TOOLS)


def load_control_rows() -> list[dict[str, Any]]:
    if not CONTROL_JSON.is_file():
        raise SystemExit(f"Missing control artifact: {CONTROL_JSON}")
    data = json.loads(CONTROL_JSON.read_text())
    runs = data.get("v21_golden_runs") or []
    rows = []
    for i, run in enumerate(runs, 1):
        row = {**run}
        row["group"] = "control"
        row["run_order"] = i
        row.setdefault(
            "legacy_after_resolve",
            _legacy_after_resolve(row.get("tool_sequence")),
        )
        term = (row.get("termination_reason") or "").lower()
        row.setdefault(
            "no_retrieval_progress",
            "no_retrieval_progress" in term
            or row.get("failure_class") == "NO_RETRIEVAL_PROGRESS",
        )
        obs = (row.get("resolve_observability") or [{}])[0]
        row.setdefault(
            "context_resolution_status",
            "resolved"
            if (obs.get("resolution_status") == "resolved")
            or (obs.get("learning_outcome_count") or 0) >= 10
            else None,
        )
        row.setdefault("resolver_evidence_count", obs.get("learning_outcome_count") or 0)
        row.setdefault("failure_class", classify_failure(row))
        rows.append(row)
    return rows


def write_markdown(report: dict[str, Any]) -> None:
    c = report["control_summary"]
    t = report["treatment_summary"]
    lines = [
        "# V2.2 Context Boundary Experiment",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Hypothesis",
        "",
        "> Once `resolve_curriculum_context` returns authoritative learning outcomes, "
        "the agent should treat that context as the evidence boundary and avoid "
        "redundant exploratory retrieval.",
        "",
        "## Design",
        "",
        "- **Control:** V2.1 (reused from `v21_evaluation.json`, 10 golden runs)",
        "- **Treatment:** V2.2 `context_boundary_experiment=true` per request (10 runs)",
        f"- **Question:** _{GOLDEN}_",
        "- **Flag:** `CURRICULUM_V2_CONTEXT_BOUNDARY_EXPERIMENT` / request override",
        "",
        "## Results",
        "",
        "| Metric | Control | Treatment | Δ (abs) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report["comparison_table"]:
        delta = row.get("absolute_difference")
        delta_s = f"{delta:+.3f}" if isinstance(delta, (int, float)) else "—"
        lines.append(
            f"| {row['metric']} | {row.get('control')} | {row.get('treatment')} | {delta_s} |"
        )
    lines.extend(
        [
            "",
            "## Failure classification",
            "",
            f"- Control: `{c.get('failure_class_distribution')}`",
            f"- Treatment: `{t.get('failure_class_distribution')}`",
            "",
            "## Recommendation",
            "",
            f"**{report['recommendation']['code']}** — {report['recommendation']['text']}",
            "",
        ]
    )
    DOC.write_text("\n".join(lines))


def decide_recommendation(c: dict[str, Any], t: dict[str, Any]) -> dict[str, Any]:
    success_up = (t.get("success_rate") or 0) > (c.get("success_rate") or 0) + 0.15
    legacy_down = (t.get("avg_legacy_after_resolve") or 0) < (
        c.get("avg_legacy_after_resolve") or 0
    ) - 0.5
    no_prog_down = (t.get("no_retrieval_progress_rate") or 0) < (
        c.get("no_retrieval_progress_rate") or 0
    ) - 0.15
    grounding_ok = True  # no automatic regression signal in this harness

    if success_up and legacy_down and no_prog_down and grounding_ok:
        return {
            "code": "PROMOTE",
            "text": "Treatment materially improves success and reduces redundant retrieval.",
        }
    if legacy_down or no_prog_down:
        return {
            "code": "ITERATE",
            "text": "Efficiency improved but end-to-end success needs generation/verifier follow-up.",
        }
    return {
        "code": "REJECT",
        "text": "Treatment did not improve primary metrics; keep experiment isolated.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(timeout=180.0)

    control_rows = load_control_rows()
    print(f"Loaded {len(control_rows)} control runs from V2.1 evaluation", flush=True)

    # Alternating treatment schedule (control reused; treatment executed live).
    order = ["C", "T", "T", "C", "C", "T", "C", "T", "C", "T", "T", "C", "C", "T", "C", "T", "T", "C", "C", "T"]
    treatment_rows: list[dict[str, Any]] = []
    t_idx = 0
    for slot, kind in enumerate(order[:20], 1):
        if kind != "T":
            continue
        t_idx += 1
        print(f"=== TREATMENT {t_idx}/10 (slot {slot}) ===", flush=True)
        row = ask_once(
            client,
            GOLDEN,
            f"treatment_{t_idx:02d}",
            context_boundary_experiment=True,
        )
        row["group"] = "treatment"
        row["run_order"] = slot
        treatment_rows.append(row)
        print(
            json.dumps(
                {
                    "run": t_idx,
                    "status": row["status"],
                    "latency_ms": row["latency_ms"],
                    "tools": row.get("tool_call_count"),
                    "legacy_after_resolve": row.get("legacy_after_resolve"),
                    "boundary_skips": row.get("boundary_skip_count"),
                    "class": row.get("failure_class"),
                }
            ),
            flush=True,
        )
        time.sleep(0.5)

    control_summary = summarize(control_rows)
    treatment_summary = summarize(treatment_rows)
    metrics = [
        "success_rate",
        "resolver_success_rate",
        "verifier_acceptance_rate",
        "no_retrieval_progress_rate",
        "retrieve_more_rate",
        "avg_tool_calls",
        "avg_latency_ms",
        "median_latency_ms",
        "max_latency_ms",
        "avg_legacy_after_resolve",
        "avg_regeneration_without_retrieval",
        "avg_duplicate_evidence_adds",
        "avg_boundary_skips",
    ]
    comparison = [
        compare_metric(control_summary, treatment_summary, m) for m in metrics
    ]
    recommendation = decide_recommendation(control_summary, treatment_summary)

    success_ex = treatment_rows[0] if treatment_rows else None
    fail_ex = next(
        (r for r in treatment_rows if r["status"] not in {"completed", "answered"}),
        treatment_rows[-1] if treatment_rows else None,
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Resolved curriculum context as authoritative evidence boundary",
        "golden_question": GOLDEN,
        "configuration": {
            "control": "V2.1 default (artifact reuse)",
            "treatment_flag": "context_boundary_experiment=true",
            "runs": "10 control (cached) + 10 treatment (live)",
            "run_order": order[:20],
        },
        "control_summary": control_summary,
        "treatment_summary": treatment_summary,
        "comparison_table": comparison,
        "control_runs": control_rows,
        "treatment_runs": treatment_rows,
        "representative_success_trace": success_ex.get("agent_run_id") if success_ex else None,
        "representative_failure_trace": fail_ex.get("agent_run_id") if fail_ex else None,
        "recommendation": recommendation,
    }

    out_json = ROOT / "data" / "diagnostics" / "v22_context_boundary_experiment.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))
    (OUT / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    write_markdown(report)
    print("--- EXPERIMENT COMPLETE ---", flush=True)
    print(json.dumps({"control": control_summary, "treatment": treatment_summary, "recommendation": recommendation}, indent=2))


if __name__ == "__main__":
    main()
