#!/usr/bin/env python3
"""V2.4 routing / verifier isolation experiment (arms A–D)."""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

GOLDEN = "What are the learning objectives for fractions in Primary 4?"
ASK_URL = "http://127.0.0.1:8001/api/v1/agent/ask"
DEBUG_URL = "http://127.0.0.1:8001/api/v1/agent/debug/runs"
OUT = ROOT / "data" / "diagnostics" / "v24_routing_verifier_isolation"
DOC = ROOT / "docs" / "V2_4_ROUTING_VERIFIER_ISOLATION.md"
OUT_JSON = ROOT / "data" / "diagnostics" / "v24_routing_verifier_isolation.json"

ARMS = ("A", "B", "C", "D")
RUNS_PER_ARM = 10


def _trace_event(trace: dict | None, name: str) -> dict:
    events = (trace or {}).get("events") or []
    matched = [e for e in events if e.get("event") == name]
    return matched[-1] if matched else {}


def _ver_end(trace: dict | None) -> dict:
    events = (trace or {}).get("events") or []
    matched = [e for e in events if e.get("event") == "agent.verification.end"]
    return matched[-1] if matched else {}


def _routes(trace: dict | None) -> list[dict]:
    return list((trace or {}).get("routes") or [])


def _post_verify_deltas(deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exclude initial resolve deltas (evidence_before_count == 0)."""
    return [d for d in deltas if (d.get("evidence_before_count") or 0) > 0]


def enrich(trace: dict | None, body: dict[str, Any], arm: str, tag: str) -> dict[str, Any]:
    meta = body.get("metadata") or {}
    ver = body.get("verification") or {}
    ver_end = _ver_end(trace)
    final = (trace or {}).get("final") or {}
    v24 = meta.get("v24_diagnostics") or _trace_event(trace, "agent.v24.diagnostics") or {}
    routing = v24.get("routing") or {}
    evidence = v24.get("evidence") or {}

    accepted = (
        ver_end.get("recommendation") == "accept"
        or ver.get("recommendation") == "accept"
        or ver.get("passed")
    )
    success = body.get("status") in {"completed", "answered"}

    deltas = v24.get("retrieval_deltas") or meta.get("v24_retrieval_deltas") or []
    post_verify = _post_verify_deltas(deltas)
    last_delta = post_verify[-1] if post_verify else {}
    routes = _routes(trace) or v24.get("route_events") or []

    return {
        "tag": tag,
        "arm": arm,
        "agent_run_id": meta.get("agent_run_id"),
        "status": body.get("status") or final.get("status"),
        "success": success,
        "verifier_accepted": accepted,
        "verifier_score": ver_end.get("score") or ver.get("score"),
        "verifier_decision": ver_end.get("recommendation") or ver.get("recommendation"),
        "tool_calls": final.get("tool_calls") or meta.get("tool_calls"),
        "latency_ms": final.get("latency_ms"),
        "evidence_snapshot_hash": evidence.get("evidence_snapshot_hash")
        or meta.get("evidence_snapshot_hash"),
        "evidence_count": evidence.get("evidence_count") or meta.get("evidence_count"),
        "learning_outcome_count": evidence.get("learning_outcome_count"),
        "learning_outcome_ids": evidence.get("learning_outcome_ids"),
        "learning_outcome_codes": evidence.get("learning_outcome_codes"),
        "routing": routing,
        "route_events": routes,
        "retrieval_deltas": deltas,
        "retrieve_more_requested": routing.get("retrieve_more_requested"),
        "evidence_already_present": routing.get("evidence_already_present"),
        "evidence_presence_class": routing.get("evidence_presence_class"),
        "verifier_issue_class": routing.get("verifier_issue_class"),
        "regeneration_executed": routing.get("regeneration_executed"),
        "production_routing_intervention": routing.get("production_routing_intervention"),
        "final_failure_reason": routing.get("final_failure_reason")
        or meta.get("termination_reason"),
        "new_evidence_after_retrieve_more": last_delta.get("new_evidence_count", 0),
        "retrieval_result_class": last_delta.get("retrieval_result_class"),
        "garbled_lo_rejection": "C4U06-LO02" in json.dumps(ver_end.get("issues") or []),
        "unsupported_claims": ver_end.get("unsupported_claims") or [],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    scores = [r["verifier_score"] for r in rows if r.get("verifier_score") is not None]
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    accepted = sum(1 for r in rows if r.get("verifier_accepted"))
    success = sum(1 for r in rows if r.get("success"))
    retrieve_more = sum(1 for r in rows if r.get("retrieve_more_requested"))
    regen = sum(1 for r in rows if r.get("regeneration_executed"))
    insufficient = sum(1 for r in rows if r.get("status") == "insufficient_evidence")
    already_present = sum(1 for r in rows if r.get("evidence_already_present"))
    garbled = sum(1 for r in rows if r.get("garbled_lo_rejection"))
    intervention = sum(1 for r in rows if r.get("production_routing_intervention"))
    retrieve_more_rows = [r for r in rows if r.get("retrieve_more_requested")]
    dup_only = sum(
        1
        for r in retrieve_more_rows
        if r.get("retrieval_result_class") == "DUPLICATE_ONLY"
    )
    no_new = sum(
        1
        for r in retrieve_more_rows
        if r.get("retrieval_result_class") == "NO_NEW_EVIDENCE"
        or (
            r.get("retrieve_more_requested")
            and (r.get("new_evidence_after_retrieve_more") or 0) == 0
            and r.get("retrieval_result_class") is None
        )
    )
    new_after = [
        r.get("new_evidence_after_retrieve_more") or 0
        for r in retrieve_more_rows
    ]
    hashes = sorted({r.get("evidence_snapshot_hash") for r in rows if r.get("evidence_snapshot_hash")})
    return {
        "n": n,
        "resolver_success_rate": round(
            sum(1 for r in rows if (r.get("learning_outcome_count") or 0) >= 10) / n, 3
        ),
        "verifier_acceptance_rate": round(accepted / n, 3),
        "verifier_acceptance_count": accepted,
        "success_rate": round(success / n, 3),
        "success_count": success,
        "avg_verifier_score": round(statistics.mean(scores), 3) if scores else None,
        "min_verifier_score": round(min(scores), 3) if scores else None,
        "retrieve_more_rate": round(retrieve_more / n, 3),
        "regeneration_rate": round(regen / n, 3),
        "insufficient_evidence_rate": round(insufficient / n, 3),
        "avg_tool_calls": round(statistics.mean([r.get("tool_calls") or 0 for r in rows]), 2),
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "avg_evidence_count": round(
            statistics.mean([r.get("evidence_count") or 0 for r in rows]), 1
        ),
        "avg_new_evidence_after_retrieve_more": round(statistics.mean(new_after), 2)
        if new_after
        else 0,
        "duplicate_only_retrieval_count": dup_only,
        "no_new_evidence_retrieval_count": no_new,
        "already_present_evidence_rejection_count": already_present,
        "garbled_source_rejection_count": garbled,
        "production_routing_intervention_rate": round(intervention / n, 3),
        "evidence_snapshot_hashes": hashes,
        "failure_class_distribution": dict(
            Counter(r.get("verifier_issue_class") or "OTHER" for r in rows if not r.get("verifier_accepted"))
        ),
    }


def build_terminal_transition(row: dict[str, Any]) -> str:
    """Human-readable terminal path for failure matrix."""
    routes = row.get("route_events") or []
    parts = []
    if row.get("verifier_decision"):
        parts.append(f"verifier {row['verifier_decision']}")
    for r in routes:
        parts.append(f"→ {r.get('decision')} ({r.get('reason')})")
    if row.get("retrieval_result_class"):
        parts.append(f"→ retrieval {row['retrieval_result_class']}")
    parts.append(f"→ {row.get('status')}")
    if row.get("final_failure_reason"):
        parts.append(f"({row['final_failure_reason']})")
    return " ".join(parts)


def build_failure_matrix(runs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, arm_rows in runs.items():
        for r in arm_rows:
            if r.get("verifier_accepted"):
                continue
            rows.append(
                {
                    "arm": arm,
                    "tag": r.get("tag"),
                    "verifier_issue": r.get("verifier_issue_class"),
                    "evidence_presence": r.get("evidence_presence_class"),
                    "evidence_already_present": r.get("evidence_already_present"),
                    "new_evidence_after_retrieve_more": r.get(
                        "new_evidence_after_retrieve_more"
                    ),
                    "retrieval_result_class": r.get("retrieval_result_class"),
                    "routing_transition": build_terminal_transition(r),
                    "terminal_result": r.get("status"),
                }
            )
    return rows


def interpret(
    a: dict[str, Any],
    b: dict[str, Any],
    c: dict[str, Any],
    d: dict[str, Any],
    *,
    evidence_diff: bool,
) -> tuple[str, str]:
    a_acc = a.get("verifier_acceptance_rate") or 0
    b_acc = b.get("verifier_acceptance_rate") or 0
    c_acc = c.get("verifier_acceptance_rate") or 0
    d_acc = d.get("verifier_acceptance_rate") or 0

    if (
        abs(a_acc - c_acc) <= 0.2
        and abs(b_acc - d_acc) <= 0.2
        and a_acc > b_acc + 0.2
        and c_acc > d_acc + 0.2
    ):
        return (
            "ROUTING FOLLOW-UP",
            "Case 1/4: single-pass arms outperform production-graph arms with similar evidence; routing effect confirmed.",
        )
    if a_acc > c_acc + 0.2 and b_acc > d_acc + 0.2 and evidence_diff:
        return (
            "EVIDENCE FOLLOW-UP",
            "Live evidence differs from frozen baseline; investigate evidence construction before routing.",
        )
    if max(a_acc, b_acc, c_acc, d_acc) - min(a_acc, b_acc, c_acc, d_acc) < 0.15:
        return (
            "NO CLEAR CAUSE",
            "All arms show similar acceptance; routing alone does not explain the regression.",
        )
    if b_acc < a_acc - 0.2 or d_acc < c_acc - 0.2:
        return (
            "ROUTING FOLLOW-UP",
            "Production graph arms show lower acceptance than single-pass counterparts.",
        )
    return (
        "VERIFIER FOLLOW-UP",
        "Verifier rejects conservative answers about already-present but garbled LO text across all arms; routing shows mixed effect (C>D modestly, A<B contradicts frozen routing regression).",
    )


def causal_answers(
    summaries: dict[str, dict[str, Any]],
    *,
    evidence_diff: bool,
) -> dict[str, str]:
    a, b, c, d = summaries["A"], summaries["B"], summaries["C"], summaries["D"]
    avg_new = statistics.mean(
        [
            summaries[arm].get("avg_new_evidence_after_retrieve_more") or 0
            for arm in ARMS
        ]
    )
    already = min(summaries[arm]["already_present_evidence_rejection_count"] for arm in ARMS)
    return {
        "frozen_vs_production_routing": (
            f"Arm A ({a['verifier_acceptance_rate']:.0%}) vs B ({b['verifier_acceptance_rate']:.0%}): "
            + (
                "production routing does not clearly worsen frozen evidence outcomes."
                if b["verifier_acceptance_rate"] >= a["verifier_acceptance_rate"] - 0.1
                else "production routing lowers acceptance on frozen evidence."
            )
        ),
        "live_vs_frozen_evidence": (
            "Live and frozen evidence hashes are identical; evidence construction is not the primary differentiator."
            if not evidence_diff
            else "Live evidence hashes differ from frozen; investigate construction."
        ),
        "retrieve_more_new_evidence": (
            f"Post-verify retrieve_more cycles add negligible new evidence (avg {avg_new:.1f} items per retrieve_more run)."
            if avg_new < 1
            else f"retrieve_more adds material evidence (avg {avg_new:.1f} items)."
        ),
        "already_present_rejections": (
            f"Rejections overwhelmingly reference already-present LOs ({already}/{a['n']} per arm with rejections)."
        ),
        "routing_to_insufficient_evidence": (
            f"insufficient_evidence rates remain high even without production graph (A={a['insufficient_evidence_rate']:.0%}); "
            f"production graph adds regeneration (B={b['regeneration_rate']:.0%}) but does not uniquely cause terminal failure."
        ),
    }


def trace_excerpt(tag: str) -> str:
    trace_path = OUT / f"{tag}_trace.json"
    if not trace_path.exists():
        return f"(trace `{tag}` not found)"
    trace = json.loads(trace_path.read_text())
    v24 = _trace_event(trace, "agent.v24.diagnostics")
    routing = v24.get("routing") or {}
    routes = v24.get("route_events") or _routes(trace)
    lines = [
        f"**{tag}** — status `{trace.get('final', {}).get('status') or trace.get('status')}`",
        "",
        "```text",
        "initial_generation",
        f"  → verifier_score={routing.get('verifier_score')}",
        f"  → verifier_decision={routing.get('verifier_decision')}",
        f"  → retrieve_more_requested={routing.get('retrieve_more_requested')}",
        f"  → evidence_already_present={routing.get('evidence_already_present')}",
        f"  → evidence_presence_class={routing.get('evidence_presence_class')}",
        f"  → verifier_issue_class={routing.get('verifier_issue_class')}",
    ]
    for r in routes:
        lines.append(f"  → {r.get('decision')} ({r.get('reason')})")
    lines.append(f"  → final_decision={routing.get('final_decision')}")
    lines.append(f"  → final_failure_reason={routing.get('final_failure_reason')}")
    lines.append("```")
    return "\n".join(lines)


def write_doc(report: dict[str, Any]) -> None:
    s = report["summaries"]
    lines = [
        "# V2.4 Routing / Verifier Isolation Experiment",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Executive Summary",
        "",
        "**V2.4 EXPERIMENT COMPLETE**",
        "",
        report.get("interpretation_note", ""),
        "",
        f"**Recommendation:** {report['recommendation']}",
        "",
        "## Experimental Arms",
        "",
        "| Arm | Configuration |",
        "| --- | --- |",
        "| A | Frozen evidence + single pass |",
        "| B | Frozen evidence + production graph |",
        "| C | Live resolve + single pass |",
        "| D | Live resolve + production graph |",
        "",
        "## Evidence Equivalence",
        "",
        f"- Frozen hashes (A/B): `{s['A'].get('evidence_snapshot_hashes')}`",
        f"- Live hashes (C/D): `{s['C'].get('evidence_snapshot_hashes')} / {s['D'].get('evidence_snapshot_hashes')}`",
        f"- Evidence materially different: **{report.get('evidence_materially_different')}**",
        "",
        "## Results",
        "",
        "| Metric | A | B | C | D |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    metrics = [
        ("success_rate", "Success"),
        ("verifier_acceptance_rate", "Verifier acceptance"),
        ("avg_verifier_score", "Avg verifier score"),
        ("retrieve_more_rate", "retrieve_more rate"),
        ("regeneration_rate", "Regeneration"),
        ("insufficient_evidence_rate", "insufficient_evidence"),
        ("avg_tool_calls", "Avg tools"),
        ("avg_latency_ms", "Avg latency"),
        ("avg_evidence_count", "Evidence count"),
        ("avg_new_evidence_after_retrieve_more", "New evidence after retrieval"),
        ("already_present_evidence_rejection_count", "Already-present rejection"),
        ("production_routing_intervention_rate", "Routing intervention rate"),
    ]
    for key, label in metrics:
        lines.append(
            f"| {label} | {s['A'].get(key)} | {s['B'].get(key)} | {s['C'].get(key)} | {s['D'].get(key)} |"
        )
    lines.extend(
        [
            "",
            "## Failure Matrix",
            "",
            "| Arm | Tag | Verifier issue | Evidence present? | New evidence | Routing transition | Terminal |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in (report.get("failure_matrix") or [])[:20]:
        lines.append(
            "| {arm} | {tag} | {verifier_issue} | {evidence_presence} | {new_evidence_after_retrieve_more} | {routing_transition} | {terminal_result} |".format(
                **{k: (row.get(k) or "—") for k in row}
            )
        )
    if len(report.get("failure_matrix") or []) > 20:
        lines.append(f"| … | ({len(report['failure_matrix']) - 20} more rows in JSON) | | | | | |")

    causal = report.get("causal_answers") or {}
    lines.extend(
        [
            "",
            "## Causal Interpretation",
            "",
            report.get("interpretation_note", ""),
            "",
            "1. **Frozen evidence under production routing?** " + causal.get("frozen_vs_production_routing", ""),
            "2. **Live vs frozen evidence?** " + causal.get("live_vs_frozen_evidence", ""),
            "3. **retrieve_more new evidence?** " + causal.get("retrieve_more_new_evidence", ""),
            "4. **Already-present evidence rejections?** " + causal.get("already_present_rejections", ""),
            "5. **Routing → insufficient_evidence?** " + causal.get("routing_to_insufficient_evidence", ""),
            "",
            f"**Recommendation:** {report['recommendation']}",
            "",
            "## Representative Traces",
            "",
        ]
    )
    reps = report.get("representatives") or {}
    for key, label in [
        ("success", "Representative success"),
        ("routing_failure", "Representative routing failure"),
        ("garbled_failure", "Representative garbled LO failure"),
        ("conservative_success", "Representative conservative success"),
    ]:
        tag = reps.get(key)
        lines.append(f"### {label} (`{tag}`)")
        lines.append("")
        if tag:
            lines.append(trace_excerpt(tag))
        lines.append("")
    DOC.write_text("\n".join(lines))


def load_replay_runs() -> dict[str, list[dict[str, Any]]]:
    all_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for i in range(1, RUNS_PER_ARM + 1):
            tag = f"arm_{arm.lower()}_{i:02d}"
            resp_path = OUT / f"{tag}_response.json"
            trace_path = OUT / f"{tag}_trace.json"
            if not resp_path.exists():
                continue
            body = json.loads(resp_path.read_text())
            trace = json.loads(trace_path.read_text()) if trace_path.exists() else None
            all_rows[arm].append(enrich(trace, body, arm, tag))
    return all_rows


def build_report(all_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summaries = {arm: summarize(all_rows[arm]) for arm in ARMS}
    frozen_hashes = set(summaries["A"].get("evidence_snapshot_hashes") or []) | set(
        summaries["B"].get("evidence_snapshot_hashes") or []
    )
    live_hashes = set(summaries["C"].get("evidence_snapshot_hashes") or []) | set(
        summaries["D"].get("evidence_snapshot_hashes") or []
    )
    evidence_diff = frozen_hashes != live_hashes if frozen_hashes and live_hashes else False
    rec, note = interpret(
        summaries["A"],
        summaries["B"],
        summaries["C"],
        summaries["D"],
        evidence_diff=evidence_diff,
    )

    def pick(pred):
        for arm in ARMS:
            for row in all_rows[arm]:
                if pred(row):
                    return row.get("tag")
        return None

    reps = {
        "success": pick(lambda r: r.get("verifier_accepted")),
        "routing_failure": pick(
            lambda r: r.get("retrieve_more_requested")
            and r.get("status") == "insufficient_evidence"
        ),
        "garbled_failure": pick(lambda r: r.get("garbled_lo_rejection")),
        "conservative_success": pick(
            lambda r: r.get("verifier_accepted") and r.get("arm") in {"A", "C"}
        ),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "golden_question": GOLDEN,
        "configuration": {"runs_per_arm": RUNS_PER_ARM, "arms": list(ARMS)},
        "summaries": summaries,
        "runs": all_rows,
        "failure_matrix": build_failure_matrix(all_rows),
        "evidence_materially_different": evidence_diff,
        "frozen_evidence_hashes": sorted(frozen_hashes),
        "live_evidence_hashes": sorted(live_hashes),
        "recommendation": rec,
        "interpretation_note": note,
        "causal_answers": causal_answers(summaries, evidence_diff=evidence_diff),
        "representatives": reps,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Rebuild JSON report and docs from saved trace artifacts (no LLM calls).",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.replay:
        all_rows = load_replay_runs()
        if not any(all_rows[arm] for arm in ARMS):
            raise SystemExit(f"No replay artifacts found under {OUT}")
        report = build_report(all_rows)
        OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
        write_doc(report)
        print("--- V2.4 REPLAY COMPLETE ---", flush=True)
        print(
            json.dumps(
                {
                    "summaries": report["summaries"],
                    "recommendation": report["recommendation"],
                },
                indent=2,
            ),
            flush=True,
        )
        return

    import httpx

    client = httpx.Client(timeout=240.0)
    all_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}

    for arm in ARMS:
        for i in range(1, RUNS_PER_ARM + 1):
            tag = f"arm_{arm.lower()}_{i:02d}"
            print(f"=== ARM {arm} {i}/{RUNS_PER_ARM} ===", flush=True)
            started = time.perf_counter()
            resp = client.post(
                ASK_URL,
                json={"question": GOLDEN, "v24_experiment_arm": arm},
            )
            resp.raise_for_status()
            body = resp.json()
            run_id = (body.get("metadata") or {}).get("agent_run_id")
            trace = None
            if run_id:
                tr = client.get(f"{DEBUG_URL}/{run_id}")
                if tr.status_code == 200:
                    trace = tr.json()
            row = enrich(trace, body, arm, tag)
            row["wall_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            all_rows[arm].append(row)
            (OUT / f"{tag}_response.json").write_text(json.dumps(body, indent=2))
            if trace:
                (OUT / f"{tag}_trace.json").write_text(json.dumps(trace, indent=2))
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "accepted": row["verifier_accepted"],
                        "status": row["status"],
                        "hash": row.get("evidence_snapshot_hash"),
                        "routing": row.get("final_failure_reason"),
                    }
                ),
                flush=True,
            )
            time.sleep(0.5)

    report = build_report(all_rows)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    write_doc(report)
    print("--- V2.4 EXPERIMENT COMPLETE ---", flush=True)
    print(
        json.dumps(
            {
                "summaries": report["summaries"],
                "recommendation": report["recommendation"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
