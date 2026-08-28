#!/usr/bin/env python3
"""Golden evaluation after V2.3 evidence-conservative productionization."""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

GOLDEN = "What are the learning objectives for fractions in Primary 4?"
ASK_URL = "http://127.0.0.1:8001/api/v1/agent/ask"
DEBUG_URL = "http://127.0.0.1:8001/api/v1/agent/debug/runs"
OUT = ROOT / "data" / "diagnostics" / "v23_productionization"
DOC = ROOT / "docs" / "V2_3_PRODUCTIONIZATION.md"
OUT_JSON = ROOT / "data" / "diagnostics" / "v23_productionization.json"

BASELINES = {
    "v22_treatment": {"verifier_acceptance_rate": 0.3, "success_rate": 0.3},
    "v23_constrained": {"verifier_acceptance_rate": 0.6, "success_rate": 0.6},
}


def _speculative(answer: str) -> bool:
    return bool(
        re.search(
            r"\blikely\b|\bprobably\b|\bthis means\b|\bappears to mean\b",
            answer or "",
            re.I,
        )
    )


def _unsupported_absence(answer: str) -> bool:
    if re.search(
        r"\b(?:resolved|supplied|available)\s+evidence\s+does not include\b",
        answer or "",
        re.I,
    ):
        return False
    return bool(
        re.search(
            r"\b(?:there are|there is)\s+no\s+(?:learning\s+outcomes?|los?)\b",
            answer or "",
            re.I,
        )
    )


def analyze_run(trace: dict[str, Any] | None, body: dict[str, Any]) -> dict[str, Any]:
    events = (trace or {}).get("events") or []
    ver_end = next(
        (e for e in reversed(events) if e.get("event") == "agent.verification.end"),
        {},
    )
    gen_diag = next(
        (e for e in reversed(events) if e.get("event") == "agent.generation.diagnostics"),
        {},
    )
    ver = body.get("verification") or {}
    meta = body.get("metadata") or {}
    final = (trace or {}).get("final") or {}
    draft = next(
        (e.get("answer") for e in reversed(events) if e.get("event") == "agent.verification.start"),
        "",
    )
    answer = draft or body.get("answer") or ""
    accepted = ver_end.get("recommendation") == "accept" or ver.get("passed")
    return {
        "status": body.get("status") or final.get("status"),
        "verifier_decision": ver_end.get("recommendation") or ver.get("recommendation"),
        "verifier_score": ver_end.get("score") or ver.get("score"),
        "verifier_accepted": accepted,
        "success": body.get("status") in {"completed", "answered"},
        "tool_calls": final.get("tool_calls") or meta.get("tool_calls"),
        "latency_ms": final.get("latency_ms"),
        "generation_policy": gen_diag.get("generation_policy") or meta.get("generation_policy"),
        "unsupported_claims": ver_end.get("unsupported_claims") or [],
        "speculative_wording": _speculative(answer),
        "unsupported_absence": _unsupported_absence(answer),
        "truncation_warning_count": gen_diag.get("truncation_warning_count"),
        "agent_run_id": meta.get("agent_run_id"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    accepted = sum(1 for r in rows if r.get("verifier_accepted"))
    success = sum(1 for r in rows if r.get("success"))
    scores = [r["verifier_score"] for r in rows if r.get("verifier_score") is not None]
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    return {
        "n": n,
        "success_rate": round(success / n, 3),
        "success_count": success,
        "verifier_acceptance_rate": round(accepted / n, 3),
        "verifier_acceptance_count": accepted,
        "avg_verifier_score": round(statistics.mean(scores), 3) if scores else None,
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "avg_tool_calls": round(
            statistics.mean([r.get("tool_calls") or 0 for r in rows]), 2
        ),
        "unsupported_claims_total": sum(len(r.get("unsupported_claims") or []) for r in rows),
        "speculative_claims_runs": sum(1 for r in rows if r.get("speculative_wording")),
        "unsupported_absence_runs": sum(1 for r in rows if r.get("unsupported_absence")),
        "truncation_warning_runs": sum(
            1 for r in rows if (r.get("truncation_warning_count") or 0) > 0
        ),
        "generation_policy": rows[0].get("generation_policy") if rows else None,
    }


def conclusion(after: dict[str, Any]) -> str:
    target = BASELINES["v23_constrained"]["verifier_acceptance_rate"]
    acc = after.get("verifier_acceptance_rate") or 0
    if acc >= target - 0.1:
        return "REPRODUCED"
    if acc >= BASELINES["v22_treatment"]["verifier_acceptance_rate"] + 0.15:
        return "PARTIALLY REPRODUCED"
    return "REGRESSION"


def write_doc(report: dict[str, Any]) -> None:
    after = report["after"]
    lines = [
        "# V2.3 Productionization — Evidence-Conservative Generation",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Change",
        "",
        "Production `AnswerGenerator` now applies `generation_policy=evidence_conservative`.",
        "Verifier, resolver, and retrieval unchanged.",
        "",
        "## Golden Question",
        "",
        f"_{GOLDEN}_ — 10 production runs (`context_boundary_experiment: true`).",
        "",
        "## Before (baselines)",
        "",
        "| Source | Verifier acceptance | Success |",
        "| --- | ---: | ---: |",
        f"| V2.2 treatment | 30% | 30% |",
        f"| V2.3 constrained experiment | 60% | 60% |",
        "",
        "## After (productionized)",
        "",
        f"| Metric | Value |",
        f"| --- | ---: |",
        f"| Verifier acceptance | {after.get('verifier_acceptance_rate')} ({after.get('verifier_acceptance_count')}/{after.get('n')}) |",
        f"| End-to-end success | {after.get('success_rate')} |",
        f"| Unsupported claims (verifier) | {after.get('unsupported_claims_total')} |",
        f"| Speculative wording runs | {after.get('speculative_claims_runs')} |",
        f"| Unsupported absence runs | {after.get('unsupported_absence_runs')} |",
        f"| Truncation warning runs | {after.get('truncation_warning_runs')} |",
        f"| Avg latency (ms) | {after.get('avg_latency_ms')} |",
        f"| Avg tool calls | {after.get('avg_tool_calls')} |",
        "",
        f"## Conclusion: **{report['conclusion']}**",
        "",
        report.get("conclusion_note", ""),
    ]
    DOC.write_text("\n".join(lines))


def main() -> None:
    import httpx

    OUT.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(timeout=180.0)
    rows: list[dict[str, Any]] = []

    for i in range(1, 11):
        print(f"=== PRODUCTION RUN {i}/10 ===", flush=True)
        started = time.perf_counter()
        resp = client.post(
            ASK_URL,
            json={
                "question": GOLDEN,
                "context_boundary_experiment": True,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        run_id = (body.get("metadata") or {}).get("agent_run_id")
        trace = None
        if run_id:
            tr = client.get(f"{DEBUG_URL}/{run_id}")
            if tr.status_code == 200:
                trace = tr.json()
        row = analyze_run(trace, body)
        row["tag"] = f"prod_{i:02d}"
        row["wall_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        rows.append(row)
        (OUT / f"prod_{i:02d}_response.json").write_text(json.dumps(body, indent=2))
        if trace:
            (OUT / f"prod_{i:02d}_trace.json").write_text(json.dumps(trace, indent=2))
        print(json.dumps(row), flush=True)
        time.sleep(0.5)

    after = summarize(rows)
    conc = conclusion(after)
    note = (
        "Productionized generation meets or approaches V2.3 constrained experiment."
        if conc == "REPRODUCED"
        else "Improved over V2.2 but below V2.3 constrained target."
        if conc == "PARTIALLY REPRODUCED"
        else "Did not improve over baselines; review traces."
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "golden_question": GOLDEN,
        "configuration": {
            "production": True,
            "context_boundary_experiment": True,
            "v23_diagnostic_experiment": False,
            "generation_policy": "evidence_conservative",
            "runs": 10,
        },
        "baselines": BASELINES,
        "after": after,
        "runs": rows,
        "conclusion": conc,
        "conclusion_note": note,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    write_doc(report)
    print("--- V2.3 PRODUCTIONIZATION EVAL ---", flush=True)
    print(json.dumps({"after": after, "conclusion": conc}, indent=2), flush=True)


if __name__ == "__main__":
    main()
