#!/usr/bin/env python3
"""Five-run fractions LO benchmark for state-aware retrieval."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

QUESTION = "What are the learning objectives for fractions in Primary 4?"
OUT = Path(__file__).resolve().parents[1] / "data" / "diagnostics" / "retrieval_state_bench"
ASK_URL = "http://127.0.0.1:8001/api/v1/agent/ask"
DEBUG_URL = "http://127.0.0.1:8001/api/v1/agent/debug/runs"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    client = httpx.Client(timeout=180.0)
    for i in range(1, 6):
        started = time.perf_counter()
        resp = client.post(ASK_URL, json={"question": QUESTION, "conversation_id": None})
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
                (OUT / f"run_{i}_trace.json").write_text(
                    json.dumps(trace, indent=2)
                )
        (OUT / f"run_{i}_response.json").write_text(json.dumps(body, indent=2))

        events = (trace or {}).get("events") or []
        tool_starts = [
            e
            for e in events
            if e.get("event") == "agent.tool.start" and not e.get("skipped")
        ]
        skips = [e for e in events if e.get("event") == "agent.tool.skip"]
        plans = [e for e in events if e.get("event") == "agent.retrieval.plan"]
        targeted = sum(
            1
            for e in plans
            if e.get("reason") and "Verifier" in str(e.get("reason"))
        )
        row = {
            "run": i,
            "agent_run_id": run_id,
            "status": body.get("status"),
            "latency_ms": latency_ms,
            "tool_calls": meta.get("tool_calls"),
            "retrieval_rounds": meta.get("retrieval_rounds"),
            "verification_attempts": meta.get("verification_attempts"),
            "verification_status": meta.get("verification_status"),
            "termination_reason": meta.get("termination_reason")
            or meta.get("fallback_reason"),
            "evidence_count": meta.get("evidence_count"),
            "verification_score": meta.get("verification_score"),
            "duplicate_tool_skips": len(skips),
            "tool_starts": len(tool_starts),
            "tools": [
                (e.get("tool_name"), e.get("iteration")) for e in tool_starts
            ],
            "targeted_plan_events": targeted,
            "retrieval_metrics": meta.get("retrieval_state"),
            "visited_nodes": meta.get("visited_nodes"),
        }
        rows.append(row)
        print(json.dumps(row, default=str))

    summary = {
        "question": QUESTION,
        "runs": rows,
        "success_count": sum(
            1 for r in rows if r["status"] in {"completed", "answered"}
        ),
        "avg_latency_ms": round(
            sum(r["latency_ms"] for r in rows) / max(len(rows), 1), 1
        ),
        "avg_tool_calls": round(
            sum((r["tool_calls"] or 0) for r in rows) / max(len(rows), 1), 2
        ),
        "avg_duplicate_skips": round(
            sum(r["duplicate_tool_skips"] for r in rows) / max(len(rows), 1), 2
        ),
        "avg_retrieval_rounds": round(
            sum((r["retrieval_rounds"] or 0) for r in rows) / max(len(rows), 1),
            2,
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
