#!/usr/bin/env python3
"""Generate controlled real QA traffic through POST /api/v1/agent/ask.

Does NOT write shadow JSONL directly. Does NOT change sample rate.
Records are only Phase 1 evidence if V2.13D samples them via the live agent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "evals" / "v213c_curriculum_qa.json"
OUT = ROOT / "data" / "diagnostics" / "v213d_phase1_traffic_run.json"


def load_questions() -> list[dict]:
    data = json.loads(EVAL.read_text())
    return list(data.get("questions") or [])


def pick_batch(questions: list[dict], n: int) -> list[dict]:
    """Round-robin across categories for a representative mix."""
    by_cat: dict[str, list[dict]] = {}
    for q in questions:
        by_cat.setdefault(str(q.get("category") or "unknown"), []).append(q)
    cats = sorted(by_cat.keys())
    selected: list[dict] = []
    idx = {c: 0 for c in cats}
    while len(selected) < n and cats:
        progressed = False
        for cat in cats:
            bucket = by_cat[cat]
            if not bucket:
                continue
            selected.append(bucket[idx[cat] % len(bucket)])
            idx[cat] += 1
            progressed = True
            if len(selected) >= n:
                break
        if not progressed:
            break
    return selected


def ask(base: str, question: str, timeout: float = 180.0) -> dict:
    payload = json.dumps(
        {"question": question, "conversation_id": None}
    ).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/api/v1/agent/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return {
                "ok": True,
                "http_status": resp.status,
                "status": body.get("status"),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "conversation_id": body.get("conversation_id"),
                "answer_present": bool(body.get("answer")),
            }
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "http_status": exc.code,
            "error": str(exc),
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "http_status": None,
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": (time.perf_counter() - started) * 1000,
        }


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    questions = load_questions()
    batch = pick_batch(questions, args.count)
    health_before = fetch_json(f"{args.base_url}/health")
    metrics_before = fetch_json(f"{args.base_url}/api/v1/agent/metrics")
    jsonl = ROOT / "data" / "diagnostics" / "v213d_shadow.jsonl"
    rows_before = (
        sum(1 for line in jsonl.read_text().splitlines() if line.strip())
        if jsonl.exists()
        else 0
    )

    print(
        json.dumps(
            {
                "event": "traffic_start",
                "count": len(batch),
                "concurrency": args.concurrency,
                "qa_before": metrics_before.get("total_requests"),
                "shadow_rows_before": rows_before,
                "categories": dict(Counter(q.get("category") for q in batch)),
            }
        ),
        flush=True,
    )

    results: list[dict] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {
            pool.submit(ask, args.base_url, q["question"], args.timeout): q
            for q in batch
        }
        done = 0
        for fut in as_completed(futures):
            q = futures[fut]
            row = fut.result()
            row["id"] = q.get("id")
            row["category"] = q.get("category")
            row["question_hash"] = __import__("hashlib").sha256(
                str(q.get("question") or "").encode()
            ).hexdigest()[:16]
            results.append(row)
            done += 1
            if done % 10 == 0 or done == len(batch):
                ok = sum(1 for r in results if r.get("ok"))
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "done": done,
                            "total": len(batch),
                            "ok": ok,
                            "elapsed_s": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )

    health_after = fetch_json(f"{args.base_url}/health")
    metrics_after = fetch_json(f"{args.base_url}/api/v1/agent/metrics")
    rows_after = (
        sum(1 for line in jsonl.read_text().splitlines() if line.strip())
        if jsonl.exists()
        else 0
    )
    report = {
        "traffic_class": "CONTROLLED_REAL_QA",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "requested": len(batch),
        "ok": sum(1 for r in results if r.get("ok")),
        "failed": sum(1 for r in results if not r.get("ok")),
        "categories": dict(Counter(q.get("category") for q in batch)),
        "qa_requests_before": metrics_before.get("total_requests"),
        "qa_requests_after": metrics_after.get("total_requests"),
        "shadow_rows_before": rows_before,
        "shadow_rows_after": rows_after,
        "health_before_v213d": health_before.get("v213d"),
        "health_after_v213d": health_after.get("v213d"),
        "funnel_after": (health_after.get("v213d_pipeline") or {}).get("stages"),
        "traffic_after": health_after.get("v213d_traffic"),
        "mean_latency_ms": (
            sum(float(r.get("latency_ms") or 0) for r in results if r.get("ok"))
            / max(sum(1 for r in results if r.get("ok")), 1)
        ),
        "note": (
            "Requests went through POST /api/v1/agent/ask only. "
            "Shadow rows appear only if V2.13D sampled them at configured rate."
        ),
        "results": results,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"event": "traffic_complete", "path": str(OUT), **{k: report[k] for k in (
        "ok","failed","qa_requests_after","shadow_rows_after","funnel_after","traffic_after"
    )}}, indent=2), flush=True)
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
