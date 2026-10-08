"""Measurement-only timing of the live curriculum structure API.

Does not change pool mode, timeouts, or the QA agent. Reads the server
timing log written by the structure API.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

SERVER_LOG = Path(
    "/home/afiz/Projects/curriculumz/curriculum-structure/data/request_timing.jsonl"
)
OUT = Path(__file__).parent
IDENTITY = "http://127.0.0.1:8000/api/v1/curricula?code=MBSSE-SSC&limit=20"
AGENT = "http://127.0.0.1:8001/api/v1/agent/ask"
CLIENT_TIMEOUT = 15.0


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return round(ordered[index], 1)


def _stats(rows: list[dict], key: str) -> dict:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return {
        "n": len(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": round(max(values), 1) if values else None,
    }


def _load_server() -> list[dict]:
    if not SERVER_LOG.exists():
        return []
    return [json.loads(line) for line in SERVER_LOG.read_text().splitlines() if line.strip()]


def _request(url: str, request_id: str, timeout: float, body: bytes | None = None) -> dict:
    started = time.perf_counter()
    headers = {"X-Request-ID": request_id, "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers)
    status = None
    error = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = f"HTTPError:{exc.code}"
    except Exception as exc:
        error = type(exc).__name__
    return {
        "request_id": request_id,
        "client_ms": round((time.perf_counter() - started) * 1000, 1),
        "client_status": status,
        "client_error": error,
        "client_timeout_s": timeout,
    }


def _run_level(prefix: str, count: int, workers: int) -> list[dict]:
    ids = [f"{prefix}-{index:03d}-{uuid4().hex[:8]}" for index in range(count)]

    def one(request_id: str) -> dict:
        return _request(IDENTITY, request_id, CLIENT_TIMEOUT)

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, request_id) for request_id in ids]
        for future in as_completed(futures):
            rows.append(future.result())
    return rows


def _dominant(row: dict) -> str:
    parts = {
        "connection checkout": row.get("checkout_ms") or 0,
        "SQL execution": row.get("sql_ms") or 0,
        "fetch/materialization": row.get("fetch_ms") or 0,
        "application processing": row.get("application_ms") or 0,
    }
    name, value = max(parts.items(), key=lambda item: item[1])
    total = row.get("total_ms") or 0
    if total >= 15000 and value < total * 0.5:
        return "unknown"
    return name


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before = len(_load_server())
    client_rows: list[dict] = []
    print("A sequential", flush=True)
    sequential = _run_level("A", 30, 1)
    for row in sequential:
        row["phase"] = "A"
        row["concurrency"] = 1
    client_rows.extend(sequential)

    for workers, count in ((2, 16), (4, 16), (8, 16)):
        print(f"B concurrency={workers}", flush=True)
        rows = _run_level(f"B{workers}", count, workers)
        for row in rows:
            row["phase"] = "B"
            row["concurrency"] = workers
        client_rows.extend(rows)

    print("D disconnect probe", flush=True)
    # Production client timeout is 15s, then the identity retry starts at once.
    # Also abort one request early so cancellation is observed even when the
    # endpoint answers before 15s. The server timeout is not changed.
    disconnect_id = f"D-abort15-{uuid4().hex[:8]}"
    started = time.perf_counter()
    aborted = _request(IDENTITY, disconnect_id, CLIENT_TIMEOUT)
    aborted["phase"] = "D"
    aborted["concurrency"] = 1
    aborted["probe"] = "client_timeout_15s"
    client_rows.append(aborted)
    overlap_id = f"D-overlap-{uuid4().hex[:8]}"
    overlap = _request(IDENTITY, overlap_id, CLIENT_TIMEOUT)
    overlap["phase"] = "D"
    overlap["concurrency"] = 1
    overlap["started_after_abort_ms"] = round((time.perf_counter() - started) * 1000, 1)
    client_rows.append(overlap)
    early_id = None
    early = None
    for attempt in range(4):
        early_id = f"D-early-{attempt}-{uuid4().hex[:8]}"
        early_started = time.perf_counter()
        early = _request(IDENTITY, early_id, 2.0)
        early["phase"] = "D"
        early["concurrency"] = 1
        early["probe"] = "client_abort_2s"
        client_rows.append(early)
        follow_id = f"D-early-follow-{attempt}-{uuid4().hex[:8]}"
        follow = _request(IDENTITY, follow_id, CLIENT_TIMEOUT)
        follow["phase"] = "D"
        follow["concurrency"] = 1
        follow["probe"] = "after_2s_abort"
        follow["started_after_abort_ms"] = round((time.perf_counter() - early_started) * 1000, 1)
        follow["pairs_with"] = early_id
        client_rows.append(follow)
        if early["client_error"]:
            break

    questions = [
        "What does Biology cover for SSS1 Sciences & Technologies?",
        "What does Calculus cover for SSS1 Mathematics & Numeracy?",
        "What does History of Sierra Leone cover for SSS1 Social & Cultural Studies?",
        "What does Principles of Accounting cover for SSS3 Economics, Business & Entrepreneurship?",
    ]
    print("C agent path", flush=True)
    agent_rows = []
    if os.environ.get("SKIP_AGENT_PATH") == "1":
        questions = []
    for index, question in enumerate(questions, start=1):
        request_id = f"C-{index}-{uuid4().hex[:8]}"
        body = json.dumps({"question": question}).encode()
        observed = _request(AGENT, request_id, 180.0, body)
        observed["phase"] = "C"
        observed["question"] = question
        agent_rows.append(observed)
        client_rows.append(observed)
        print(f"C {index} {observed['client_ms']}ms {observed['client_error']}", flush=True)

    deadline = time.perf_counter() + 45
    expected_ids = {row["request_id"] for row in client_rows if row.get("phase") == "D"}
    while time.perf_counter() < deadline:
        seen = {row.get("request_id") for row in _load_server()}
        if expected_ids <= seen:
            break
        time.sleep(0.5)
    time.sleep(0.5)
    server = _load_server()[before:]
    by_id = {row["request_id"]: row for row in server}
    joined = []
    for client in client_rows:
        server_row = by_id.get(client["request_id"])
        # Agent calls use their own curriculum request ids. Keep client rows
        # and attach server rows for direct identity calls.
        merged = dict(client)
        if server_row:
            merged.update({f"server_{key}": value for key, value in server_row.items()})
            merged["server"] = server_row
        joined.append(merged)

    def server_for(prefix: str, concurrency: int | None = None) -> list[dict]:
        rows = []
        for client in joined:
            if not str(client["request_id"]).startswith(prefix):
                continue
            if concurrency is not None and client.get("concurrency") != concurrency:
                continue
            if client.get("server"):
                rows.append(client["server"])
        return rows

    identity_table = []
    for label, concurrency, rows in (
        ("1 (sequential)", 1, server_for("A-")),
        ("2", 2, server_for("B2-")),
        ("4", 4, server_for("B4-")),
        ("8", 8, server_for("B8-")),
    ):
        identity_table.append(
            {
                "concurrency": label,
                "n": len(rows),
                "checkout_p50": _stats(rows, "checkout_ms")["p50"],
                "checkout_p95": _stats(rows, "checkout_ms")["p95"],
                "sql_p50": _stats(rows, "sql_ms")["p50"],
                "sql_p95": _stats(rows, "sql_ms")["p95"],
                "total_p50": _stats(rows, "total_ms")["p50"],
                "total_p95": _stats(rows, "total_ms")["p95"],
                "timeouts_ge_15s": sum(1 for row in rows if (row.get("total_ms") or 0) >= 15000),
                "checkout_still_open": sum(1 for row in rows if row.get("checkout_still_open")),
                "connection_returned": sum(1 for row in rows if row.get("connection_returned")),
            }
        )

    route_rows: dict[str, list[dict]] = {}
    for row in server:
        route_rows.setdefault(row.get("route") or row.get("path"), []).append(row)
    route_table = []
    for route, rows in sorted(route_rows.items(), key=lambda item: -len(item[1])):
        route_table.append(
            {
                "route": route,
                "n": len(rows),
                "checkout_p50": _stats(rows, "checkout_ms")["p50"],
                "checkout_p95": _stats(rows, "checkout_ms")["p95"],
                "sql_p50": _stats(rows, "sql_ms")["p50"],
                "sql_p95": _stats(rows, "sql_ms")["p95"],
                "total_p50": _stats(rows, "total_ms")["p50"],
                "total_p95": _stats(rows, "total_ms")["p95"],
            }
        )

    slow = [row for row in server if (row.get("total_ms") or 0) >= 10000]
    for row in slow:
        row["dominant_delay"] = _dominant(row)
    abort = by_id.get(disconnect_id)
    follow = by_id.get(overlap_id)
    overlap_window = None
    if abort and follow:
        # ISO timestamps plus durations tell us whether the aborted request
        # was still running when the next request started.
        overlap_window = {
            "abort_request_id": disconnect_id,
            "abort_client_ms": aborted["client_ms"],
            "abort_server_total_ms": abort.get("total_ms"),
            "abort_checkout_ms": abort.get("checkout_ms"),
            "abort_sql_ms": abort.get("sql_ms"),
            "abort_client_disconnected": abort.get("client_disconnected"),
            "abort_cancelled_while_waiting": abort.get("cancelled_while_waiting"),
            "abort_connection_acquired": abort.get("connection_acquired"),
            "abort_connection_returned": abort.get("connection_returned"),
            "abort_checkout_still_open": abort.get("checkout_still_open"),
            "abort_dominant": _dominant(abort),
            "follow_server_total_ms": follow.get("total_ms"),
            "follow_checkout_ms": follow.get("checkout_ms"),
            "follow_started_after_abort_client_ms": overlap.get("started_after_abort_ms"),
        }

    client_vs_server = []
    for client in joined:
        server_row = client.get("server")
        if not server_row:
            continue
        client_vs_server.append(
            {
                "request_id": client["request_id"],
                "client_ms": client["client_ms"],
                "server_total_ms": server_row.get("total_ms"),
                "gap_ms": round(client["client_ms"] - (server_row.get("total_ms") or 0), 1),
            }
        )
    gaps = [row["gap_ms"] for row in client_vs_server]

    summary = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "pool_mode": (server[0].get("pool_mode") if server else None),
        "pool_class": (server[0].get("pool_class") if server else None),
        "db_host": (server[0].get("db_host") if server else None),
        "db_port": (server[0].get("db_port") if server else None),
        "measurement_client_timeout_s": CLIENT_TIMEOUT,
        "identity_table": identity_table,
        "route_table": route_table,
        "slow_requests": slow,
        "disconnect_probe": overlap_window,
        "disconnect_rows": [
            {
                "request_id": row.get("request_id"),
                "probe": next(
                    (
                        client.get("probe")
                        for client in joined
                        if client["request_id"] == row.get("request_id")
                    ),
                    None,
                ),
                "client_ms": next(
                    (
                        client.get("client_ms")
                        for client in joined
                        if client["request_id"] == row.get("request_id")
                    ),
                    None,
                ),
                "client_error": next(
                    (
                        client.get("client_error")
                        for client in joined
                        if client["request_id"] == row.get("request_id")
                    ),
                    None,
                ),
                "server_total_ms": row.get("total_ms"),
                "checkout_ms": row.get("checkout_ms"),
                "sql_ms": row.get("sql_ms"),
                "client_disconnected": row.get("client_disconnected"),
                "disconnect_ms": row.get("disconnect_ms"),
                "handler_continued_after_disconnect": row.get(
                    "handler_continued_after_disconnect"
                ),
                "cancelled_while_waiting": row.get("cancelled_while_waiting"),
                "connection_acquired": row.get("connection_acquired"),
                "connection_returned": row.get("connection_returned"),
                "checkout_still_open": row.get("checkout_still_open"),
                "dominant": _dominant(row),
            }
            for row in server
            if str(row.get("request_id", "")).startswith("D-")
        ],
        "client_server_gap_p50": _percentile(gaps, 50),
        "client_server_gap_p95": _percentile(gaps, 95),
        "agent_requests": agent_rows,
        "server_rows": len(server),
    }
    (OUT / "client_observations_15s.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in joined)
    )
    (OUT / "summary_15s.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in summary if key != "slow_requests"}, indent=2), flush=True)
    print("TIMING_DONE", flush=True)


if __name__ == "__main__":
    main()
