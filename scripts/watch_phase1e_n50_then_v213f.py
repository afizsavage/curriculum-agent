#!/usr/bin/env python3
"""Watch Phase 1E shadows until post_corpus_n >= target, then freeze + V2.13F eval."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data" / "diagnostics" / "v213d_shadow.jsonl"
TARGET = int(os.environ.get("V213F_N50_TARGET", "50"))
TRAFFIC_PID = int(os.environ.get("V213F_TRAFFIC_PID", "0") or 0)


def post_ok_count() -> int:
    if not JSONL.is_file():
        return 0
    n = 0
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("replay_id"):
            continue
        if r.get("corpus_epoch") == "pre_corpus":
            continue
        if (r.get("comparison") or {}).get("classification") == "DOCUMENT_CORPUS_UNAVAILABLE":
            continue
        if (r.get("shadow") or {}).get("error"):
            continue
        n += 1
    return n


def stop_traffic() -> None:
    # Prefer explicit PID; else match the PHASE1E_TO_N50 generator.
    pids = []
    if TRAFFIC_PID:
        pids.append(TRAFFIC_PID)
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "generate_v213d_phase1_traffic.py.*PHASE1E_TO_N50"],
            text=True,
        )
        pids.extend(int(x) for x in out.split() if x.strip().isdigit())
    except subprocess.CalledProcessError:
        pass
    for pid in sorted(set(pids)):
        try:
            os.kill(pid, signal.SIGTERM)
            print(json.dumps({"event": "stopped_traffic", "pid": pid}), flush=True)
        except ProcessLookupError:
            pass


def main() -> int:
    print(json.dumps({"event": "watch_start", "target": TARGET, "post_ok": post_ok_count()}), flush=True)
    stagnant = 0
    last_n = post_ok_count()
    while True:
        n = post_ok_count()
        traffic_alive = False
        try:
            subprocess.check_output(
                ["pgrep", "-f", "generate_v213d_phase1_traffic.py"],
                text=True,
            )
            traffic_alive = True
        except subprocess.CalledProcessError:
            traffic_alive = False
        print(
            json.dumps(
                {
                    "event": "progress",
                    "post_ok": n,
                    "target": TARGET,
                    "traffic_alive": traffic_alive,
                }
            ),
            flush=True,
        )
        if n >= TARGET:
            break
        if n == last_n:
            stagnant += 1
        else:
            stagnant = 0
            last_n = n
        # If traffic died before target, exit so an operator can restart.
        if not traffic_alive and stagnant >= 3:
            print(
                json.dumps(
                    {
                        "event": "traffic_dead",
                        "post_ok": n,
                        "target": TARGET,
                        "note": "Restart generate_v213d_phase1_traffic.py and this watcher.",
                    }
                ),
                flush=True,
            )
            return 2
        time.sleep(60)
    stop_traffic()
    # Give in-flight asks a moment to finish.
    time.sleep(45)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    steps = [
        [sys.executable, "scripts/eval_v213d_production_shadow.py", "--from-jsonl"],
        [sys.executable, "scripts/freeze_v213d_baseline_for_v213f.py"],
        [
            sys.executable,
            "scripts/build_v213f_replay_dataset.py",
            "--jsonl",
            "data/diagnostics/v213f_baseline/v213d_shadow_freeze.jsonl",
            "--min-size",
            "30",
            "--max-size",
            "50",
        ],
        [sys.executable, "scripts/eval_v213f_document_arbitration.py"],
    ]
    for cmd in steps:
        print(json.dumps({"event": "run", "cmd": cmd}), flush=True)
        subprocess.check_call(cmd, cwd=ROOT, env=env)
    print(json.dumps({"event": "complete", "post_ok": post_ok_count()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
