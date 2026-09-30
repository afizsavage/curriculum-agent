#!/usr/bin/env python3
"""Watch V2.13G until healthy valid LIVE_TRAFFIC rows reach the n>=50 milestone.

Stage-1 structured_sufficient_with_docs (target 20) is recorded but does not
stop collection. Reaching n>=50 does not enable production arbitration.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# First healthy-live milestone. Not the sufficient-with-docs target of 20.
TARGET = int(os.environ.get("V213G_HEALTHY_TARGET", "50"))


def coverage() -> dict:
    from app.agent.v213g_live_shadow import count_sufficient_with_docs, load_v213g_records

    rows = load_v213g_records()
    return count_sufficient_with_docs(rows)


def stop_traffic() -> None:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "generate_v213d_phase1_traffic.py.*V213G_HEALTHY"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return
    for pid in sorted({int(x) for x in out.split() if x.strip().isdigit()}):
        try:
            os.kill(pid, signal.SIGTERM)
            print(json.dumps({"event": "stopped_traffic", "pid": pid}), flush=True)
        except ProcessLookupError:
            pass


def main() -> int:
    print(
        json.dumps(
            {
                "event": "watch_start",
                "target_metric": "healthy_valid",
                "target": TARGET,
                **coverage(),
            }
        ),
        flush=True,
    )
    stagnant = 0
    last = int(coverage().get("healthy_valid") or coverage().get("valid_healthy") or 0)
    while True:
        cov = coverage()
        n = int(cov.get("healthy_valid") or cov.get("valid_healthy") or 0)
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
                    "target_metric": "healthy_valid",
                    "target": TARGET,
                    "traffic_alive": traffic_alive,
                    **cov,
                }
            ),
            flush=True,
        )
        if n >= TARGET:
            break
        if n == last:
            stagnant += 1
        else:
            stagnant = 0
            last = n
        if not traffic_alive and stagnant >= 3:
            print(
                json.dumps(
                    {
                        "event": "traffic_dead",
                        "note": "Restart traffic and this watcher.",
                        **cov,
                    }
                ),
                flush=True,
            )
            return 2
        time.sleep(60)
    stop_traffic()
    time.sleep(30)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.check_call(
        [sys.executable, "scripts/eval_v213g_live_arbitration_shadow.py"],
        cwd=ROOT,
        env=env,
    )
    subprocess.check_call(
        [sys.executable, "scripts/eval_v213g_cohort_coverage.py"],
        cwd=ROOT,
        env=env,
    )
    print(json.dumps({"event": "complete", **coverage()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
