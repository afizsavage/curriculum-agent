#!/usr/bin/env python3
"""Build a frozen V2.13F replay dataset from real post-corpus shadow JSONL."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def post_ok(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("replay_id"):
            continue
        if r.get("corpus_epoch") == "pre_corpus":
            continue
        if (r.get("comparison") or {}).get("classification") == "DOCUMENT_CORPUS_UNAVAILABLE":
            continue
        if (r.get("shadow") or {}).get("error"):
            continue
        out.append(r)
    return out


def assign_group(record: dict) -> str | None:
    control = record.get("control") or {}
    shadow = record.get("shadow") or {}
    comparison = record.get("comparison") or {}
    docs = int(shadow.get("document_evidence_count") or 0)
    c_ok = bool(control.get("final_accepted"))
    s_ok = bool(shadow.get("final_accepted"))
    if comparison.get("control_correct_shadow_worse") or (c_ok and not s_ok):
        return "D_regression"
    if c_ok and docs > 0:
        return "A_structured_sufficient"
    if (not c_ok) and s_ok:
        return "B_structured_insufficient_recovery"
    if docs > 0 and c_ok == s_ok:
        return "C_neutral"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=ROOT / "data" / "diagnostics" / "v213d_shadow.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "diagnostics" / "v213f_replay" / "dataset.json",
    )
    parser.add_argument("--min-size", type=int, default=30)
    parser.add_argument("--max-size", type=int, default=50)
    args = parser.parse_args()

    rows = post_ok(load_rows(args.jsonl))
    buckets: dict[str, list[dict]] = {
        "D_regression": [],
        "B_structured_insufficient_recovery": [],
        "A_structured_sufficient": [],
        "C_neutral": [],
    }
    for r in rows:
        g = assign_group(r)
        if g:
            buckets[g].append(r)

    selected: list[dict] = []
    seen_keys: set[str] = set()

    def _key(r: dict) -> str:
        # Prefer distinct shadow observations (request_id); fall back to question hash.
        rid = str(r.get("request_id") or "")
        if rid:
            return f"rid:{rid}"
        return f"qh:{(r.get('question') or {}).get('hash')}"

    # Always include all regressions first.
    for r in buckets["D_regression"]:
        k = _key(r)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        selected.append({**r, "v213f_group": "D_regression"})
    for r in buckets["B_structured_insufficient_recovery"]:
        if len(selected) >= args.max_size:
            break
        k = _key(r)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        selected.append({**r, "v213f_group": "B_structured_insufficient_recovery"})
    for r in buckets["A_structured_sufficient"]:
        if len(selected) >= args.max_size:
            break
        k = _key(r)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        selected.append({**r, "v213f_group": "A_structured_sufficient"})
    for r in buckets["C_neutral"]:
        if len(selected) >= args.max_size:
            break
        k = _key(r)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        selected.append({**r, "v213f_group": "C_neutral"})

    # Prefer at least min_size if available.
    dataset = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_jsonl": str(args.jsonl),
        "post_corpus_pool": len(rows),
        "group_pool_sizes": {k: len(v) for k, v in buckets.items()},
        "n": len(selected),
        "min_size": args.min_size,
        "max_size": args.max_size,
        "target_met": len(selected) >= args.min_size,
        "group_counts": {
            g: sum(1 for c in selected if c.get("v213f_group") == g)
            for g in buckets
        },
        "cases": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dataset, indent=2))
    print(
        json.dumps(
            {
                "path": str(args.out),
                "n": dataset["n"],
                "group_pool_sizes": dataset["group_pool_sizes"],
                "selected_groups": {
                    g: sum(1 for c in selected if c.get("v213f_group") == g)
                    for g in buckets
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
