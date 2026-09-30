#!/usr/bin/env python3
"""V2.13G TARGETED_REPLAY dual-arm validation (separate from LIVE_TRAFFIC).

Runs legitimate eval questions through CurriculumQAAgent.ask (same control path),
then forces dual-arm shadow for every case. Does NOT raise production sample_rate
and does NOT enable V2.13F production arbitration.

Writes only to data/diagnostics/v213g_arbitration_targeted_replay.jsonl.
Never mixes into live hypothesis n without an explicit separate summary.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "evals" / "v213c_curriculum_qa.json"
OUT_JSONL = ROOT / "data" / "diagnostics" / "v213g_arbitration_targeted_replay.jsonl"
OUT_SUMMARY = ROOT / "data" / "diagnostics" / "v213g_targeted_replay_summary.json"


def load_questions(categories: list[str] | None) -> list[dict]:
    data = json.loads(EVAL.read_text())
    qs = list(data.get("questions") or [])
    allow = {c.strip() for c in (categories or []) if c.strip()} or None
    if allow is None:
        return qs
    return [q for q in qs if str(q.get("category") or "") in allow]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument(
        "--categories",
        default="structured_fact,structured_plus_document,source_grounding",
    )
    parser.add_argument("--out-jsonl", type=Path, default=OUT_JSONL)
    parser.add_argument("--out-summary", type=Path, default=OUT_SUMMARY)
    parser.add_argument(
        "--target-sufficient-with-docs",
        type=int,
        default=20,
        help="Stop early once this many VALID sufficient+docs rows are written",
    )
    args = parser.parse_args()

    from app.agent.orchestrator import CurriculumQAAgent
    from app.agent.v213g_live_shadow import (
        persist_v213g_record,
        probe_structured_api,
        run_dual_arm_shadow_pipeline,
    )
    from app.config import get_settings

    settings = get_settings()
    if bool(getattr(settings, "v213f_document_arbitration_experiment", False)):
        raise SystemExit("Refuse: V213F production arbitration must stay false")
    if float(getattr(settings, "v213d_shadow_sample_rate", 0.0) or 0.0) > 0.01 + 1e-9:
        raise SystemExit("Refuse: sample_rate must remain 0.01")

    infra = probe_structured_api(settings)
    if not infra.get("structured_api_available"):
        raise SystemExit(f"Refuse: structured API unavailable: {infra}")

    cats = [c.strip() for c in str(args.categories or "").split(",") if c.strip()]
    questions = load_questions(cats)
    if not questions:
        raise SystemExit(f"No questions for categories={cats}")

    batch: list[dict] = []
    i = 0
    while len(batch) < args.count:
        batch.append(questions[i % len(questions)])
        i += 1

    agent = CurriculumQAAgent(settings=settings)
    written = 0
    sufficient = 0
    role_counts: Counter[str] = Counter()
    suff_counts: Counter[str] = Counter()
    errors = 0
    started = time.perf_counter()

    print(
        json.dumps(
            {
                "event": "targeted_replay_start",
                "count": len(batch),
                "categories": cats,
                "target_sufficient_with_docs": args.target_sufficient_with_docs,
                "structured_api": infra,
                "isolation": {
                    "v213f_document_arbitration_experiment": False,
                    "v213d_shadow_sample_rate": settings.v213d_shadow_sample_rate,
                    "v213g_live_arbitration_shadow": settings.v213g_live_arbitration_shadow,
                },
            }
        ),
        flush=True,
    )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    for idx, q in enumerate(batch, start=1):
        request_id = f"v213g-tr-{uuid.uuid4().hex[:12]}"
        question = str(q.get("question") or "")
        t0 = time.perf_counter()
        try:
            # Keep production 1% shadow off for this harness so TARGETED_REPLAY
            # owns the dual-arm write (no double LLM, no source mixing).
            with patch(
                "app.agent.v213d_shadow.maybe_schedule_v213d_shadow",
                return_value=None,
            ):
                state = agent.ask(question, request_id=request_id)
            record = run_dual_arm_shadow_pipeline(
                agent, state, request_id=request_id
            )
            record["source"] = "TARGETED_REPLAY"
            record["batch_id"] = "batch_2_targeted_replay"
            record["traffic_class"] = "V213G_TARGETED_REPLAY"
            record["eval_category"] = q.get("category")
            record["eval_id"] = q.get("id")
            infra_now = probe_structured_api(settings)
            record["infrastructure"] = {
                **(record.get("infrastructure") or {}),
                **infra_now,
            }
            if not infra_now.get("structured_api_available"):
                record["infrastructure"]["validity"] = "INFRASTRUCTURE_INVALID"
            persist_v213g_record(record, path=args.out_jsonl)
            written += 1
            arb = record.get("arbitration") or {}
            suff = str(arb.get("structured_sufficiency") or "")
            role = str(arb.get("document_role") or "")
            docs = int(arb.get("document_count") or 0)
            sc = int(arb.get("structured_count") or 0)
            suff_counts[suff] += 1
            role_counts[role] += 1
            valid = (
                (record.get("infrastructure") or {}).get("validity")
                != "INFRASTRUCTURE_INVALID"
            )
            if valid and suff == "SUFFICIENT" and docs > 0:
                sufficient += 1
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "done": idx,
                        "total": len(batch),
                        "written": written,
                        "sufficient_with_docs": sufficient,
                        "structured_count": sc,
                        "document_count": docs,
                        "sufficiency": suff,
                        "role": role,
                        "control_accepted": (record.get("control") or {}).get(
                            "final_accepted"
                        ),
                        "latency_s": round(time.perf_counter() - t0, 1),
                        "question_category": q.get("category"),
                    }
                ),
                flush=True,
            )
            if sufficient >= args.target_sufficient_with_docs:
                print(
                    json.dumps(
                        {
                            "event": "target_met",
                            "sufficient_with_docs": sufficient,
                            "written": written,
                        }
                    ),
                    flush=True,
                )
                break
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(
                json.dumps(
                    {
                        "event": "error",
                        "done": idx,
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_s": round(time.perf_counter() - t0, 1),
                        "question_category": q.get("category"),
                    }
                ),
                flush=True,
            )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "TARGETED_REPLAY",
        "batch_id": "batch_2_targeted_replay",
        "n_written": written,
        "errors": errors,
        "structured_sufficient_with_docs": sufficient,
        "sufficiency_counts": dict(suff_counts),
        "role_counts": dict(role_counts),
        "elapsed_s": round(time.perf_counter() - started, 1),
        "jsonl": str(args.out_jsonl),
        "note": (
            "Separate from LIVE_TRAFFIC. Do not merge into live n without "
            "explicit dual reporting."
        ),
        "production_isolation": {
            "v213f_document_arbitration_experiment": False,
            "v213d_shadow_sample_rate": float(settings.v213d_shadow_sample_rate),
            "forced_dual_arm": True,
            "sample_rate_unchanged": True,
        },
    }
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", **summary}, indent=2), flush=True)
    return 0 if sufficient >= args.target_sufficient_with_docs else 2


if __name__ == "__main__":
    raise SystemExit(main())
