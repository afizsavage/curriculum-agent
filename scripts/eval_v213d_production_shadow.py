#!/usr/bin/env python3
"""V2.13D production-shadow evaluation (replay + aggregate)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "diagnostics" / "v213d_shadow"
DOC = ROOT / "docs" / "V2_13D_PRODUCTION_SHADOW.md"
OUT_JSON = ROOT / "data" / "diagnostics" / "v213d_shadow_summary.json"
JSONL = ROOT / "data" / "diagnostics" / "v213d_shadow.jsonl"


def write_doc(report: dict) -> None:
    lines = [
        "# V2.13D — Production Shadow for Context-Hybrid Curriculum Document Evidence",
        "",
        f"Generated: `{report.get('generated_at')}`",
        "",
        f"**Canary recommendation: {report.get('canary_recommendation')}**",
        "",
        report.get("canary_note", ""),
        "",
        "## Executive summary",
        "",
        "V2.13D is a **shadow-only** evaluation. Production LangGraph responses are unchanged. "
        "Document retrieval runs after the user response is determined, in a failure-isolated thread.",
        "",
        report.get("v213c_comparison_note", ""),
        "",
        "## Replay vs production traffic",
        "",
        "This report is from **controlled local replay** (Phase 0), not observed production traffic. "
        "Do not treat these rates as equivalent to V2.13C's 72-question harness.",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps({k: v for k, v in report.items() if k not in {"records", "generated_at"}}, indent=2),
        "```",
        "",
        "## Safety gates",
        "",
        "```json",
        json.dumps(report.get("safety_metrics", {}), indent=2),
        "```",
        "",
        "## Production integrity",
        "",
        "- LangGraph production path unchanged (shadow scheduled after response)",
        "- Answer generator / verifier / mapper / V2.9 / V2.11 unchanged",
        "- `/api/v1` unchanged",
        "- `v213d_shadow_enabled=false`, `v213d_shadow_sample_rate=0` by default",
        "- Shadow exceptions cannot propagate to the production caller",
        "",
        "## Recommendation",
        "",
        report.get("canary_recommendation", "CANARY_NOT_READY"),
        "",
        "Do **not** automatically enable document retrieval or increase sample rate.",
        "",
        "Initial production shadow configuration if operators choose to collect traffic:",
        "",
        "```text",
        "v213d_shadow_enabled=true",
        "v213d_shadow_sample_rate=0.01",
        "v213d_shadow_document_retrieval=true",
        "v213d_shadow_retrieval_variant=context_hybrid",
        "```",
    ]
    DOC.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="V2.13D production-shadow eval")
    parser.add_argument("--replay", action="store_true", default=True)
    args = parser.parse_args()

    from app.agent.orchestrator import CurriculumQAAgent
    from app.agent.v213d_shadow import (
        aggregate_records,
        persist_record,
        prepare_replay_corpus,
        replay_fixtures,
    )
    from app.config import Settings
    from app.llm.provider import StubLLMProvider

    OUT.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        v213d_shadow_enabled=True,
        v213d_shadow_sample_rate=1.0,
        llm_provider="stub",
    )
    agent = CurriculumQAAgent(settings=settings, llm=StubLLMProvider())
    retrieval = prepare_replay_corpus(OUT / "documents", OUT / "index")
    records = replay_fixtures(
        agent,
        retrieval=retrieval,
        inject_failures={"V213C-F01": "retrieval"},
    )
    if args.replay:
        JSONL.parent.mkdir(parents=True, exist_ok=True)
        if JSONL.exists():
            JSONL.write_text("")
        for record in records:
            persist_record(record, JSONL)
    summary = aggregate_records(records)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["mode"] = "controlled_replay"
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    write_doc(summary)
    print(f"Canary: {summary['canary_recommendation']}")
    print(f"Report: {DOC}")
    print(f"JSON: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
