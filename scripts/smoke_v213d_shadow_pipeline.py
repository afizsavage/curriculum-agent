#!/usr/bin/env python3
"""V2.13D Phase 1 pipeline smoke test (NOT production traffic).

Forces 100% sampling in a test-only override and writes to a dedicated JSONL.
Never writes to data/diagnostics/v213d_shadow.jsonl.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SMOKE_JSONL = ROOT / "data" / "diagnostics" / "v213d_shadow_smoke.jsonl"
SMOKE_FUNNEL = ROOT / "data" / "diagnostics" / "v213d_pipeline_funnel_smoke.json"
SMOKE_TRAFFIC = ROOT / "data" / "diagnostics" / "v213d_traffic_smoke.json"
SMOKE_CORPUS = ROOT / "data" / "diagnostics" / "v213d_shadow_smoke_corpus"


def main() -> int:
    from unittest.mock import patch

    from app.agent.orchestrator import CurriculumQAAgent
    from app.agent.state import CurriculumQAState
    from app.agent.v213d_shadow import (
        maybe_schedule_v213d_shadow,
        prepare_replay_corpus,
        run_production_shadow,
    )
    from app.config import Settings
    from app.llm.provider import StubLLMProvider

    settings = Settings(
        _env_file=None,
        v213d_shadow_enabled=True,
        v213d_shadow_sample_rate=1.0,  # test-only force sample
        llm_provider="stub",
    )
    agent = CurriculumQAAgent(settings=settings, llm=StubLLMProvider())
    retrieval = prepare_replay_corpus(SMOKE_CORPUS / "documents", SMOKE_CORPUS / "index")
    state = CurriculumQAState.initial(
        question="What does the MBSSE curriculum say about the purpose of mathematics education?"
    )
    state.subject = "MATHEMATICS"
    state.final_answer = "production-answer-unchanged"

    SMOKE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    if SMOKE_JSONL.exists():
        SMOKE_JSONL.write_text("")

    with (
        patch("app.agent.v213d_shadow._FUNNEL", SMOKE_FUNNEL),
        patch("app.agent.v213d_shadow._TRAFFIC", SMOKE_TRAFFIC),
        patch(
            "app.agent.v213d_shadow.default_retrieval_service",
            return_value=retrieval,
        ),
    ):
        # Direct path (deterministic persistence)
        record = run_production_shadow(
            agent,
            state,
            request_id="smoke-v213d-direct",
            retrieval=retrieval,
            jsonl_path=SMOKE_JSONL,
        )
        record["traffic_class"] = "SMOKE_TEST"
        # Async schedule path
        thread = maybe_schedule_v213d_shadow(
            agent,
            state,
            request_id="smoke-v213d-async",
            jsonl_path=SMOKE_JSONL,
        )
        if thread:
            thread.join(timeout=30)

    assert state.final_answer == "production-answer-unchanged"
    lines = [ln for ln in SMOKE_JSONL.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 1, "smoke JSONL empty"
    for line in lines:
        row = json.loads(line)
        assert row.get("experiment") == "v2.13d"
        # Ensure we did not write into production log via this script's path
        assert SMOKE_JSONL.name == "v213d_shadow_smoke.jsonl"

    funnel = json.loads(SMOKE_FUNNEL.read_text()) if SMOKE_FUNNEL.exists() else {}
    report = {
        "traffic_class": "SMOKE_TEST",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "smoke_jsonl": str(SMOKE_JSONL),
        "production_jsonl_untouched": str(
            ROOT / "data" / "diagnostics" / "v213d_shadow.jsonl"
        ),
        "records_written": len(lines),
        "direct_shadow_error": (record.get("shadow") or {}).get("error"),
        "funnel_stages": funnel.get("stages"),
        "production_answer_unchanged": True,
        "note": "Do not mix these records into Phase 1 production analysis.",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
