#!/usr/bin/env python3
"""Post-corpus V2.13D smoke: production store + forced sample → smoke JSONL only.

Does NOT write to data/diagnostics/v213d_shadow.jsonl.
Does NOT change live sample_rate.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SMOKE_JSONL = ROOT / "data" / "diagnostics" / "v213d_shadow_smoke.jsonl"
SMOKE_FUNNEL = ROOT / "data" / "diagnostics" / "v213d_pipeline_funnel_smoke.json"
SMOKE_TRAFFIC = ROOT / "data" / "diagnostics" / "v213d_traffic_smoke.json"
PROD_JSONL = ROOT / "data" / "diagnostics" / "v213d_shadow.jsonl"


def main() -> int:
    from app.agent.orchestrator import CurriculumQAAgent
    from app.agent.state import CurriculumQAState
    from app.agent.v213d_shadow import (
        default_retrieval_service,
        production_corpus_status,
        run_production_shadow,
    )
    from app.config import Settings
    from app.llm.provider import StubLLMProvider
    from app.tools.registry import build_default_registry

    corpus = production_corpus_status()
    if not corpus["available"]:
        print(json.dumps({"ok": False, "error": "production corpus unavailable", **corpus}))
        return 1

    settings = Settings(
        _env_file=None,
        v213d_shadow_enabled=True,
        v213d_shadow_sample_rate=1.0,  # test-only override
        llm_provider="stub",
    )
    # Production tools remain without document search unless experiment flags on.
    registry = build_default_registry(settings=Settings(_env_file=None))
    assert "search_curriculum_documents" not in registry.names()

    agent = CurriculumQAAgent(settings=settings, llm=StubLLMProvider())
    retrieval = default_retrieval_service(settings)
    state = CurriculumQAState.initial(
        question="What does the MBSSE curriculum say about the purpose of mathematics education?"
    )
    state.subject = "MATHEMATICS"
    state.final_answer = "production-answer-unchanged"

    before = PROD_JSONL.read_text() if PROD_JSONL.exists() else ""
    SMOKE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_JSONL.write_text("")

    with (
        patch("app.agent.v213d_shadow._FUNNEL", SMOKE_FUNNEL),
        patch("app.agent.v213d_shadow._TRAFFIC", SMOKE_TRAFFIC),
    ):
        record = run_production_shadow(
            agent,
            state,
            request_id="smoke-post-corpus",
            retrieval=retrieval,
            jsonl_path=SMOKE_JSONL,
        )
    record["traffic_class"] = "SMOKE_TEST_POST_CORPUS"
    after = PROD_JSONL.read_text() if PROD_JSONL.exists() else ""
    assert after == before, "production JSONL must not change during smoke"
    assert state.final_answer == "production-answer-unchanged"

    shadow = record.get("shadow") or {}
    docs = int(shadow.get("document_evidence_count") or 0)
    report = {
        "traffic_class": "SMOKE_TEST_POST_CORPUS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stages": {
            "request": "PASS",
            "sampled": "PASS",
            "retrieved": "PASS" if docs > 0 else "FAIL",
            "provenance": "PASS" if shadow.get("provenance_complete") else "FAIL",
            "persisted": "PASS" if SMOKE_JSONL.stat().st_size > 0 else "FAIL",
            "production_jsonl_untouched": "PASS" if after == before else "FAIL",
            "production_answer_unchanged": "PASS",
        },
        "document_evidence_count": docs,
        "classification": (record.get("comparison") or {}).get("classification"),
        "corpus": corpus,
        "smoke_jsonl": str(SMOKE_JSONL),
    }
    report["ok"] = all(v == "PASS" for v in report["stages"].values())
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
