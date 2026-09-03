#!/usr/bin/env python3
"""V2.13D production-shadow evaluation (Phase 1 aggregate + optional replay)."""

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
PHASE1_DOC = ROOT / "docs" / "V2_13D_PHASE1_OBSERVATION.md"
OUT_JSON = ROOT / "data" / "diagnostics" / "v213d_shadow_summary.json"
PHASE1_JSON = ROOT / "data" / "diagnostics" / "v213d_phase1_summary.json"
JSONL = ROOT / "data" / "diagnostics" / "v213d_shadow.jsonl"


def _examples(records: list[dict]) -> dict[str, list[dict]]:
    buckets = {
        "document_improved": [],
        "structured_sufficient": [],
        "document_did_not_help": [],
        "document_noise": [],
        "regression": [],
        "safety": [],
        "shadow_failure": [],
    }
    for record in records:
        comparison = record.get("comparison") or {}
        label = comparison.get("classification")
        anon = {
            "request_id": record.get("request_id"),
            "question_hash": (record.get("question") or {}).get("hash"),
            "category": (record.get("question") or {}).get("category"),
            "classification": label,
            "control_route": (record.get("control") or {}).get("final_route"),
            "shadow_route": (record.get("shadow") or {}).get("final_route"),
            "document_evidence_count": (record.get("shadow") or {}).get(
                "document_evidence_count"
            ),
        }
        if comparison.get("newly_recoverable") or label == "DOCUMENT_ADDED_MISSING_CONTEXT":
            buckets["document_improved"].append(anon)
        elif label == "STRUCTURED_DATA_ALREADY_SUFFICIENT":
            buckets["structured_sufficient"].append(anon)
        elif label == "DOCUMENT_DID_NOT_HELP":
            buckets["document_did_not_help"].append(anon)
        elif label == "DOCUMENT_NOISE":
            buckets["document_noise"].append(anon)
        elif comparison.get("regressed") or comparison.get("control_correct_shadow_worse"):
            buckets["regression"].append(anon)
        elif label == "WRONG_CONTEXT" or (
            (record.get("grounding") or {}).get("wrong_context")
            and (record.get("shadow") or {}).get("final_accepted")
        ):
            buckets["safety"].append(anon)
        if (record.get("shadow") or {}).get("error"):
            buckets["shadow_failure"].append(anon)
    return {k: v[:5] for k, v in buckets.items()}


def write_phase1_doc(report: dict, *, config: dict) -> None:
    examples = report.get("examples") or {}
    lines = [
        "# V2.13D Phase 1 — Production Shadow Observation",
        "",
        f"Generated: `{report.get('generated_at')}`",
        "",
        f"## Executive status",
        "",
        f"**{report.get('phase1_status')}**",
        "",
        report.get("canary_note", ""),
        "",
        "## Configuration",
        "",
        "```text",
        f"v213d_shadow_enabled={config.get('shadow_enabled')}",
        f"v213d_shadow_sample_rate={config.get('sample_rate')}",
        f"v213d_shadow_document_retrieval={config.get('document_retrieval')}",
        f"v213d_shadow_retrieval_variant={config.get('retrieval_variant')}",
        f"v213d_shadow_timeout_seconds={config.get('timeout_seconds')}",
        "```",
        "",
        "## Sample",
        "",
        "```json",
        json.dumps(
            {
                "total_production_requests": report.get("total_production_requests"),
                "sampled": report.get("sampled_requests"),
                "completed": report.get("successful_shadow_evaluations"),
                "errors": report.get("shadow_errors"),
                "timeouts": report.get("shadow_timeouts"),
                "observation_target": [
                    report.get("observation_target_min"),
                    report.get("observation_target_max"),
                ],
                "source": report.get("source"),
            },
            indent=2,
        ),
        "```",
        "",
        "## Retrieval performance",
        "",
        "```json",
        json.dumps(
            {
                "retrieval_success_rate": report.get("retrieval_success_rate"),
                "mean_retrieval_latency": report.get("mean_retrieval_latency"),
                "p95_retrieval_latency": report.get("p95_retrieval_latency"),
                "mean_passages_retrieved": report.get("mean_passages_retrieved"),
                "provenance_complete_rate": report.get("provenance_complete_rate"),
            },
            indent=2,
        ),
        "```",
        "",
        "## Grounding safety",
        "",
        "```json",
        json.dumps(report.get("safety_metrics", {}), indent=2),
        "```",
        "",
        "## Product impact",
        "",
        "```json",
        json.dumps(
            {
                "newly_recoverable": report.get("newly_recoverable_count"),
                "newly_recoverable_rate": report.get("newly_recoverable_rate"),
                "improvements": report.get("improvements"),
                "unchanged": report.get("unchanged"),
                "regressions": report.get("regressions"),
                "control_correct_shadow_worse": report.get(
                    "control_correct_shadow_worse"
                ),
                "document_added_explanation": report.get("document_added_explanation"),
                "document_disambiguated_context": report.get(
                    "document_disambiguated_context"
                ),
                "document_did_not_help": report.get("document_did_not_help"),
                "document_noise": report.get("document_noise"),
            },
            indent=2,
        ),
        "```",
        "",
        "## Qualitative examples (anonymized)",
        "",
        "1. Document retrieval improving an answer:",
        "",
        "```json",
        json.dumps(examples.get("document_improved") or [], indent=2),
        "```",
        "",
        "2. Structured data already sufficient:",
        "",
        "```json",
        json.dumps(examples.get("structured_sufficient") or [], indent=2),
        "```",
        "",
        "3. Document retrieval did not help:",
        "",
        "```json",
        json.dumps(examples.get("document_did_not_help") or [], indent=2),
        "```",
        "",
        "4. Document noise:",
        "",
        "```json",
        json.dumps(examples.get("document_noise") or [], indent=2),
        "```",
        "",
        "5. Regressions:",
        "",
        "```json",
        json.dumps(examples.get("regression") or [], indent=2),
        "```",
        "",
        "6. Safety violations / shadow failures:",
        "",
        "```json",
        json.dumps(
            {
                "safety": examples.get("safety") or [],
                "shadow_failure": examples.get("shadow_failure") or [],
            },
            indent=2,
        ),
        "```",
        "",
        "## Distinctions",
        "",
        "- This report counts **real production shadow** records only when "
        "`source=production_shadow`.",
        "- Controlled V2.13C / Phase 0 replay is **not** mixed into Phase 1 success claims.",
        "",
        report.get("v213c_comparison_note", ""),
        "",
        "## Recommendation",
        "",
        report.get("phase1_recommendation", "CONTINUE SHADOW"),
        "",
    ]
    PHASE1_DOC.write_text("\n".join(lines))


def write_doc(report: dict) -> None:
    lines = [
        "# V2.13D — Production Shadow for Context-Hybrid Curriculum Document Evidence",
        "",
        f"Generated: `{report.get('generated_at')}`",
        "",
        f"**Phase 1 status: {report.get('phase1_status')}**",
        "",
        f"**Recommendation: {report.get('phase1_recommendation')}**",
        "",
        report.get("canary_note", ""),
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(
            {k: v for k, v in report.items() if k not in {"records", "examples"}},
            indent=2,
        ),
        "```",
        "",
    ]
    DOC.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="V2.13D production-shadow eval")
    parser.add_argument(
        "--from-jsonl",
        action="store_true",
        help="Aggregate real/production JSONL (default when --replay is absent)",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Run controlled fixtures (does not claim Phase 1 success)",
    )
    parser.add_argument(
        "--include-replay-records",
        action="store_true",
        help="When aggregating JSONL, keep rows that have replay_id",
    )
    args = parser.parse_args()

    from app.agent.v213d_shadow import (
        aggregate_records,
        load_jsonl_records,
        load_traffic_counters,
        persist_record,
        prepare_replay_corpus,
        production_shadow_records,
        replay_fixtures,
        v213d_runtime_config,
    )
    from app.config import Settings, get_settings

    if args.replay:
        from app.agent.orchestrator import CurriculumQAAgent
        from app.llm.provider import StubLLMProvider

        OUT.mkdir(parents=True, exist_ok=True)
        settings = Settings(
            _env_file=None,
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
        replay_jsonl = OUT / "phase0_replay.jsonl"
        replay_jsonl.write_text("")
        for record in records:
            persist_record(record, replay_jsonl)
        summary = aggregate_records(records, source="controlled_replay")
        summary["generated_at"] = datetime.now(timezone.utc).isoformat()
        summary["mode"] = "controlled_replay"
        OUT_JSON.write_text(json.dumps(summary, indent=2))
        write_doc(summary)
        print(f"Replay status: {summary['phase1_status']}")
        print(f"Report: {DOC}")
        return 0

    # Phase 1: real traffic JSONL
    settings = get_settings()
    config = v213d_runtime_config(settings)
    records = load_jsonl_records(JSONL)
    if not args.include_replay_records:
        records = production_shadow_records(records)
    summary = aggregate_records(
        records,
        traffic=load_traffic_counters(),
        source="production_shadow",
    )
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["active_configuration"] = config
    summary["examples"] = _examples(records)
    summary["real_traffic_observed"] = bool(records)
    PHASE1_JSON.write_text(json.dumps(summary, indent=2))
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    write_phase1_doc(summary, config=config)
    write_doc(summary)
    print(f"Phase1 status: {summary['phase1_status']}")
    print(f"Recommendation: {summary['phase1_recommendation']}")
    print(f"Report: {PHASE1_DOC}")
    print(f"JSON: {PHASE1_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
