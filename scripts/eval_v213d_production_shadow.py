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


def classify_pipeline(
    *,
    live_qa_requests: int | None,
    production_rows: int,
    config: dict,
    funnel: dict,
) -> dict:
    stages = (funnel or {}).get("stages") or {}
    enabled = bool(config.get("shadow_enabled"))
    rate = float(config.get("sample_rate") or 0.0)
    if not enabled or rate <= 0:
        classification = "CONFIGURATION_MISMATCH"
    elif live_qa_requests is not None and live_qa_requests == 0 and production_rows == 0:
        classification = "TRAFFIC_NOT_REACHING_QA"
    elif production_rows > 0:
        classification = "PIPELINE_OPERATIONAL"
    elif int(stages.get("shadow_sampled") or 0) > 0 and int(
        stages.get("shadow_persisted") or 0
    ) == 0:
        classification = "PERSISTENCE_FAILURE"
    elif int(stages.get("shadow_sampled") or 0) > 0 and int(
        stages.get("shadow_started") or 0
    ) == 0:
        classification = "SHADOW_NOT_EXECUTING"
    elif live_qa_requests and live_qa_requests > 0 and production_rows == 0:
        # Requests exist but no shadows yet — likely low volume at 1%.
        classification = "TRAFFIC_IS_SIMPLY_TOO_LOW"
    else:
        classification = "TRAFFIC_NOT_REACHING_QA"
    return {
        "classification": classification,
        "live_qa_metrics_total_requests": live_qa_requests,
        "production_jsonl_rows": production_rows,
        "funnel_stages": stages,
        "config_enabled": enabled,
        "sample_rate": rate,
        "jsonl_path": (funnel or {}).get("jsonl_path"),
        "stages_checklist": {
            "qa_request": (
                "PASS"
                if (live_qa_requests or 0) > 0
                else "NOT OBSERVED"
            ),
            "hook": (
                "PASS"
                if int(stages.get("request_seen") or 0) > 0
                else "NOT OBSERVED"
            ),
            "sampling": "PASS" if enabled and rate > 0 else "FAIL",
            "shadow": (
                "PASS"
                if int(stages.get("shadow_completed") or 0) > 0
                else "NOT OBSERVED"
            ),
            "persistence": (
                "PASS"
                if production_rows > 0 or int(stages.get("shadow_persisted") or 0) > 0
                else "NOT OBSERVED"
            ),
        },
    }


def write_phase1_doc(report: dict, *, config: dict) -> None:
    examples = report.get("examples") or {}
    pipeline = report.get("pipeline_verification") or {}
    classifications = report.get("classifications") or {}
    activation_path = Path("data/diagnostics/v213d_corpus_activation.json")
    activation = {}
    if activation_path.is_file():
        try:
            activation = json.loads(activation_path.read_text(encoding="utf-8"))
        except Exception:
            activation = {}
    phase1d_path = Path("data/diagnostics/v213d_phase1d_traffic_run.json")
    phase1d_traffic: dict = {}
    if phase1d_path.is_file():
        try:
            raw = json.loads(phase1d_path.read_text(encoding="utf-8"))
            phase1d_traffic = {
                "traffic_class": raw.get("traffic_class"),
                "requested": raw.get("requested"),
                "ok": raw.get("ok"),
                "failed": raw.get("failed"),
                "elapsed_s": raw.get("elapsed_s"),
                "categories": raw.get("categories"),
                "shadow_rows_before": raw.get("shadow_rows_before"),
                "shadow_rows_after": raw.get("shadow_rows_after"),
                "funnel_before": raw.get("funnel_before"),
                "funnel_after": raw.get("funnel_after"),
                "traffic_before": raw.get("traffic_before"),
                "traffic_after": raw.get("traffic_after"),
                "mean_latency_ms": raw.get("mean_latency_ms"),
            }
        except Exception:
            phase1d_traffic = {}
    lines = [
        "# V2.13D Phase 1 Observation Report",
        "",
        f"Generated: `{report.get('generated_at')}`",
        "",
        "## Executive Summary",
        "",
        f"**Status: `{report.get('phase1_status')}`**",
        "",
        f"**Recommendation: `{report.get('phase1_recommendation')}`**",
        "",
        f"**Pipeline classification: `{pipeline.get('classification', 'UNKNOWN')}`**",
        "",
        report.get("canary_note", ""),
        "",
        "Pre-corpus real shadows are infrastructure/corpus-availability failures "
        "(`DOCUMENT_CORPUS_UNAVAILABLE`), not retrieval-algorithm failures. "
        "Post-corpus observation sufficiency is measured separately.",
        "",
        "## Phase 1 Timeline",
        "",
        "```text",
        "Phase 1A — pipeline verification",
        "    0 production QA requests (TRAFFIC_NOT_REACHING_QA initially)",
        "",
        "Phase 1B — first real traffic",
        "    ~121 QA requests",
        "    2 real shadows",
        "    corpus unavailable (empty data/documents)",
        "    classification: DOCUMENT_CORPUS_UNAVAILABLE (reclassified)",
        "",
        "Phase 1C — corpus activation",
        "    trusted V2.13A–C BENCHMARK_SOURCES activated",
        f"    documents={activation.get('counts', {}).get('documents', 'n/a')}",
        f"    passages={activation.get('counts', {}).get('passages', 'n/a')}",
        f"    index_entries={activation.get('counts', {}).get('index_entries', 'n/a')}",
        f"    activation_ok={activation.get('activation_ok')}",
        f"    expected_hashes_matched={activation.get('expected_hashes_matched')}",
        "",
        "Phase 1D — post-corpus observation",
        f"    post-corpus real shadows: {report.get('post_corpus_successful_shadow_evaluations', 0)}",
        "    sample_rate remains 0.01 (no forced sampling)",
        f"    metrics_scope: {report.get('metrics_scope', 'post_corpus')}",
        f"    retrieval_success (post-corpus): {report.get('retrieval_success_rate')}",
        f"    newly_recoverable: {report.get('newly_recoverable_count')}",
        f"    regressions (control_correct_shadow_worse): {report.get('control_correct_shadow_worse')}",
        "```",
        "",
        "## Active Configuration",
        "",
        "```text",
        f"v213d_shadow_enabled={config.get('shadow_enabled')}",
        f"v213d_shadow_sample_rate={config.get('sample_rate')}",
        f"v213d_shadow_document_retrieval={config.get('document_retrieval')}",
        f"v213d_shadow_retrieval_variant={config.get('retrieval_variant')}",
        f"v213d_shadow_timeout_seconds={config.get('timeout_seconds')}",
        "```",
        "",
        "No rollout escalation. No V2.13E. Document evidence does not enter the "
        "user-facing production answer path.",
        "",
        "## Corpus Activation (Phase 1C)",
        "",
        "```json",
        json.dumps(
            {
                "corpus_family": activation.get("corpus_family"),
                "counts": activation.get("counts"),
                "document_hashes": (activation.get("ingestion") or {}).get(
                    "document_hashes"
                ),
                "discrepancies": activation.get("discrepancies"),
                "orphaned": (activation.get("counts") or {}).get(
                    "orphaned_document_dirs"
                ),
                "passages_missing_provenance": (activation.get("counts") or {}).get(
                    "passages_missing_provenance"
                ),
                "hierarchy": activation.get("hierarchy"),
                "activated_at": activation.get("activated_at"),
            },
            indent=2,
        ),
        "```",
        "",
        "## Phase 1D Traffic Batch",
        "",
        "```json",
        json.dumps(phase1d_traffic, indent=2),
        "```",
        "",
        "## Pre- vs Post-Corpus Real Shadows",
        "",
        "```json",
        json.dumps(
            {
                "pre_corpus_shadow_evaluations": report.get(
                    "pre_corpus_shadow_evaluations"
                ),
                "post_corpus_shadow_evaluations": report.get(
                    "post_corpus_shadow_evaluations"
                ),
                "post_corpus_successful_shadow_evaluations": report.get(
                    "post_corpus_successful_shadow_evaluations"
                ),
                "corpus_unavailable_count": report.get("corpus_unavailable_count"),
                "metrics_scope": report.get("metrics_scope"),
                "classifications": classifications,
                "post_corpus_classifications": report.get("post_corpus_classifications"),
            },
            indent=2,
        ),
        "```",
        "",
        "## Phase 1D Post-Corpus Performance (primary)",
        "",
        "```json",
        json.dumps(
            {
                "retrieval_success_rate": report.get("retrieval_success_rate"),
                "no_match_rate": report.get("no_match_rate"),
                "mean_passages_retrieved": report.get("mean_passages_retrieved"),
                "provenance_complete_rate": report.get("provenance_complete_rate"),
                "metadata_valid_rate": report.get("metadata_valid_rate"),
                "newly_recoverable_count": report.get("newly_recoverable_count"),
                "newly_recoverable_rate": report.get("newly_recoverable_rate"),
                "improvement_rate": report.get("improvement_rate"),
                "regression_rate": report.get("regression_rate"),
                "control_correct_shadow_worse": report.get(
                    "control_correct_shadow_worse"
                ),
                "document_added_missing_context": report.get(
                    "document_added_missing_context"
                ),
                "document_added_explanation": report.get("document_added_explanation"),
                "document_disambiguated_context": report.get(
                    "document_disambiguated_context"
                ),
                "document_provided_source": report.get("document_provided_source"),
                "document_did_not_help": report.get("document_did_not_help"),
                "document_noise": report.get("document_noise"),
                "structured_data_already_sufficient": report.get(
                    "structured_data_already_sufficient"
                ),
                "latency_metrics": report.get("latency_metrics"),
            },
            indent=2,
        ),
        "```",
        "",
        "Primary performance metrics above are scoped to **post-corpus** shadows. "
        "Pre-corpus `DOCUMENT_CORPUS_UNAVAILABLE` rows remain historical infrastructure "
        "failures and are excluded from retrieval-quality rates.",
        "",
        "## Phase 1 Traffic Pipeline Verification",
        "",
        "```json",
        json.dumps(pipeline, indent=2),
        "```",
        "",
        "## Real-Traffic Sample",
        "",
        "```json",
        json.dumps(
            {
                "total_production_requests": report.get("total_production_requests"),
                "live_qa_metrics_total_requests": pipeline.get(
                    "live_qa_metrics_total_requests"
                ),
                "sampled": report.get("sampled_requests"),
                "completed": report.get("successful_shadow_evaluations"),
                "errors": report.get("shadow_errors"),
                "timeouts": report.get("shadow_timeouts"),
                "observation_target": [
                    report.get("observation_target_min"),
                    report.get("observation_target_max"),
                ],
                "source": report.get("source"),
                "real_traffic_observed": report.get("real_traffic_observed"),
            },
            indent=2,
        ),
        "```",
        "",
        "## Retrieval Performance",
        "",
        "```json",
        json.dumps(
            {
                "retrieval_success_rate": report.get("retrieval_success_rate"),
                "no_match_rate": report.get("no_match_rate"),
                "mean_retrieval_latency": report.get("mean_retrieval_latency"),
                "p95_retrieval_latency": report.get("p95_retrieval_latency"),
                "mean_passages_retrieved": report.get("mean_passages_retrieved"),
                "provenance_complete_rate": report.get("provenance_complete_rate"),
                "metrics_scope": report.get("metrics_scope"),
            },
            indent=2,
        ),
        "```",
        "",
        "## Grounding and Safety",
        "",
        "```json",
        json.dumps(report.get("safety_metrics", {}), indent=2),
        "```",
        "",
        "## Outcome Metrics",
        "",
        "```json",
        json.dumps(
            {
                "newly_recoverable": report.get("newly_recoverable_count"),
                "newly_recoverable_rate": report.get("newly_recoverable_rate"),
                "improvements": report.get("improvements"),
                "improvement_rate": report.get("improvement_rate"),
                "unchanged": report.get("unchanged"),
                "regressions": report.get("regressions"),
                "regression_rate": report.get("regression_rate"),
                "control_correct_shadow_worse": report.get(
                    "control_correct_shadow_worse"
                ),
            },
            indent=2,
        ),
        "```",
        "",
        "## Qualitative Examples (anonymized)",
        "",
        "```json",
        json.dumps(examples, indent=2),
        "```",
        "",
        "## Distinctions",
        "",
        "- Production analysis uses only `v213d_shadow.jsonl` (no `replay_id`).",
        "- Smoke/test records must live in `v213d_shadow_smoke.jsonl` only.",
        "- Phase 0 replay is excluded from Phase 1 claims.",
        "- Pre-corpus vs post-corpus shadows are analyzed separately.",
        "- Phase 1D primary rates are post-corpus only.",
        "",
        report.get("v213c_comparison_note", ""),
        "",
        "## Recommendation",
        "",
        report.get("phase1_recommendation", "CONTINUE SHADOW"),
        "",
        "Keep `sample_rate=0.01`. Do not enable V2.13E until enough **post-corpus** "
        "real shadows exist to judge document-layer value in production.",
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
        load_pipeline_funnel,
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
    live_qa = None
    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:8001/api/v1/agent/metrics", timeout=2) as resp:
            live_qa = json.loads(resp.read().decode()).get("total_requests")
    except Exception:
        live_qa = None
    funnel = load_pipeline_funnel()
    summary["pipeline_verification"] = classify_pipeline(
        live_qa_requests=live_qa,
        production_rows=len(records),
        config=config,
        funnel=funnel,
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
    print(
        f"Pipeline: {summary['pipeline_verification']['classification']}"
    )
    print(f"Report: {PHASE1_DOC}")
    print(f"JSON: {PHASE1_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
