#!/usr/bin/env python3
"""Freeze V2.13D/Phase 1E baseline artifacts for the V2.13F experiment."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data" / "diagnostics" / "v213d_shadow.jsonl"
OUT_DIR = ROOT / "data" / "diagnostics" / "v213f_baseline"
FREEZE_JSONL = OUT_DIR / "v213d_shadow_freeze.jsonl"
ACTIVATION = ROOT / "data" / "diagnostics" / "v213d_corpus_activation.json"
INDEX_MANIFEST = ROOT / "data" / "document_index" / "feature-hash-v1" / "manifest.json"
TRAFFIC = ROOT / "data" / "diagnostics" / "v213d_traffic.json"
FUNNEL = ROOT / "data" / "diagnostics" / "v213d_pipeline_funnel.json"


def main() -> int:
    from app.agent.v213d_shadow import (
        aggregate_records,
        load_jsonl_records,
        load_traffic_counters,
        production_corpus_status,
    )
    from app.config import Settings

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not JSONL.is_file():
        raise SystemExit(f"missing {JSONL}")
    shutil.copy2(JSONL, FREEZE_JSONL)
    raw = FREEZE_JSONL.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    rows = load_jsonl_records(FREEZE_JSONL)
    metrics = aggregate_records(rows, traffic=load_traffic_counters(), source="production_shadow")
    settings = Settings()
    activation = {}
    if ACTIVATION.is_file():
        activation = json.loads(ACTIVATION.read_text(encoding="utf-8"))
    index_manifest = {}
    if INDEX_MANIFEST.is_file():
        index_manifest = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
    if TRAFFIC.is_file():
        shutil.copy2(TRAFFIC, OUT_DIR / "v213d_traffic.json")
    if FUNNEL.is_file():
        shutil.copy2(FUNNEL, OUT_DIR / "v213d_pipeline_funnel.json")
    if INDEX_MANIFEST.is_file():
        shutil.copy2(INDEX_MANIFEST, OUT_DIR / "index_manifest.json")

    freeze = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "phase": "v213d_phase1e_baseline_freeze_for_v213f",
        "config": {
            "v213d_shadow_enabled": settings.v213d_shadow_enabled,
            "v213d_shadow_sample_rate": settings.v213d_shadow_sample_rate,
            "v213d_shadow_document_retrieval": settings.v213d_shadow_document_retrieval,
            "v213d_shadow_retrieval_variant": settings.v213d_shadow_retrieval_variant,
            "v213d_shadow_timeout_seconds": settings.v213d_shadow_timeout_seconds,
            "v213f_document_arbitration_experiment": settings.v213f_document_arbitration_experiment,
            "v213e_note": "disabled / not present",
        },
        "corpus": production_corpus_status(),
        "activation_hashes": (activation.get("ingestion") or {}).get("document_hashes")
        or activation.get("document_hashes"),
        "index_manifest": {
            "embedding_model": index_manifest.get("embedding_model"),
            "embedding_dimension": index_manifest.get("embedding_dimension"),
            "indexed_at": index_manifest.get("indexed_at"),
            "documents": index_manifest.get("documents"),
        },
        "jsonl_sha256": sha,
        "jsonl_rows": len(rows),
        "post_corpus_successful": metrics.get("post_corpus_successful_shadow_evaluations"),
        "document_helped": metrics.get("document_helped"),
        "document_neutral": metrics.get("document_neutral"),
        "document_hurt": metrics.get("document_hurt"),
        "control_correct_shadow_worse": metrics.get("control_correct_shadow_worse"),
        "safety_metrics": metrics.get("safety_metrics"),
        "phase1_status": metrics.get("phase1_status"),
        "phase1_recommendation": metrics.get("phase1_recommendation"),
    }
    (OUT_DIR / "freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "freeze": str(OUT_DIR / "freeze.json"), **{
        k: freeze[k]
        for k in (
            "jsonl_rows",
            "post_corpus_successful",
            "document_helped",
            "document_neutral",
            "document_hurt",
            "control_correct_shadow_worse",
            "phase1_status",
            "jsonl_sha256",
        )
    }}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
