"""V2.13G.2 preserve-control paired validation tests."""

from __future__ import annotations

from app.agent.v213g2_preserve_control import (
    REGRESSION_IDS,
    aggregate_v213g2,
    apply_preserve_to_record,
    paired_case_row,
)
from app.agent.v213g1_provenance_only import load_targeted_records
from pathlib import Path


def test_frozen_cohort_mode_b_meets_success_criteria():
    path = Path("data/diagnostics/v213g_arbitration_targeted_replay.jsonl")
    if not path.is_file():
        return
    records = load_targeted_records(path)
    assert len(records) >= 20
    summary = aggregate_v213g2(records)
    assert summary["n_provenance_only_redundant"] == 17
    assert summary["success_criteria"]["S1_control_preservation"] is True
    assert summary["success_criteria"]["S2_no_generation_leakage"] is True
    assert summary["success_criteria"]["S3_no_unnecessary_regeneration"] is True
    assert summary["success_criteria"]["S4_regression_elimination"] is True
    assert summary["success_criteria"]["S5_h2_preservation"] is True
    assert summary["decision"] == "PRESERVE_CONTROL_CONFIRMED"
    table = summary["comparison_table"]
    assert table["control_hash_matches"]["preserve_control"] == 17
    assert table["regeneration_attempted"]["preserve_control"] == 0
    assert table["generation_document_leaks"]["preserve_control"] == 0
    assert table["answer_regressions"]["preserve_control"] == 0


def test_prior_arbitrated_regressions_cleared_under_preserve():
    path = Path("data/diagnostics/v213g_arbitration_targeted_replay.jsonl")
    if not path.is_file():
        return
    records = load_targeted_records(path)
    by_hash = {(r.get("question") or {}).get("hash"): r for r in records}
    for qh in ("fd894182926a5f8b", "3445d54939ac9ed9", "7fb53cf210f937c5"):
        row = paired_case_row(by_hash[qh])
        assert row["mode_a_regeneration"]["regression"] is True
        assert row["mode_b_preserve"]["regression"] is False
        assert row["mode_b_preserve"]["hash_match_control"] is True
        assert row["mode_b_preserve"]["generation_attempted"] is False
        assert row["mode_b_preserve"]["generation_document_count"] == 0


def test_h2_decisive_not_preserved():
    path = Path("data/diagnostics/v213g_arbitration_targeted_replay.jsonl")
    if not path.is_file():
        return
    records = [r for r in load_targeted_records(path) if True]
    decisive = []
    for r in records:
        arb = r.get("arbitration") or {}
        if (
            arb.get("structured_sufficiency") == "INSUFFICIENT"
            and arb.get("document_role") == "DECISIVE"
        ):
            decisive.append(r)
    assert decisive
    for r in decisive:
        b = apply_preserve_to_record(r)
        assert b.get("preserve_applied") is False
        assert (b.get("arbitrated_shadow") or {}).get("generation_attempted") is True


def test_regression_id_set_complete():
    assert len(REGRESSION_IDS) == 5
