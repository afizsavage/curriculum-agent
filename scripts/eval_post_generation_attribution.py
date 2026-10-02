"""Attribute frozen production answers. Does not regenerate them."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.post_generation_attribution import attribute_existing_answer
from app.config import get_settings
from app.llm.provider import build_llm_provider
from scripts.eval_claim_shadow import LIVE_CASES, _evidence_from_tool

PRIOR = Path("data/diagnostics/claim_shadow/claim_shadow_results.json")
OUT = Path("data/diagnostics/claim_shadow/post_generation_attribution.json")

FIXTURES = [
    {
        "id": "fixture-p3-fractions",
        "fixture": "p3",
        "malformed_ids": ["lo-p3-equivalent-incomplete"],
    },
    {
        "id": "fixture-p4-fractions",
        "fixture": "p4",
        "malformed_ids": ["lo-multiply-garbled"],
    },
]


def _specs() -> dict[str, dict]:
    specs = {case["id"]: case for case in LIVE_CASES}
    specs.update({case["id"]: case for case in FIXTURES})
    return specs


def _aggregate(rows: list[dict]) -> dict:
    measured = [row for row in rows if row.get("metrics")]
    def total(key: str) -> int:
        return sum(int(row["metrics"].get(key) or 0) for row in measured)
    buckets = {"1": 0, "2": 0, "3": 0, "4+": 0}
    for row in measured:
        for key, count in (row["metrics"].get("record_buckets") or {}).items():
            buckets[key] = buckets.get(key, 0) + int(count)
    substantive = total("substantive_claims")
    returned = total("returned_mappings")
    covered = total("covered_claims")
    complete = sum(1 for row in measured if row["metrics"].get("complete"))
    preserved = sum(1 for row in measured if row.get("answer_preserved"))
    return {
        "evidence_bearing_cases": len(measured),
        "substantive_claims": substantive,
        "returned_mappings": returned,
        "exact_matches": total("exact_matches"),
        "near_misses": total("near_misses"),
        "missing_mappings": total("missing_mappings"),
        "returned_invalid_mappings": total("returned_invalid_mappings"),
        "extra_mappings": total("extra_mappings"),
        "invalid_refs": total("invalid_refs"),
        "unsupported_refs": total("unsupported_refs"),
        "record_buckets": buckets,
        "malformed_evidence_mappings": total("malformed_evidence_mappings"),
        "complete_cases": complete,
        "claim_coverage": (covered / substantive) if substantive else None,
        "exact_span_compliance": (total("exact_matches") / returned) if returned else None,
        "case_level_complete_attribution": (complete / len(measured)) if measured else None,
        "unsupported_ref_rate": (total("unsupported_refs") / returned) if returned else None,
        "production_answer_preservation": (preserved / len(measured)) if measured else None,
        "production_refs_covered": total("production_refs_covered_count"),
        "production_refs_unattributed": total("production_refs_unattributed_count"),
        "claim_refs_not_in_production": total("claim_refs_not_in_production_count"),
        "evidence_id_mismatches": sum(1 for row in measured if row.get("evidence_id_mismatch")),
    }


def _run(llm, spec: dict, prior: dict) -> dict:
    answer = prior["production_answer"]
    evidence = _evidence_from_tool(spec)
    stored_ids = set(prior.get("evidence_ids") or [])
    fetched_ids = {item.entity_id for item in evidence if item.entity_id}
    before = answer
    result = attribute_existing_answer(
        llm,
        question=prior["question"],
        answer=answer,
        evidence=evidence,
        production_refs=list(prior.get("production_refs") or []),
        malformed_ids=spec.get("malformed_ids"),
    )
    metrics = result["metrics"]
    metrics["production_refs_covered_count"] = len(metrics["production_refs_covered"])
    metrics["production_refs_unattributed_count"] = len(metrics["production_refs_unattributed"])
    metrics["claim_refs_not_in_production_count"] = len(metrics["claim_refs_not_in_production"])
    return {
        "id": prior["id"],
        "shape": prior.get("shape"),
        "question": prior["question"],
        "production_answer": before,
        "production_refs": list(prior.get("production_refs") or []),
        "shadow_claims": result["shadow_claims"],
        "validated_shadow_claims": metrics["validation"].get("valid_claims"),
        "attribution_failures": {
            "omitted_claims": metrics["omitted_claims"],
            "returned_invalid_claims": metrics["returned_invalid_claims"],
            "extra_classes": metrics["text_classes"],
            "invalid_refs": metrics["validation"].get("invalid_refs"),
            "unsupported_claims": metrics["validation"].get("unsupported_claims"),
            "wide_returned_claims": metrics["wide_returned_claims"],
            "wide_valid_claims": metrics["wide_valid_claims"],
        },
        "metrics": metrics,
        "answer_preserved": result["production_answer"] == before == answer,
        "evidence_id_mismatch": stored_ids != fetched_ids,
        "discarded_answer_field": result["discarded_answer_field"],
    }


def main() -> None:
    settings = get_settings()
    prior_rows = {
        row["id"]: row
        for row in json.loads(PRIOR.read_text()).get("cases") or []
        if row.get("evidence_count") and row.get("production_answer")
    }
    specs = _specs()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if OUT.exists():
        existing = {
            row["id"]: row
            for row in json.loads(OUT.read_text()).get("cases") or []
            if row.get("id") and row.get("answer_preserved") and not row.get("error")
        }
    llm = build_llm_provider(settings)
    rows: list[dict] = []
    ordered = [prior_rows[key] for key in prior_rows]
    for index, prior in enumerate(ordered, start=1):
        spec = specs.get(prior["id"])
        if spec is None:
            print(f"[{index}/{len(ordered)}] {prior['id']} skipped: no evidence spec", flush=True)
            continue
        if prior["id"] in existing:
            rows.append(existing[prior["id"]])
            print(f"[{index}/{len(ordered)}] {prior['id']} (cached)", flush=True)
            continue
        print(f"[{index}/{len(ordered)}] {prior['id']}", flush=True)
        try:
            row = _run(llm, spec, prior)
        except Exception as exc:
            row = {"id": prior["id"], "question": prior.get("question"), "error": str(exc), "trace": traceback.format_exc()}
            print(f"  ERROR {exc}", flush=True)
        else:
            metrics = row["metrics"]
            print(
                f"  preserved={row['answer_preserved']} exact={metrics['exact_matches']}/{metrics['returned_mappings']} "
                f"covered={metrics['covered_claims']}/{metrics['substantive_claims']} "
                f"omitted={metrics['missing_mappings']} invalid={metrics['returned_invalid_mappings']} "
                f"wide={metrics['record_buckets']['4+']}",
                flush=True,
            )
        rows.append(row)
        OUT.write_text(json.dumps({
            "model": settings.llm_model,
            "provider": settings.llm_provider,
            "aggregate": _aggregate(rows),
            "cases": rows,
        }, indent=2))
    summary = {
        "model": settings.llm_model,
        "provider": settings.llm_provider,
        "aggregate": _aggregate(rows),
        "cases": rows,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
