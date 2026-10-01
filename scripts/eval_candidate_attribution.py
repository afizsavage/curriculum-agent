"""Attribute frozen claims from constrained candidate sets. Does not regenerate answers."""

from __future__ import annotations

import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.candidate_attribution import (
    attribute_one_claim,
    candidate_size_summary,
    extract_answer_claims,
    measure_candidate_case,
    select_candidates,
)
from app.config import get_settings
from app.llm.provider import build_llm_provider
from scripts.eval_claim_shadow import LIVE_CASES, _evidence_from_tool

PRIOR = Path("data/diagnostics/claim_shadow/post_generation_attribution.json")
CORPUS = Path("data/diagnostics/claim_shadow/claim_shadow_results.json")
OUT = Path("data/diagnostics/claim_shadow/candidate_attribution.json")

FIXTURES = [
    {"id": "fixture-p3-fractions", "fixture": "p3", "malformed_ids": ["lo-p3-equivalent-incomplete"]},
    {"id": "fixture-p4-fractions", "fixture": "p4", "malformed_ids": ["lo-multiply-garbled"]},
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
    lexical_buckets = {"1": 0, "2": 0, "3": 0, "4+": 0}
    counts: list[int] = []
    for row in measured:
        for key, count in (row["metrics"].get("record_buckets") or {}).items():
            buckets[key] = buckets.get(key, 0) + int(count)
        for key, count in (row["metrics"].get("lexical_record_buckets") or {}).items():
            lexical_buckets[key] = lexical_buckets.get(key, 0) + int(count)
        counts.extend(row["metrics"].get("candidate_counts") or [])
    substantive = total("substantive_claims")
    returned = total("returned_mappings")
    covered = total("covered_claims")
    complete = sum(1 for row in measured if row["metrics"].get("complete"))
    with_claims = sum(1 for row in measured if row["metrics"].get("substantive_claims"))
    preserved = sum(1 for row in measured if row.get("answer_preserved"))
    sizes = candidate_size_summary(counts, total("truncations"))
    return {
        "evidence_bearing_cases": len(measured),
        "cases_with_claims": with_claims,
        "substantive_claims": substantive,
        "returned_mappings": returned,
        "exact_matches": total("exact_matches"),
        "exact_span_compliance": (total("exact_matches") / substantive) if substantive else None,
        "claim_coverage": (covered / substantive) if substantive else None,
        "model_coverage": (total("model_covered_claims") / substantive) if substantive else None,
        "unsupported_refs": total("unsupported_refs"),
        "unsupported_ref_rate": (total("unsupported_refs") / returned) if returned else None,
        "unknown_refs": total("unknown_refs"),
        "outside_candidate_refs": total("outside_candidate_refs"),
        "empty_content_refs": total("empty_content_refs"),
        "empty_mappings": total("empty_mappings"),
        "no_candidate_claims": total("no_candidate_claims"),
        "complete_cases": complete,
        "case_level_complete_attribution": (complete / with_claims) if with_claims else None,
        "model_complete_cases": sum(1 for row in measured if row["metrics"].get("model_complete")),
        "record_buckets": buckets,
        "lexical_record_buckets": lexical_buckets,
        "malformed_evidence_mappings": total("malformed_evidence_mappings"),
        "malformed_model_selections": total("malformed_model_selections"),
        "candidate_size": sizes,
        "production_answer_preservation": (preserved / len(measured)) if measured else None,
        "evidence_id_mismatches": sum(1 for row in measured if row.get("evidence_id_mismatch")),
        "claim_errors": sum(1 for row in measured for claim in row.get("claims") or [] if claim.get("error")),
    }


def _attribute_claim(llm, question: str, claim: dict, evidence, production_refs: list[str]) -> dict:
    selected = select_candidates(claim, evidence, production_refs)
    try:
        selection = attribute_one_claim(
            llm,
            question=question,
            claim_text=str(claim["text"]),
            candidates=selected["candidates"],
            evidence=evidence,
        )
        error = None
    except Exception as exc:
        selection = {
            "model_refs": [],
            "model-selected-empty": False,
            "model-selected-valid-lexically": [],
            "model-selected-but-lexically-rejected": [],
            "model-selected-unknown": [],
            "model-selected-outside-candidates": [],
            "empty_content_selected": [],
            "no_candidates": False,
            "model_called": True,
        }
        error = str(exc)
    return {
        "text": claim["text"],
        "kind": claim["kind"],
        "heading": claim.get("heading"),
        "candidate_count": selected["candidate_count"],
        "candidate_count_before_cap": selected["candidate_count_before_cap"],
        "truncated": selected["truncated"],
        "omitted_ids": selected["omitted_ids"],
        "candidates": [
            {
                "entity_id": item["entity_id"],
                "entity_type": item["entity_type"],
                "name": item["name"],
                "content": (item.get("content") or "")[:240],
                "empty_content": item["empty_content"],
                "score": item["score"],
                "reasons": item["reasons"],
            }
            for item in selected["candidates"]
        ],
        "selection": selection,
        "error": error,
    }


def _run(llm, spec: dict, prior: dict) -> dict:
    answer = prior["production_answer"]
    before = answer
    evidence = _evidence_from_tool(spec)
    stored_ids = set(prior.get("evidence_ids") or [])
    fetched_ids = {item.entity_id for item in evidence if item.entity_id}
    claims = extract_answer_claims(answer)
    rows: list[dict | None] = [None] * len(claims)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                _attribute_claim,
                llm,
                prior["question"],
                claim,
                evidence,
                list(prior.get("production_refs") or []),
            ): index
            for index, claim in enumerate(claims)
        }
        for future in as_completed(futures):
            index = futures[future]
            rows[index] = future.result()
    claim_rows = [row for row in rows if row is not None]
    metrics = measure_candidate_case(
        answer=before,
        claim_rows=claim_rows,
        malformed_ids=spec.get("malformed_ids"),
    )
    return {
        "id": prior["id"],
        "question": prior["question"],
        "production_answer": before,
        "production_refs": list(prior.get("production_refs") or []),
        "claims": claim_rows,
        "metrics": metrics,
        "answer_preserved": before == answer == prior["production_answer"],
        "evidence_id_mismatch": stored_ids != fetched_ids,
    }


def main() -> None:
    settings = get_settings()
    source = json.loads(PRIOR.read_text())
    corpus_ids = {
        row["id"]: row.get("evidence_ids") or []
        for row in json.loads(CORPUS.read_text()).get("cases") or []
        if row.get("id")
    }
    prior_rows = []
    for row in source.get("cases") or []:
        if row.get("production_answer") and not row.get("error"):
            row = dict(row)
            row["evidence_ids"] = corpus_ids.get(row["id"]) or row.get("evidence_ids") or []
            prior_rows.append(row)
    specs = _specs()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if OUT.exists():
        existing = {
            row["id"]: row
            for row in json.loads(OUT.read_text()).get("cases") or []
            if row.get("id") and row.get("answer_preserved") and row.get("metrics")
            and not any(claim.get("error") for claim in row.get("claims") or [])
        }
    llm = build_llm_provider(settings)
    rows: list[dict] = []
    for index, prior in enumerate(prior_rows, start=1):
        spec = specs.get(prior["id"])
        if spec is None:
            print(f"[{index}/{len(prior_rows)}] {prior['id']} skipped: no evidence spec", flush=True)
            continue
        if prior["id"] in existing:
            rows.append(existing[prior["id"]])
            print(f"[{index}/{len(prior_rows)}] {prior['id']} (cached)", flush=True)
            continue
        print(f"[{index}/{len(prior_rows)}] {prior['id']}", flush=True)
        try:
            row = _run(llm, spec, prior)
        except Exception as exc:
            row = {
                "id": prior["id"],
                "question": prior.get("question"),
                "error": str(exc),
                "trace": traceback.format_exc(),
            }
            print(f"  ERROR {exc}", flush=True)
        else:
            metrics = row["metrics"]
            print(
                f"  preserved={row['answer_preserved']} claims={metrics['substantive_claims']} "
                f"lexical={metrics['covered_claims']} model={metrics['model_covered_claims']} "
                f"empty={metrics['empty_mappings']} none={metrics['no_candidate_claims']} "
                f"wide={metrics['record_buckets']['4+']} trunc={metrics['truncations']}",
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
