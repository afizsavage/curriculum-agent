"""Shadow experiment: narrow support rules, note-cap protection, and note selectivity.

Uses the frozen production answers from the previous attribution run.
Does not call production generation and does not modify those answers.
The smallest-sufficient pass is recorded separately and does not replace the selection.
"""

from __future__ import annotations

import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.candidate_attribution import (
    attribute_one_claim,
    attribute_smallest_note_set,
    candidate_size_summary,
    classify_truncation,
    extract_answer_claims,
    group_evidence_note,
    measure_candidate_case,
    select_candidates,
)
from app.config import get_settings
from app.llm.provider import build_llm_provider
from scripts.eval_claim_shadow import LIVE_CASES, _evidence_from_tool

PRIOR = Path("data/diagnostics/claim_shadow/candidate_attribution.json")
CORPUS = Path("data/diagnostics/claim_shadow/claim_shadow_results.json")
OUT = Path("data/diagnostics/claim_shadow/note_selectivity.json")

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

    def merged(field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in measured:
            for key, count in (row["metrics"].get(field) or {}).items():
                counts[key] = counts.get(key, 0) + int(count)
        return counts

    substantive = total("substantive_claims")
    returned = total("returned_mappings")
    covered = total("covered_claims")
    relaxed = total("relaxed_covered_claims")
    with_claims = sum(1 for row in measured if row["metrics"].get("substantive_claims"))
    preserved = sum(1 for row in measured if row.get("answer_preserved"))
    refs_preserved = sum(1 for row in measured if row.get("refs_preserved"))
    counts: list[int] = []
    for row in measured:
        counts.extend(row["metrics"].get("candidate_counts") or [])
    truncation = merged("truncation_classes")
    return {
        "evidence_bearing_cases": len(measured),
        "cases_with_claims": with_claims,
        "substantive_claims": substantive,
        "returned_mappings": returned,
        "exact_matches": total("exact_matches"),
        "exact_span_compliance": (total("exact_matches") / substantive) if substantive else None,
        "model_coverage": (total("model_covered_claims") / substantive) if substantive else None,
        "legacy_coverage": (covered / substantive) if substantive else None,
        "relaxed_coverage": (relaxed / substantive) if substantive else None,
        "legacy_accepted_refs": total("legacy_accepted_refs"),
        "legacy_rejected_refs": total("unsupported_refs"),
        "relaxed_accepted_refs": total("relaxed_accepted_refs"),
        "relaxed_rejected_refs": total("relaxed_rejected_refs"),
        "empty_mappings": total("empty_mappings"),
        "unknown_refs": total("unknown_refs"),
        "outside_candidate_refs": total("outside_candidate_refs"),
        "empty_content_refs": total("empty_content_refs"),
        "no_candidate_claims": total("no_candidate_claims"),
        "record_buckets": merged("record_buckets"),
        "lexical_record_buckets": merged("lexical_record_buckets"),
        "relaxed_record_buckets": merged("relaxed_record_buckets"),
        "candidate_size": candidate_size_summary(counts, total("truncations")),
        "duplicates_removed": total("duplicates_removed"),
        "cap_removed": total("cap_removed"),
        "specific_content_retained": total("specific_content_retained"),
        "generic_structural_retained": total("generic_structural_retained"),
        "rescued_specific": total("rescued_specific"),
        "truncation_classes": truncation,
        "production_answer_preservation": (preserved / len(measured)) if measured else None,
        "production_refs_preservation": (refs_preserved / len(measured)) if measured else None,
        "evidence_id_mismatches": sum(1 for row in measured if row.get("evidence_id_mismatch")),
        "claim_errors": sum(1 for row in measured for claim in row.get("claims") or [] if claim.get("error")),
        "note_protected_candidates": sum(len(claim.get("note_protected_candidates") or []) for row in measured for claim in row.get("claims") or []),
        "note_protected_candidates_that_would_have_been_truncated": sum(len(claim.get("note_protected_candidates_that_would_have_been_truncated") or []) for row in measured for claim in row.get("claims") or []),
        "evidence_notes": sum(1 for row in measured for claim in row.get("claims") or [] if claim.get("kind") == "evidence_note"),
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
            kind=claim.get("kind"),
        )
        error = None
    except Exception as exc:
        selection = {
            "model_refs": [],
            "model-selected-empty": False,
            "model-selected-valid-lexically": [],
            "model-selected-but-lexically-rejected": [],
            "relaxed-supported": [],
            "relaxed-rejected": [],
            "model-selected-unknown": [],
            "model-selected-outside-candidates": [],
            "empty_content_selected": [],
            "no_candidates": False,
            "model_called": True,
            "support_labels": [],
        }
        error = str(exc)
    smallest = None
    note_group = None
    if claim.get("kind") == "evidence_note" and not error:
        try:
            smallest = attribute_smallest_note_set(
                llm,
                question=question,
                claim_text=str(claim["text"]),
                candidates=selected["candidates"],
                selected_refs=list(selection.get("model_refs") or []),
                evidence=evidence,
            )
        except Exception as exc:
            smallest = {"refs": [], "error": str(exc)}
        note_group = group_evidence_note(
            str(claim["text"]),
            selected["candidates"],
            list(selection.get("model_refs") or []),
        )
    truncation_class = classify_truncation(
        claim_text=str(claim["text"]),
        kind=claim.get("kind"),
        retained=selected["candidates"],
        removed_duplicates=selected["removed_duplicates"],
        removed_by_cap=selected["removed_by_cap"],
    )
    return {
        "text": claim["text"],
        "kind": claim["kind"],
        "heading": claim.get("heading"),
        "candidate_count": selected["candidate_count"],
        "candidates_considered": selected["candidates_considered"],
        "candidates_retained": selected["candidates_retained"],
        "candidates_removed_as_duplicates": selected["candidates_removed_as_duplicates"],
        "candidates_removed_by_cap": selected["candidates_removed_by_cap"],
        "truncated": selected["truncated"],
        "truncation_class": truncation_class,
        "removed_duplicates": selected["removed_duplicates"],
        "removed_by_cap": selected["removed_by_cap"],
        "rescued_specific_ids": selected["rescued_specific_ids"],
        "note_protected_candidates": selected["note_protected_candidates"],
        "note_protected_candidates_that_would_have_been_truncated": selected["note_protected_candidates_that_would_have_been_truncated"],
        "note_group": note_group,
        "smallest_sufficient": smallest,
        "specific_content_retained": selected["specific_content_retained"],
        "generic_structural_retained": selected["generic_structural_retained"],
        "candidates": [
            {
                "entity_id": item["entity_id"],
                "entity_type": item["entity_type"],
                "name": item["name"],
                "content": (item.get("content") or "")[:240],
                "empty_content": item["empty_content"],
                "generic_structural": item["generic_structural"],
                "reasons": item["reasons"],
            }
            for item in selected["candidates"]
        ],
        "selection": selection,
        "error": error,
    }


def _run(llm, spec: dict, prior: dict) -> dict:
    answer = prior["production_answer"]
    refs = list(prior.get("production_refs") or [])
    evidence = _evidence_from_tool(spec)
    stored_ids = set(prior.get("evidence_ids") or [])
    fetched_ids = {item.entity_id for item in evidence if item.entity_id}
    claims = extract_answer_claims(answer)
    rows: list[dict | None] = [None] * len(claims)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_attribute_claim, llm, prior["question"], claim, evidence, refs): index
            for index, claim in enumerate(claims)
        }
        for future in as_completed(futures):
            rows[futures[future]] = future.result()
    claim_rows = [row for row in rows if row is not None]
    metrics = measure_candidate_case(
        answer=answer,
        claim_rows=claim_rows,
        malformed_ids=spec.get("malformed_ids"),
    )
    return {
        "id": prior["id"],
        "question": prior["question"],
        "production_answer": answer,
        "production_refs": refs,
        "claims": claim_rows,
        "metrics": metrics,
        "answer_preserved": answer == prior["production_answer"],
        "refs_preserved": refs == list(prior.get("production_refs") or []),
        "evidence_id_mismatch": stored_ids != fetched_ids,
    }


def main() -> None:
    settings = get_settings()
    corpus_ids = {
        row["id"]: row.get("evidence_ids") or []
        for row in json.loads(CORPUS.read_text()).get("cases") or []
        if row.get("id")
    }
    prior_rows = []
    for row in json.loads(PRIOR.read_text()).get("cases") or []:
        if row.get("production_answer") and not row.get("error"):
            copied = dict(row)
            copied["evidence_ids"] = corpus_ids.get(row["id"]) or []
            prior_rows.append(copied)
    specs = _specs()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if OUT.exists():
        existing = {
            row["id"]: row
            for row in json.loads(OUT.read_text()).get("cases") or []
            if row.get("answer_preserved") and row.get("metrics")
            and not any(claim.get("error") for claim in row.get("claims") or [])
        }
    llm = build_llm_provider(settings)
    rows: list[dict] = []
    for index, prior in enumerate(prior_rows, start=1):
        spec = specs.get(prior["id"])
        if spec is None:
            print(f"[{index}/{len(prior_rows)}] {prior['id']} skipped", flush=True)
            continue
        if prior["id"] in existing:
            rows.append(existing[prior["id"]])
            print(f"[{index}/{len(prior_rows)}] {prior['id']} (cached)", flush=True)
            continue
        print(f"[{index}/{len(prior_rows)}] {prior['id']}", flush=True)
        try:
            row = _run(llm, spec, prior)
        except Exception as exc:
            row = {"id": prior["id"], "error": str(exc), "trace": traceback.format_exc()}
            print(f"  ERROR {exc}", flush=True)
        else:
            metrics = row["metrics"]
            print(
                f"  preserved={row['answer_preserved']} refs={row['refs_preserved']} "
                f"claims={metrics['substantive_claims']} model={metrics['model_covered_claims']} "
                f"legacy={metrics['covered_claims']} relaxed={metrics['relaxed_covered_claims']} "
                f"dupes={metrics['duplicates_removed']} trunc={metrics['truncations']} "
                f"classes={metrics['truncation_classes']}",
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
