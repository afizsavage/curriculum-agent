#!/usr/bin/env python3
"""Phase 1E offline diagnosis for a production-shadow regression case.

Does not modify production behavior. Reads existing shadow JSONL only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent.v213d_phase1e_diagnostics import (  # noqa: E402
    enrich_existing_record,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash-prefix", default="dd1c57")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=ROOT / "data" / "diagnostics" / "v213d_shadow.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "diagnostics" / "v213d_phase1e_regression_dd1c57.json",
    )
    args = parser.parse_args()

    rows = []
    for line in args.jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qh = str((row.get("question") or {}).get("hash") or "")
        if qh.startswith(args.hash_prefix):
            rows.append(enrich_existing_record(row))
    if not rows:
        print(json.dumps({"error": "no matching rows", "prefix": args.hash_prefix}))
        return 1

    # Prefer the post-corpus DOCUMENT_NOISE instance if multiple.
    target = rows[-1]
    for row in rows:
        if (row.get("comparison") or {}).get("control_correct_shadow_worse"):
            target = row
            break

    control = target.get("control") or {}
    shadow = target.get("shadow") or {}
    diagnostics = target.get("diagnostics") or {}
    report = {
        "case": "phase1e_regression_diagnosis",
        "question_hash": (target.get("question") or {}).get("hash"),
        "timestamp": target.get("timestamp"),
        "corpus_epoch": target.get("corpus_epoch"),
        "classification": (target.get("comparison") or {}).get("classification"),
        "regression_cause": diagnostics.get("regression_cause"),
        "control": {
            "evidence_count": control.get("evidence_count"),
            "verifier_decision": control.get("verifier_decision"),
            "verifier_score": control.get("verifier_score"),
            "mapper_recommendation": control.get("mapper_recommendation"),
            "final_route": control.get("final_route"),
            "final_accepted": control.get("final_accepted"),
            "unsupported_claims": control.get("unsupported_claims"),
            "answer_hash": control.get("answer_hash"),
        },
        "shadow": {
            "structured_evidence_count": shadow.get("structured_evidence_count"),
            "document_evidence_count": shadow.get("document_evidence_count"),
            "merged_evidence_count": shadow.get("evidence_count"),
            "document_passages": shadow.get("document_passages"),
            "verifier_decision": shadow.get("verifier_decision"),
            "verifier_score": shadow.get("verifier_score"),
            "mapper_recommendation": shadow.get("mapper_recommendation"),
            "final_route": shadow.get("final_route"),
            "final_accepted": shadow.get("final_accepted"),
            "unsupported_claims": shadow.get("unsupported_claims"),
            "answer_hash": shadow.get("answer_hash"),
        },
        "transitions": (diagnostics.get("deltas") or {}),
        "document_value": diagnostics.get("document_value"),
        "analysis": {
            "control_retrieved": (
                "Structured curriculum entities only "
                f"(n={control.get('evidence_count')}); no documents."
            ),
            "shadow_passages": [
                {
                    "rank": p.get("retrieval_rank"),
                    "document_id": p.get("document_id"),
                    "source_url": p.get("source_url"),
                    "page": p.get("page_number"),
                    "subject": p.get("subject"),
                    "grade": p.get("grade"),
                    "topic": p.get("topic"),
                    "score": p.get("retrieval_score"),
                }
                for p in (shadow.get("document_passages") or [])
            ],
            "semantic_relevance_note": (
                "Top hits include BEC framework (general mathematics) and one "
                "math-primary money passage. Question subject was null/mixed; "
                "passages are topically adjacent to Class-4 mathematics/money "
                "but not strictly filtered by subject."
            ),
            "generator_input_change": (
                "Shadow merged structured evidence with 5 document passages "
                f"(merged_count={shadow.get('evidence_count')}). "
                "Answer hash changed vs control."
            ),
            "claim_change": (
                "Shadow introduced unsupported claim(s) aligned with document "
                "guidance wording about Class-4 money teaching."
            ),
            "verifier_change": (
                f"{control.get('verifier_decision')}@{control.get('verifier_score')} → "
                f"{shadow.get('verifier_decision')}@{shadow.get('verifier_score')}"
            ),
            "mapper_change": (
                f"{control.get('mapper_recommendation')} → "
                f"{shadow.get('mapper_recommendation')}"
            ),
            "route_change": (
                f"{control.get('final_route')}/{control.get('final_accepted')} → "
                f"{shadow.get('final_route')}/{shadow.get('final_accepted')}"
            ),
            "decisive_transition": (
                "finish/accepted → fallback/not-accepted after document-conditioned "
                "regeneration produced a claim the verifier did not accept."
            ),
            "cause_rationale": (
                "Primary cause GENERATOR_DOCUMENT_DRIFT: answer changed after "
                "documents were added and unsupported claims increased, driving "
                "verifier accept→retrieve_more and mapper accept→reject. "
                "Not metadata-blocked; not wrong-context."
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "ok": True,
        "path": str(args.out),
        "regression_cause": report["regression_cause"],
        "route": report["analysis"]["route_change"],
        "verifier": report["analysis"]["verifier_change"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
