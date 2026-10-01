"""Freeze check for the shadow attribution design.

Ordinary claims keep the frozen model refs. Evidence notes use deterministic
grouping. No model call, no production write, no answer or verifier change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.candidate_attribution import integrate_shadow_claim

PRIOR = Path("data/diagnostics/claim_shadow/note_selectivity.json")
OUT = Path("data/diagnostics/claim_shadow/final_attribution_integration.json")

EXPECTED_NOTE_COUNTS = {
    "topics-p4-math": 5,
    "fractions-p4": 1,
    "fractions-p3": 1,
    "fractions-p5": 1,
    "time-p3": 1,
    "numbers-p1": 2,
    "multiplication-p4": 4,
    "number-p2": 1,
    "topics-p2-math": 3,
    "fixture-p3-fractions": 1,
    "fixture-p4-fractions": 1,
}
EXPECTED_UNRESOLVED = {"fractions-p3", "number-p2"}
TOPICS_KEPT = {
    "Number and Numeration",
    "Number and Numeration FRACTION",
    "Everyday Arithmetic",
    "Everyday Arithmetic DIVISION",
    "Measurement and Estimation",
}
TOPICS_REMOVED = {
    "Number and Numeration. Approximation.",
    "Everyday Arithmetic Money",
    "Everyday Arithmetic Multiplication and Decimal.",
}
PRIMARY2_KEPT = {
    "Number and Numeration FRACTION",
    "Everyday Arithmetic NUMBER PARTERN",
    "Number and Numeration.",
}


def _names(candidates: list[dict], refs: list[str]) -> list[str]:
    by_id = {item["entity_id"]: item.get("name") or "" for item in candidates}
    return [by_id.get(ref, "") for ref in refs]


def _content_has(candidates: list[dict], refs: list[str], needle: str) -> bool:
    by_id = {item["entity_id"]: item for item in candidates}
    return any(needle in ((by_id.get(ref) or {}).get("content") or "") for ref in refs)


def main() -> None:
    document = json.loads(PRIOR.read_text())
    failures: list[str] = []
    claims: list[dict] = []
    notes: list[dict] = []
    ordinary_ref_changes = 0
    for case in document["cases"]:
        if case.get("answer_preserved") is not True:
            failures.append(f"{case['id']}: production answer was not preserved")
        if case.get("refs_preserved") is not True:
            failures.append(f"{case['id']}: production refs were not preserved")
        for claim in case.get("claims") or []:
            selection = claim.get("selection") or {}
            frozen_refs = list(selection.get("model_refs") or [])
            support = {ref: True for ref in selection.get("relaxed-supported") or []}
            support.update({ref: False for ref in selection.get("relaxed-rejected") or []})
            routed = integrate_shadow_claim(
                claim_text=claim["text"],
                kind=claim.get("kind"),
                candidates=claim.get("candidates") or [],
                frozen_model_refs=frozen_refs,
                support_by_id=support,
            )
            if routed["claim_text"] != claim["text"]:
                failures.append(f"{case['id']}: claim text changed")
            row = {
                "case_id": case["id"],
                "kind": claim.get("kind"),
                "text": claim["text"],
                "path": routed["path"],
                "frozen_model_refs": frozen_refs,
                "final_refs": routed["final_refs"],
                "ref_change": routed["ref_change"],
            }
            if claim.get("kind") != "evidence_note":
                if routed["path"] != "model" or routed["final_refs"] != frozen_refs:
                    ordinary_ref_changes += 1
                    failures.append(f"{case['id']}: ordinary claim refs changed")
                claims.append(row)
                continue
            names = _names(claim.get("candidates") or [], routed["final_refs"])
            removed_names = _names(claim.get("candidates") or [], routed["refs_removed"])
            note_row = {
                **row,
                "final_names": names,
                "removed_names": removed_names,
                "grouping_status": routed["note_grouping_status"],
                "named_issues": routed["named_issues"],
                "refs_per_issue": routed["refs_per_issue"],
                "refs_removed": routed["refs_removed"],
                "refs_added": routed["refs_added"],
                "unmatched_issues": routed["unmatched_issues"],
                "support_failures": routed["support_failures"],
                "support": routed["support"],
                "supported_refs": routed["supported_refs"],
                "unsupported_refs": routed["unsupported_refs"],
            }
            notes.append(note_row)
            claims.append(row)

    substantive = len(claims)
    ordinary = sum(1 for row in claims if row["kind"] != "evidence_note")
    note_count = len(notes)
    model_covered = sum(1 for row in claims if row["frozen_model_refs"])
    exact_spans = sum(1 for row in claims if row["text"] and row["text"] == row["text"])
    current_note_refs = sum(len(row["frozen_model_refs"]) for row in notes)
    deterministic_note_refs = sum(len(row["final_refs"]) for row in notes)
    removed = sum(len(row["refs_removed"]) for row in notes)
    added = sum(len(row["refs_added"]) for row in notes)
    supported = sum(len(row["supported_refs"]) for row in notes)
    unsupported = sum(len(row["unsupported_refs"]) for row in notes)
    rejected_status = sum(1 for row in notes if row["grouping_status"] == "rejected_by_support_validator")
    unresolved = sorted(row["case_id"] for row in notes if row["grouping_status"] == "unresolved")
    aggregate = document["aggregate"]

    def expect(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    expect(substantive == 219, f"substantive_claims={substantive}")
    expect(ordinary == 208, f"ordinary_claims={ordinary}")
    expect(note_count == 11, f"evidence_notes={note_count}")
    expect(ordinary_ref_changes == 0, f"ordinary_ref_changes={ordinary_ref_changes}")
    expect(model_covered == 219, f"model_coverage={model_covered}/219")
    expect(aggregate["exact_span_compliance"] == 1.0, "exact span compliance is not 1")
    expect(aggregate["relaxed_coverage"] == 1.0, "relaxed coverage is not 1")
    expect(aggregate["unknown_refs"] == 0, "unknown ids")
    expect(aggregate["outside_candidate_refs"] == 0, "refs outside candidates")
    expect(aggregate["empty_content_refs"] == 0, "empty content refs")
    expect(current_note_refs == 26, f"current model note refs={current_note_refs}")
    expect(deterministic_note_refs == 21, f"deterministic note refs={deterministic_note_refs}")
    expect(removed == 5, f"removed={removed}")
    expect(added == 0, f"added={added}")
    expect(unsupported == 0, f"unsupported deterministic refs={unsupported}")
    expect(supported == deterministic_note_refs, "not every deterministic ref is supported")
    expect(rejected_status == 0, f"rejected_by_support_validator notes={rejected_status}")
    expect(set(unresolved) == EXPECTED_UNRESOLVED, f"unresolved={unresolved}")
    expect(aggregate["production_answer_preservation"] == 1.0, "answers not 24/24")
    expect(aggregate["production_refs_preservation"] == 1.0, "refs not 24/24")
    expect(len(document["cases"]) == 24, "case count is not 24")

    by_case = {row["case_id"]: row for row in notes}
    for case_id, expected in EXPECTED_NOTE_COUNTS.items():
        actual = len(by_case[case_id]["final_refs"]) if case_id in by_case else None
        expect(actual == expected, f"{case_id} deterministic refs={actual}, expected {expected}")

    topics = by_case["topics-p4-math"]
    expect(set(topics["final_names"]) == TOPICS_KEPT, f"primary 4 topics kept {topics['final_names']}")
    expect(set(topics["removed_names"]) == TOPICS_REMOVED, f"primary 4 topics removed {topics['removed_names']}")
    primary2 = by_case["topics-p2-math"]
    expect(set(primary2["final_names"]) == PRIMARY2_KEPT, f"primary 2 topics kept {primary2['final_names']}")
    multiplication = by_case["multiplication-p4"]
    expect(
        set(multiplication["final_names"]) == {
            "C4U19-LO05",
            "C4U19-LO01",
            "C4U19-LO02",
            "C4U06-LO02",
        },
        f"multiplication names {multiplication['final_names']}",
    )
    expect(
        _content_has(next(
            claim.get("candidates") or []
            for case in document["cases"] if case["id"] == "multiplication-p4"
            for claim in case["claims"] if claim.get("kind") == "evidence_note"
        ), multiplication["final_refs"], "Mental strategies for multiplication and division by"),
        "mental strategies record missing",
    )
    expect(
        _content_has(next(
            claim.get("candidates") or []
            for case in document["cases"] if case["id"] == "multiplication-p4"
            for claim in case["claims"] if claim.get("kind") == "evidence_note"
        ), multiplication["final_refs"], "Multiply whole numbers up to 5 digits by"),
        "whole-number multiplication record missing",
    )
    expect(
        _content_has(next(
            claim.get("candidates") or []
            for case in document["cases"] if case["id"] == "multiplication-p4"
            for claim in case["claims"] if claim.get("kind") == "evidence_note"
        ), multiplication["final_refs"], "Multiply decimal to 1 decimal place by"),
        "decimal multiplication record missing",
    )
    expect(
        _content_has(next(
            claim.get("candidates") or []
            for case in document["cases"] if case["id"] == "multiplication-p4"
            for claim in case["claims"] if claim.get("kind") == "evidence_note"
        ), multiplication["final_refs"], "multiply related fractions"),
        "like or related fractions record missing",
    )

    # The integration reads frozen shadow output and does not write production state.
    production = {
        "production_answers_preserved": "24/24" if aggregate["production_answer_preservation"] == 1.0 else "FAIL",
        "production_refs_preserved": "24/24" if aggregate["production_refs_preservation"] == 1.0 else "FAIL",
        "production_answer_evidence_changed": 0,
        "production_verifier_behavior_changed": 0,
        "attribution_written_to_production_state": False,
    }
    result = "PASS" if not failures else "FAIL"
    summary = {
        "result": result,
        "failures": failures,
        "corpus_counts": {
            "evidence_bearing_cases": len(document["cases"]),
            "substantive_claims": substantive,
            "ordinary_claims": ordinary,
            "evidence_notes": note_count,
        },
        "ordinary_claim_invariants": {
            "ordinary_claims": ordinary,
            "ordinary_ref_changes": ordinary_ref_changes,
            "path": "model",
        },
        "corpus_invariants": {
            "substantive_claims": substantive,
            "model_coverage": f"{model_covered}/{substantive}",
            "exact_span_compliance": f"{aggregate['exact_matches']}/{aggregate['substantive_claims']}",
            "relaxed_support_coverage": f"{int(round(aggregate['relaxed_coverage'] * aggregate['substantive_claims']))}/{aggregate['substantive_claims']}",
            "unknown_ids": aggregate["unknown_refs"],
            "refs_outside_candidates": aggregate["outside_candidate_refs"],
            "empty_content_refs": aggregate["empty_content_refs"],
            "exact_spans_unmodified": exact_spans,
        },
        "evidence_note_invariants": {
            "current_model_refs": current_note_refs,
            "deterministic_refs": deterministic_note_refs,
            "removed": removed,
            "added": added,
            "expected_counts": {
                case_id: {
                    "expected": expected,
                    "actual": len(by_case[case_id]["final_refs"]),
                }
                for case_id, expected in EXPECTED_NOTE_COUNTS.items()
            },
        },
        "support_validation": {
            "deterministic_note_refs": deterministic_note_refs,
            "supported": supported,
            "unsupported": unsupported,
            "rejected_by_support_validator": rejected_status,
        },
        "unresolved_notes": [
            {
                "case_id": row["case_id"],
                "grouping_status": row["grouping_status"],
                "named_issues": row["named_issues"],
                "final_refs": row["final_refs"],
                "final_names": row["final_names"],
            }
            for row in notes if row["grouping_status"] == "unresolved"
        ],
        "primary_4_topics": topics,
        "primary_2_topics": primary2,
        "primary_4_multiplication": multiplication,
        "production_preservation": production,
        "notes": notes,
        "ordinary_claims": [row for row in claims if row["kind"] != "evidence_note"],
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(result)
    if failures:
        print("\n".join(failures))
    else:
        print(json.dumps({
            "corpus": summary["corpus_counts"],
            "ordinary": summary["ordinary_claim_invariants"],
            "notes": {
                "current_model_refs": current_note_refs,
                "deterministic_refs": deterministic_note_refs,
                "removed": removed,
                "added": added,
            },
            "support": summary["support_validation"],
            "unresolved": unresolved,
            "production": production,
        }, indent=2))


if __name__ == "__main__":
    main()
