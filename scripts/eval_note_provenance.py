"""Shadow comparison of model evidence-note refs with deterministic grouping.

Reads the frozen note-selectivity run. Does not call a model, does not
regenerate answers, and does not change ordinary claim attribution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.candidate_attribution import select_deterministic_note_refs

PRIOR = Path("data/diagnostics/claim_shadow/note_selectivity.json")
OUT = Path("data/diagnostics/claim_shadow/note_provenance.json")

# Human review of every ref that appears in the model selection or the
# deterministic selection. These labels are not produced by the grouper.
HUMAN_LABELS = {
    "topics-p4-math": {
        "f293e00a-bae7-44db-ad55-3c1263e20387": "NECESSARY",
        "0f90b316-6be5-4b60-a2b1-2ac91a603d49": "NECESSARY",
        "20b6fae8-05d2-4c27-9b2b-2d04f91e7422": "SUPPORTED BUT REDUNDANT",
        "684ebd01-8dc0-459b-a610-9d8dfea454ff": "NECESSARY",
        "152b4a5a-c080-403c-a780-9940c818bc4b": "NECESSARY",
        "1997d3c8-5ec2-43d7-9dac-9afcf97c1734": "SUPPORTED BUT REDUNDANT",
        "29558081-a6cf-4721-b52c-f6f3e59b12a7": "SUPPORTED BUT REDUNDANT",
        "c0a9dc68-e3f3-424e-8873-d2642e7cc71b": "NECESSARY",
    },
    "fractions-p4": {
        "51c4c17f-6b68-49bb-b025-f377e4e6f8a5": "NECESSARY",
    },
    "fractions-p3": {
        "1dd051f9-b841-4733-bf6b-704c819c46ba": "NECESSARY",
    },
    "fractions-p5": {
        "610e543f-0a3b-46ee-8d39-acc2480c5805": "NECESSARY",
        "e4710e7b-a33c-487f-91ac-636fa21fdb15": "UNSUPPORTED",
    },
    "time-p3": {
        "2ce1b82d-edf6-4287-a229-b15b999eb8e4": "NECESSARY",
    },
    "numbers-p1": {
        "1a97eec4-aa71-435f-98c0-99490e47eda6": "NECESSARY",
        "95b34608-df73-40b9-89d7-9e50e1e12054": "NECESSARY",
        "a79eb6ea-b0e7-4ff0-9e5d-aeeb98f8f80a": "SUPPORTED BUT REDUNDANT",
    },
    "multiplication-p4": {
        "351ef533-e32c-4582-8cd0-ecad956377e8": "NECESSARY",
        "36d36607-fa63-431b-9fd5-dedc3db40322": "NECESSARY",
        "0731389e-56b3-4f6f-a530-8c173659bee1": "NECESSARY",
        "51c4c17f-6b68-49bb-b025-f377e4e6f8a5": "NECESSARY",
    },
    "number-p2": {
        "8f782c51-1f25-429a-ba31-538834471f7e": "NECESSARY",
    },
    "topics-p2-math": {
        "660d3ebd-6809-4940-ba31-90a5c88da348": "NECESSARY",
        "0f85f7a4-9eee-435f-96d8-22899075d761": "NECESSARY",
        "82734b4a-42b3-483e-ba76-24a0f10762d1": "NECESSARY",
    },
    "fixture-p3-fractions": {
        "lo-p3-equivalent-incomplete": "NECESSARY",
    },
    "fixture-p4-fractions": {
        "lo-multiply-garbled": "NECESSARY",
    },
}


def _rates(labels: list[str]) -> dict[str, float | int]:
    total = len(labels)
    necessary = sum(label == "NECESSARY" for label in labels)
    redundant = sum(label == "SUPPORTED BUT REDUNDANT" for label in labels)
    unsupported = sum(label == "UNSUPPORTED" for label in labels)
    supported = necessary + redundant

    def rate(count: int) -> float | None:
        return (count / total) if total else None

    return {
        "refs": total,
        "necessary_refs": necessary,
        "redundant_refs": redundant,
        "unsupported_refs": unsupported,
        "supported_refs": supported,
        "necessary_ref_rate": rate(necessary),
        "redundant_ref_rate": rate(redundant),
        "unsupported_ref_rate": rate(unsupported),
        "supported_ref_rate": rate(supported),
    }


def main() -> None:
    document = json.loads(PRIOR.read_text())
    ordinary_claims = 0
    ordinary_ref_changes = 0
    notes = []
    current_labels: list[str] = []
    deterministic_labels: list[str] = []
    for case in document["cases"]:
        for claim in case.get("claims") or []:
            selection = claim.get("selection") or {}
            model_refs = list(selection.get("model_refs") or [])
            if claim.get("kind") != "evidence_note":
                ordinary_claims += 1
                ordinary_ref_changes += 0
                continue
            support = {ref: True for ref in selection.get("relaxed-supported") or []}
            support.update({ref: False for ref in selection.get("relaxed-rejected") or []})
            grouped = select_deterministic_note_refs(
                claim["text"],
                claim.get("candidates") or [],
                model_refs,
                support_by_id=support,
            )
            by_id = {item["entity_id"]: item for item in claim.get("candidates") or []}
            review = HUMAN_LABELS[case["id"]]
            union = list(dict.fromkeys([*model_refs, *grouped["deterministic_refs"]]))
            labeled = []
            for ref in union:
                label = review[ref]
                labeled.append({
                    "ref": ref,
                    "name": (by_id.get(ref) or {}).get("name"),
                    "in_current": ref in model_refs,
                    "in_deterministic": ref in grouped["deterministic_refs"],
                    "label": label,
                })
                if ref in model_refs:
                    current_labels.append(label)
                if ref in grouped["deterministic_refs"]:
                    deterministic_labels.append(label)
            notes.append({
                "case_id": case["id"],
                "note": claim["text"],
                "current_refs": model_refs,
                "deterministic_refs": grouped["deterministic_refs"],
                "refs_removed": grouped["refs_removed"],
                "refs_added": grouped["refs_added"],
                "named_issues": grouped["named_issues"],
                "refs_per_issue": grouped["refs_per_issue"],
                "grouping_status": grouped["note_grouping_status"],
                "unmatched_issues": grouped["unmatched_issues"],
                "support_failures": grouped["support_failures"],
                "human_review": labeled,
            })
    current_counts = [len(note["current_refs"]) for note in notes]
    deterministic_counts = [len(note["deterministic_refs"]) for note in notes]
    aggregate = document["aggregate"]
    summary = {
        "ordinary_claims": ordinary_claims,
        "ordinary_claim_ref_changes": ordinary_ref_changes,
        "evidence_notes": len(notes),
        "current_refs": {
            "average": mean(current_counts) if current_counts else None,
            "median": median(current_counts) if current_counts else None,
            "maximum": max(current_counts) if current_counts else None,
        },
        "deterministic_refs": {
            "average": mean(deterministic_counts) if deterministic_counts else None,
            "median": median(deterministic_counts) if deterministic_counts else None,
            "maximum": max(deterministic_counts) if deterministic_counts else None,
        },
        "refs_removed": sum(len(note["refs_removed"]) for note in notes),
        "refs_added": sum(len(note["refs_added"]) for note in notes),
        "current_human": _rates(current_labels),
        "deterministic_human": _rates(deterministic_labels),
        "invariants": {
            "substantive_claims": aggregate["substantive_claims"],
            "model_coverage": aggregate["model_coverage"],
            "exact_span_compliance": aggregate["exact_span_compliance"],
            "relaxed_support_coverage": aggregate["relaxed_coverage"],
            "unknown_ids": aggregate["unknown_refs"],
            "refs_outside_candidates": aggregate["outside_candidate_refs"],
            "empty_content_refs": aggregate["empty_content_refs"],
            "production_answer_preservation": aggregate["production_answer_preservation"],
            "production_refs_preservation": aggregate["production_refs_preservation"],
        },
    }
    OUT.write_text(json.dumps({"summary": summary, "notes": notes}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
