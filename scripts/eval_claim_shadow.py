"""Shadow live-model claim-span measurement. Does not change production answers."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.answer_generator import AnswerGenerator
from app.agent.claim_shadow import measure_shadow_case
from app.agent.state import CurriculumQAState
from app.config import get_settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.deps import get_tool_registry
from app.llm.provider import build_llm_provider
from tests.agent.test_answer_synthesis import (
    _primary3_fractions_evidence,
    _primary4_fractions_evidence,
)

OUT_DIR = Path("data/diagnostics/claim_shadow")

# Representative curriculum questions. Retrieval uses the existing curriculum
# tools; generation is the only place the shadow prompt is added.
LIVE_CASES: list[dict] = [
    {"id": "subjects-p3", "shape": "simple-list", "question": "What are the subjects in Primary 3?", "tool": "get_curriculum_structure", "args": {"grade": "CLASS_3"}, "grade": "CLASS_3"},
    {"id": "subjects-p4", "shape": "simple-list", "question": "What subjects are taught in Primary 4?", "tool": "get_curriculum_structure", "args": {"grade": "CLASS_4"}, "grade": "CLASS_4"},
    {"id": "topics-p4-math", "shape": "simple-list", "question": "What topics are taught in Primary 4 Mathematics?", "tool": "get_curriculum_structure", "args": {"grade": "CLASS_4", "subject": "MATHEMATICS"}, "grade": "CLASS_4", "subject": "MATHEMATICS"},
    {"id": "fractions-p4", "shape": "multi-unit", "question": "What does Primary 4 Mathematics teach about fractions?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_4", "subject": "MATHEMATICS", "topic": "fractions"}, "grade": "CLASS_4", "subject": "MATHEMATICS", "topic": "fractions"},
    {"id": "fractions-p3", "shape": "multi-unit", "question": "What should a Primary 3 pupil learn about fractions?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_3", "subject": "MATHEMATICS", "topic": "fractions"}, "grade": "CLASS_3", "subject": "MATHEMATICS", "topic": "fractions"},
    {"id": "fractions-p5", "shape": "multi-claim", "question": "What are the learning outcomes for fractions in Primary 5 Mathematics?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_5", "subject": "MATHEMATICS", "topic": "fractions"}, "grade": "CLASS_5", "subject": "MATHEMATICS", "topic": "fractions"},
    {"id": "money-p4", "shape": "multi-claim", "question": "What should Primary 4 pupils learn about money?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_4", "subject": "MATHEMATICS", "topic": "money"}, "grade": "CLASS_4", "subject": "MATHEMATICS", "topic": "money"},
    {"id": "time-p3", "shape": "multi-claim", "question": "What does Primary 3 Mathematics teach about time?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_3", "subject": "MATHEMATICS", "topic": "time"}, "grade": "CLASS_3", "subject": "MATHEMATICS", "topic": "time"},
    {"id": "shapes-p2", "shape": "short", "question": "What shapes do Primary 2 pupils learn?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_2", "subject": "MATHEMATICS", "topic": "shapes"}, "grade": "CLASS_2", "subject": "MATHEMATICS", "topic": "shapes"},
    {"id": "measurement-p4", "shape": "multi-claim", "question": "What should pupils learn about measurement in Primary 4 Mathematics?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_4", "subject": "MATHEMATICS", "topic": "measurement"}, "grade": "CLASS_4", "subject": "MATHEMATICS", "topic": "measurement"},
    {"id": "decimals-p5", "shape": "multi-claim", "question": "What does Primary 5 Mathematics teach about decimals?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_5", "subject": "MATHEMATICS", "topic": "decimals"}, "grade": "CLASS_5", "subject": "MATHEMATICS", "topic": "decimals"},
    {"id": "percentages-p6", "shape": "short", "question": "What should Primary 6 pupils learn about percentages?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_6", "subject": "MATHEMATICS", "topic": "percentage"}, "grade": "CLASS_6", "subject": "MATHEMATICS", "topic": "percentage"},
    {"id": "reading-p3", "shape": "multi-claim", "question": "What does Primary 3 English teach about reading?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_3", "subject": "ENGLISH", "topic": "reading"}, "grade": "CLASS_3", "subject": "ENGLISH", "topic": "reading"},
    {"id": "writing-p4", "shape": "multi-claim", "question": "What should Primary 4 pupils learn about writing?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_4", "subject": "ENGLISH", "topic": "writing"}, "grade": "CLASS_4", "subject": "ENGLISH", "topic": "writing"},
    {"id": "grammar-p5", "shape": "multi-claim", "question": "What grammar is taught in Primary 5 English?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_5", "subject": "ENGLISH", "topic": "grammar"}, "grade": "CLASS_5", "subject": "ENGLISH", "topic": "grammar"},
    {"id": "plants-p4", "shape": "multi-claim", "question": "What should Primary 4 pupils learn about plants?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_4", "subject": "SCIENCE", "topic": "plants"}, "grade": "CLASS_4", "subject": "SCIENCE", "topic": "plants"},
    {"id": "animals-p3", "shape": "multi-claim", "question": "What does Primary 3 Science teach about animals?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_3", "subject": "SCIENCE", "topic": "animals"}, "grade": "CLASS_3", "subject": "SCIENCE", "topic": "animals"},
    {"id": "human-body-p5", "shape": "multi-claim", "question": "What should Primary 5 pupils learn about the human body?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_5", "subject": "SCIENCE", "topic": "human body"}, "grade": "CLASS_5", "subject": "SCIENCE", "topic": "human body"},
    {"id": "water-p4", "shape": "short", "question": "What does Primary 4 Science say about water?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_4", "subject": "SCIENCE", "topic": "water"}, "grade": "CLASS_4", "subject": "SCIENCE", "topic": "water"},
    {"id": "environment-p6", "shape": "multi-claim", "question": "What should Primary 6 pupils learn about the environment?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_6", "subject": "SCIENCE", "topic": "environment"}, "grade": "CLASS_6", "subject": "SCIENCE", "topic": "environment"},
    {"id": "social-p4", "shape": "simple-list", "question": "What topics are in Primary 4 Social Studies?", "tool": "get_curriculum_structure", "args": {"grade": "CLASS_4", "subject": "SOCIAL STUDIES"}, "grade": "CLASS_4", "subject": "SOCIAL STUDIES"},
    {"id": "limited-quantum", "shape": "limited-evidence", "question": "What should Primary 4 pupils learn about quantum computing?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_4", "subject": "MATHEMATICS", "topic": "quantum computing"}, "grade": "CLASS_4", "subject": "MATHEMATICS", "topic": "quantum computing"},
    {"id": "numbers-p1", "shape": "short", "question": "What numbers do Primary 1 pupils learn?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_1", "subject": "MATHEMATICS", "topic": "numbers"}, "grade": "CLASS_1", "subject": "MATHEMATICS", "topic": "numbers"},
    {"id": "addition-p2", "shape": "multi-record", "question": "What should Primary 2 pupils learn about addition?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_2", "subject": "MATHEMATICS", "topic": "addition"}, "grade": "CLASS_2", "subject": "MATHEMATICS", "topic": "addition"},
    {"id": "length-p4", "shape": "short", "question": "What should Primary 4 pupils learn about length?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_4", "subject": "MATHEMATICS", "topic": "length"}, "grade": "CLASS_4", "subject": "MATHEMATICS", "topic": "length"},
    {"id": "multiplication-p4", "shape": "multi-claim", "question": "What does Primary 4 Mathematics teach about multiplication?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_4", "subject": "MATHEMATICS", "topic": "multiplication"}, "grade": "CLASS_4", "subject": "MATHEMATICS", "topic": "multiplication"},
    {"id": "number-p2", "shape": "multi-claim", "question": "What should Primary 2 pupils learn about numbers?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_2", "subject": "MATHEMATICS", "topic": "number"}, "grade": "CLASS_2", "subject": "MATHEMATICS", "topic": "number"},
    {"id": "ratio-p6", "shape": "multi-claim", "question": "What should Primary 6 pupils learn about ratio?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_6", "subject": "MATHEMATICS", "topic": "ratio"}, "grade": "CLASS_6", "subject": "MATHEMATICS", "topic": "ratio"},
    {"id": "geometry-p5", "shape": "multi-claim", "question": "What does Primary 5 Mathematics teach about geometry?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_5", "subject": "MATHEMATICS", "topic": "geometry"}, "grade": "CLASS_5", "subject": "MATHEMATICS", "topic": "geometry"},
    {"id": "topics-p2-math", "shape": "simple-list", "question": "What topics are taught in Primary 2 Mathematics?", "tool": "get_curriculum_structure", "args": {"grade": "CLASS_2", "subject": "MATHEMATICS"}, "grade": "CLASS_2", "subject": "MATHEMATICS"},
    {"id": "statistics-p4", "shape": "short", "question": "What should Primary 4 pupils learn about statistics?", "tool": "resolve_curriculum_context", "args": {"grade": "CLASS_4", "subject": "MATHEMATICS", "topic": "statistics"}, "grade": "CLASS_4", "subject": "MATHEMATICS", "topic": "statistics"},
]


def _evidence_from_tool(case: dict) -> list[CurriculumEvidence]:
    if case.get("fixture") == "p3":
        return _primary3_fractions_evidence()
    if case.get("fixture") == "p4":
        return _primary4_fractions_evidence()
    result = get_tool_registry().execute(case["tool"], **case["args"])
    if not result.success:
        return []
    raw = (result.data or {}).get("evidence") or []
    evidence = []
    for item in raw:
        if isinstance(item, CurriculumEvidence):
            evidence.append(item)
        elif isinstance(item, dict):
            evidence.append(CurriculumEvidence.model_validate(item))
    return evidence


def _state_for(case: dict, evidence: list[CurriculumEvidence]) -> CurriculumQAState:
    state = CurriculumQAState.initial(question=case["question"])
    state.grade = case.get("grade")
    state.subject = case.get("subject")
    state.topic = case.get("topic")
    state.evidence = evidence
    state.evidence_status = EvidenceStatus.FOUND if evidence else EvidenceStatus.NOT_FOUND
    return state


def _run_case(generator: AnswerGenerator, case: dict) -> dict:
    evidence = _evidence_from_tool(case)
    production_state = _state_for(case, evidence)
    production = generator.generate(production_state)
    shadow_state = _state_for(case, evidence)
    shadow_state.metadata["claim_shadow"] = True
    shadow = generator.generate(shadow_state)
    metrics = measure_shadow_case(
        production_answer=production.answer,
        shadow_answer=shadow.answer,
        claim_report=shadow_state.metadata.get("claim_attribution"),
        malformed_ids=case.get("malformed_ids"),
    )
    return {
        "id": case["id"],
        "shape": case["shape"],
        "question": case["question"],
        "evidence_ids": [item.entity_id for item in evidence if item.entity_id],
        "evidence_count": len(evidence),
        "production_answer": production.answer,
        "production_refs": [ref.entity_id for ref in production.evidence],
        "shadow_answer": shadow.answer,
        "shadow_claims": (shadow_state.metadata.get("claim_attribution") or {}).get("returned_claims"),
        "validation": {
            "status": (shadow_state.metadata.get("claim_attribution") or {}).get("status"),
            "valid_claims": (shadow_state.metadata.get("claim_attribution") or {}).get("valid_claims"),
            "absent_claims": (shadow_state.metadata.get("claim_attribution") or {}).get("absent_claims"),
            "unsupported_claims": (shadow_state.metadata.get("claim_attribution") or {}).get("unsupported_claims"),
            "invalid_refs": (shadow_state.metadata.get("claim_attribution") or {}).get("invalid_refs"),
            "unattributed_claims": (shadow_state.metadata.get("claim_attribution") or {}).get("unattributed_claims"),
        },
        "metrics": metrics,
    }


def _aggregate(rows: list[dict]) -> dict:
    metrics = [row["metrics"] for row in rows if "metrics" in row]
    def total(key: str) -> int:
        return sum(int(item.get(key) or 0) for item in metrics)
    returned = total("returned_mappings")
    exact = total("exact_matches")
    substantive = total("substantive_claims")
    covered = total("covered_claims")
    invalid = total("invalid_refs") + total("unsupported_refs")
    deltas: dict[str, int] = {}
    shapes: dict[str, int] = {}
    for row in rows:
        if "metrics" not in row:
            continue
        delta = row["metrics"]["answer_delta"]
        deltas[delta] = deltas.get(delta, 0) + 1
        shapes[row["shape"]] = shapes.get(row["shape"], 0) + 1
    complete = sum(1 for item in metrics if item.get("complete"))
    return {
        "cases": len(metrics),
        "failed_cases": sum(1 for row in rows if row.get("error")),
        "shapes": shapes,
        "substantive_claims": substantive,
        "returned_mappings": returned,
        "exact_matches": exact,
        "near_misses": total("near_misses"),
        "missing_mappings": total("missing_mappings"),
        "extra_mappings": total("extra_mappings"),
        "invalid_refs": total("invalid_refs"),
        "unsupported_refs": total("unsupported_refs"),
        "multi_record_claims": total("multi_record_claims"),
        "malformed_evidence_mappings": total("malformed_evidence_mappings"),
        "exact_span_match_rate": (exact / returned) if returned else None,
        "claim_coverage": (covered / substantive) if substantive else None,
        "invalid_attribution_rate": (invalid / returned) if returned else None,
        "case_level_complete_attribution": (complete / len(metrics)) if metrics else None,
        "answer_delta": deltas,
        "over_attribution_cases": sum(1 for item in metrics if item.get("over_attribution")),
        "under_attribution_cases": sum(1 for item in metrics if item.get("under_attribution")),
    }


def main() -> None:
    settings = get_settings()
    cases = list(LIVE_CASES)
    cases.extend(
        [
            {
                "id": "fixture-p3-fractions",
                "shape": "malformed-evidence",
                "question": "What should a Primary 3 pupil learn about fractions?",
                "fixture": "p3",
                "grade": "CLASS_3",
                "subject": "MATHEMATICS",
                "topic": "fractions",
                "malformed_ids": ["lo-p3-equivalent-incomplete"],
            },
            {
                "id": "fixture-p4-fractions",
                "shape": "malformed-evidence",
                "question": "What does Primary 4 Mathematics teach about fractions?",
                "fixture": "p4",
                "grade": "CLASS_4",
                "subject": "MATHEMATICS",
                "topic": "fractions",
                "malformed_ids": ["lo-multiply-garbled"],
            },
        ]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "claim_shadow_results.json"
    existing: dict[str, dict] = {}
    if results_path.exists():
        prior = json.loads(results_path.read_text())
        existing = {row["id"]: row for row in prior.get("cases") or [] if row.get("id") and not row.get("error")}
    generator = AnswerGenerator(build_llm_provider(settings))
    rows: list[dict] = []
    for index, case in enumerate(cases, start=1):
        if case["id"] in existing:
            rows.append(existing[case["id"]])
            print(f"[{index}/{len(cases)}] {case['id']} (cached)", flush=True)
            continue
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        try:
            row = _run_case(generator, case)
        except Exception as exc:
            row = {"id": case["id"], "shape": case["shape"], "question": case["question"], "error": str(exc), "trace": traceback.format_exc()}
            print(f"  ERROR {exc}", flush=True)
        else:
            metrics = row["metrics"]
            print(
                f"  evidence={row['evidence_count']} delta={metrics['answer_delta']} "
                f"exact={metrics['exact_matches']}/{metrics['returned_mappings']} "
                f"missing={metrics['missing_mappings']} near={metrics['near_misses']}",
                flush=True,
            )
        rows.append(row)
        results_path.write_text(json.dumps({
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
    results_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
