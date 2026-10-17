"""Separate Marcus evaluation set.

This module does not call a model and does not write the frozen gold set, the
assistant behaviour set, or historical trace reports. Official scores use
curriculum gold scoring v1. v2 and v3 are secondary labels only.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from app.assistant.curriculum_gold import (
    SCORING_VERSION,
    SCORING_VERSION_V2,
    SCORING_VERSION_V3,
    aggregate,
    file_sha256,
    load_dataset as load_gold_dataset,
    score_observation,
)
from app.assistant.evaluation import dataset_path as behaviour_dataset_path

DATASET_VERSION = "marcus-evaluation-v1"
OFFICIAL_SCORING_VERSION = SCORING_VERSION
SECONDARY_SCORING_VERSIONS = (SCORING_VERSION_V2, SCORING_VERSION_V3)
_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "evaluation" / "marcus_evaluation_v1.json"
)
_BASELINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "evaluation"
    / "marcus_evaluation_v1_baseline.json"
)
_BATCH17_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "traces"
    / "curriculum_qa_gold_v1_batch17.json"
)
_ALLOWED_OUTPUT_NAMES = frozenset(
    {"marcus_evaluation_v1.json", "marcus_evaluation_v1_baseline.json"}
)
_STAGE_REPORT_NAME = "marcus_evaluation_v1_stage2_absence.json"
_STATUS_FIELDS = (
    "ready_to_score",
    "requires_fact_enumeration",
    "saved_answer_available",
    "requires_live_run",
)


class MarcusNotScoreable(ValueError):
    """The case has no enumerated factual contract."""


class MarcusReportRefused(RuntimeError):
    """The output path could overwrite a historical evaluation artifact."""


def dataset_path() -> Path:
    return _DATASET_PATH


def baseline_path() -> Path:
    return _BASELINE_PATH


def assert_safe_output(path: Path) -> Path:
    """Refuse every path except the two new Marcus files.

    Existing report writers accept any path. This guard is the one used for
    Marcus output so a baseline run cannot replace gold, behaviour, or Batch 17.
    """
    resolved = path.expanduser().resolve()
    allowed = _ALLOWED_OUTPUT_NAMES | {_STAGE_REPORT_NAME}
    if resolved.name not in allowed:
        raise MarcusReportRefused(
            f"Refusing to write '{resolved.name}'. Marcus output is limited to "
            "marcus_evaluation_v1.json, marcus_evaluation_v1_baseline.json, and "
            f"{_STAGE_REPORT_NAME}."
        )
    if resolved.name == _STAGE_REPORT_NAME and resolved.exists():
        raise MarcusReportRefused(f"Refusing to overwrite the existing stage report {resolved}.")
    blocked = {
        _gold_path().resolve(),
        behaviour_dataset_path().resolve(),
        _BATCH17_PATH.resolve(),
        (_gold_path().parent / "curriculum_qa_eval_v1.json").resolve(),
    }
    if resolved in blocked:
        raise MarcusReportRefused(f"Refusing to overwrite {resolved}.")
    if "traces" in resolved.parts:
        raise MarcusReportRefused(
            "Refusing to write a Marcus file into data/traces, where historical reports live."
        )
    return resolved


def _gold_path() -> Path:
    from app.assistant.curriculum_gold import dataset_path as gold_dataset_path

    return gold_dataset_path()


def load_dataset(path: Path | None = None) -> dict[str, Any]:
    source = path or _DATASET_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate_dataset(payload)
    return payload


def _validate_dataset(payload: dict[str, Any]) -> None:
    if payload.get("version") != DATASET_VERSION:
        raise ValueError(f"dataset version must be {DATASET_VERSION}")
    if payload.get("official_scoring_version") != OFFICIAL_SCORING_VERSION:
        raise ValueError("Marcus official scoring version must stay v1")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 24:
        raise ValueError("Marcus dataset must contain the 24 proposed cases")
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or case_id in seen:
            raise ValueError("Marcus case ids must be unique strings")
        seen.add(case_id)
        for field in _STATUS_FIELDS:
            if not isinstance(case.get(field), bool):
                raise ValueError(f"{case_id} is missing boolean status {field}")
        facts = case.get("required_facts")
        if case["requires_fact_enumeration"] or not case["ready_to_score"]:
            if facts is not None:
                raise ValueError(
                    f"{case_id} is not scoreable and must not carry a fact list"
                )
        elif not isinstance(facts, list):
            raise ValueError(f"{case_id} is scoreable but has no fact list")
        if case["question_match"] == "exact" and case.get("historical_question"):
            if case["question"] != case["historical_question"]:
                raise ValueError(f"{case_id} is marked exact but the questions differ")
        if case["question_match"] == "approximate":
            if not case.get("question_difference"):
                raise ValueError(f"{case_id} approximate match needs a written difference")
            if case["question"] == case.get("historical_question"):
                raise ValueError(f"{case_id} is marked approximate but the questions match")
        _validate_evaluation_contract(case)


def score_marcus_case(
    case: dict[str, Any],
    observation: dict[str, Any],
    *,
    scoring_version: str = OFFICIAL_SCORING_VERSION,
) -> dict[str, Any]:
    """Score one answer. Unenumerated cases cannot pass by having no facts."""
    if scoring_version not in {OFFICIAL_SCORING_VERSION, *SECONDARY_SCORING_VERSIONS}:
        raise ValueError(f"unsupported scoring version {scoring_version}")
    if case.get("requires_fact_enumeration") or not case.get("ready_to_score"):
        raise MarcusNotScoreable(
            f"{case.get('id')} is not factually scoreable: "
            f"{case.get('not_scoreable_reason') or 'facts are not enumerated'}"
        )
    if not isinstance(case.get("required_facts"), list):
        raise MarcusNotScoreable(f"{case.get('id')} has no enumerated fact list")
    metrics = score_observation(case, observation, scoring_version=scoring_version)
    metrics["evaluation"] = "marcus"
    metrics["evaluation_type"] = case.get("evaluation_type")
    metrics["question_match"] = case.get("question_match")
    metrics["execution"] = "rescore_or_supplied_answer"
    # v1 treats a blank answer plus an empty fact list as an abstention when the
    # status is already insufficient_evidence. An empty fact list is not itself a pass.
    if case.get("required_facts") == [] and not (observation.get("answer") or "").strip():
        metrics["deterministic_pass"] = False
        categories = list(metrics.get("failure_categories") or [])
        if "blank_answer_is_not_an_abstention" not in categories:
            categories.append("blank_answer_is_not_an_abstention")
        metrics["failure_categories"] = categories
    return metrics


def _validate_evaluation_contract(case: dict[str, Any]) -> None:
    """Check a typed contract when the case declares one.

    Cases built by ``construct_dataset`` omit ``evaluation_type``. Those remain valid.
    """
    evaluation_type = case.get("evaluation_type")
    if not evaluation_type:
        return
    case_id = case["id"]
    facts = case.get("required_facts")
    if evaluation_type == "factual_list":
        if not case["ready_to_score"] or not facts:
            raise ValueError(f"{case_id} factual list needs enumerated facts")
        for fact in facts:
            if not fact.get("source_ids") or not fact.get("match_any"):
                raise ValueError(f"{case_id} fact {fact.get('fact_id')} has no source")
    elif evaluation_type == "identity_lookup":
        if not case["ready_to_score"] or not facts:
            raise ValueError(f"{case_id} identity lookup needs enumerated identities")
        if len(facts) > 8:
            raise ValueError(f"{case_id} identity lookup is not a full unit list")
        for fact in facts:
            if not fact.get("source_ids") or not fact.get("match_any"):
                raise ValueError(f"{case_id} fact {fact.get('fact_id')} has no source")
    elif evaluation_type == "catalogue_abstention":
        if not case["ready_to_score"]:
            raise ValueError(f"{case_id} catalogue abstention is marked unready")
        if facts != []:
            raise ValueError(f"{case_id} abstention must not invent required sentences")
        if case.get("support") != "insufficient" or not case.get("absence"):
            raise ValueError(f"{case_id} abstention needs an insufficient-support contract")
        if not case.get("source_records") and not case.get("forbidden_substitutions"):
            raise ValueError(f"{case_id} abstention has no source records")
    elif evaluation_type == "lesson_plan":
        if case["ready_to_score"] or facts is not None:
            raise ValueError(f"{case_id} lesson plan must stay unscored until its own rubric exists")
        if not case.get("curriculum_scope_facts") or not case.get("lesson_rubric"):
            raise ValueError(f"{case_id} lesson plan needs scope facts and a lesson rubric")
    elif evaluation_type == "blocked_enumeration":
        if case["ready_to_score"] or not case["requires_fact_enumeration"] or facts is not None:
            raise ValueError(f"{case_id} blocked case must not pass factual scoring")
        if not case.get("source_limitation"):
            raise ValueError(f"{case_id} blocked case must document the missing records")
    elif evaluation_type == "not_applicable":
        if case["ready_to_score"] or facts is not None:
            raise ValueError(f"{case_id} has no curriculum contract")
    elif evaluation_type in {"partial_fact_list"}:
        if not case["ready_to_score"] or not facts:
            raise ValueError(f"{case_id} partial list needs enumerated facts")
    else:
        raise ValueError(f"{case_id} has an unknown evaluation type {evaluation_type}")


def construct_dataset() -> dict[str, Any]:
    """Build the Marcus document from the frozen gold file. Gold is only read."""
    gold = load_gold_dataset()
    by_id = {case["id"]: case for case in gold["cases"]}
    cases: list[dict[str, Any]] = []
    for spec in _EXACT_LINKS:
        cases.append(_from_gold(by_id[spec[2]], spec, question_match="exact"))
    for spec in _APPROXIMATE_LINKS:
        cases.append(_from_gold(by_id[spec[2]], spec, question_match="approximate"))
    cases.append(_identity_case(by_id["gold-c4-math-units-complete"]))
    cases.extend(_lesson_fact_cases(by_id))
    cases.extend(_UNSCOREABLE)
    cases.sort(key=lambda case: _CASE_ORDER.index(case["id"]))
    return {
        "version": DATASET_VERSION,
        "description": (
            "Teacher-assistant evaluation separate from curriculum gold and from "
            "assistant behaviour. Facts are copied from the frozen gold set. "
            "Cases without an enumerated contract are not factually scoreable."
        ),
        "official_scoring_version": OFFICIAL_SCORING_VERSION,
        "secondary_scoring_versions": list(SECONDARY_SCORING_VERSIONS),
        "gold_dataset": gold["version"],
        "gold_dataset_sha256": file_sha256(),
        "behaviour_dataset_sha256": _sha256(behaviour_dataset_path()),
        "live_executed": False,
        "cases": cases,
    }


def _from_gold(
    gold_case: dict[str, Any],
    spec: tuple[Any, ...],
    *,
    question_match: str,
) -> dict[str, Any]:
    case_id, group, gold_id = spec[:3]
    question = gold_case["question"] if question_match == "exact" else spec[3]
    difference = None if question_match == "exact" else spec[4]
    saved = True
    return {
        "id": case_id,
        "group": group,
        "question": question,
        "grade_code": gold_case["grade_code"],
        "subject_code": gold_case["subject_code"],
        "grade_labels": list(gold_case["grade_labels"]),
        "subject_labels": list(gold_case["subject_labels"]),
        "support": gold_case["support"],
        "must_be_complete": gold_case["must_be_complete"],
        "uncertainty": gold_case["uncertainty"],
        "required_facts": deepcopy(gold_case["required_facts"]),
        "forbidden_substitutions": deepcopy(gold_case.get("forbidden_substitutions") or []),
        "unsupported_patterns": deepcopy(gold_case.get("unsupported_patterns") or []),
        "absence": deepcopy(gold_case.get("absence")),
        "ready_to_score": True,
        "requires_fact_enumeration": False,
        "saved_answer_available": saved,
        "requires_live_run": question_match != "exact",
        "question_match": question_match,
        "historical_case_id": gold_id,
        "historical_question": gold_case["question"],
        "question_difference": difference,
        "comparability": (
            "Exact question, grade, subject, and fact contract."
            if question_match == "exact"
            else difference
        ),
        "fact_source": "curriculum-qa-gold-v1",
        "factual_scoring": True,
        "not_scoreable_reason": None,
    }


def _identity_case(math_units: dict[str, Any]) -> dict[str, Any]:
    fact = next(item for item in math_units["required_facts"] if item["source_code"] == "C4-U01")
    return {
        "id": "marcus-c4-math-u01-identity",
        "group": "curriculum_lookup",
        "question": (
            "What is the content code and name of the Class 4 Mathematics unit "
            "Number and Numeration?"
        ),
        "grade_code": math_units["grade_code"],
        "subject_code": math_units["subject_code"],
        "grade_labels": list(math_units["grade_labels"]),
        "subject_labels": list(math_units["subject_labels"]),
        "support": "answerable",
        "must_be_complete": True,
        "uncertainty": "not_applicable",
        "required_facts": [deepcopy(fact)],
        "forbidden_substitutions": [],
        "unsupported_patterns": [],
        "absence": None,
        "ready_to_score": True,
        "requires_fact_enumeration": False,
        "saved_answer_available": False,
        "requires_live_run": True,
        "question_match": "not_comparable",
        "historical_case_id": "gold-c4-math-units-complete",
        "historical_question": math_units["question"],
        "question_difference": (
            "The historical question asks for every Class 4 Mathematics unit. "
            "This case asks only for the C4-U01 code and name. The saved list "
            "answer is not scored as this lookup."
        ),
        "comparability": "Fact copied from the frozen unit list. Answer not transferable.",
        "fact_source": "curriculum-qa-gold-v1",
        "factual_scoring": True,
        "not_scoreable_reason": None,
    }


def _lesson_fact_cases(by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        (
            "marcus-teach-c4-u01",
            "gold-c4-math-u01-outcomes-complete",
            (
                "Plan a lesson from the published Class 4 Mathematics C4-U01 outcomes, "
                "and suggest how to introduce them."
            ),
            "Curriculum channel must contain the five frozen outcomes. "
            "Teaching suggestions stay off that channel. No period count.",
        ),
        (
            "marcus-teach-solar",
            "gold-c4-science-solar-outcomes-complete",
            "Draft lesson objectives from the Class 4 Science solar-system outcomes only.",
            "Curriculum channel must contain the seven frozen outcomes and no further outcome.",
        ),
        (
            "marcus-teach-bio-topics",
            "gold-sss1-bio-intro-topics-complete",
            (
                "Prepare a lesson using only the topics under SSS 1 Biology, "
                "Introduction to Biology."
            ),
            "Curriculum channel must contain the three frozen topics. LO04 is not required.",
        ),
    ]
    built = []
    for case_id, gold_id, question, channel in specs:
        gold_case = by_id[gold_id]
        built.append(
            {
                "id": case_id,
                "group": "lesson_preparation",
                "question": question,
                "grade_code": gold_case["grade_code"],
                "subject_code": gold_case["subject_code"],
                "grade_labels": list(gold_case["grade_labels"]),
                "subject_labels": list(gold_case["subject_labels"]),
                "support": gold_case["support"],
                "must_be_complete": gold_case["must_be_complete"],
                "uncertainty": gold_case["uncertainty"],
                "required_facts": deepcopy(gold_case["required_facts"]),
                "forbidden_substitutions": deepcopy(gold_case.get("forbidden_substitutions") or []),
                "unsupported_patterns": deepcopy(gold_case.get("unsupported_patterns") or []),
                "absence": deepcopy(gold_case.get("absence")),
                "ready_to_score": True,
                "requires_fact_enumeration": False,
                "saved_answer_available": False,
                "requires_live_run": True,
                "question_match": "not_comparable",
                "historical_case_id": gold_id,
                "historical_question": gold_case["question"],
                "question_difference": (
                    "The historical question asks for the published list. "
                    "This question also asks for lesson preparation. The saved list "
                    "answer is not a lesson answer and is not scored here."
                ),
                "comparability": "Facts copied. Historical answer not transferable.",
                "channel_expectation": channel,
                "fact_source": "curriculum-qa-gold-v1",
                "factual_scoring": True,
                "not_scoreable_reason": None,
            }
        )
    return built


def write_dataset(path: Path | None = None) -> Path:
    destination = assert_safe_output(path or _DATASET_PATH)
    payload = construct_dataset()
    _validate_dataset(payload)
    _write_json(destination, payload)
    return destination


def build_baseline_report(
    *,
    batch17_path: Path | None = None,
) -> dict[str, Any]:
    """Rescore comparable saved answers and compare selector windows.

    This reads Batch 17. It does not write it and does not call a model.
    """
    dataset = construct_dataset()
    _validate_dataset(dataset)
    source = batch17_path or _BATCH17_PATH
    if source.resolve() == _gold_path().resolve():
        raise MarcusReportRefused("The gold file is not an answer report.")
    report = json.loads(source.read_text(encoding="utf-8"))
    saved = {row["case_id"]: row for row in report["cases"]}
    exact_rows: list[dict[str, Any]] = []
    approximate_rows: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    traced: set[str] = set()
    for case in dataset["cases"]:
        result, selector = _assess_case(case, saved, traced)
        case_results.append(result)
        if selector is not None:
            selector_rows.append(selector)
        if result["execution"] == "rescored_historical" and result["question_match"] == "exact":
            exact_rows.append(result["official_v1"])
        elif result["execution"] == "approximate_rescore":
            approximate_rows.append(result["official_v1"])
    return {
        "dataset": DATASET_VERSION,
        "official_scoring_version": OFFICIAL_SCORING_VERSION,
        "secondary_scoring_versions": list(SECONDARY_SCORING_VERSIONS),
        "mode": "rescore_saved_answers",
        "live_executed": False,
        "assistant_enabled": False,
        "gold_dataset_sha256": file_sha256(),
        "behaviour_dataset_sha256": _sha256(behaviour_dataset_path()),
        "source_report": str(source),
        "source_report_sha256": _sha256(source),
        "case_count": 24,
        "exact_saved_answer_overlaps": sum(
            1 for row in case_results if row["question_match"] == "exact" and row["trace_question_matches"]
        ),
        "approximate_saved_answer_overlaps": sum(
            1 for row in case_results if row["execution"] == "approximate_rescore"
        ),
        "official_metrics_exact_v1": aggregate(exact_rows),
        "secondary_metrics_exact": {
            version: aggregate(
                [
                    row["secondary"][version]
                    for row in case_results
                    if row["question_match"] == "exact" and row.get("secondary")
                ]
            )
            for version in SECONDARY_SCORING_VERSIONS
        },
        "approximate_metrics_v1": aggregate(approximate_rows) if approximate_rows else None,
        "selector_difference_count": sum(
            1 for row in selector_rows if row["result"] != "no selector difference"
        ),
        "cases": case_results,
        "selector_audit": selector_rows,
    }


def write_stage_report(payload: dict[str, Any], path: Path | None = None) -> Path:
    """Write one new stage report. An existing file is refused."""
    destination = assert_safe_output(
        path or (_DATASET_PATH.parent / _STAGE_REPORT_NAME)
    )
    if destination.name != _STAGE_REPORT_NAME:
        raise MarcusReportRefused("Stage reports use the stage-2 absence filename.")
    _write_json(destination, payload)
    return destination


def write_baseline_report(path: Path | None = None, *, batch17_path: Path | None = None) -> Path:
    destination = assert_safe_output(path or _BASELINE_PATH)
    payload = build_baseline_report(batch17_path=batch17_path)
    _write_json(destination, payload)
    return destination


def _assess_case(
    case: dict[str, Any],
    saved: dict[str, dict[str, Any]],
    traced: set[str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    historical_id = case.get("historical_case_id")
    row = saved.get(historical_id) if historical_id else None
    trace_question = None
    trace_id = None
    trace_path = None
    if row is not None:
        trace_id = row["trace"]["agent_run_id"]
        trace_path = Path("data/traces") / f"{trace_id}.json"
        if trace_path.is_file():
            trace_question = json.loads(trace_path.read_text(encoding="utf-8")).get("question")
    trace_matches_gold = bool(
        trace_question and trace_question == case.get("historical_question")
    )
    question_match = case["question_match"]
    execution = "not_executed"
    official = None
    secondary = None
    if (
        question_match == "exact"
        and case["ready_to_score"]
        and row is not None
        and trace_matches_gold
        and case["question"] == trace_question
    ):
        execution = "rescored_historical"
        official, secondary = _score_saved(case, row)
    elif (
        question_match == "approximate"
        and case["ready_to_score"]
        and row is not None
        and trace_matches_gold
    ):
        execution = "approximate_rescore"
        official, secondary = _score_saved(case, row)
    selector = None
    if row is not None and trace_path is not None and trace_path.is_file() and historical_id not in traced:
        selector = _selector_audit(
            case_id=historical_id,
            question=case.get("historical_question") or "",
            trace_path=trace_path,
            trace_id=trace_id,
        )
        traced.add(historical_id)
    result = {
        "id": case["id"],
        "group": case["group"],
        "question_match": question_match,
        "ready_to_score": case["ready_to_score"],
        "requires_fact_enumeration": case["requires_fact_enumeration"],
        "saved_answer_available": bool(row) and question_match in {"exact", "approximate"},
        "requires_live_run": execution != "rescored_historical",
        "execution": execution,
        "scoreability": _scoreability(case, execution),
        "evidence_provenance": {
            "historical_case_id": historical_id,
            "agent_run_id": trace_id,
            "trace_question_matches_gold": trace_matches_gold,
            "source": "batch17" if execution != "not_executed" else None,
        },
        "trace_question_matches": trace_matches_gold and case["question"] == trace_question,
        "comparability": case.get("comparability"),
        "question_difference": case.get("question_difference"),
        "official_scoring_version": OFFICIAL_SCORING_VERSION if official else None,
        "official_v1": _public_metrics(official) if official else None,
        "secondary": (
            {version: _public_metrics(metrics) for version, metrics in secondary.items()}
            if secondary
            else None
        ),
        "selector_result": selector["result"] if selector else "no saved evidence bag",
    }
    return result, selector


def _score_saved(
    case: dict[str, Any], row: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    observation = {
        "answer": row.get("answer") or "",
        "status": row.get("status"),
        "evidence_status": row.get("evidence_status"),
        "coverage_decision": row.get("coverage_decision"),
        "verification_passed": row.get("verification_passed"),
        "termination_reason": row.get("termination_reason"),
        "evidence": list(row.get("evidence") or row.get("evidence_refs") or []),
    }
    official = score_marcus_case(case, observation, scoring_version=OFFICIAL_SCORING_VERSION)
    secondary = {
        version: score_marcus_case(case, observation, scoring_version=version)
        for version in SECONDARY_SCORING_VERSIONS
    }
    return official, secondary


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": metrics["case_id"],
        "scoring_version": metrics["scoring_version"],
        "factual_recall": metrics.get("factual_recall"),
        "factual_precision": metrics.get("factual_precision"),
        "completeness": metrics.get("completeness"),
        "evidence_abstention": metrics.get("evidence_abstention"),
        "subject_grade_accuracy": metrics.get("subject_grade_accuracy"),
        "unsupported_claims": metrics.get("unsupported_claims"),
        "unsupported_claim_rate": metrics.get("unsupported_claim_rate"),
        "deterministic_pass": metrics.get("deterministic_pass"),
        "failure_categories": metrics.get("failure_categories") or [],
        "diagnostic_notes": metrics.get("diagnostic_notes") or [],
        "termination_reason": metrics.get("termination_reason"),
        "scored": True,
    }


def _scoreability(case: dict[str, Any], execution: str) -> str:
    if case["requires_fact_enumeration"]:
        return "not_scoreable_until_facts_are_enumerated"
    if not case["ready_to_score"]:
        return "not_factually_scoreable"
    if execution == "rescored_historical":
        return "rescored_exact_historical_answer"
    if execution == "approximate_rescore":
        return "rescored_approximate_historical_answer"
    return "scoreable_when_a_live_answer_exists"


def _selector_audit(
    *,
    case_id: str,
    question: str,
    trace_path: Path,
    trace_id: str | None,
) -> dict[str, Any]:
    from app.agent.answer_generator import select_evidence_for_prompt
    from app.agent.question_purpose import classify_question_purpose
    from app.curriculum.evidence import CurriculumEvidence

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    evidence = _evidence_from_trace(trace, CurriculumEvidence)
    purpose = classify_question_purpose(question)
    _old, old_ids = select_evidence_for_prompt(
        evidence, question=question, question_purpose=None
    )
    _new, new_ids = select_evidence_for_prompt(
        evidence, question=question, question_purpose=purpose
    )
    same = old_ids == new_ids
    return {
        "historical_case_id": case_id,
        "agent_run_id": trace_id,
        "question_purpose": purpose,
        "evidence_records": len(evidence),
        "purpose_omitted_count": len(old_ids),
        "purpose_aware_count": len(new_ids),
        "ids_and_order_equal": same,
        "result": "no selector difference" if same else "window differs",
        "counted_as_improvement": False,
    }


def _evidence_from_trace(trace: dict[str, Any], evidence_type: type) -> list[Any]:
    items = []
    seen: set[str] = set()
    for event in trace.get("events") or []:
        if event.get("event") != "agent.evidence.add" or event.get("duplicate"):
            continue
        if event.get("disposition") not in {None, "new"}:
            continue
        entity_id = event.get("id")
        if not entity_id or entity_id in seen:
            continue
        seen.add(entity_id)
        items.append(
            evidence_type(
                entity_type=event.get("entity_type") or "unknown",
                entity_id=entity_id,
                name=event.get("name"),
                grade=event.get("grade"),
                subject=event.get("subject"),
                topic=event.get("topic"),
                content=event.get("content_preview"),
                metadata={},
                source=event.get("source") or "curriculum_api",
            )
        )
    return items


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_EXACT_LINKS: tuple[tuple[str, str, str], ...] = (
    ("marcus-c4-science-first", "curriculum_lookup", "gold-c4-science-not-integrated"),
    ("marcus-jss1-is-first", "curriculum_lookup", "gold-jss1-integrated-science-first-unit"),
    ("marcus-bio-intro-topics", "curriculum_lookup", "gold-sss1-bio-intro-topics-complete"),
    ("marcus-c4-science-all", "complete_list", "gold-c4-science-content-complete"),
    ("marcus-c4-math-units", "complete_list", "gold-c4-math-units-complete"),
    ("marcus-c3-u15-outcomes", "complete_list", "gold-c3-math-u15-outcomes-complete"),
    ("marcus-c4-solar-outcomes", "complete_list", "gold-c4-science-solar-outcomes-complete"),
    ("marcus-c4-ela-term1", "complete_list", "gold-c4-ela-term1-units-complete"),
    ("marcus-c4-french", "unavailable_source", "gold-c4-french-subject-absent"),
    ("marcus-c4-astronomy", "unavailable_source", "gold-c4-math-astronomy-absent"),
)

_APPROXIMATE_LINKS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "marcus-c4-u01-outcomes",
        "complete_list",
        "gold-c4-math-u01-outcomes-complete",
        "What are all the learning outcomes for Class 4 Mathematics unit C4-U01?",
        "Proposed wording drops ', Number and Numeration' from the historical question. "
        "The grade, subject, and five outcome facts are the same. The saved answer was "
        "produced for the longer question, so it is an approximate rescore, not an exact one.",
    ),
    (
        "marcus-c4-u01-periods",
        "unavailable_source",
        "gold-c4-math-u01-periods-partial",
        "What are the C4-U01 outcomes, and how many recommended periods does that unit assign?",
        "Proposed wording is shorter than the historical question. Both ask for the "
        "C4-U01 outcomes and the recommended period count. The saved answer was produced "
        "for the longer question, so it is an approximate rescore, not an exact one.",
    ),
)

_UNSCOREABLE: tuple[dict[str, Any], ...] = (
    {
        "id": "marcus-c4-u12-absent",
        "group": "curriculum_lookup",
        "question": "Is C4-U12 a published Class 4 Mathematics unit?",
        "grade_code": "CLASS_4",
        "subject_code": "MATHEMATICS",
        "ready_to_score": False,
        "requires_fact_enumeration": True,
        "saved_answer_available": False,
        "requires_live_run": True,
        "question_match": "none",
        "required_facts": None,
        "factual_scoring": False,
        "fact_status": "grounded_note_without_a_score_contract",
        "grounding": "The frozen gold source notes say Class 4 Mathematics codes skip C4-U12.",
        "not_scoreable_reason": "No abstention contract has been enumerated for this yes/no question.",
        "behavior_expectation": "Say the code is not published. Do not invent a unit title.",
    },
    {
        "id": "marcus-c4-english-subject",
        "group": "curriculum_lookup",
        "question": "What units does the Class 4 English subject cover?",
        "grade_code": "CLASS_4",
        "subject_code": "ENGLISH",
        "ready_to_score": False,
        "requires_fact_enumeration": True,
        "saved_answer_available": False,
        "requires_live_run": True,
        "question_match": "none",
        "required_facts": None,
        "factual_scoring": False,
        "fact_status": "grounded_note_without_a_score_contract",
        "grounding": "The frozen gold source notes say Class 4 subject rows do not include ENGLISH.",
        "not_scoreable_reason": "No absence contract has been enumerated for subject ENGLISH.",
        "behavior_expectation": "Do not present English Language units as the English subject.",
    },
    {
        "id": "marcus-lesson-opener",
        "group": "lesson_preparation",
        "question": "Give me a lesson idea for introducing a new idea before I open the textbook.",
        "grade_code": None,
        "subject_code": None,
        "ready_to_score": False,
        "requires_fact_enumeration": False,
        "saved_answer_available": False,
        "requires_live_run": True,
        "question_match": "not_applicable",
        "required_facts": None,
        "factual_scoring": False,
        "fact_status": "not_applicable",
        "behaviour_case_id": "gen-lesson-idea",
        "not_scoreable_reason": "This is a general-channel question. It has no curriculum facts to score.",
        "behavior_expectation": "General AI assistance only. Do not state a syllabus fact.",
    },
    {
        "id": "marcus-plan-unspecified",
        "group": "lesson_preparation",
        "question": "Plan tomorrow's lesson.",
        "grade_code": None,
        "subject_code": None,
        "ready_to_score": False,
        "requires_fact_enumeration": False,
        "saved_answer_available": False,
        "requires_live_run": True,
        "question_match": "not_applicable",
        "required_facts": None,
        "factual_scoring": False,
        "fact_status": "not_applicable",
        "not_scoreable_reason": "No grade or subject is given, and no curriculum facts apply.",
        "behavior_expectation": "Ask for the grade and subject. Do not invent a unit or outcomes.",
    },
    {
        "id": "marcus-teach-fractions",
        "group": "lesson_preparation",
        "question": "How should I teach fractions to Class 4?",
        "grade_code": "CLASS_4",
        "subject_code": "MATHEMATICS",
        "ready_to_score": False,
        "requires_fact_enumeration": True,
        "saved_answer_available": False,
        "requires_live_run": True,
        "question_match": "none",
        "required_facts": None,
        "factual_scoring": False,
        "fact_status": "not_enumerated",
        "not_scoreable_reason": "No Class 4 fraction outcomes are frozen, so a list cannot be scored.",
        "behavior_expectation": "Do not invent fraction outcomes. Teaching advice is not a syllabus fact.",
    },
    {
        "id": "marcus-c4-integrated-science",
        "group": "unavailable_source",
        "question": "What is the first published unit of Class 4 Integrated Science?",
        "grade_code": "CLASS_4",
        "subject_code": "INTEGRATED_SCIENCE",
        "ready_to_score": False,
        "requires_fact_enumeration": True,
        "saved_answer_available": False,
        "requires_live_run": True,
        "question_match": "none",
        "required_facts": None,
        "factual_scoring": False,
        "fact_status": "grounded_note_without_a_score_contract",
        "grounding": (
            "The JSS 1 gold rationale says Class 4 does not publish an Integrated Science "
            "grade curriculum. The source notes say Class 4 subject rows do not include "
            "INTEGRATED_SCIENCE."
        ),
        "not_scoreable_reason": "The absence is noted in gold prose, not as a scoreable absence contract.",
        "behavior_expectation": "Abstain. Do not return the JSS 1 unit or Class 4 Science C4-U01.",
    },
    {
        "id": "marcus-p4-astronomy-subject",
        "group": "unavailable_source",
        "question": "What topics are taught in Primary 4 Astronomy?",
        "grade_code": "CLASS_4",
        "subject_code": "ASTRONOMY",
        "ready_to_score": False,
        "requires_fact_enumeration": True,
        "saved_answer_available": False,
        "requires_live_run": True,
        "question_match": "none",
        "required_facts": None,
        "factual_scoring": False,
        "fact_status": "not_enumerated",
        "behaviour_case_id": "curr-astronomy-missing",
        "not_scoreable_reason": "The frozen gold file does not establish whether a Primary 4 Astronomy subject exists.",
        "behavior_expectation": "Abstain only after the catalogue is checked. Do not invent topics.",
    },
    {
        "id": "marcus-c4-french-appreciation",
        "group": "unavailable_source",
        "question": "What units does Class 4 French Appreciation cover?",
        "grade_code": "CLASS_4",
        "subject_code": "FRENCH_APPRECIATION",
        "ready_to_score": False,
        "requires_fact_enumeration": True,
        "saved_answer_available": False,
        "requires_live_run": True,
        "question_match": "none",
        "required_facts": None,
        "factual_scoring": False,
        "fact_status": "not_enumerated",
        "grounding": (
            "The Class 4 French gold absence says FRENCH_APPRECIATION is a different subject "
            "and must not be substituted for FRENCH. Its unit sentences are not frozen."
        ),
        "not_scoreable_reason": "French Appreciation unit sentences have not been enumerated.",
        "behavior_expectation": "Do not label those rows as the Class 4 French subject.",
    },
)

_CASE_ORDER: tuple[str, ...] = (
    "marcus-c4-science-first",
    "marcus-jss1-is-first",
    "marcus-bio-intro-topics",
    "marcus-c4-math-u01-identity",
    "marcus-c4-u12-absent",
    "marcus-c4-english-subject",
    "marcus-c4-science-all",
    "marcus-c4-math-units",
    "marcus-c4-u01-outcomes",
    "marcus-c3-u15-outcomes",
    "marcus-c4-solar-outcomes",
    "marcus-c4-ela-term1",
    "marcus-lesson-opener",
    "marcus-plan-unspecified",
    "marcus-teach-c4-u01",
    "marcus-teach-solar",
    "marcus-teach-bio-topics",
    "marcus-teach-fractions",
    "marcus-c4-french",
    "marcus-c4-astronomy",
    "marcus-c4-u01-periods",
    "marcus-c4-integrated-science",
    "marcus-p4-astronomy-subject",
    "marcus-c4-french-appreciation",
)
