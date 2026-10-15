"""Score curriculum answers against the frozen gold set.

The dataset is the authority for expected facts. This module does not call a
model unless ``run_live`` is used, and that path is refused unless the process
opts in. Scoring rules are fixed by ``SCORING_VERSION`` before any live run.

A plausible sentence that does not contain a gold fact is not marked correct.
If a non-empty answer misses every gold fact, the case is sent to semantic
review instead of being treated as a supported paraphrase.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from app.config import Settings

DATASET_VERSION = "curriculum-qa-gold-v1"
SCORING_VERSION = "curriculum-qa-gold-scoring-v1"
_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "evaluation"
    / "curriculum_qa_gold_v1.json"
)
_GENERAL_LABEL = "General AI assistance"
_ABSTAIN_STATUSES = frozenset({"insufficient_evidence", "needs_clarification"})
_MISSING_PART_PHRASES = (
    "not recorded",
    "not published",
    "no recommended",
    "does not assign",
    "does not say",
    "not available",
    "not in the syllabus",
    "not in the curriculum",
    "cannot",
    "no period",
)
_NEGATIONS = (
    " not ",
    " no ",
    " instead",
    " cannot",
    " can't",
    " do not",
    " does not",
    " isn't",
    " is not",
)


class GoldEvaluationRefused(RuntimeError):
    """The live gold run is not allowed in this process."""


def dataset_path() -> Path:
    return _DATASET_PATH


def load_dataset(path: Path | None = None) -> dict[str, Any]:
    source = path or _DATASET_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("version") != DATASET_VERSION:
        raise ValueError(f"dataset version must be {DATASET_VERSION}")
    if payload.get("scoring_version") != SCORING_VERSION:
        raise ValueError(f"scoring version must be {SCORING_VERSION}")
    seen: set[str] = set()
    for case in payload["cases"]:
        case_id = case["id"]
        if case_id in seen:
            raise ValueError(f"duplicate case id {case_id}")
        seen.add(case_id)
        _validate_case(case)
    return payload


def file_sha256(path: Path | None = None) -> str:
    source = path or _DATASET_PATH
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return digest


def run_audit(*, path: Path | None = None) -> dict[str, Any]:
    """Validate the frozen set. This does not call the QA agent."""
    source = path or _DATASET_PATH
    payload = load_dataset(source)
    cases = payload["cases"]
    scored = [case for case in cases if case.get("factual_scoring", True)]
    return {
        "dataset": DATASET_VERSION,
        "scoring_version": SCORING_VERSION,
        "dataset_sha256": file_sha256(source),
        "mode": "audit",
        "live_executed": False,
        "case_count": len(cases),
        "factual_scoring_count": len(scored),
        "required_fact_count": sum(len(case["required_facts"]) for case in scored),
        "by_support": _count(cases, "support"),
        "by_subject": _count(cases, "subject_code"),
        "by_grade": _count(cases, "grade_code"),
        "complete_list_cases": sum(1 for case in cases if case["must_be_complete"]),
        "source_limitations": _limitations(payload),
        "metrics": {
            "factual_precision": _unmeasured("Live answers were not scored."),
            "factual_recall": _unmeasured("Live answers were not scored."),
            "unsupported_claim_rate": _unmeasured("Live answers were not scored."),
            "completeness": _unmeasured("Live answers were not scored."),
            "subject_grade_accuracy": _unmeasured("Live answers were not scored."),
            "evidence_abstention": _unmeasured("Live answers were not scored."),
            "channel_integrity": _unmeasured(
                "Channel integrity applies when an assistant response separates general advice."
            ),
        },
        "failures": [],
        "semantic_review": [],
    }


def score_observation(case: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    """Score one answer. ``observation`` is not written back into the dataset."""
    if not case.get("factual_scoring", True):
        return {
            "case_id": case["id"],
            "scored": False,
            "excluded_reason": "Authoritative records for this case could not be enumerated.",
            "failure_categories": ["source_data_missing_or_inconsistent"],
            "semantic_review_required": False,
        }
    answer = observation.get("answer") or ""
    recalled = []
    missed = []
    for fact in case["required_facts"]:
        if _fact_recalled(fact, answer):
            recalled.append(fact["fact_id"])
        else:
            missed.append(fact["fact_id"])
    asserted_forbidden = []
    negated_forbidden = []
    for item in case.get("forbidden_substitutions") or []:
        if _forbidden_asserted(item, answer):
            asserted_forbidden.append(item["text"])
        elif _forbidden_present(item, answer):
            negated_forbidden.append(item["text"])
    pattern_hits = _pattern_hits(case, answer)
    unsupported = len(asserted_forbidden) + len(pattern_hits)
    supported = len(recalled)
    checkable = supported + unsupported
    required = case["required_facts"]
    recall = (supported / len(required)) if required else None
    precision = (supported / checkable) if checkable else None
    unsupported_rate = (unsupported / checkable) if checkable else None
    completeness = _completeness(case, recall, unsupported)
    subject_grade = _subject_grade(case, observation, answer)
    abstention = _abstention(case, observation, answer, asserted_forbidden, pattern_hits)
    channel = _channel_integrity(observation, answer)
    missing_answer = not answer.strip()
    semantic = bool(
        (not missing_answer and required and recall == 0)
        or negated_forbidden
    )
    metrics = {
        "case_id": case["id"],
        "subject_code": case.get("subject_code"),
        "grade_code": case.get("grade_code"),
        "support": case["support"],
        "scored": True,
        "recalled_fact_ids": recalled,
        "missed_fact_ids": missed,
        "factual_recall": recall,
        "factual_precision": precision,
        "unsupported_claim_rate": unsupported_rate,
        "unsupported_claims": unsupported,
        "completeness": completeness,
        "subject_grade_accuracy": subject_grade,
        "evidence_abstention": abstention,
        "channel_integrity": channel,
        "semantic_review_required": semantic,
        "missing_answer": missing_answer,
    }
    metrics["deterministic_pass"] = _passed(metrics)
    metrics["failure_categories"] = diagnose(case, observation, metrics)
    return metrics


def diagnose(
    case: dict[str, Any],
    observation: dict[str, Any],
    metrics: dict[str, Any],
) -> list[str]:
    """Classify a failed observation. This does not change the QA pipeline."""
    if metrics.get("deterministic_pass") and not metrics.get("semantic_review_required"):
        if observation.get("verification_passed") is False:
            return ["verification_incorrectly_rejected"]
        return []
    categories: list[str] = []
    if metrics.get("semantic_review_required"):
        categories.append("semantic_review_required")
    status = observation.get("status")
    evidence_status = observation.get("evidence_status")
    if case["support"] != "insufficient" and (
        evidence_status in {"not_found", "error"} or status == "insufficient_evidence"
    ):
        categories.append("retrieval_missed_relevant_records")
    evidence = list(observation.get("evidence") or [])
    evidence_ids = {item.get("entity_id") for item in evidence if item.get("entity_id")}
    if _wrong_scope(case, evidence):
        categories.append("context_preparation_wrong_scope")
    missed_ids = set(metrics.get("missed_fact_ids") or [])
    omitted = False
    for fact in case.get("required_facts") or []:
        if fact["fact_id"] in missed_ids and any(
            source_id in evidence_ids for source_id in fact["source_ids"]
        ):
            omitted = True
    if omitted:
        categories.append("generation_omitted_evidence")
    if metrics.get("unsupported_claims") and evidence:
        categories.append("generation_misstated_evidence")
    if (
        case.get("must_be_complete")
        and metrics.get("completeness") is False
        and observation.get("coverage_decision") == "accept"
    ):
        categories.append("coverage_validation_missed_incomplete_answer")
    failed_answer = (
        metrics.get("completeness") is False
        or metrics.get("unsupported_claims")
        or metrics.get("evidence_abstention") is False
        or metrics.get("factual_recall") not in {None, 1}
    )
    if observation.get("verification_passed") is True and failed_answer:
        categories.append("verification_incorrectly_accepted")
    if "truncat" in (observation.get("termination_reason") or "").lower():
        categories.append("answer_rendering_or_truncation")
    if case["support"] == "insufficient" and metrics.get("evidence_abstention") is False:
        categories.append("insufficient_evidence_behaviour_failed")
    return categories


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in results if row.get("scored")]
    return {
        "factual_precision": _ratio(scored, "factual_precision"),
        "factual_recall": _ratio(scored, "factual_recall"),
        "unsupported_claim_rate": _ratio(scored, "unsupported_claim_rate"),
        "completeness": _pass_ratio(scored, "completeness"),
        "subject_grade_accuracy": _pass_ratio(scored, "subject_grade_accuracy"),
        "evidence_abstention": _pass_ratio(scored, "evidence_abstention"),
        "channel_integrity": _pass_ratio(scored, "channel_integrity"),
    }


def assert_live_allowed(settings: Settings) -> None:
    if os.environ.get("CURRICULUM_GOLD_EVAL_LIVE") != "1":
        raise GoldEvaluationRefused(
            "Set CURRICULUM_GOLD_EVAL_LIVE=1 to score the gold set against the QA agent."
        )
    if str(settings.environment or "").strip().lower() == "production":
        raise GoldEvaluationRefused("Live gold evaluation is refused in production.")
    if settings.assistant_enabled:
        raise GoldEvaluationRefused("Turn ASSISTANT_ENABLED off before the live gold evaluation.")
    provider = (settings.llm_provider or "").strip().lower()
    if provider in {"", "stub"}:
        raise GoldEvaluationRefused("Live gold evaluation needs a non-stub LLM provider.")
    key = (settings.llm_api_key or settings.openrouter_api_key or "").strip()
    if not key:
        raise GoldEvaluationRefused("Live gold evaluation needs a configured provider credential.")


def run_live(settings: Settings, *, include_answers: bool = False) -> dict[str, Any]:
    """Call the current QA agent in process. Experiment flags are not forwarded.

    A provider or infrastructure exception is recorded as a technical failure.
    It is not given a factual recall of zero.
    """
    assert_live_allowed(settings)
    from app.assistant.contract import AssistantTurnRequest
    from app.assistant.qa_adapter import CurriculumQAAdapter
    from app.deps import get_agent

    dataset = load_dataset()
    before_hash = file_sha256()
    agent = get_agent()
    adapter = CurriculumQAAdapter(agent)
    results = []
    for case in dataset["cases"]:
        request_id = f"gold-{case['id']}"
        print(f"gold case {case['id']} start", file=sys.stderr, flush=True)
        started = time.perf_counter()
        try:
            outcome = adapter(
                AssistantTurnRequest(
                    question=case["question"],
                    conversation_id=None,
                    request_id=request_id,
                )
            )
        except Exception as exc:
            row = _technical_failure_row(
                case,
                exc,
                latency_ms=(time.perf_counter() - started) * 1000,
                request_id=request_id,
            )
            results.append(row)
            print(
                f"gold case {case['id']} technical {row['technical_failure_class']}",
                file=sys.stderr,
                flush=True,
            )
            continue
        latency_ms = (time.perf_counter() - started) * 1000
        observation = _observation_from_outcome(outcome)
        metrics = score_observation(case, observation)
        metrics.update(_live_diagnostics(case, observation, latency_ms, request_id))
        print(
            f"gold case {case['id']} done status={metrics.get('status')} "
            f"latency_ms={metrics.get('latency_ms')}",
            file=sys.stderr,
            flush=True,
        )
        if include_answers:
            metrics["answer"] = observation.get("answer")
        results.append(metrics)
    report = run_audit()
    if file_sha256() != before_hash:
        raise RuntimeError("The gold dataset changed during the live run.")
    factual_rows = [row for row in results if not row.get("technical_failure")]
    report["mode"] = "live"
    report["live_executed"] = True
    report["include_answers"] = include_answers
    report["dataset_unchanged"] = True
    report["assistant_enabled"] = settings.assistant_enabled
    report["metrics"] = aggregate(factual_rows)
    technical_count = sum(1 for row in results if row.get("technical_failure"))
    report["metrics"]["technical_failure_rate"] = {
        "measured": True,
        "failed": technical_count,
        "denominator": len(results),
        "value": (technical_count / len(results)) if results else None,
    }
    latencies = [
        float(row["latency_ms"]) for row in results if row.get("latency_ms") is not None
    ]
    report["metrics"]["latency_ms"] = _latency(latencies)
    report["cases"] = results
    report["failures"] = [
        {
            "id": row["case_id"],
            "subject_code": row.get("subject_code"),
            "grade_code": row.get("grade_code"),
            "failure_categories": row.get("failure_categories") or [],
            "technical_failure": bool(row.get("technical_failure")),
        }
        for row in results
        if row.get("failure_categories") or row.get("technical_failure")
    ]
    report["semantic_review"] = [
        row["case_id"] for row in results if row.get("semantic_review_required")
    ]
    return report


def format_summary(report: dict[str, Any]) -> str:
    """Short summary. Answer text is not included."""
    lines = [
        f"Curriculum QA gold evaluation ({report['dataset']})",
        f"Scoring: {report['scoring_version']}",
        f"Dataset sha256: {report['dataset_sha256']}",
        f"Mode: {report['mode']}",
        f"Cases: {report['case_count']}",
        f"Required facts: {report['required_fact_count']}",
        "Support: " + ", ".join(
            f"{name}={count}" for name, count in sorted(report["by_support"].items())
        ),
    ]
    metrics = report["metrics"]
    for name in (
        "factual_precision",
        "factual_recall",
        "unsupported_claim_rate",
        "completeness",
        "subject_grade_accuracy",
        "evidence_abstention",
        "channel_integrity",
    ):
        metric = metrics[name]
        if metric.get("measured") is False:
            lines.append(f"{name}: {metric['reason']}")
        else:
            lines.append(
                f"{name}: {metric['value']} ({metric['applicable']}/{metric.get('denominator', metric['applicable'])} applicable)"
            )
    technical = metrics.get("technical_failure_rate")
    if technical is not None:
        lines.append(
            "Technical failures: "
            f"{technical['failed']}/{technical['denominator']}"
        )
    latency = metrics.get("latency_ms")
    if isinstance(latency, dict) and latency.get("n"):
        lines.append(
            "Latency ms: "
            f"n={latency['n']} median={latency['median']:.1f} "
            f"max={latency.get('max')} "
            f"p95={latency['p95'] if latency['p95'] is not None else 'not reported'}"
        )
    if report.get("failures"):
        lines.append("Failures:")
        for failure in report["failures"]:
            categories = ", ".join(failure["failure_categories"])
            lines.append(
                f"  {failure['id']} {failure.get('grade_code')} {failure.get('subject_code')}: {categories}"
            )
    else:
        lines.append("Case-level failures: none in this mode.")
    if report.get("semantic_review"):
        lines.append("Semantic review: " + ", ".join(report["semantic_review"]))
    if report.get("source_limitations"):
        lines.append("Source limitations:")
        for note in report["source_limitations"]:
            lines.append(f"  {note}")
    return "\n".join(lines)


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _validate_case(case: dict[str, Any]) -> None:
    support = case["support"]
    if support not in {"answerable", "partially_answerable", "insufficient"}:
        raise ValueError(f"{case['id']} has an unknown support value")
    facts = case["required_facts"]
    if support == "insufficient" and facts:
        raise ValueError(f"{case['id']} is insufficient but has required facts")
    if support == "insufficient" and not (case.get("absence") or {}).get("why_insufficient"):
        raise ValueError(f"{case['id']} does not explain the evidence gap")
    if case["must_be_complete"] and not facts:
        raise ValueError(f"{case['id']} requires a complete list without facts")
    if case.get("factual_scoring", True) and support != "insufficient" and not facts:
        raise ValueError(f"{case['id']} has no gold facts")
    for fact in facts:
        if not fact.get("source_ids") or not fact.get("match_any"):
            raise ValueError(f"{case['id']} fact {fact.get('fact_id')} lacks source evidence")


def _fact_recalled(fact: dict[str, Any], answer: str) -> bool:
    return any(_contains(answer, text) for text in fact["match_any"])


def _forbidden_present(item: dict[str, Any], answer: str) -> bool:
    return any(_contains(answer, text) for text in item.get("match_any") or [])


def _forbidden_asserted(item: dict[str, Any], answer: str) -> bool:
    if not answer:
        return False
    normalized = _norm(answer)
    for text in item.get("match_any") or []:
        needle = _norm(text)
        start = normalized.find(needle)
        if start < 0:
            continue
        window = normalized[max(0, start - 80) : start + len(needle) + 80]
        padded = f" {window} "
        if any(negation in padded for negation in _NEGATIONS):
            continue
        return True
    return False


def _pattern_hits(case: dict[str, Any], answer: str) -> list[str]:
    hits = []
    for pattern in case.get("unsupported_patterns") or []:
        if re.search(pattern, answer or "", flags=re.IGNORECASE):
            hits.append(pattern)
    return hits


def _completeness(case: dict[str, Any], recall: float | None, unsupported: int) -> bool | None:
    if not case["must_be_complete"]:
        return None
    if recall is None:
        return False
    return recall == 1 and unsupported == 0


def _subject_grade(
    case: dict[str, Any],
    observation: dict[str, Any],
    answer: str,
) -> bool | None:
    if not answer.strip():
        return None
    grade_ok = _label_found(case.get("grade_labels") or [], answer) or _evidence_matches(
        observation, "grade", case.get("grade_code"), case.get("grade_labels") or []
    )
    subject_ok = _label_found(case.get("subject_labels") or [], answer) or _evidence_matches(
        observation, "subject", case.get("subject_code"), case.get("subject_labels") or []
    )
    return bool(grade_ok and subject_ok)


def _abstention(
    case: dict[str, Any],
    observation: dict[str, Any],
    answer: str,
    asserted_forbidden: list[str],
    pattern_hits: list[str],
) -> bool | None:
    if case["support"] == "insufficient":
        status = observation.get("status")
        if asserted_forbidden or pattern_hits:
            return False
        if status in _ABSTAIN_STATUSES:
            return True
        if status == "completed" and answer.strip():
            return False
        return not answer.strip()
    if case.get("uncertainty") == "state_missing_part":
        if pattern_hits:
            return False
        return _label_found(list(_MISSING_PART_PHRASES), answer)
    return None


def _channel_integrity(observation: dict[str, Any], answer: str) -> bool | None:
    general = observation.get("general_text")
    label = observation.get("general_label")
    if general is None and label is None:
        return None
    if label != _GENERAL_LABEL:
        return False
    if general and _contains(answer, general):
        return False
    return True


def _passed(metrics: dict[str, Any]) -> bool:
    if metrics.get("semantic_review_required"):
        return False
    if metrics.get("factual_recall") not in {None, 1}:
        return False
    if metrics.get("completeness") is False:
        return False
    if metrics.get("subject_grade_accuracy") is False:
        return False
    if metrics.get("evidence_abstention") is False:
        return False
    if metrics.get("channel_integrity") is False:
        return False
    if metrics.get("unsupported_claims"):
        return False
    if metrics.get("factual_precision") not in {None, 1}:
        return False
    return True


def _wrong_scope(case: dict[str, Any], evidence: list[dict[str, Any]]) -> bool:
    grades = set(case.get("grade_labels") or [])
    grades.add(case.get("grade_code"))
    subjects = set(case.get("subject_labels") or [])
    subjects.add(case.get("subject_code"))
    for item in evidence:
        grade = item.get("grade")
        subject = item.get("subject")
        if grade and grade not in grades:
            return True
        if subject and subject not in subjects:
            return True
    return False


def _technical_failure_row(
    case: dict[str, Any],
    exc: Exception,
    *,
    latency_ms: float,
    request_id: str,
) -> dict[str, Any]:
    """Record an infrastructure failure without a factual score."""
    return {
        "case_id": case["id"],
        "subject_code": case.get("subject_code"),
        "grade_code": case.get("grade_code"),
        "curriculum_code": case.get("curriculum_code"),
        "curriculum_version": case.get("curriculum_version"),
        "support": case["support"],
        "must_be_complete": case.get("must_be_complete"),
        "scored": False,
        "technical_failure": True,
        "technical_failure_class": type(exc).__name__,
        "technical_failure_code": getattr(exc, "code", None),
        "latency_ms": round(latency_ms, 1),
        "request_id": request_id,
        "trace": _trace_reference(request_id),
        "failure_categories": ["provider_or_infrastructure_failure"],
        "factual_recall": None,
        "factual_precision": None,
        "unsupported_claim_rate": None,
        "completeness": None,
        "subject_grade_accuracy": None,
        "evidence_abstention": None,
        "channel_integrity": None,
        "semantic_review_required": False,
        "missing_answer": True,
    }


def _live_diagnostics(
    case: dict[str, Any],
    observation: dict[str, Any],
    latency_ms: float,
    request_id: str,
) -> dict[str, Any]:
    evidence = list(observation.get("evidence") or [])
    return {
        "curriculum_code": case.get("curriculum_code"),
        "curriculum_version": case.get("curriculum_version"),
        "must_be_complete": case.get("must_be_complete"),
        "technical_failure": False,
        "status": observation.get("status"),
        "evidence_status": observation.get("evidence_status"),
        "verification_passed": observation.get("verification_passed"),
        "coverage_decision": observation.get("coverage_decision"),
        "termination_reason": observation.get("termination_reason"),
        "latency_ms": round(latency_ms, 1),
        "request_id": request_id,
        "evidence_refs": [
            {
                "entity_id": item.get("entity_id"),
                "entity_type": item.get("entity_type"),
                "name": item.get("name"),
                "grade": item.get("grade"),
                "subject": item.get("subject"),
            }
            for item in evidence
        ],
        "trace": _trace_reference(request_id),
    }


def _trace_reference(request_id: str) -> dict[str, Any] | None:
    from app.agent.trace import get_trace_store

    store = get_trace_store()
    for run_id in reversed(store.list_ids()):
        trace = store.get(run_id)
        if trace is None or trace.request_id != request_id:
            continue
        path = Path(__file__).resolve().parents[2] / "data" / "traces" / f"{trace.agent_run_id}.json"
        final = trace.final or {}
        return {
            "agent_run_id": trace.agent_run_id,
            "trace_path": str(path),
            "trace_latency_ms": final.get("latency_ms"),
            "trace_status": final.get("status"),
            "evidence_count": final.get("evidence_count"),
        }
    return None


def _latency(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    mid = ordered[len(ordered) // 2]
    if len(ordered) % 2 == 0:
        mid = (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
    p95 = None
    if len(ordered) >= 20:
        import statistics

        p95 = statistics.quantiles(ordered, n=20, method="inclusive")[18]
    return {
        "n": len(ordered),
        "median": round(mid, 1),
        "p95": None if p95 is None else round(p95, 1),
        "max": round(ordered[-1], 1),
    }


def _observation_from_outcome(outcome: Any) -> dict[str, Any]:
    response = outcome.response
    evidence = []
    answer = None
    status = None
    evidence_status = None
    verification_passed = None
    termination = None
    if response is not None:
        answer = response.answer
        status = response.status
        evidence_status = response.metadata.evidence_status
        termination = response.metadata.termination_reason
        if response.verification is not None:
            verification_passed = response.verification.passed
        for item in response.evidence or []:
            evidence.append(
                {
                    "entity_id": item.entity_id,
                    "entity_type": item.entity_type,
                    "name": item.name,
                    "grade": item.grade,
                    "subject": item.subject,
                }
            )
    return {
        "answer": answer,
        "status": status,
        "evidence_status": evidence_status,
        "coverage_decision": outcome.coverage_decision,
        "verification_passed": verification_passed,
        "termination_reason": termination,
        "evidence": evidence,
        "general_text": None,
        "general_label": None,
    }


def _label_found(labels: list[str], answer: str) -> bool:
    return any(_contains(answer, label) for label in labels)


def _evidence_matches(
    observation: dict[str, Any],
    field: str,
    code: str | None,
    labels: list[str],
) -> bool:
    allowed = set(labels)
    if code:
        allowed.add(code)
    return any((item.get(field) in allowed) for item in observation.get("evidence") or [])


def _contains(haystack: str | None, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return _norm(needle) in _norm(haystack)


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _count(cases: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        key = str(case.get(field))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _limitations(payload: dict[str, Any]) -> list[str]:
    notes = list(payload.get("source", {}).get("notes") or [])
    for case in payload["cases"]:
        for note in case.get("source_limitations") or []:
            line = f"{case['id']}: {note}"
            if line not in notes:
                notes.append(line)
    return notes


def _unmeasured(reason: str) -> dict[str, Any]:
    return {"measured": False, "value": None, "applicable": 0, "reason": reason}


def _ratio(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [row[field] for row in rows if row.get(field) is not None]
    if not values:
        return _unmeasured("No applicable cases.")
    return {
        "measured": True,
        "value": sum(values) / len(values),
        "applicable": len(values),
        "denominator": len(rows),
        "reason": None,
    }


def _pass_ratio(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [row[field] for row in rows if row.get(field) is not None]
    if not values:
        return _unmeasured("No applicable cases.")
    return {
        "measured": True,
        "value": sum(1 for value in values if value) / len(values),
        "applicable": len(values),
        "denominator": len(rows),
        "reason": None,
    }
