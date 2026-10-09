"""Replay official SSS questions that timed out on GET /api/v1/curricula.

Uses the current agent, including the one bounded identity-lookup retry.
Does not change verifier, routing, prompts, or the baseline result files.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections import Counter
from pathlib import Path
from uuid import uuid4

from app.agent.context import ConversationStore
from app.agent.orchestrator import CurriculumQAAgent
from app.config import get_settings
from app.llm.provider import build_llm_provider
from app.tools.registry import build_default_registry

from run_experiment import (
    MANIFEST,
    ROOT,
    TimingClient,
    _observe,
    _percentile,
    _record,
    _syllabus_index,
)

RESULTS_A = ROOT / "results_A.jsonl"
RESULTS_B = ROOT / "results_B.jsonl"
OUT = ROOT / "results_identity_retry.jsonl"
SUMMARY = ROOT / "summary_identity_retry.json"
RESOLVED = {
    "found",
    "subject_not_in_stream",
    "ambiguous_subject",
    "grade_content_missing",
    "not_found",
}
STREAM_ORDER = {
    "Mathematics & Numeracy": 0,
    "Sciences & Technologies": 1,
    "Languages & Literatures": 2,
    "Social & Cultural Studies": 3,
    "Economics, Business & Entrepreneurship": 4,
}


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _themes(manifest: dict) -> dict[tuple[str, str], list[str]]:
    found: dict[tuple[str, str], list[str]] = {}
    for row in manifest["syllabi"]:
        names = [
            item["name"]
            for item in row.get("content") or []
            if item.get("type") == "theme" and item.get("name")
        ]
        found[(row["subject_name"], row["grade"])] = names
    return found


def _subject_ids(manifest: dict) -> dict[tuple[str, str], str]:
    return {
        (row["subject_name"], row["grade"]): row["subject_id"]
        for row in manifest["syllabi"]
        if row.get("subject_id")
    }


def rescore(row: dict, themes: dict[tuple[str, str], list[str]]) -> str:
    """Score with the baseline evidence rules.

    A timeout that never resolves curriculum context is an infrastructure
    failure. Topic-title overlap with another grade is not contamination.
    """
    resolution = row.get("resolution_status")
    timeout = bool(row.get("timeout"))
    if row.get("exception") or (timeout and resolution not in RESOLVED):
        return "infra_failure"
    grades = {grade for grade in (row.get("content_grades") or []) if grade}
    streams = {stream for stream in (row.get("content_streams") or []) if stream}
    if grades and grades != {row["grade"]}:
        return "fail_contamination"
    if streams and row["stream"] not in streams:
        return "fail_contamination"
    if resolution != "found":
        return "fail_resolution"
    context_ok = (
        row.get("resolved_grade") == row["grade"]
        and row.get("resolved_stream") == row["stream"]
        and row.get("resolved_subject") == row["subject"]
    )
    source = row.get("source_document")
    sources = set(row.get("source_references") or [])
    answer = row.get("answer") or ""
    source_ok = (not source) or source in sources or source in answer
    if not context_ok or not source_ok:
        return "fail_retrieval"
    if not row.get("verification_passed"):
        return "fail_verification"
    expected = themes.get((row["subject"], row["grade"]), [])
    if expected and not all(name.casefold() in answer.casefold() for name in expected):
        return "fail_generation"
    return "grounded"


def select_cases(rows_a: list[dict], rows_b: list[dict]) -> list[dict]:
    official = [
        row
        for row in rows_a
        if row.get("kind") == "valid" and row.get("formulation") == "official"
    ]
    recovered = {
        row["question"]
        for row in rows_b
        if row.get("decision") == "grounded"
    }
    selected: list[dict] = []
    for row in official:
        unresolved_timeout = bool(row.get("timeout")) and row.get("resolution_status") is None
        if not unresolved_timeout:
            continue
        identity = (row.get("api_calls") or 0) <= 1
        cohort = "identity_timeout" if identity else "multi_call_timeout"
        if row["question"] in recovered:
            cohort = f"{cohort}_run_b_recovered"
        selected.append({**row, "cohort": cohort, "priority": 0 if identity else 1})

    def controls(stream: str, limit: int) -> None:
        seen: set[str] = set()
        for row in official:
            if row.get("stream") != stream or row.get("decision") != "grounded":
                continue
            if row["subject"] in seen:
                continue
            seen.add(row["subject"])
            selected.append({**row, "cohort": "control", "priority": 2})
            if len(seen) == limit:
                return

    controls("Social & Cultural Studies", 3)
    controls("Economics, Business & Entrepreneurship", 3)
    selected.sort(
        key=lambda row: (
            row["priority"],
            STREAM_ORDER.get(row["stream"], 9),
            row["subject"],
            row["grade"],
        )
    )
    return selected


def _identity_fields(lookups: list[dict]) -> dict:
    first = lookups[0] if lookups else {}
    statuses = [item.get("status") for item in lookups]
    if "failed" in statuses:
        final_status = "failed"
    elif "recovered" in statuses:
        final_status = "recovered"
    elif statuses:
        final_status = statuses[-1]
    else:
        final_status = "not_called"
    return {
        "curriculum_identity_attempts": first.get("curriculum_identity_attempts"),
        "first_attempt_duration_ms": first.get("first_attempt_duration_ms"),
        "retry_used": any(item.get("retry_used") for item in lookups),
        "retry_duration_ms": next(
            (item.get("retry_duration_ms") for item in lookups if item.get("retry_used")),
            None,
        ),
        "identity_status": final_status,
        "identity_lookups": lookups,
        "identity_lookup_count": len(lookups),
        "identity_total_duration_ms": round(
            sum(item.get("total_duration_ms") or 0 for item in lookups), 1
        ),
    }


def _latency(rows: list[dict], key: str = "latency_ms") -> dict:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return {
        "n": len(values),
        "median_ms": _percentile(values, 50),
        "p75_ms": _percentile(values, 75),
        "p90_ms": _percentile(values, 90),
        "p95_ms": _percentile(values, 95),
        "max_ms": round(max(values), 1) if values else None,
    }


def summarize(records: list[dict], baseline: list[dict]) -> dict:
    decisions = Counter(row["decision"] for row in records)
    base = Counter(row["baseline_decision"] for row in records)
    identity_rows = [row for row in records if row["cohort"].startswith("identity_timeout")]
    retried = [row for row in records if row.get("retry_used")]
    recovered = [row for row in retried if row.get("identity_status") == "recovered"]
    still_failed = [row for row in retried if row.get("identity_status") == "failed"]
    return {
        "questions": len(records),
        "cohorts": dict(Counter(row["cohort"] for row in records)),
        "baseline_decisions": dict(base),
        "decisions": dict(decisions),
        "identity_timeout_questions": len(identity_rows),
        "identity_retries": len(retried),
        "identity_recovered": len(recovered),
        "identity_still_failed": len(still_failed),
        "retry_success_rate": round(len(recovered) / len(retried), 3) if retried else None,
        "infrastructure_before": base["infra_failure"],
        "infrastructure_after": decisions["infra_failure"],
        "contamination_after": decisions["fail_contamination"],
        "latency_total_ms": _latency(records),
        "latency_identity_ms": _latency(records, "identity_total_duration_ms"),
        "latency_retry_recovered_ms": _latency(
            [row for row in records if row.get("identity_status") == "recovered"]
        ),
        "latency_no_retry_success_ms": _latency(
            [
                row
                for row in records
                if not row.get("retry_used")
                and row["decision"] in {"grounded", "correct_rejection"}
            ]
        ),
        "latency_infra_ms": _latency(
            [row for row in records if row["decision"] == "infra_failure"]
        ),
        "baseline_rows": len(baseline),
    }


def main() -> None:
    logging.getLogger("app.curriculum.client").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = json.loads(MANIFEST.read_text())
    themes = _themes(manifest)
    subject_ids = _subject_ids(manifest)
    syllabi = _syllabus_index(manifest)
    rows_a = _load(RESULTS_A)
    rows_b = _load(RESULTS_B)
    cases = select_cases(rows_a, rows_b)
    done = {row["question"] for row in _load(OUT)}
    settings = get_settings()
    client = TimingClient(settings=settings)
    agent = CurriculumQAAgent(
        settings=settings,
        llm=build_llm_provider(settings),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
    )
    print(
        f"RETRY cases={len(cases)} already_done={len(done)} "
        f"provider={settings.llm_provider} model={settings.llm_model} "
        f"timeout={settings.curriculum_api_timeout_seconds}",
        flush=True,
    )
    for index, case in enumerate(cases, start=1):
        if case["question"] in done:
            continue
        client.calls = []
        client.identity_lookups = []
        started = time.perf_counter()
        state = None
        error = None
        try:
            state = agent.ask(case["question"], conversation_id=str(uuid4()))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        observed = _observe(state, list(client.calls), started, error)
        replay_case = {
            "run": "identity_retry",
            "case_id": case["case_id"],
            "kind": case["kind"],
            "formulation": case["formulation"],
            "question": case["question"],
            "stream": case["stream"],
            "grade": case["grade"],
            "subject": case["subject"],
            "subject_id": subject_ids.get((case["subject"], case["grade"])),
            "expected_resolution": case.get("expected_resolution") or "found",
            "home_stream": case["stream"],
        }
        record = _record(replay_case, observed, {"decision": None, "failure_class": None}, syllabi)
        record.update(_identity_fields(list(client.identity_lookups)))
        record["cohort"] = case["cohort"]
        record["baseline_decision"] = case.get("decision")
        record["baseline_latency_ms"] = case.get("latency_ms")
        record["baseline_api_calls"] = case.get("api_calls")
        record["decision"] = rescore(record, themes)
        record["final_outcome"] = record["decision"]
        with OUT.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"{index}/{len(cases)} {record['decision']} {case['cohort']} "
            f"identity={record['identity_status']} retry={record['retry_used']} "
            f"{case['grade']} {case['subject'][:40]} {record['latency_ms']}ms",
            flush=True,
        )
    records = _load(OUT)
    summary = summarize(records, rows_a)
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"RETRY_DONE questions={len(records)}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
