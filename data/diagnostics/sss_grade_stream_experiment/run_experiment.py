"""Baseline SSS grade + stream + subject reliability run.

Uses the current QA agent and the frozen manifest. It does not change
resolver, retrieval, tools, verifier, generation, or timeouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.agent.context import ConversationStore
from app.agent.orchestrator import CurriculumQAAgent
from app.config import get_settings
from app.curriculum.client import CurriculumAPIClient
from app.llm.provider import build_llm_provider
from app.tools.registry import build_default_registry

ROOT = Path(__file__).parent
MANIFEST = ROOT / "manifest.json"
GRADES = ("SSS_1", "SSS_2", "SSS_3")
GRADE_LABEL = {"SSS_1": "SSS1", "SSS_2": "SSS2", "SSS_3": "SSS3"}
SPOKEN_STREAM = {
    "Sciences & Technologies": "Science and Technology",
    "Mathematics & Numeracy": "Mathematics and Numeracy",
    "Languages & Literatures": "Languages and Literatures",
    "Social & Cultural Studies": "Social and Cultural Studies",
    "Economics, Business & Entrepreneurship": "Economics, Business and Entrepreneurship",
}
GRADE_FORMS = {
    "SSS_1": [
        "SSS1",
        "SSS 1",
        "SSS-1",
        "SSS_1",
        "Senior Secondary 1",
        "Senior Secondary School 1",
    ],
    "SSS_2": ["SSS2", "SSS 2", "SSS-2", "SSS_2", "Senior Secondary 2", "Senior Secondary School 2"],
    "SSS_3": ["SSS3", "SSS 3", "SSS-3", "SSS_3", "Senior Secondary 3", "Senior Secondary School 3"],
}
SHORT_STREAMS = ("Science", "Business", "Humanities", "Social Science")
UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.I,
)
LESSON_CODE_RE = re.compile(r"\bC\d+-U\d+\b|\bC\d+U\d+-LO\d+\b", re.I)
GENERIC_INSUFFICIENT = "couldn't find sufficient"


class TimingClient(CurriculumAPIClient):
    """Same client, with a per-request timing log for the experiment."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[dict] = []

    def get(self, path: str, *, params=None, request_id=None):  # type: ignore[override]
        started = time.perf_counter()
        error = None
        try:
            return super().get(path, params=params, request_id=request_id)
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            self.calls.append(
                {
                    "path": path,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "error": error,
                }
            )


def _syllabus_index(manifest: dict) -> dict[tuple[str, str], dict]:
    index = {}
    for row in manifest["syllabi"]:
        if row.get("grade") in GRADES and row.get("subject_id"):
            index[(row["subject_id"], row["grade"])] = row
    return index


def _exclusive_names(manifest: dict) -> dict[tuple[str, str], set[str]]:
    by_subject: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in manifest["syllabi"]:
        names = {
            item["name"].casefold()
            for item in row.get("content") or []
            if item.get("name") and item.get("type") in {"theme", "topic"}
        }
        by_subject[row["subject_id"]][row["grade"]].update(names)
    exclusive: dict[tuple[str, str], set[str]] = {}
    for subject_id, grades in by_subject.items():
        for grade, names in grades.items():
            others: set[str] = set()
            for other, other_names in grades.items():
                if other != grade:
                    others.update(other_names)
            exclusive[(subject_id, grade)] = {
                name for name in names - others if len(name) >= 8
            }
    return exclusive


def _question(subject: str, grade_text: str, stream: str) -> str:
    return f"What does {subject} cover for {grade_text} {stream}?"


def _acronym(name: str) -> str | None:
    found = re.findall(r"\(([A-Za-z][A-Za-z0-9]{1,12})\)", name or "")
    return found[-1] if len(found) == 1 else None


def build_cases(manifest: dict, run: str) -> list[dict]:
    syllabi = _syllabus_index(manifest)
    streams = manifest["streams"]
    official = {stream["name"] for stream in streams}
    membership: dict[str, set[str]] = defaultdict(set)
    for stream in streams:
        for subject in stream["subjects"]:
            membership[subject["id"]].add(stream["name"])

    valid: list[dict] = []
    missing: list[dict] = []
    for stream in streams:
        for subject in stream["subjects"]:
            for grade in GRADES:
                syllabus = syllabi.get((subject["id"], grade))
                base = {
                    "stream": stream["name"],
                    "grade": grade,
                    "subject": subject["name"],
                    "subject_id": subject["id"],
                    "formulation": "official",
                }
                if syllabus and not syllabus.get("content_error"):
                    valid.append(
                        {
                            **base,
                            "kind": "valid",
                            "expected_resolution": "found",
                            "question": _question(
                                subject["name"], GRADE_LABEL[grade], stream["name"]
                            ),
                        }
                    )
                elif syllabus and syllabus.get("content_error"):
                    missing.append(
                        {
                            **base,
                            "kind": "syllabus_unreadable",
                            "expected_resolution": "found_or_infra",
                            "question": _question(
                                subject["name"], GRADE_LABEL[grade], stream["name"]
                            ),
                        }
                    )
                else:
                    missing.append(
                        {
                            **base,
                            "kind": "missing_grade",
                            "expected_resolution": "grade_content_missing",
                            "question": _question(
                                subject["name"], GRADE_LABEL[grade], stream["name"]
                            ),
                        }
                    )

    wrong: list[dict] = []
    for index, stream in enumerate(streams):
        other = streams[(index + 1) % len(streams)]
        chosen = []
        for subject in stream["subjects"]:
            if other["name"] in membership[subject["id"]]:
                continue
            if (subject["id"], "SSS_1") not in syllabi:
                continue
            chosen.append(subject)
            if len(chosen) == 2:
                break
        for subject in chosen:
            wrong.append(
                {
                    "kind": "wrong_stream",
                    "stream": other["name"],
                    "grade": "SSS_1",
                    "subject": subject["name"],
                    "subject_id": subject["id"],
                    "home_stream": stream["name"],
                    "formulation": "official",
                    "expected_resolution": "subject_not_in_stream",
                    "question": _question(subject["name"], "SSS1", other["name"]),
                }
            )

    biology = next(
        (
            subject
            for stream in streams
            if stream["name"] == "Sciences & Technologies"
            for subject in stream["subjects"]
            if subject["name"] == "Biology"
        ),
        None,
    )
    if biology:
        for stream in streams:
            if stream["name"] == "Sciences & Technologies":
                continue
            if stream["name"] in membership[biology["id"]]:
                continue
            wrong.append(
                {
                    "kind": "wrong_stream",
                    "stream": stream["name"],
                    "grade": "SSS_2",
                    "subject": "Biology",
                    "subject_id": biology["id"],
                    "home_stream": "Sciences & Technologies",
                    "formulation": "official",
                    "expected_resolution": "subject_not_in_stream",
                    "question": _question("Biology", "SSS2", stream["name"]),
                }
            )

    unsupported = []
    for stream in streams:
        unsupported.append(
            {
                "kind": "unsupported_subject",
                "stream": stream["name"],
                "grade": "SSS_1",
                "subject": "Basket Weaving",
                "subject_id": None,
                "formulation": "unsupported",
                "expected_resolution": "subject_not_in_stream",
                "question": _question("Basket Weaving", "SSS1", stream["name"]),
            }
        )

    ambiguous = []
    science = next(
        (stream for stream in streams if stream["name"] == "Sciences & Technologies"),
        None,
    )
    if science:
        ambiguous.append(
            {
                "kind": "ambiguous_phrase",
                "stream": science["name"],
                "grade": "SSS_1",
                "subject": "ICT literacy",
                "subject_id": None,
                "formulation": "ambiguous",
                "expected_resolution": "ambiguous_subject",
                "question": _question(
                    "ICT literacy", "SSS1", "Science and Technology"
                ),
            }
        )

    terminology: list[dict] = []
    for stream in streams:
        subject = next(
            (
                item
                for item in stream["subjects"]
                if (item["id"], "SSS_1") in syllabi
                and not syllabi[(item["id"], "SSS_1")].get("content_error")
            ),
            None,
        )
        if subject is None:
            continue
        spoken = SPOKEN_STREAM.get(stream["name"], stream["name"])
        terminology.append(
            {
                "kind": "valid",
                "stream": stream["name"],
                "grade": "SSS_1",
                "subject": subject["name"],
                "subject_id": subject["id"],
                "formulation": "spoken_stream",
                "expected_resolution": "found",
                "question": _question(subject["name"], "SSS1", spoken),
            }
        )
        acronym = _acronym(subject["name"])
        if acronym:
            terminology.append(
                {
                    "kind": "valid",
                    "stream": stream["name"],
                    "grade": "SSS_1",
                    "subject": subject["name"],
                    "subject_id": subject["id"],
                    "formulation": "official_acronym",
                    "expected_resolution": "found",
                    "question": _question(acronym, "SSS1", stream["name"]),
                }
            )

    for stream in streams:
        for subject in stream["subjects"]:
            acronym = _acronym(subject["name"])
            if not acronym:
                continue
            grade = next(
                (
                    code
                    for code in GRADES
                    if (subject["id"], code) in syllabi
                    and not syllabi[(subject["id"], code)].get("content_error")
                ),
                None,
            )
            if grade is None:
                continue
            bare = re.sub(r"\s*\([^)]*\)", "", subject["name"]).strip()
            terminology.append(
                {
                    "kind": "valid",
                    "stream": stream["name"],
                    "grade": grade,
                    "subject": subject["name"],
                    "subject_id": subject["id"],
                    "formulation": "official_acronym",
                    "expected_resolution": "found",
                    "question": _question(acronym, GRADE_LABEL[grade], stream["name"]),
                }
            )
            if bare and bare != subject["name"]:
                terminology.append(
                    {
                        "kind": "valid",
                        "stream": stream["name"],
                        "grade": grade,
                        "subject": subject["name"],
                        "subject_id": subject["id"],
                        "formulation": "name_without_acronym",
                        "expected_resolution": "found",
                        "question": _question(bare, GRADE_LABEL[grade], stream["name"]),
                    }
                )
            break

    anchor = None
    anchor_stream = None
    if biology and all((biology["id"], grade) in syllabi for grade in GRADES):
        anchor = biology
        anchor_stream = "Sciences & Technologies"
    else:
        for stream in streams:
            for subject in stream["subjects"]:
                if all((subject["id"], grade) in syllabi for grade in GRADES):
                    anchor = subject
                    anchor_stream = stream["name"]
                    break
            if anchor:
                break
    if anchor and anchor_stream:
        for grade, forms in GRADE_FORMS.items():
            for form in forms:
                if form == GRADE_LABEL[grade]:
                    continue
                terminology.append(
                    {
                        "kind": "valid",
                        "stream": anchor_stream,
                        "grade": grade,
                        "subject": anchor["name"],
                        "subject_id": anchor["id"],
                        "formulation": "grade_variant",
                        "expected_resolution": "found",
                        "question": _question(anchor["name"], form, anchor_stream),
                    }
                )

    short = []
    probe_subject = anchor["name"] if anchor else "Biology"
    for word in SHORT_STREAMS:
        short.append(
            {
                "kind": "short_stream",
                "stream": word,
                "grade": "SSS_1",
                "subject": probe_subject,
                "subject_id": anchor["id"] if anchor else None,
                "formulation": "unsupported_short_stream",
                "expected_resolution": "not_a_stream",
                "question": _question(probe_subject, "SSS1", word),
            }
        )

    def _dedupe(rows: list[dict]) -> list[dict]:
        seen = set()
        kept = []
        for row in rows:
            key = row["question"]
            if key in seen:
                continue
            seen.add(key)
            kept.append(row)
        return kept

    valid = _dedupe(valid)
    terminology = _dedupe(terminology)
    negatives = _dedupe(missing + wrong + unsupported + ambiguous + short)
    if run == "A":
        selected = valid + terminology + negatives
    else:
        sample = []
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in valid:
            grouped[(row["stream"], row["grade"])].append(row)
        for key in sorted(grouped):
            rows = sorted(grouped[key], key=lambda item: item["subject"])
            sample.append(rows[0])
            if len(rows) > 1:
                sample.append(rows[-1])
        if anchor:
            sample.extend(
                row
                for row in valid
                if row["subject_id"] == anchor["id"]
            )
        if len(valid) <= 80:
            selected = valid + terminology + negatives
        else:
            selected = _dedupe(sample) + terminology + negatives
    for index, row in enumerate(selected, start=1):
        row["case_id"] = f"{run}-{index:04d}"
        row["run"] = run
    return selected


def _content_names(syllabus: dict | None) -> set[tuple[str, str]]:
    if not syllabus:
        return set()
    return {
        (item.get("type"), item.get("name"))
        for item in syllabus.get("content") or []
        if item.get("type") in {"theme", "topic", "learning_outcome"} and item.get("name")
    }


def _classify(case: dict, observed: dict, syllabi: dict, exclusive: dict) -> dict:
    answer = observed.get("answer") or ""
    answer_cf = answer.casefold()
    resolution = observed.get("resolution")
    expected = case["expected_resolution"]
    kind = case["kind"]
    timeout = bool(observed.get("timeout"))
    syllabus = None
    if case.get("subject_id") and case.get("grade") in GRADES:
        syllabus = syllabi.get((case["subject_id"], case["grade"]))
    expected_names = _content_names(syllabus)
    got = set(observed.get("content_names") or [])
    grades = set(observed.get("content_grades") or [])
    sources = set(observed.get("source_references") or [])
    foreign = False
    if case.get("subject_id") and case.get("grade") in GRADES:
        for other in GRADES:
            if other == case["grade"]:
                continue
            other_only = exclusive.get((case["subject_id"], other), set())
            if any(name in answer_cf for name in other_only):
                foreign = True
        if grades and grades != {case["grade"]}:
            foreign = True
    content_loaded = bool(got) or bool(observed.get("content_urls"))
    generic = GENERIC_INSUFFICIENT in answer_cf
    infra = bool(observed.get("exception")) or (
        timeout and resolution not in {
            "found",
            "subject_not_in_stream",
            "ambiguous_subject",
            "grade_content_missing",
            "not_found",
        }
    )
    failure = None
    decision = None
    if infra:
        failure = "infrastructure"
        decision = "infra_failure"
    elif kind == "short_stream":
        official_hit = observed.get("resolved_stream") in set(SPOKEN_STREAM)
        if official_hit or "sciences & technologies" in answer_cf and case["stream"] == "Science":
            failure = "resolution"
            decision = "fail_resolution"
        elif case["stream"] == "Business" and "economics, business" in answer_cf:
            failure = "resolution"
            decision = "fail_resolution"
        elif case["stream"] == "Social Science" and "social & cultural" in answer_cf:
            failure = "resolution"
            decision = "fail_resolution"
        elif case["stream"] == "Humanities" and "social & cultural" in answer_cf:
            failure = "resolution"
            decision = "fail_resolution"
        else:
            decision = "correct_rejection"
    elif foreign and kind == "valid":
        failure = "cross_context"
        decision = "fail_contamination"
    elif kind == "valid":
        themes = {name for typ, name in expected_names if typ == "theme"}
        covered = all(name.casefold() in answer_cf for name in themes) if themes else True
        context_ok = (
            observed.get("resolved_grade") == case["grade"]
            and observed.get("resolved_stream") == case["stream"]
            and observed.get("resolved_subject") == case["subject"]
        )
        evidence_ok = (
            resolution == "found"
            and context_ok
            and bool(expected_names)
            and expected_names <= got
            and not foreign
        )
        source_ok = True
        if syllabus and syllabus.get("source_reference"):
            source_ok = syllabus["source_reference"] in sources or syllabus[
                "source_reference"
            ] in answer
        if resolution != "found":
            failure = "resolution"
            decision = "fail_resolution"
        elif not evidence_ok:
            failure = "retrieval"
            decision = "fail_retrieval"
        elif foreign:
            failure = "cross_context"
            decision = "fail_contamination"
        elif not observed.get("verification_passed"):
            failure = "verification"
            decision = "fail_verification"
        elif not covered or not source_ok or UUID_RE.search(answer) or LESSON_CODE_RE.search(answer):
            failure = "generation"
            decision = "fail_generation"
        else:
            decision = "grounded"
    elif kind == "missing_grade":
        other_grade_loaded = grades and grades != {case["grade"]}
        if resolution == "grade_content_missing" and not content_loaded and not other_grade_loaded:
            decision = "correct_rejection"
        elif content_loaded or other_grade_loaded:
            failure = "cross_context" if other_grade_loaded or got else "retrieval"
            decision = "fail_contamination" if other_grade_loaded or got else "fail_retrieval"
        elif resolution != "grade_content_missing":
            failure = "resolution"
            decision = "fail_resolution"
        else:
            decision = "correct_rejection"
    elif kind in {"wrong_stream", "unsupported_subject"}:
        if content_loaded or (got and expected_names and expected_names <= got):
            failure = "cross_context"
            decision = "fail_contamination"
        elif resolution == expected and not generic:
            decision = "correct_rejection"
        elif resolution == expected and generic:
            failure = "generation"
            decision = "fail_generation"
        else:
            failure = "resolution"
            decision = "fail_resolution"
    elif kind == "ambiguous_phrase":
        if content_loaded:
            failure = "cross_context"
            decision = "fail_contamination"
        elif resolution == "ambiguous_subject" and not generic and "if you mean" in answer_cf:
            decision = "correct_rejection"
        elif resolution == "ambiguous_subject":
            failure = "generation"
            decision = "fail_generation"
        else:
            failure = "resolution"
            decision = "fail_resolution"
    elif kind == "syllabus_unreadable":
        failure = "infrastructure"
        decision = "infra_failure"
    else:
        failure = "resolution"
        decision = "fail_resolution"
    return {"failure_class": failure, "decision": decision}


def _observe(state, calls: list[dict], started: float, error: str | None) -> dict:
    evidence = list(getattr(state, "evidence", []) or [])
    content = [
        item
        for item in evidence
        if (item.entity_type or "").lower() in {"theme", "topic", "learning_outcome"}
    ]
    verification = getattr(state, "verification", None)
    meta = dict(getattr(state, "metadata", {}) or {})
    tool_latency = [
        record.latency_ms
        for record in getattr(state, "retrieval_history", []) or []
        if record.latency_ms is not None
    ]
    timeouts = [call for call in calls if call.get("error") == "CurriculumTimeoutError"]
    paths = [call["path"] for call in calls]
    retry = any(
        call.get("error") == "CurriculumTimeoutError"
        and any(
            later["path"] == call["path"] and not later.get("error")
            for later in calls[index + 1 :]
        )
        for index, call in enumerate(calls)
    )
    return {
        "answer": getattr(state, "final_answer", None) if state else None,
        "status": getattr(getattr(state, "status", None), "value", None) if state else None,
        "resolved_grade": getattr(state, "grade", None) if state else None,
        "resolved_stream": meta.get("resolved_stream_name") or meta.get("stream_name"),
        "resolved_subject": meta.get("subject_name"),
        "resolution": meta.get("sss_stream_resolution"),
        "intent": getattr(state, "intent", None) if state else None,
        "verification_passed": bool(verification.passed) if verification else False,
        "verification_status": getattr(
            getattr(state, "verification_status", None), "value", None
        )
        if state
        else None,
        "verification_recommendation": getattr(
            getattr(verification, "recommendation", None), "value", None
        )
        if verification
        else None,
        "content_names": [
            ((item.entity_type or "").lower(), item.name) for item in content if item.name
        ],
        "content_grades": [(item.grade or "") for item in content],
        "content_subjects": sorted({item.subject for item in content if item.subject}),
        "content_streams": sorted(
            {
                str((item.metadata or {}).get("stream_name"))
                for item in content
                if (item.metadata or {}).get("stream_name")
            }
        ),
        "source_references": sorted(
            {item.source_reference for item in content if item.source_reference}
        ),
        "content_urls": [
            call["path"] for call in calls if str(call["path"]).endswith("/content")
        ],
        "tool_calls": getattr(state, "tool_calls", None) if state else None,
        "retrieval_rounds": getattr(state, "retrieval_rounds", None) if state else None,
        "tool_latency_ms": round(sum(tool_latency), 1) if tool_latency else None,
        "generation_latency_ms": meta.get("generation_latency_ms"),
        "verification_latency_ms": (verification.metadata or {}).get(
            "verification_latency_ms"
        )
        if verification
        else None,
        "api_latency_ms": round(sum(call["latency_ms"] for call in calls), 1),
        "api_calls": len(calls),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "timeout": bool(timeouts),
        "timeout_count": len(timeouts),
        "retry_used": retry,
        "exception": error,
        "paths": paths,
    }


def _record(case: dict, observed: dict, verdict: dict, syllabi: dict) -> dict:
    syllabus = None
    if case.get("subject_id"):
        syllabus = syllabi.get((case["subject_id"], case["grade"]))
    evidence_hash = hashlib.sha256(
        json.dumps(observed.get("content_names") or [], ensure_ascii=False).encode()
    ).hexdigest()
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "run": case["run"],
        "case_id": case["case_id"],
        "kind": case["kind"],
        "formulation": case["formulation"],
        "question": case["question"],
        "stream": case["stream"],
        "grade": case["grade"],
        "subject": case["subject"],
        "expected_resolution": case["expected_resolution"],
        "resolved_grade": observed.get("resolved_grade"),
        "resolved_stream": observed.get("resolved_stream"),
        "resolved_subject": observed.get("resolved_subject"),
        "resolution_status": observed.get("resolution"),
        "agent_status": observed.get("status"),
        "intent": observed.get("intent"),
        "verification_passed": observed.get("verification_passed"),
        "verification_status": observed.get("verification_status"),
        "decision": verdict["decision"],
        "failure_class": verdict["failure_class"],
        "source_document": (syllabus or {}).get("source_reference"),
        "source_grade": (syllabus or {}).get("grade"),
        "source_subject": (syllabus or {}).get("subject_name"),
        "source_stream": case.get("home_stream") or case["stream"],
        "expected_evidence_sha256": (syllabus or {}).get("evidence_sha256"),
        "retrieved_evidence_sha256": evidence_hash,
        "latency_ms": observed.get("latency_ms"),
        "api_latency_ms": observed.get("api_latency_ms"),
        "tool_latency_ms": observed.get("tool_latency_ms"),
        "generation_latency_ms": observed.get("generation_latency_ms"),
        "verification_latency_ms": observed.get("verification_latency_ms"),
        "tool_calls": observed.get("tool_calls"),
        "retrieval_iterations": observed.get("retrieval_rounds"),
        "api_calls": observed.get("api_calls"),
        "timeout": observed.get("timeout"),
        "timeout_count": observed.get("timeout_count"),
        "retry_used": observed.get("retry_used"),
        "exception": observed.get("exception"),
        "content_grades": observed.get("content_grades"),
        "content_subjects": observed.get("content_subjects"),
        "content_streams": observed.get("content_streams"),
        "source_references": observed.get("source_references"),
        "answer": observed.get("answer"),
    }


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return round(ordered[index], 1)


def _summary(records: list[dict], run: str) -> dict:
    def latency_block(rows: list[dict]) -> dict:
        values = [row["latency_ms"] for row in rows if isinstance(row.get("latency_ms"), (int, float))]
        return {
            "n": len(values),
            "median_ms": _percentile(values, 50),
            "p75_ms": _percentile(values, 75),
            "p90_ms": _percentile(values, 90),
            "p95_ms": _percentile(values, 95),
            "max_ms": round(max(values), 1) if values else None,
        }

    decisions = defaultdict(int)
    by_grade = defaultdict(lambda: defaultdict(int))
    by_stream = defaultdict(lambda: defaultdict(int))
    by_subject = defaultdict(lambda: defaultdict(int))
    by_form = defaultdict(lambda: defaultdict(int))
    for row in records:
        decisions[row["decision"]] += 1
        if row["failure_class"]:
            decisions[f"class:{row['failure_class']}"] += 1
        bucket = row["decision"]
        by_grade[row["grade"]][bucket] += 1
        by_grade[row["grade"]]["tested"] += 1
        by_stream[row["stream"]][bucket] += 1
        by_stream[row["stream"]]["tested"] += 1
        by_subject[row["subject"]][bucket] += 1
        by_subject[row["subject"]]["tested"] += 1
        by_form[row["formulation"]][bucket] += 1
        by_form[row["formulation"]]["tested"] += 1
    success = [row for row in records if row["decision"] in {"grounded", "correct_rejection"}]
    timeouts = [row for row in records if row.get("timeout")]
    infra = [row for row in records if row["decision"] == "infra_failure"]
    return {
        "run": run,
        "questions": len(records),
        "decisions": dict(decisions),
        "by_grade": {key: dict(value) for key, value in by_grade.items()},
        "by_stream": {key: dict(value) for key, value in by_stream.items()},
        "by_subject": {key: dict(value) for key, value in by_subject.items()},
        "by_formulation": {key: dict(value) for key, value in by_form.items()},
        "latency_success_ms": latency_block(success),
        "latency_timeout_ms": latency_block(timeouts),
        "latency_infra_ms": latency_block(infra),
        "latency_all_ms": latency_block(records),
    }


def run(run_name: str) -> None:
    logging.getLogger("app.curriculum.client").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    manifest = json.loads(MANIFEST.read_text())
    cases = build_cases(manifest, run_name)
    syllabi = _syllabus_index(manifest)
    exclusive = _exclusive_names(manifest)
    out = ROOT / f"results_{run_name}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done.add(row["question"])
    settings = get_settings()
    client = TimingClient(settings=settings)
    agent = CurriculumQAAgent(
        settings=settings,
        llm=build_llm_provider(settings),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
    )
    print(
        f"RUN {run_name} cases={len(cases)} already_done={len(done)} "
        f"provider={settings.llm_provider} model={settings.llm_model}",
        flush=True,
    )
    for index, case in enumerate(cases, start=1):
        if case["question"] in done:
            continue
        client.calls = []
        started = time.perf_counter()
        state = None
        error = None
        try:
            state = agent.ask(case["question"], conversation_id=str(uuid4()))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        observed = _observe(state, list(client.calls), started, error)
        verdict = _classify(case, observed, syllabi, exclusive)
        record = _record(case, observed, verdict, syllabi)
        with out.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"{index}/{len(cases)} {verdict['decision']} {case['kind']} "
            f"{case['grade']} {case['subject'][:40]} {observed['latency_ms']}ms",
            flush=True,
        )
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    summary = _summary(records, run_name)
    (ROOT / f"summary_{run_name}.json").write_text(json.dumps(summary, indent=2))
    print(f"RUN_{run_name}_DONE questions={len(records)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=("A", "B"), required=True)
    args = parser.parse_args()
    run(args.run)


if __name__ == "__main__":
    main()
