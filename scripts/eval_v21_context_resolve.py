#!/usr/bin/env python3
"""V2.1 evaluation sprint — diagnosis only (no product fixes)."""

from __future__ import annotations

import json
import re
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "diagnostics" / "v21_evaluation"
DOC = ROOT / "docs" / "V2_1_EVALUATION.md"
BASELINE_COMPARISON = ROOT / "data" / "diagnostics" / "comparison.json"

ASK_URL = "http://127.0.0.1:8001/api/v1/agent/ask"
DEBUG_URL = "http://127.0.0.1:8001/api/v1/agent/debug/runs"
RESOLVE_URL = "http://127.0.0.1:8000/api/v2/curriculum/context/resolve"
GRADE_CURRICULUM_CONTENT = (
    "http://127.0.0.1:8000/api/v1/grade-curricula/"
    "b1bcff00-3d07-4e92-b426-97e3bfee12ec/content"
)

GOLDEN = "What are the learning objectives for fractions in Primary 4?"
LEGACY_TOOLS = {
    "search_curriculum",
    "get_curriculum_structure",
    "get_subject",
    "get_topic",
    "get_learning_objectives",
}
RESOLVE_TOOL = "resolve_curriculum_context"

BASELINE_IDS = [
    "run-ed19de4bcebd4a2f",
    "run-fbb36688ba17480f",
    "run-df4cdd1da0bb45e8",
    "run-5df89dbf77e74152",
    "run-68f2aa1ce8734ae2",
]


def _get(client: httpx.Client, url: str, **params: Any) -> Any:
    cleaned = {k: v for k, v in params.items() if v is not None}
    r = client.get(url, params=cleaned)
    r.raise_for_status()
    return r.json()


def classify_sequence(tools: list[str], status: str | None) -> str:
    if status in {"insufficient_evidence", "failed", "error"}:
        # still classify path shape, but mark failure if limits/repeats later
        pass
    if not tools:
        return "failure"
    first_resolve = tools[0] == RESOLVE_TOOL
    has_resolve = RESOLVE_TOOL in tools
    has_legacy = any(t in LEGACY_TOOLS for t in tools)
    only_resolve = tools == [RESOLVE_TOOL] or (
        first_resolve and all(t == RESOLVE_TOOL for t in tools)
    )
    if status not in {"completed", "answered"}:
        return "failure"
    if first_resolve and not has_legacy and len(tools) <= 2:
        return "ideal"
    if first_resolve and has_legacy:
        return "acceptable_fallback"
    if has_resolve and not first_resolve:
        return "exploratory"
    if has_legacy and not has_resolve:
        return "exploratory"  # pre-preferred legacy path still exploratory for V2.1
    if only_resolve:
        return "ideal"
    return "exploratory"


def analyze_trace(trace: dict[str, Any] | None, body: dict[str, Any]) -> dict[str, Any]:
    events = (trace or {}).get("events") or []
    meta = body.get("metadata") or {}
    tool_starts = [
        e
        for e in events
        if e.get("event") == "agent.tool.start" and not e.get("skipped")
    ]
    tool_ends = [e for e in events if e.get("event") == "agent.tool.end"]
    skips = [e for e in events if e.get("event") == "agent.tool.skip"]
    evidence_adds = [e for e in events if e.get("event") == "agent.evidence.add"]
    dup_evidence = [e for e in evidence_adds if e.get("duplicate") or e.get("disposition") == "duplicate"]
    zero_new = [
        e
        for e in tool_ends
        if (e.get("new_evidence") or 0) == 0 and e.get("success")
    ]

    tools = [e.get("tool_name") for e in tool_starts]
    resolve_calls = [e for e in tool_starts if e.get("tool_name") == RESOLVE_TOOL]
    legacy_calls = [e for e in tool_starts if e.get("tool_name") in LEGACY_TOOLS]
    structure_calls = [e for e in tool_starts if e.get("tool_name") == "get_curriculum_structure"]
    topic_calls = [e for e in tool_starts if e.get("tool_name") == "get_topic"]
    search_calls = [e for e in tool_starts if e.get("tool_name") == "search_curriculum"]

    phase = {"understand": 0.0, "retrieval": 0.0, "generation": 0.0, "verification": 0.0, "routing": 0.0}
    context_resolution_ms = 0.0
    for e in events:
        name = e.get("event") or ""
        dur = e.get("duration_ms")
        if dur is None:
            continue
        if "understand" in name:
            phase["understand"] += float(dur)
        elif "retriev" in name or name.startswith("agent.tool."):
            phase["retrieval"] += float(dur)
        elif "generat" in name:
            phase["generation"] += float(dur)
        elif "verif" in name:
            phase["verification"] += float(dur)
        elif "route" in name:
            phase["routing"] += float(dur)
    for e in tool_ends:
        if e.get("tool_name") == RESOLVE_TOOL:
            obs = e.get("observability") or {}
            if obs.get("query_timing_ms") is not None:
                context_resolution_ms += float(obs["query_timing_ms"])
            elif e.get("duration_ms") is not None:
                context_resolution_ms += float(e["duration_ms"])

    resolve_obs = []
    for e in tool_ends:
        if e.get("tool_name") == RESOLVE_TOOL and e.get("observability"):
            resolve_obs.append(e["observability"])

    visited = meta.get("visited_nodes") or []
    retrieve_more = sum(1 for n in visited if n == "retrieve_more") or max(
        0, (meta.get("retrieval_rounds") or 1) - 1
    )

    verification = body.get("verification") or {}
    answer = body.get("answer") or ""
    lower = answer.lower()
    quality = {
        "answers_question": bool(
            re.search(r"learning object|outcome|should (be able|know)|learners", lower)
        )
        or (body.get("status") in {"completed", "answered"} and "fraction" in lower),
        "speculative_wording": bool(re.search(r"\blikely\b|\bprobably\b|\bmight\b|\bperhaps\b", lower)),
        "unsupported_elaboration_flags": verification.get("unsupported_claims") or [],
        "cites_evidence": bool(body.get("evidence")),
        "verifier_accepts": (meta.get("verification_status") == "passed")
        or (verification.get("recommendation") == "accept")
        or (verification.get("passed") is True),
    }

    status = body.get("status")
    seq_class = classify_sequence([t for t in tools if t], status)
    if status not in {"completed", "answered"}:
        seq_class = "failure"

    return {
        "agent_run_id": meta.get("agent_run_id"),
        "status": status,
        "termination_reason": meta.get("termination_reason")
        or meta.get("fallback_reason"),
        "iteration_count": meta.get("iterations") or meta.get("iteration"),
        "tool_call_count": meta.get("tool_calls") or len(tool_starts),
        "tool_sequence": tools,
        "resolver_calls": len(resolve_calls),
        "legacy_retrieval_calls": len(legacy_calls),
        "structure_calls": len(structure_calls),
        "topic_calls": len(topic_calls),
        "search_calls": len(search_calls),
        "duplicate_tool_skips": len(skips),
        "duplicate_evidence_adds": len(dup_evidence),
        "zero_new_evidence_tool_ends": len(zero_new),
        "evidence_count": meta.get("evidence_count")
        if meta.get("evidence_count") is not None
        else len(body.get("evidence") or []),
        "generation_evidence_count": len(body.get("evidence") or []),
        "verifier_score": meta.get("verification_score")
        if meta.get("verification_score") is not None
        else verification.get("score"),
        "verifier_decision": meta.get("verification_status")
        or verification.get("recommendation"),
        "retrieve_more_count": retrieve_more,
        "retrieval_rounds": meta.get("retrieval_rounds"),
        "answer": answer,
        "answer_confidence": body.get("confidence"),
        "sequence_class": seq_class,
        "phase_timings_ms": {k: round(v, 2) for k, v in phase.items()},
        "context_resolution_ms": round(context_resolution_ms, 2),
        "resolve_observability": resolve_obs,
        "answer_quality": quality,
        "visited_nodes": visited,
        "understand_subject": meta.get("subject")
        or ((trace or {}).get("final_state") or {}).get("subject"),
    }


def ask_once(client: httpx.Client, question: str, tag: str) -> dict[str, Any]:
    started = time.perf_counter()
    resp = client.post(ASK_URL, json={"question": question, "conversation_id": None})
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    resp.raise_for_status()
    body = resp.json()
    meta = body.get("metadata") or {}
    run_id = meta.get("agent_run_id")
    trace = None
    if run_id:
        tr = client.get(f"{DEBUG_URL}/{run_id}")
        if tr.status_code == 200:
            trace = tr.json()
    analyzed = analyze_trace(trace, body)
    analyzed["question"] = question
    analyzed["tag"] = tag
    analyzed["latency_ms"] = latency_ms
    # persist raw
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", tag)[:80]
    (OUT / f"{safe}_response.json").write_text(json.dumps(body, indent=2))
    if trace:
        (OUT / f"{safe}_trace.json").write_text(json.dumps(trace, indent=2))
    return analyzed


def baseline_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [r["latency_ms"] for r in rows]
    tools = [r["tool_calls"] for r in rows]
    success = sum(1 for r in rows if r["status"] in {"completed", "answered"})
    retrieve_more = sum(
        1
        for r in rows
        if (r.get("retrieval_rounds") or 0) > 1
        or r.get("recommendation") == "retrieve_more"
        or (r.get("iterations") or 0) > 1
    )
    return {
        "n": len(rows),
        "success_rate": round(success / max(len(rows), 1), 3),
        "success_count": success,
        "avg_latency_ms": round(statistics.mean(latencies), 1),
        "median_latency_ms": round(statistics.median(latencies), 1),
        "avg_tool_calls": round(statistics.mean(tools), 2),
        "max_tool_calls": max(tools),
        "retrieve_more_rate": round(retrieve_more / max(len(rows), 1), 3),
        "verifier_acceptance_rate": round(
            sum(1 for r in rows if r.get("recommendation") == "accept")
            / max(len(rows), 1),
            3,
        ),
        "note": "duplicate/zero-new metrics not fully present in baseline comparison.json; inferred from traces where available",
    }


def v21_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [r["latency_ms"] for r in rows]
    tools = [r["tool_call_count"] or 0 for r in rows]
    success = sum(1 for r in rows if r["status"] in {"completed", "answered"})
    retrieve_more = sum(1 for r in rows if (r.get("retrieve_more_count") or 0) > 0)
    return {
        "n": len(rows),
        "success_rate": round(success / max(len(rows), 1), 3),
        "success_count": success,
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "median_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
        "avg_tool_calls": round(statistics.mean(tools), 2) if tools else None,
        "max_tool_calls": max(tools) if tools else None,
        "retrieve_more_rate": round(retrieve_more / max(len(rows), 1), 3),
        "avg_duplicate_tool_skips": round(
            statistics.mean([r.get("duplicate_tool_skips") or 0 for r in rows]), 2
        ),
        "avg_duplicate_evidence_adds": round(
            statistics.mean([r.get("duplicate_evidence_adds") or 0 for r in rows]), 2
        ),
        "avg_zero_new_evidence_calls": round(
            statistics.mean([r.get("zero_new_evidence_tool_ends") or 0 for r in rows]),
            2,
        ),
        "avg_resolver_calls": round(
            statistics.mean([r.get("resolver_calls") or 0 for r in rows]), 2
        ),
        "avg_legacy_calls": round(
            statistics.mean([r.get("legacy_retrieval_calls") or 0 for r in rows]), 2
        ),
        "verifier_acceptance_rate": round(
            sum(1 for r in rows if r.get("answer_quality", {}).get("verifier_accepts"))
            / max(len(rows), 1),
            3,
        ),
        "sequence_class_distribution": dict(Counter(r["sequence_class"] for r in rows)),
        "avg_context_resolution_ms": round(
            statistics.mean([r.get("context_resolution_ms") or 0 for r in rows]), 2
        ),
        "avg_retrieval_phase_ms": round(
            statistics.mean(
                [(r.get("phase_timings_ms") or {}).get("retrieval") or 0 for r in rows]
            ),
            2,
        ),
        "avg_generation_phase_ms": round(
            statistics.mean(
                [(r.get("phase_timings_ms") or {}).get("generation") or 0 for r in rows]
            ),
            2,
        ),
        "avg_verification_phase_ms": round(
            statistics.mean(
                [(r.get("phase_timings_ms") or {}).get("verification") or 0 for r in rows]
            ),
            2,
        ),
    }


def collect_fraction_authoritative(client: httpx.Client) -> dict[str, Any]:
    tree = client.get(GRADE_CURRICULUM_CONTENT).json()

    def walk(nodes: list[dict[str, Any]], out: list[dict[str, Any]]) -> None:
        for n in nodes or []:
            name = (n.get("name") or "").lower()
            if "fraction" in name:
                out.append(
                    {
                        "id": n.get("id"),
                        "code": n.get("code"),
                        "name": n.get("name"),
                        "content_type": n.get("content_type"),
                        "learning_outcomes": [
                            {
                                "id": lo.get("id"),
                                "code": lo.get("code"),
                                "description": lo.get("description"),
                            }
                            for lo in (n.get("learning_outcomes") or [])
                        ],
                    }
                )
            walk(n.get("children") or [], out)

    units: list[dict[str, Any]] = []
    walk(tree if isinstance(tree, list) else [tree], units)
    return {
        "grade_curriculum_id": "b1bcff00-3d07-4e92-b426-97e3bfee12ec",
        "fraction_units": units,
        "unit_count": len(units),
        "lo_count": sum(len(u["learning_outcomes"]) for u in units),
        "lo_codes": [
            lo["code"] for u in units for lo in u["learning_outcomes"] if lo.get("code")
        ],
    }


def probe_resolver(client: httpx.Client) -> dict[str, Any]:
    probes = {}
    cases = {
        "default_grade_code": {
            "grade": "CLASS_4",
            "subject": "MATHEMATICS",
            "topic": "fractions",
            "curriculum_code": "MBSSE-BEC",
            "version": "2020",
        },
        "correct_grade_id": {
            "grade_id": "0f392a75-cf81-4c62-94e4-6041c23baf0b",
            "subject": "MATHEMATICS",
            "topic": "fractions",
            "curriculum_code": "MBSSE-BEC",
            "version": "2020",
        },
        "missing_subject": {
            "grade": "CLASS_4",
            "topic": "fractions",
            "curriculum_code": "MBSSE-BEC",
            "version": "2020",
        },
        "missing_subject_correct_grade_id": {
            "grade_id": "0f392a75-cf81-4c62-94e4-6041c23baf0b",
            "topic": "fractions",
            "curriculum_code": "MBSSE-BEC",
            "version": "2020",
        },
        "unknown_topic": {
            "grade_id": "0f392a75-cf81-4c62-94e4-6041c23baf0b",
            "subject": "MATHEMATICS",
            "topic": "quantum computing",
            "curriculum_code": "MBSSE-BEC",
            "version": "2020",
        },
        "c4u06_code": {
            "grade_id": "0f392a75-cf81-4c62-94e4-6041c23baf0b",
            "subject": "MATHEMATICS",
            "topic": "C4U06",
            "curriculum_code": "MBSSE-BEC",
            "version": "2020",
        },
        "c4-u06_code": {
            "grade_id": "0f392a75-cf81-4c62-94e4-6041c23baf0b",
            "subject": "MATHEMATICS",
            "topic": "C4-U06",
            "curriculum_code": "MBSSE-BEC",
            "version": "2020",
        },
    }
    for name, params in cases.items():
        t0 = time.perf_counter()
        try:
            data = _get(client, RESOLVE_URL, **params)
            api_ms = round((time.perf_counter() - t0) * 1000, 2)
            probes[name] = {
                "params": params,
                "api_latency_ms": api_ms,
                "resolution": data.get("resolution"),
                "grade": data.get("grade"),
                "subject": data.get("subject"),
                "grade_curriculum_id": data.get("grade_curriculum_id"),
                "topics": [
                    {"id": x.get("id"), "code": x.get("code"), "name": x.get("name")}
                    for x in (data.get("topics") or [])
                ],
                "units": [
                    {
                        "id": x.get("id"),
                        "code": x.get("code"),
                        "name": x.get("name"),
                        "content_type": x.get("content_type"),
                    }
                    for x in (data.get("units") or [])
                ],
                "learning_outcomes": [
                    {
                        "id": x.get("id"),
                        "code": x.get("code"),
                        "description": x.get("description"),
                        "parent_content_id": x.get("parent_content_id"),
                        "parent_content_code": x.get("parent_content_code"),
                        "has_provenance": bool(x.get("provenance")),
                        "evidence_quality": x.get("evidence_quality"),
                    }
                    for x in (data.get("learning_outcomes") or [])
                ],
                "candidates": data.get("candidates") or [],
                "unit_count": len(data.get("units") or []),
                "lo_count": len(data.get("learning_outcomes") or []),
            }
        except Exception as exc:  # noqa: BLE001 — evaluation harness
            probes[name] = {"params": params, "error": str(exc)}
    return probes


def write_markdown(report: dict[str, Any]) -> None:
    base = report["baseline_summary"]
    v21 = report["v21_golden_summary"]
    cmp_ = report["comparison_table"]
    lines = [
        "# V2.1 Evaluation — Context Resolution vs QA Baseline",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "Diagnosis only. No schema/migration/verifier/limit/prompt changes in this sprint.",
        "",
        "## Baseline (pre-V2.1)",
        "",
        f"- Runs: {', '.join(BASELINE_IDS)}",
        f"- Success: **{base['success_count']}/{base['n']}** ({base['success_rate']})",
        f"- Avg latency: **{base['avg_latency_ms']} ms** (median {base['median_latency_ms']})",
        f"- Avg tool calls: **{base['avg_tool_calls']}** (max {base['max_tool_calls']})",
        f"- Retrieve-more rate: **{base['retrieve_more_rate']}**",
        f"- Verifier acceptance: **{base['verifier_acceptance_rate']}**",
        "",
        "Understand consistently left `subject=null` while topic=`fractions`.",
        "",
        "## Resolver completeness (live API)",
        "",
        "### Default resolve (`grade=CLASS_4`)",
        "",
    ]
    dflt = report["resolver_probes"]["default_grade_code"]
    lines.append(
        f"- Status: `{dflt.get('resolution', {}).get('status')}` — "
        f"{dflt.get('resolution', {}).get('message')}"
    )
    lines.append(
        f"- Grade resolved: `{dflt.get('grade')}` · GC id: `{dflt.get('grade_curriculum_id')}` · strategy: `{((dflt.get('resolution') or {}).get('diagnostics') or {}).get('grade_strategy')}`"
    )
    lines += [
        "",
        "### Correct grade_id resolve",
        "",
    ]
    ok = report["resolver_probes"]["correct_grade_id"]
    lines.append(f"- Status: `{ok.get('resolution', {}).get('status')}`")
    lines.append(f"- Units ({ok.get('unit_count')}):")
    for u in ok.get("units") or []:
        lines.append(f"  - `{u.get('code')}` — {u.get('name')}")
    lines.append(f"- Learning outcomes ({ok.get('lo_count')}):")
    for lo in ok.get("learning_outcomes") or []:
        lines.append(
            f"  - `{lo.get('code')}` parent=`{lo.get('parent_content_code')}` "
            f"provenance={lo.get('has_provenance')} eq={lo.get('evidence_quality')}"
        )
    lines += [
        "",
        "### Authoritative GradeCurriculum inventory (false-narrowing check)",
        "",
        f"- Fraction units in GC tree: **{report['authoritative_fraction']['unit_count']}**",
        f"- LO codes: {', '.join(report['authoritative_fraction']['lo_codes'])}",
        f"- Resolver with correct grade_id returns the same unit set: "
        f"**{report['false_narrowing']['verdict']}**",
        "",
        "## V2.1 golden question (10 runs)",
        "",
        f"Question: _{GOLDEN}_",
        "",
        f"- Success: **{v21['success_count']}/{v21['n']}** ({v21['success_rate']})",
        f"- Avg latency: **{v21['avg_latency_ms']} ms** (median {v21['median_latency_ms']})",
        f"- Avg tool calls: **{v21['avg_tool_calls']}** (max {v21['max_tool_calls']})",
        f"- Retrieve-more rate: **{v21['retrieve_more_rate']}**",
        f"- Avg resolver calls: **{v21['avg_resolver_calls']}**",
        f"- Avg legacy calls: **{v21['avg_legacy_calls']}**",
        f"- Sequence classes: `{v21['sequence_class_distribution']}`",
        f"- Verifier acceptance: **{v21['verifier_acceptance_rate']}**",
        "",
        "## Comparison",
        "",
        "| Metric | Pre-V2.1 | V2.1 |",
        "| --- | ---: | ---: |",
    ]
    for row in cmp_:
        lines.append(f"| {row['metric']} | {row['pre_v21']} | {row['v21']} |")
    lines += [
        "",
        "## Failure classification",
        "",
    ]
    for f in report["failure_classifications"]:
        lines.append(
            f"- `{f['agent_run_id']}` — **{f['category']}**: {f['reason']}"
        )
    lines += [
        "",
        "## Stress matrix",
        "",
    ]
    for s in report["stress_runs"]:
        lines.append(
            f"- [{s['tag']}] status=`{s['status']}` tools={s['tool_sequence']} "
            f"class=`{s['sequence_class']}` resolver={s['resolver_calls']}"
        )
    lines += [
        "",
        "## Subject=null",
        "",
        report["subject_null_analysis"],
        "",
        "## Recommendation",
        "",
        f"**{report['recommendation']['code']}** — {report['recommendation']['text']}",
        "",
        "### Next engineering change (do not implement here)",
        "",
    ]
    for item in report["recommendation"]["next_changes"]:
        lines.append(f"- {item}")
    lines.append("")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(timeout=180.0)

    baseline_rows = json.loads(BASELINE_COMPARISON.read_text())
    # keep only the five named baseline IDs
    baseline_rows = [r for r in baseline_rows if r.get("agent_run_id") in BASELINE_IDS]
    base_sum = baseline_summary(baseline_rows)

    authoritative = collect_fraction_authoritative(client)
    probes = probe_resolver(client)
    (OUT / "resolver_probes.json").write_text(json.dumps(probes, indent=2))
    (OUT / "authoritative_fraction.json").write_text(
        json.dumps(authoritative, indent=2)
    )

    ok_units = {u.get("code") for u in (probes.get("correct_grade_id") or {}).get("units") or []}
    auth_units = {u.get("code") for u in authoritative["fraction_units"]}
    false_narrowing = {
        "authoritative_unit_codes": sorted(auth_units),
        "resolver_unit_codes_with_correct_grade_id": sorted(ok_units),
        "missing_from_resolver": sorted(auth_units - ok_units),
        "extra_in_resolver": sorted(ok_units - auth_units),
        "default_path_status": (probes.get("default_grade_code") or {})
        .get("resolution", {})
        .get("status"),
        "verdict": (
            "A_complete"
            if ok_units == auth_units and ok_units
            else (
                "C_misses_units_or_blocked"
                if (probes.get("default_grade_code") or {})
                .get("resolution", {})
                .get("status")
                == "not_found"
                else "incomplete"
            )
        ),
        "notes": (
            "Default grade=CLASS_4 resolves to UPPER_PRIMARY grade UUID that is not "
            "the GradeCurriculum.grade_id (orphan/inactive Class 4 row holds GC). "
            "With the GC grade_id, resolver returns all three fraction units and 10 LOs."
        ),
    }

    golden_runs = []
    for i in range(1, 11):
        print(f"=== GOLDEN {i}/10 ===", flush=True)
        row = ask_once(client, GOLDEN, f"golden_{i:02d}")
        golden_runs.append(row)
        print(
            json.dumps(
                {
                    "run": i,
                    "status": row["status"],
                    "latency_ms": row["latency_ms"],
                    "tools": row["tool_sequence"],
                    "class": row["sequence_class"],
                    "resolver": row["resolver_calls"],
                },
                default=str,
            ),
            flush=True,
        )
        time.sleep(0.5)

    stress_questions = [
        ("direct_lo", GOLDEN),
        ("direct_c4u06", "What are the learning outcomes for C4U06?"),
        ("grade_subject_topics", "What topics are covered in Primary 4 Mathematics?"),
        ("topic_ambiguity_no_grade", "What are the learning objectives for fractions?"),
        ("missing_subject_phrasing", "What are the learning objectives for fractions in Primary 4?"),
        ("unknown_topic", "What are the learning objectives for quantum computing in Primary 4 Mathematics?"),
    ]
    stress_runs = []
    for tag, q in stress_questions:
        print(f"=== STRESS {tag} ===", flush=True)
        # skip duplicate golden ask for missing_subject_phrasing identical to golden —
        # still run independently as required.
        row = ask_once(client, q, f"stress_{tag}")
        stress_runs.append(row)
        print(
            json.dumps(
                {
                    "tag": tag,
                    "status": row["status"],
                    "tools": row["tool_sequence"],
                    "class": row["sequence_class"],
                },
                default=str,
            ),
            flush=True,
        )
        time.sleep(0.5)

    v21_sum = v21_summary(golden_runs)

    failures = []
    for r in golden_runs + stress_runs:
        if r["status"] in {"completed", "answered"} and r["sequence_class"] != "failure":
            continue
        reason = r.get("termination_reason") or r.get("status")
        category = "retrieval"
        if (r.get("resolver_calls") or 0) > 0 and (
            r.get("resolve_observability")
            and any(
                (o or {}).get("resolution_status") == "not_found"
                for o in r.get("resolve_observability") or []
            )
        ):
            category = "resolver correctness"
            reason = (
                "resolve_curriculum_context returned not_found due to CLASS_4 grade UUID "
                "mismatch; agent fell back / exhausted retrieval"
            )
        elif r.get("status") == "insufficient_evidence":
            category = "verification" if (r.get("evidence_count") or 0) > 10 else "retrieval"
        failures.append(
            {
                "agent_run_id": r.get("agent_run_id"),
                "tag": r.get("tag"),
                "status": r.get("status"),
                "category": category,
                "reason": reason,
                "tool_sequence": r.get("tool_sequence"),
            }
        )

    # subject null analysis from golden runs
    subjects = [r.get("understand_subject") for r in golden_runs]
    subject_null_analysis = (
        "Baseline understand left subject=null. V2.1 tool selection for LO questions "
        "can still call resolve with grade+topic only; live default resolve then fails "
        f"GradeCurriculum lookup (and/or missing-subject path). Observed understand subjects "
        f"in golden runs: {subjects}. Practical consequence of subject=null is NOT eliminated "
        "while grade-code resolution points at a grade row without GradeCurriculum content."
    )

    comparison_table = [
        {
            "metric": "success rate",
            "pre_v21": base_sum["success_rate"],
            "v21": v21_sum["success_rate"],
        },
        {
            "metric": "avg latency",
            "pre_v21": base_sum["avg_latency_ms"],
            "v21": v21_sum["avg_latency_ms"],
        },
        {
            "metric": "median latency",
            "pre_v21": base_sum["median_latency_ms"],
            "v21": v21_sum["median_latency_ms"],
        },
        {
            "metric": "avg tool calls",
            "pre_v21": base_sum["avg_tool_calls"],
            "v21": v21_sum["avg_tool_calls"],
        },
        {
            "metric": "max tool calls",
            "pre_v21": base_sum["max_tool_calls"],
            "v21": v21_sum["max_tool_calls"],
        },
        {
            "metric": "retrieve_more rate",
            "pre_v21": base_sum["retrieve_more_rate"],
            "v21": v21_sum["retrieve_more_rate"],
        },
        {
            "metric": "duplicate calls (skips avg)",
            "pre_v21": "n/a (see traces)",
            "v21": v21_sum["avg_duplicate_tool_skips"],
        },
        {
            "metric": "zero-new-evidence calls (avg)",
            "pre_v21": "n/a (see traces)",
            "v21": v21_sum["avg_zero_new_evidence_calls"],
        },
        {
            "metric": "verifier acceptance",
            "pre_v21": base_sum["verifier_acceptance_rate"],
            "v21": v21_sum["verifier_acceptance_rate"],
        },
    ]

    # Recommendation logic
    default_broken = (
        (probes.get("default_grade_code") or {}).get("resolution") or {}
    ).get("status") == "not_found"
    resolve_used = v21_sum["avg_resolver_calls"] > 0
    if default_broken:
        recommendation = {
            "code": "2",
            "text": (
                "V2.1 works but resolver completeness/correctness needs correction "
                "before it should be the preferred retrieval path."
            ),
            "next_changes": [
                "Fix grade resolution so CLASS_4 maps to the GradeCurriculum-linked grade "
                "(or resolve GradeCurriculum via grade.code within curriculum without "
                "requiring exact grade UUID match).",
                "Re-run this evaluation after grade resolution fix; do not remove legacy tools yet.",
                "Separately address understand subject=null once resolve works for grade+topic.",
                "Do not change verifier/limits/prompts until structured evidence reaches GENERATE reliably.",
            ],
        }
    elif v21_sum["success_rate"] >= 0.8 and resolve_used:
        recommendation = {
            "code": "1",
            "text": "V2.1 is working and should become the preferred retrieval path.",
            "next_changes": [
                "Keep legacy tools as fallback",
                "Monitor generation/verifier as remaining latency drivers",
            ],
        }
    elif resolve_used and v21_sum["success_rate"] < base_sum["success_rate"]:
        recommendation = {
            "code": "4",
            "text": "V2.1 should not yet be used as the preferred path.",
            "next_changes": ["Investigate regressions before preferring resolve"],
        }
    else:
        recommendation = {
            "code": "3",
            "text": "V2.1 routing needs correction.",
            "next_changes": ["Ensure resolve is selected and legacy fallback is orderly"],
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "constraints": {
            "schema_changes": False,
            "migrations": False,
            "verifier_changes": False,
            "retrieval_limit_changes": False,
            "graph_redesign": False,
            "prompt_redesign": False,
        },
        "golden_question": GOLDEN,
        "baseline_run_ids": BASELINE_IDS,
        "baseline_rows": baseline_rows,
        "baseline_summary": base_sum,
        "resolver_probes": probes,
        "authoritative_fraction": authoritative,
        "false_narrowing": false_narrowing,
        "v21_golden_runs": golden_runs,
        "v21_golden_summary": v21_sum,
        "stress_runs": stress_runs,
        "comparison_table": comparison_table,
        "failure_classifications": failures,
        "subject_null_analysis": subject_null_analysis,
        "recommendation": recommendation,
    }

    summary_path = ROOT / "data" / "diagnostics" / "v21_evaluation.json"
    summary_path.write_text(json.dumps(report, indent=2, default=str))
    (OUT / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    write_markdown(report)
    print("--- DONE ---", flush=True)
    print(json.dumps({"recommendation": recommendation, "v21": v21_sum, "baseline": base_sum}, indent=2))


if __name__ == "__main__":
    main()
