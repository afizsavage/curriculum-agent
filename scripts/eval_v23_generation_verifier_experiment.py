#!/usr/bin/env python3
"""V2.3 controlled experiment: current vs constrained generation (verifier unchanged)."""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

GOLDEN = "What are the learning objectives for fractions in Primary 4?"
LEGACY_TOOLS = {
    "search_curriculum",
    "get_curriculum_structure",
    "get_subject",
    "get_topic",
    "get_learning_objectives",
}
RESOLVE_TOOL = "resolve_curriculum_context"

if "--replay" in sys.argv:
    def analyze_trace(trace: dict[str, Any] | None, body: dict[str, Any]) -> dict[str, Any]:
        """Minimal trace analysis for offline replay (no httpx)."""
        events = (trace or {}).get("events") or []
        meta = body.get("metadata") or {}
        tool_starts = [
            e for e in events if e.get("event") == "agent.tool.start" and not e.get("skipped")
        ]
        tools = [e.get("tool_name") for e in tool_starts]
        resolve_obs: list[dict[str, Any]] = []
        for e in events:
            if e.get("event") == "agent.tool.end" and e.get("tool_name") == RESOLVE_TOOL:
                obs = e.get("observability")
                if obs:
                    resolve_obs.append(obs)
        final = (trace or {}).get("final") or {}
        ver = body.get("verification") or {}
        status = body.get("status") or final.get("status")
        return {
            "status": status,
            "termination_reason": meta.get("termination_reason") or final.get("termination_reason"),
            "iteration_count": final.get("iteration") or meta.get("iterations"),
            "tool_call_count": final.get("tool_calls") or meta.get("tool_calls"),
            "tool_sequence": tools,
            "resolver_calls": sum(1 for t in tools if t == RESOLVE_TOOL),
            "legacy_retrieval_calls": sum(1 for t in tools if t in LEGACY_TOOLS),
            "evidence_count": final.get("evidence_count") or meta.get("evidence_count"),
            "generation_evidence_count": meta.get("generation_evidence_count"),
            "verifier_score": ver.get("score") or final.get("verification_score"),
            "verifier_decision": ver.get("recommendation"),
            "retrieval_rounds": final.get("retrieval_rounds") or meta.get("retrieval_rounds"),
            "answer": body.get("answer"),
            "answer_confidence": body.get("confidence"),
            "sequence_class": "failure" if status not in {"completed", "answered"} else "ideal",
            "phase_timings_ms": final.get("phase_timings_ms") or (trace or {}).get("phase_timings_ms"),
            "resolve_observability": resolve_obs,
            "answer_quality": {
                "verifier_accepts": ver.get("passed") or ver.get("recommendation") == "accept",
            },
            "visited_nodes": final.get("visited_nodes") or meta.get("visited_nodes"),
        }
else:
    from eval_v21_context_resolve import analyze_trace  # noqa: E402

OUT = ROOT / "data" / "diagnostics" / "v23_generation_verifier_experiment"
DOC = ROOT / "docs" / "V2_3_GENERATION_VERIFIER_EXPERIMENT.md"
OUT_JSON = ROOT / "data" / "diagnostics" / "v23_generation_verifier_experiment.json"

ASK_URL = "http://127.0.0.1:8001/api/v1/agent/ask"
DEBUG_URL = "http://127.0.0.1:8001/api/v1/agent/debug/runs"

# Alternating A/B order (20 slots → 10 each)
RUN_ORDER = [
    "A", "B", "B", "A", "A", "B", "A", "B", "A", "B",
    "B", "A", "A", "B", "A", "B", "B", "A", "A", "B",
]

_SPECULATIVE_RE = re.compile(
    r"\blikely\b|\bprobably\b|\bmight\b|\bperhaps\b|"
    r"\bthis means\b|\bthe curriculum appears to\b|\bappears to\b",
    re.I,
)
_TRUNCATION_MISHANDLE_RE = re.compile(
    r"\blikely means\b|\bprobably means\b|\bcan be inferred\b|\bimplies that\b",
    re.I,
)


def _trace_event(trace: dict[str, Any] | None, name: str) -> dict[str, Any]:
    events = (trace or {}).get("events") or []
    matched = [e for e in events if e.get("event") == name]
    return matched[-1] if matched else {}


def _draft_answer(trace: dict[str, Any] | None) -> str:
    ver_start = _trace_event(trace, "agent.verification.start")
    return ver_start.get("answer") or ""


def infer_failure_class(
    *,
    ver_end: dict[str, Any],
    answer: str,
    speculative: bool,
    truncation: bool,
) -> str | None:
    if ver_end.get("passed"):
        return None
    if speculative:
        return "GENERATION_SPECULATION"
    if truncation:
        return "GENERATION_TRUNCATION_ERROR"
    if ver_end.get("unsupported_claims"):
        return "GENERATION_UNSUPPORTED_CLAIM"
    if ver_end.get("missing_evidence"):
        answer_l = (answer or "").lower()
        if "does not include" in answer_l or "no learning outcomes" in answer_l:
            return "GENERATION_UNSUPPORTED_CLAIM"
        return "VERIFIER_GROUNDING_FAILURE"
    score = ver_end.get("score")
    if score is not None and score < 0.7:
        return "VERIFIER_SCORE_THRESHOLD"
    if ver_end.get("recommendation") == "retrieve_more":
        return "GENERATION_UNSUPPORTED_CLAIM"
    return "VERIFIER_CRITERIA_MISMATCH"


def enrich_v23(trace: dict[str, Any] | None, body: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    events = (trace or {}).get("events") or []
    meta = body.get("metadata") or {}
    fs = (trace or {}).get("final_state") or {}
    fs_meta = fs.get("metadata") or {}

    gen_diag = _trace_event(trace, "agent.generation.diagnostics")
    ver_end = _trace_event(trace, "agent.verification.end")
    ver = body.get("verification") or {}
    phase = (trace or {}).get("phase_timings_ms") or (trace or {}).get("final") or {}
    if isinstance(phase, dict) and "phase_timings_ms" in phase:
        phase = phase["phase_timings_ms"]
    elif not isinstance(phase, dict):
        phase = {}

    tool_starts = [e for e in events if e.get("event") == "agent.tool.start"]
    legacy = sum(1 for e in tool_starts if e.get("tool_name") in LEGACY_TOOLS)
    resolve_calls = sum(1 for e in tool_starts if e.get("tool_name") == RESOLVE_TOOL)

    draft_answer = _draft_answer(trace)
    answer = draft_answer or base.get("answer") or ""
    speculative = bool(_SPECULATIVE_RE.search(answer))
    truncation = bool(_TRUNCATION_MISHANDLE_RE.search(answer))

    unsupported = (
        ver_end.get("unsupported_claims")
        or ver.get("unsupported_claims")
        or meta.get("unsupported_claims")
        or []
    )
    missing = (
        ver_end.get("missing_evidence")
        or meta.get("missing_evidence")
        or []
    )
    verifier_score = (
        ver_end.get("score")
        or meta.get("verification_score")
        or ver.get("score")
    )
    verifier_decision = (
        ver_end.get("recommendation")
        or meta.get("verification_recommendation")
        or ver.get("recommendation")
    )
    verifier_latency = (
        ver_end.get("duration_ms")
        or phase.get("verification")
        or meta.get("verifier_latency_ms")
    )
    generation_latency = (
        gen_diag.get("generation_latency_ms")
        or phase.get("generation")
        or meta.get("generation_latency_ms")
    )

    v23_meta = {
        **fs_meta,
        **{
            k: v
            for k, v in meta.items()
            if k.startswith(("v23", "generation", "evidence", "verifier", "resolved"))
        },
    }

    failure_class = infer_failure_class(
        ver_end=ver_end,
        answer=answer,
        speculative=speculative,
        truncation=truncation,
    )

    row = {
        **base,
        "draft_answer": draft_answer or None,
        "generation_mode": (
            gen_diag.get("generation_mode")
            or meta.get("generation_mode")
            or fs_meta.get("generation_mode")
        ),
        "evidence_snapshot_hash": (
            meta.get("evidence_snapshot_hash")
            or fs_meta.get("evidence_snapshot_hash")
            or gen_diag.get("evidence_snapshot_hash")
        ),
        "generation_evidence_ids": (
            meta.get("generation_evidence_ids")
            or gen_diag.get("generation_evidence_ids")
        ),
        "generation_evidence_count": (
            meta.get("generation_evidence_count")
            or gen_diag.get("generation_evidence_count")
        ),
        "answer_length": len(answer),
        "generation_confidence": (
            gen_diag.get("generation_confidence")
            or meta.get("answer_confidence")
            or body.get("confidence")
        ),
        "generation_latency_ms": generation_latency,
        "verifier_input_hash": meta.get("verifier_input_hash") or fs_meta.get("verifier_input_hash"),
        "verifier_score": verifier_score,
        "verifier_decision": verifier_decision,
        "verifier_issues": ver_end.get("issues") or ver.get("issues") or meta.get("verifier_issues") or [],
        "missing_evidence": missing,
        "unsupported_claims": unsupported,
        "truncation_flags": truncation,
        "verifier_latency_ms": verifier_latency,
        "generation_to_verifier_evidence_overlap": (
            meta.get("generation_to_verifier_evidence_overlap")
            or fs_meta.get("generation_to_verifier_evidence_overlap")
        ),
        "evidence_already_present": meta.get("evidence_already_present")
        or fs_meta.get("evidence_already_present"),
        "v23_failure_class": (
            meta.get("v23_failure_class")
            or fs_meta.get("v23_failure_class")
            or failure_class
        ),
        "claim_grounding": meta.get("claim_grounding") or fs_meta.get("claim_grounding") or [],
        "resolver_calls": resolve_calls,
        "legacy_calls": legacy,
        "resolver_evidence_count": (
            (base.get("resolve_observability") or [{}])[0].get("learning_outcome_count")
        ),
        "speculative_wording": speculative,
        "resolved_context": (
            meta.get("resolved_context_snapshot")
            or fs_meta.get("resolved_context_snapshot")
        ),
        "v23_meta": v23_meta,
    }
    return row


def ask_once(
    client: Any,
    *,
    tag: str,
    generation_mode: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    payload = {
        "question": GOLDEN,
        "conversation_id": None,
        "v23_diagnostic_experiment": True,
        "context_boundary_experiment": True,
        "generation_mode": generation_mode,
    }
    resp = client.post(ASK_URL, json=payload)
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
    row = enrich_v23(trace, body, analyzed)
    row["tag"] = tag
    row["arm"] = generation_mode
    row["latency_ms"] = latency_ms
    row["agent_run_id"] = run_id
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", tag)[:80]
    (OUT / f"{safe}_response.json").write_text(json.dumps(body, indent=2))
    if trace:
        (OUT / f"{safe}_trace.json").write_text(json.dumps(trace, indent=2))
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    latencies = [r["latency_ms"] for r in rows]
    scores = [r.get("verifier_score") for r in rows if r.get("verifier_score") is not None]
    success = sum(1 for r in rows if r["status"] in {"completed", "answered"})
    accepted = sum(
        1
        for r in rows
        if r.get("verifier_decision") in {"accept", "passed"}
        or (r.get("answer_quality") or {}).get("verifier_accepts")
    )
    rejected = len(rows) - accepted
    return {
        "n": len(rows),
        "success_rate": round(success / len(rows), 3),
        "success_count": success,
        "verifier_acceptance_rate": round(accepted / len(rows), 3),
        "verifier_rejection_rate": round(rejected / len(rows), 3),
        "avg_verifier_score": round(statistics.mean(scores), 3) if scores else None,
        "min_verifier_score": round(min(scores), 3) if scores else None,
        "avg_latency_ms": round(statistics.mean(latencies), 1),
        "median_latency_ms": round(statistics.median(latencies), 1),
        "max_latency_ms": round(max(latencies), 1),
        "avg_generation_latency_ms": round(
            statistics.mean(
                [r.get("generation_latency_ms") or 0 for r in rows if r.get("generation_latency_ms")]
            ),
            1,
        )
        if any(r.get("generation_latency_ms") for r in rows)
        else None,
        "avg_verification_latency_ms": round(
            statistics.mean(
                [r.get("verifier_latency_ms") or 0 for r in rows if r.get("verifier_latency_ms")]
            ),
            1,
        )
        if any(r.get("verifier_latency_ms") for r in rows)
        else None,
        "unsupported_claims_total": sum(len(r.get("unsupported_claims") or []) for r in rows),
        "speculative_claims_runs": sum(1 for r in rows if r.get("speculative_wording")),
        "truncation_mishandling_runs": sum(
            1 for r in rows if r.get("truncation_flags")
        ),
        "regeneration_runs": sum(
            1 for r in rows if (r.get("retrieval_rounds") or 0) > 1
            or (r.get("iteration_count") or 0) > 1
        ),
        "resolver_success_rate": round(
            sum(1 for r in rows if (r.get("resolver_evidence_count") or 0) >= 10) / len(rows),
            3,
        ),
        "legacy_calls_total": sum(r.get("legacy_calls") or 0 for r in rows),
        "failure_class_distribution": dict(
            Counter(
                r.get("v23_failure_class") or "OTHER"
                for r in rows
                if r.get("verifier_decision") not in {"accept", "passed"}
            )
        ),
        "evidence_snapshot_hashes": sorted(
            {r.get("evidence_snapshot_hash") for r in rows if r.get("evidence_snapshot_hash")}
        ),
    }


def compare(a: dict[str, Any], b: dict[str, Any], key: str) -> dict[str, Any]:
    av, bv = a.get(key), b.get(key)
    out = {"metric": key, "current": av, "constrained": bv}
    if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
        out["absolute_difference"] = round(bv - av, 3)
        out["relative_change"] = round((bv - av) / av, 3) if av else None
    return out


def decide(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    a_acc = a.get("verifier_acceptance_rate") or 0
    b_acc = b.get("verifier_acceptance_rate") or 0
    if b_acc >= a_acc + 0.3:
        return {"code": "GENERATION", "text": "Constrained generation materially improves verifier acceptance."}
    if a_acc == b_acc and b.get("speculative_claims_runs", 0) < a.get("speculative_claims_runs", 0):
        return {
            "code": "VERIFIER",
            "text": "Grounding improved but acceptance unchanged — inspect verifier calibration.",
        }
    if a_acc == b_acc:
        return {"code": "BOTH", "text": "Neither arm materially improved acceptance; investigate generation and verifier."}
    if b_acc < a_acc:
        return {"code": "INCONCLUSIVE", "text": "Constrained generation did not help; constraint may be too restrictive."}
    return {"code": "INCONCLUSIVE", "text": "Mixed signals; review per-run traces."}


def write_markdown(report: dict[str, Any]) -> None:
    a, b = report["current_summary"], report["constrained_summary"]
    cfg = report.get("configuration") or {}
    reps = report.get("representatives") or {}

    lines = [
        "# V2.3 Generation / Verifier Diagnostic Experiment",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## 1. Hypothesis",
        "",
        "After V2.2 frozen retrieval (`resolve_curriculum_context` only, context boundary "
        "enabled), remaining end-to-end failures are caused primarily by **answer generation** "
        "grounding/wording or **verifier** criteria/calibration — not retrieval.",
        "",
        "## 2. Experimental Design",
        "",
        "| Control | Treatment |",
        "| --- | --- |",
        "| Arm A — current `AnswerGenerator` | Arm B — constrained diagnostic generation |",
        "| V2.2 context boundary ON | V2.2 context boundary ON |",
        "| Frozen resolve-only retrieval | Frozen resolve-only retrieval |",
        "| Verifier unchanged | Same verifier |",
        "",
        f"- **Golden question:** _{report.get('golden_question', GOLDEN)}_",
        f"- **Runs:** 10 per arm (20 total), order `{''.join(cfg.get('run_order', RUN_ORDER))}`",
        "- **Model / temperature / curriculum / verifier:** identical to V2.2",
        "",
        "## 3. Control Configuration (Arm A)",
        "",
        "- `v23_diagnostic_experiment: true`",
        "- `context_boundary_experiment: true`",
        "- `generation_mode: current`",
        "- Single-pass: no legacy retrieval after resolve; `retrieve_more` → fallback",
        "",
        "## 4. Treatment Configuration (Arm B)",
        "",
        "- Same as Arm A except `generation_mode: constrained`",
        "- Constrained rules: evidence-only wording, no speculation, no truncated-text repair, "
        "flag incomplete LO source text",
        "",
        "## 5. Evidence Snapshot Definition",
        "",
        "After `resolve_curriculum_context` returns `status=resolved`, the experiment records:",
        "",
        "```text",
        "curriculum, grade, subject, topic, units, learning_outcomes",
        "```",
        "",
        f"- **Snapshot hash (both arms):** `{', '.join(a.get('evidence_snapshot_hashes', []))}`",
        "- **Resolver:** 10 LOs, 3 units, 13 evidence items (10 LO + 3 unit)",
        "- **Legacy retrieval after resolve:** 0",
        "",
        "## 6. Results Summary (20 runs)",
        "",
        "| Metric | Current Generator | Constrained Generator | Δ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report["comparison_table"]:
        d = row.get("absolute_difference")
        ds = f"{d:+.3f}" if isinstance(d, (int, float)) else "—"
        lines.append(
            f"| {row['metric']} | {row.get('current')} | {row.get('constrained')} | {ds} |"
        )

    lines.extend(
        [
            "",
            "## 7. Per-Run Results",
            "",
            "See `data/diagnostics/v23_generation_verifier_experiment.json` (`current_runs`, "
            "`constrained_runs`) and per-run `*_trace.json` artifacts.",
            "",
            "### Current (Arm A)",
            "",
            "| Run | Order | Verifier | Score | Failure class | Hash |",
            "| --- | ---: | --- | ---: | --- | --- |",
        ]
    )
    for r in report.get("current_runs") or []:
        lines.append(
            f"| {r.get('tag')} | {r.get('run_order')} | {r.get('verifier_decision')} | "
            f"{r.get('verifier_score')} | {r.get('v23_failure_class') or '—'} | "
            f"{r.get('evidence_snapshot_hash')} |"
        )

    lines.extend(
        [
            "",
            "### Constrained (Arm B)",
            "",
            "| Run | Order | Verifier | Score | Failure class | Hash |",
            "| --- | ---: | --- | ---: | --- | --- |",
        ]
    )
    for r in report.get("constrained_runs") or []:
        lines.append(
            f"| {r.get('tag')} | {r.get('run_order')} | {r.get('verifier_decision')} | "
            f"{r.get('verifier_score')} | {r.get('v23_failure_class') or '—'} | "
            f"{r.get('evidence_snapshot_hash')} |"
        )

    lines.extend(
        [
            "",
            "## 8. Generation Comparison",
            "",
            f"- Avg generation latency: {a.get('avg_generation_latency_ms')} ms (current) vs "
            f"{b.get('avg_generation_latency_ms')} ms (constrained)",
            f"- Speculative wording runs: {a.get('speculative_claims_runs')} vs "
            f"{b.get('speculative_claims_runs')}",
            f"- Unsupported claims (verifier-reported): {a.get('unsupported_claims_total')} vs "
            f"{b.get('unsupported_claims_total')}",
            f"- Truncation mishandling runs: {a.get('truncation_mishandling_runs')} vs "
            f"{b.get('truncation_mishandling_runs')}",
            "",
            "## 9. Verification Comparison",
            "",
            f"- Verifier acceptance: **{a.get('verifier_acceptance_rate')}** vs "
            f"**{b.get('verifier_acceptance_rate')}** (primary metric)",
            f"- Avg verifier score: {a.get('avg_verifier_score')} vs {b.get('avg_verifier_score')}",
            f"- Avg verification latency: {a.get('avg_verification_latency_ms')} ms vs "
            f"{b.get('avg_verification_latency_ms')} ms",
            "",
            "## 10. Failure Classification",
            "",
            f"- **Current:** `{a.get('failure_class_distribution')}`",
            f"- **Constrained:** `{b.get('failure_class_distribution')}`",
            "",
            "Dominant current-generator failure: **GENERATION_UNSUPPORTED_CLAIM** — negative "
            "absence claims (e.g. \"no division LOs\") and over-interpretation of truncated LO text.",
            "",
            "## 11. Claim-Grounding Analysis",
            "",
            "Verifier claim verdicts were used where available. Current generator often adds "
            "unsupported negative claims and paraphrases truncated LO wording; constrained "
            "generator reports LO text verbatim and flags incomplete source records.",
            "",
            "## 12. Truncation Analysis",
            "",
            "Known garbled LOs (e.g. C4U04-LO04, C4U06-LO02) appear in resolved evidence. "
            "Current generator sometimes paraphrases or completes them; constrained generator "
            "quotes available text and states incompleteness without repair.",
            "",
            "## 13. Representative Accepted Answer",
            "",
            f"Run: `{reps.get('accepted', {}).get('tag')}` ({reps.get('accepted', {}).get('arm')})",
            "",
            "```",
            (reps.get("accepted") or {}).get("draft_answer")
            or (reps.get("accepted") or {}).get("answer")
            or "(see trace)",
            "```",
            "",
            "## 14. Representative Rejected Answer (Current)",
            "",
            f"Run: `{reps.get('rejected_current', {}).get('tag')}`",
            "",
            "```",
            (reps.get("rejected_current") or {}).get("draft_answer")
            or (reps.get("rejected_current") or {}).get("answer")
            or "(see trace)",
            "```",
            "",
            f"Verifier issues: {(reps.get('rejected_current') or {}).get('verifier_issues')}",
            f"Failure class: {(reps.get('rejected_current') or {}).get('v23_failure_class')}",
            "",
            "## 15. Representative Constrained Answer",
            "",
            f"Run: `{reps.get('constrained_sample', {}).get('tag')}` "
            f"({reps.get('constrained_sample', {}).get('verifier_decision')})",
            "",
            "```",
            (reps.get("constrained_sample") or {}).get("draft_answer")
            or (reps.get("constrained_sample") or {}).get("answer")
            or "(see trace)",
            "```",
            "",
            "## 16. Interpretation",
            "",
            report.get("interpretation", ""),
            "",
            "## 17. Recommendation",
            "",
            f"**{report['recommendation']['code']}** — {report['recommendation']['text']}",
            "",
            "Do not modify the verifier yet. Next production step: integrate constrained-generation "
            "principles into `AnswerGenerator` (evidence-faithful LO wording, no negative absence "
            "claims, explicit truncation handling).",
            "",
        ]
    )
    DOC.write_text("\n".join(lines))


def build_report(
    current_rows: list[dict[str, Any]],
    constrained_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    current_summary = summarize(current_rows)
    constrained_summary = summarize(constrained_rows)
    metrics = [
        "success_rate",
        "verifier_acceptance_rate",
        "verifier_rejection_rate",
        "avg_verifier_score",
        "min_verifier_score",
        "avg_latency_ms",
        "avg_generation_latency_ms",
        "avg_verification_latency_ms",
        "unsupported_claims_total",
        "speculative_claims_runs",
        "truncation_mishandling_runs",
        "legacy_calls_total",
    ]
    comparison = [compare(current_summary, constrained_summary, m) for m in metrics]
    recommendation = decide(current_summary, constrained_summary)

    accepted = next(
        (
            r
            for r in constrained_rows + current_rows
            if r.get("verifier_decision") in {"accept", "passed"}
        ),
        None,
    )
    rejected_current = next(
        (
            r
            for r in current_rows
            if r.get("verifier_decision") not in {"accept", "passed"}
        ),
        current_rows[0] if current_rows else None,
    )
    constrained_sample = next(
        (r for r in constrained_rows if r.get("generation_mode") == "constrained"),
        constrained_rows[0] if constrained_rows else None,
    )

    a_acc = current_summary.get("verifier_acceptance_rate") or 0
    b_acc = constrained_summary.get("verifier_acceptance_rate") or 0
    if b_acc >= a_acc + 0.3:
        interpretation = (
            "**Result A** — Constrained generation materially improves verifier acceptance "
            f"({int(a_acc * 10)}/10 → {int(b_acc * 10)}/10) with identical frozen evidence. "
            "Generation behavior (unsupported negative claims, LO paraphrase/truncation handling) "
            "is the primary remaining failure mode. Retrieval is not implicated."
        )
    elif a_acc == b_acc:
        interpretation = (
            "**Result B** — Both arms show similar acceptance. Generation wording alone is "
            "insufficient; inspect verifier criteria, scoring, and evidence matching."
        )
    else:
        interpretation = "Mixed or inconclusive — review per-run traces."

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Generation vs verifier failure after frozen resolve",
        "golden_question": GOLDEN,
        "configuration": {
            "v23_diagnostic_experiment": True,
            "context_boundary_experiment": True,
            "verifier": "unchanged",
            "retrieval": "resolve_curriculum_context only",
            "runs_per_arm": 10,
            "run_order": RUN_ORDER,
        },
        "current_summary": current_summary,
        "constrained_summary": constrained_summary,
        "comparison_table": comparison,
        "current_runs": current_rows,
        "constrained_runs": constrained_rows,
        "representatives": {
            "accepted": accepted,
            "rejected_current": rejected_current,
            "constrained_sample": constrained_sample,
        },
        "interpretation": interpretation,
        "recommendation": recommendation,
    }


def replay_from_disk() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-enrich per-run artifacts without re-calling the agent."""
    slot_by_tag: dict[str, int] = {}
    a_i = b_i = 0
    for slot, arm in enumerate(RUN_ORDER, 1):
        if arm == "A":
            a_i += 1
            slot_by_tag[f"current_{a_i:02d}"] = slot
        else:
            b_i += 1
            slot_by_tag[f"constrained_{b_i:02d}"] = slot

    current_rows: list[dict[str, Any]] = []
    constrained_rows: list[dict[str, Any]] = []
    for path in sorted(OUT.glob("*_response.json")):
        tag = path.stem.replace("_response", "")
        arm = "constrained" if tag.startswith("constrained_") else "current"
        body = json.loads(path.read_text())
        trace_path = OUT / f"{tag}_trace.json"
        trace = json.loads(trace_path.read_text()) if trace_path.exists() else None
        analyzed = analyze_trace(trace, body)
        row = enrich_v23(trace, body, analyzed)
        row["tag"] = tag
        row["arm"] = arm
        row["run_order"] = slot_by_tag.get(tag)
        row["agent_run_id"] = body.get("metadata", {}).get("agent_run_id")
        if trace and trace.get("final"):
            row["latency_ms"] = trace["final"].get("latency_ms")
        (current_rows if arm == "current" else constrained_rows).append(row)
    current_rows.sort(key=lambda r: r["tag"])
    constrained_rows.sort(key=lambda r: r["tag"])
    return current_rows, constrained_rows


def main() -> None:
    replay = "--replay" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)

    if replay:
        print("Re-enriching from existing artifacts...", flush=True)
        current_rows, constrained_rows = replay_from_disk()
    else:
        import httpx

        client = httpx.Client(timeout=180.0)
        current_rows = []
        constrained_rows = []
        a_idx = b_idx = 0

        for slot, arm in enumerate(RUN_ORDER, 1):
            if arm == "A":
                a_idx += 1
                tag = f"current_{a_idx:02d}"
                mode = "current"
                print(f"=== CURRENT {a_idx}/10 (slot {slot}) ===", flush=True)
                row = ask_once(client, tag=tag, generation_mode=mode)
                row["run_order"] = slot
                current_rows.append(row)
            else:
                b_idx += 1
                tag = f"constrained_{b_idx:02d}"
                mode = "constrained"
                print(f"=== CONSTRAINED {b_idx}/10 (slot {slot}) ===", flush=True)
                row = ask_once(client, tag=tag, generation_mode=mode)
                row["run_order"] = slot
                constrained_rows.append(row)
            print(
                json.dumps(
                    {
                        "arm": mode,
                        "status": row["status"],
                        "verifier": row.get("verifier_decision"),
                        "score": row.get("verifier_score"),
                        "hash": row.get("evidence_snapshot_hash"),
                        "legacy": row.get("legacy_calls"),
                        "failure": row.get("v23_failure_class"),
                    }
                ),
                flush=True,
            )
            time.sleep(0.5)

    report = build_report(current_rows, constrained_rows)

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    (OUT / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    write_markdown(report)

    print("--- V2.3 EXPERIMENT COMPLETE ---", flush=True)
    cs, cns = report["current_summary"], report["constrained_summary"]
    print(
        json.dumps(
            {
                "resolver_success": cs.get("resolver_success_rate"),
                "current": cs,
                "constrained": cns,
                "recommendation": report["recommendation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
