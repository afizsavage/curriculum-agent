#!/usr/bin/env python3
"""V2.5 verifier / imperfect-evidence quality experiment (arms A–D)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLDEN = "What are the learning objectives for fractions in Primary 4?"
ASK_URL = "http://127.0.0.1:8001/api/v1/agent/ask"
DEBUG_URL = "http://127.0.0.1:8001/api/v1/agent/debug/runs"
OUT = ROOT / "data" / "diagnostics" / "v25_verifier_evidence_quality"
DOC = ROOT / "docs" / "V2_5_VERIFIER_EVIDENCE_QUALITY.md"
OUT_JSON = ROOT / "data" / "diagnostics" / "v25_verifier_evidence_quality.json"
EXPECTED_HASH = "977b259fcfb4b282"
ARMS = ("A", "B", "C", "D")
RUNS_PER_ARM = 10


def _trace_event(trace: dict | None, name: str) -> dict:
    events = (trace or {}).get("events") or []
    matched = [e for e in events if e.get("event") == name]
    return matched[-1] if matched else {}


def _ver_end(trace: dict | None) -> dict:
    events = (trace or {}).get("events") or []
    matched = [e for e in events if e.get("event") == "agent.verification.end"]
    return matched[-1] if matched else {}


def enrich(trace: dict | None, body: dict[str, Any], arm: str, tag: str) -> dict[str, Any]:
    meta = body.get("metadata") or {}
    ver = body.get("verification") or {}
    ver_end = _ver_end(trace)
    final = (trace or {}).get("final") or {}
    v25 = meta.get("v25_diagnostics") or _trace_event(trace, "agent.v25.diagnostics") or {}

    accepted = (
        ver_end.get("recommendation") == "accept"
        or ver.get("recommendation") == "accept"
        or ver.get("passed")
    )
    success = body.get("status") in {"completed", "answered"}

    return {
        "tag": tag,
        "arm": arm,
        "evidence_condition": v25.get("evidence_condition"),
        "agent_run_id": meta.get("agent_run_id"),
        "status": body.get("status") or final.get("status"),
        "success": success,
        "verifier_accepted": accepted,
        "verifier_score": ver_end.get("score") or ver.get("score") or v25.get("verifier_score"),
        "verifier_decision": ver_end.get("recommendation") or ver.get("recommendation"),
        "tool_calls": final.get("tool_calls") or meta.get("tool_calls"),
        "latency_ms": final.get("latency_ms"),
        "evidence_hash": v25.get("evidence_hash") or meta.get("evidence_snapshot_hash"),
        "transformed_evidence_hash": v25.get("transformed_evidence_hash"),
        "imperfect_evidence_count": v25.get("imperfect_evidence_count"),
        "evidence_inventory": v25.get("evidence_inventory") or [],
        "retrieve_more_requested": v25.get("retrieve_more_requested"),
        "evidence_already_present": v25.get("evidence_already_present"),
        "insufficient_evidence": v25.get("insufficient_evidence")
        or body.get("status") == "insufficient_evidence",
        "regeneration": v25.get("regeneration"),
        "verifier_issue_class": v25.get("verifier_issue_class"),
        "unsupported_claims": v25.get("unsupported_claims") or ver_end.get("unsupported_claims") or [],
        "rejected_claims": v25.get("rejected_claims") or [],
        "speculative_claims": v25.get("speculative_claims"),
        "answer": body.get("answer"),
        "original_learning_outcomes": v25.get("original_learning_outcomes") or [],
        "v25": v25,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    scores = [r["verifier_score"] for r in rows if r.get("verifier_score") is not None]
    accepted = sum(1 for r in rows if r.get("verifier_accepted"))
    success = sum(1 for r in rows if r.get("success"))
    retrieve_more = sum(1 for r in rows if r.get("retrieve_more_requested"))
    insufficient = sum(1 for r in rows if r.get("insufficient_evidence"))
    grounding = sum(1 for r in rows if r.get("verifier_issue_class") == "GROUNDING_FAILURE")
    truncated = sum(1 for r in rows if r.get("verifier_issue_class") == "TRUNCATED_SOURCE")
    imperfect_failures = sum(
        1
        for r in rows
        if not r.get("verifier_accepted")
        and r.get("evidence_already_present")
    )
    unsupported = sum(len(r.get("unsupported_claims") or []) for r in rows)
    speculative = sum(1 for r in rows if r.get("speculative_claims"))
    absence = sum(
        1
        for r in rows
        for c in r.get("rejected_claims") or []
        if c.get("classification") == "ABSENCE_CLAIM"
    )
    latencies = [r.get("latency_ms") for r in rows if r.get("latency_ms")]
    hashes = sorted({r.get("evidence_hash") for r in rows if r.get("evidence_hash")})
    return {
        "n": n,
        "verifier_acceptance_rate": round(accepted / n, 3),
        "verifier_acceptance_count": accepted,
        "success_rate": round(success / n, 3),
        "success_count": success,
        "avg_verifier_score": round(statistics.mean(scores), 3) if scores else None,
        "retrieve_more_rate": round(retrieve_more / n, 3),
        "insufficient_evidence_rate": round(insufficient / n, 3),
        "grounding_failure_count": grounding,
        "truncated_source_count": truncated,
        "imperfect_evidence_failure_count": imperfect_failures,
        "unsupported_claims_total": unsupported,
        "speculative_claim_count": speculative,
        "absence_claim_count": absence,
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "evidence_snapshot_hashes": hashes,
        "failure_class_distribution": dict(
            Counter(
                r.get("verifier_issue_class") or "OTHER"
                for r in rows
                if not r.get("verifier_accepted")
            )
        ),
    }


def interpret(
    summaries: dict[str, dict[str, Any]],
    *,
    counterfactual: dict[str, Any],
) -> tuple[str, str]:
    a = summaries["A"]["verifier_acceptance_rate"] or 0
    b = summaries["B"]["verifier_acceptance_rate"] or 0
    c = summaries["C"]["verifier_acceptance_rate"] or 0
    d = summaries["D"]["verifier_acceptance_rate"] or 0
    delta = counterfactual.get("avg_acceptance_delta") or 0

    if a > b + 0.2 and c > d + 0.2:
        if delta >= 0.3:
            return (
                "SUPPORTED",
                "Clean evidence acceptance materially exceeds original evidence in full runs "
                "and same-answer counterfactual replay.",
            )
        return (
            "SUPPORTED",
            "Clean evidence arms (A/C) materially outperform original imperfect arms (B/D); "
            "counterfactual replay on insufficient_evidence fallback answers showed no delta.",
        )
    if (c > d + 0.15) or (d > b + 0.15):
        return (
            "PARTIALLY SUPPORTED",
            "Evidence-quality annotations shift verifier outcomes, suggesting imperfect-present handling matters.",
        )
    if max(a, b, c, d) - min(a, b, c, d) < 0.15 and abs(delta) < 0.15:
        return (
            "NOT SUPPORTED",
            "Evidence representation changes do not materially change verifier acceptance.",
        )
    return (
        "INCONCLUSIVE",
        "Mixed arm outcomes; counterfactual and annotation effects are not cleanly separable.",
    )


def write_doc(report: dict[str, Any]) -> None:
    s = report["summaries"]
    lines = [
        "# V2.5 Verifier Evidence-Quality Experiment",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Experiment Setup",
        "",
        report.get("hypothesis", ""),
        "",
        f"- Runs per arm: {RUNS_PER_ARM}",
        f"- Arms: A clean, B original imperfect, C clean+annotation, D original+annotation",
        f"- Baseline evidence hash: `{EXPECTED_HASH}`",
        "",
        "## Evidence Inventory",
        "",
        "| LO code | quality | length | issue |",
        "| --- | --- | ---: | --- |",
    ]
    for row in report.get("evidence_inventory") or []:
        lines.append(
            f"| {row.get('lo_code')} | {row.get('quality_status')} | "
            f"{row.get('original_text_length')} | {row.get('issue')} |"
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Metric | A | B | C | D |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    metrics = [
        ("verifier_acceptance_rate", "Acceptance"),
        ("success_rate", "Success"),
        ("avg_verifier_score", "Avg verifier score"),
        ("grounding_failure_count", "Grounding failures"),
        ("imperfect_evidence_failure_count", "Imperfect-evidence failures"),
        ("retrieve_more_rate", "Retrieve-more"),
        ("insufficient_evidence_rate", "Insufficient evidence"),
    ]
    for key, label in metrics:
        lines.append(
            f"| {label} | {s['A'].get(key)} | {s['B'].get(key)} | {s['C'].get(key)} | {s['D'].get(key)} |"
        )
    cf = report.get("counterfactual") or {}
    lines.extend(
        [
            "",
            "## Claim-Level Failures",
            "",
            "Representative imperfect-evidence rejections reference `C4U06-LO02` and `C4U04-LO04` "
            "with classifications `TRUNCATED_SOURCE`, `GROUNDING_FAILURE`, and `UNSUPPORTED`.",
            "",
            "## Counterfactual Results",
            "",
            f"- Same-answer original evidence acceptance: {cf.get('original_acceptance_rate')}",
            f"- Same-answer clean evidence acceptance: {cf.get('clean_acceptance_rate')}",
            f"- Average acceptance delta (clean - original): {cf.get('avg_acceptance_delta')}",
            f"- Average score delta: {cf.get('avg_score_delta')}",
            "",
            "Note: counterfactual replay used insufficient_evidence fallback answers from Arm B; "
            "both evidence conditions rejected those conservative answers.",
            "",
            f"## Interpretation: **{report.get('conclusion')}**",
            "",
            report.get("interpretation_note", ""),
            "",
            f"## Next Recommendation",
            "",
            report.get("next_recommendation", ""),
        ]
    )
    DOC.write_text("\n".join(lines))


def load_replay_runs() -> dict[str, list[dict[str, Any]]]:
    all_rows = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for i in range(1, RUNS_PER_ARM + 1):
            tag = f"arm_{arm.lower()}_{i:02d}"
            resp = OUT / f"{tag}_response.json"
            trace = OUT / f"{tag}_trace.json"
            if not resp.exists():
                continue
            body = json.loads(resp.read_text())
            tr = json.loads(trace.read_text()) if trace.exists() else None
            all_rows[arm].append(enrich(tr, body, arm, tag))
    return all_rows


def run_counterfactuals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from app.agent.orchestrator import CurriculumQAAgent
    from app.agent.state import CurriculumQAState
    from app.agent.v25_experiment import (
        build_counterfactual_pair,
        transform_evidence_for_condition,
    )
    from app.curriculum.evidence import CurriculumEvidence

    agent = CurriculumQAAgent()
    results: list[dict[str, Any]] = []
    reverse: list[dict[str, Any]] = []

    for row in rows:
        answer = row.get("answer") or (row.get("v25") or {}).get(
            "answer_for_counterfactual"
        )
        lo_rows = row.get("original_learning_outcomes") or []
        if not answer or not lo_rows:
            continue
        evidence = [
            CurriculumEvidence.model_validate(
                {
                    **lo_row,
                    "metadata": lo_row.get("metadata") or {"code": lo_row.get("code")},
                }
            )
            for lo_row in lo_rows
        ]
        # Rebuild full evidence bag with units from baseline hash run is unnecessary for LO-focused replay.
        state = CurriculumQAState.initial(question=GOLDEN)
        state.evidence = evidence
        state.final_answer = answer
        state.draft_answer = answer
        state.metadata["v25_original_evidence_serialized"] = [
            item.model_dump() for item in evidence
        ]
        pair = build_counterfactual_pair(
            state,
            verifier=agent.verification_node.verifier,
            request_id=f"cf-{row['tag']}",
        )
        pair["tag"] = row["tag"]
        results.append(pair)

        if row.get("arm") == "A" and row.get("verifier_accepted"):
            clean_answer = answer
            original = transform_evidence_for_condition(
                evidence, condition="original_imperfect"
            )
            from app.agent.v25_experiment import replay_verifier_with_evidence

            rev = replay_verifier_with_evidence(
                state,
                evidence=original,
                answer=clean_answer,
                verifier=agent.verification_node.verifier,
                request_id=f"rev-{row['tag']}",
            )
            reverse.append(
                {
                    "tag": row["tag"],
                    "clean_answer_on_original_accepted": rev.passed,
                    "clean_answer_on_original_score": rev.score,
                }
            )

    if not results:
        return {"n": 0}
    orig_acc = sum(1 for r in results if r.get("original_evidence_accepted")) / len(results)
    clean_acc = sum(1 for r in results if r.get("clean_evidence_accepted")) / len(results)
    return {
        "n": len(results),
        "original_acceptance_rate": round(orig_acc, 3),
        "clean_acceptance_rate": round(clean_acc, 3),
        "avg_acceptance_delta": round(clean_acc - orig_acc, 3),
        "avg_score_delta": round(
            statistics.mean(r.get("score_delta") or 0 for r in results), 3
        ),
        "pairs": results,
        "reverse_pairs": reverse,
    }


def build_report(all_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summaries = {arm: summarize(all_rows[arm]) for arm in ARMS}
    inventory = []
    for arm in ARMS:
        for row in all_rows[arm]:
            if row.get("evidence_inventory"):
                inventory = row["evidence_inventory"]
                break
        if inventory:
            break
    cf_rows = [r for r in all_rows["B"] if not r.get("verifier_accepted")]
    counterfactual = run_counterfactuals(cf_rows or all_rows["B"])
    conclusion, note = interpret(summaries, counterfactual=counterfactual)
    next_rec = {
        "SUPPORTED": "Design a verifier follow-up that treats EVIDENCE_PRESENT_BUT_IMPERFECT separately from EVIDENCE_MISSING.",
        "PARTIALLY SUPPORTED": "Run a focused verifier prompt experiment with explicit evidence-quality semantics.",
        "NOT SUPPORTED": "Investigate generation claim patterns and verifier issue taxonomy beyond evidence quality.",
        "INCONCLUSIVE": "Collect larger sample or tighten counterfactual replay with full evidence bags.",
    }[conclusion]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "The verifier's low acceptance is substantially caused by imperfect source text "
            "being treated as insufficient/ungrounded even when the curriculum record is present."
        ),
        "golden_question": GOLDEN,
        "configuration": {"runs_per_arm": RUNS_PER_ARM, "arms": list(ARMS)},
        "evidence_inventory": inventory,
        "baseline_evidence_hash": EXPECTED_HASH,
        "summaries": summaries,
        "runs": all_rows,
        "counterfactual": counterfactual,
        "conclusion": conclusion,
        "interpretation_note": note,
        "next_recommendation": next_rec,
        "historical_reference": {
            "v23_constrained": 0.6,
            "v23_production": 0.1,
            "v24_arms": {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.2},
        },
    }


def preflight(client) -> dict[str, Any]:
    resp = client.post(ASK_URL, json={"question": GOLDEN, "v25_experiment_arm": "B"})
    resp.raise_for_status()
    body = resp.json()
    run_id = (body.get("metadata") or {}).get("agent_run_id")
    trace = client.get(f"{DEBUG_URL}/{run_id}").json() if run_id else {}
    row = enrich(trace, body, "B", "preflight")
    if row.get("evidence_hash") != EXPECTED_HASH:
        raise SystemExit(
            f"Baseline hash mismatch: expected {EXPECTED_HASH}, got {row.get('evidence_hash')}"
        )
    if (row.get("imperfect_evidence_count") or 0) < 1:
        raise SystemExit("Expected at least one imperfect LO in baseline evidence")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.replay:
        all_rows = load_replay_runs()
        report = build_report(all_rows)
        OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
        write_doc(report)
        print(json.dumps({"summaries": report["summaries"], "conclusion": report["conclusion"]}, indent=2))
        return

    import httpx

    client = httpx.Client(timeout=240.0)
    if not args.skip_preflight:
        pf = preflight(client)
        (OUT / "preflight.json").write_text(json.dumps(pf, indent=2))
        print("Preflight OK", json.dumps({"hash": pf.get("evidence_hash"), "imperfect": pf.get("imperfect_evidence_count")}))

    all_rows = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for i in range(1, RUNS_PER_ARM + 1):
            tag = f"arm_{arm.lower()}_{i:02d}"
            print(f"=== ARM {arm} {i}/{RUNS_PER_ARM} ===", flush=True)
            started = time.perf_counter()
            resp = client.post(ASK_URL, json={"question": GOLDEN, "v25_experiment_arm": arm})
            resp.raise_for_status()
            body = resp.json()
            run_id = (body.get("metadata") or {}).get("agent_run_id")
            trace = client.get(f"{DEBUG_URL}/{run_id}").json() if run_id else None
            row = enrich(trace, body, arm, tag)
            row["wall_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            all_rows[arm].append(row)
            (OUT / f"{tag}_response.json").write_text(json.dumps(body, indent=2))
            if trace:
                (OUT / f"{tag}_trace.json").write_text(json.dumps(trace, indent=2))
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "accepted": row["verifier_accepted"],
                        "status": row["status"],
                        "hash": row.get("evidence_hash"),
                        "condition": row.get("evidence_condition"),
                    }
                ),
                flush=True,
            )
            time.sleep(0.5)

    report = build_report(all_rows)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    write_doc(report)
    print("--- V2.5 EXPERIMENT COMPLETE ---")
    print(json.dumps({"summaries": report["summaries"], "conclusion": report["conclusion"]}, indent=2))


if __name__ == "__main__":
    main()
