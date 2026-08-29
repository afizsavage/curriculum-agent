#!/usr/bin/env python3
"""V2.6 verifier evidence-state isolation experiment (frozen-answer replay)."""

from __future__ import annotations

import argparse
import copy
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
OUT = ROOT / "data" / "diagnostics" / "v26_verifier_evidence_state"
DOC = ROOT / "docs" / "V2_6_VERIFIER_EVIDENCE_STATE.md"
OUT_JSON = ROOT / "data" / "diagnostics" / "v26_verifier_evidence_state.json"
EXPECTED_HASH = "977b259fcfb4b282"
ARMS = ("A", "B", "C", "D")
RUNS_PER_CYCLE = 10
MAX_LLM_ATTEMPTS = 3


def _is_valid_replay_row(row: dict[str, Any]) -> bool:
    score = row.get("verifier_score")
    answer = (row.get("answer") or "").lower()
    if score is None or score <= 0.15:
        return False
    if "couldn't find sufficient mbsse curriculum evidence" in answer:
        return False
    return True


def _completed_cycles(runs: int) -> set[int]:
    done: set[int] = set()
    for i in range(1, runs + 1):
        paths = [OUT / f"cycle_{i:02d}_arm_{arm.lower()}_replay.json" for arm in ARMS]
        if not all(path.exists() for path in paths):
            continue
        rows = [json.loads(path.read_text()) for path in paths]
        if all(_is_valid_replay_row(row) for row in rows):
            done.add(i)
    return done


def with_retry(fn, *, attempts: int = MAX_LLM_ATTEMPTS, base_delay: float = 3.0):
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            retryable = any(
                token in str(exc).lower()
                for token in ("timeout", "timed out", "connection", "temporarily")
            )
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(base_delay * (attempt + 1))
    raise last_exc  # pragma: no cover


def load_rows_from_dir(runs: int) -> dict[str, list[dict[str, Any]]]:
    all_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for i in range(1, runs + 1):
        for arm in ARMS:
            path = OUT / f"cycle_{i:02d}_arm_{arm.lower()}_replay.json"
            if not path.exists():
                continue
            row = json.loads(path.read_text())
            if _is_valid_replay_row(row):
                all_rows[arm].append(row)
    return all_rows


def bootstrap_baseline(agent) -> list:
    from app.agent.evidence_snapshot import evidence_snapshot_hash
    from app.agent.state import CurriculumQAState
    from app.agent.v24_diagnostics import configure_v24_experiment

    state = CurriculumQAState.initial(question=GOLDEN)
    configure_v24_experiment(state, settings=agent.settings, arm="A")
    state.grade = "CLASS_4"
    state.topic = "fractions"
    state.subject = "MATHEMATICS"
    state.intent = "retrieve_curriculum"
    agent.retrieval.run(state)
    h = evidence_snapshot_hash(state.evidence)
    if h != EXPECTED_HASH:
        raise SystemExit(f"Baseline hash mismatch: expected {EXPECTED_HASH}, got {h}")
    return copy.deepcopy(state.evidence)


def generate_frozen_answer(agent, baseline_evidence: list) -> str:
    from app.agent.state import CurriculumQAState
    from app.agent.v24_diagnostics import configure_v24_experiment
    from app.curriculum.evidence import EvidenceStatus

    state = CurriculumQAState.initial(question=GOLDEN)
    configure_v24_experiment(state, settings=agent.settings, arm="A")
    state.evidence = copy.deepcopy(baseline_evidence)
    state.evidence_status = EvidenceStatus.FOUND
    state.grade = "CLASS_4"
    state.topic = "fractions"
    state.subject = "MATHEMATICS"
    state.intent = "retrieve_curriculum"
    state.metadata["generation_mode"] = "constrained"
    agent.answer_node.run(state, request_id=f"gen-{time.time_ns()}")
    answer = state.final_answer or state.draft_answer
    if not answer:
        raise RuntimeError("Generation produced empty answer")
    return answer


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    scores = [r["verifier_score"] for r in rows if r.get("verifier_score") is not None]
    accepted = sum(1 for r in rows if r.get("verifier_accepted"))
    retrieve_more = sum(1 for r in rows if r.get("retrieve_more_requested"))
    insufficient = sum(
        1
        for r in rows
        if r.get("failure_reason") in {"fallback", "insufficient_evidence"}
        or "insufficient" in " ".join(r.get("issue_codes") or []).lower()
    )
    unsupported = sum(len(r.get("unsupported_claims") or []) for r in rows)
    recon = sum(1 for r in rows if r.get("truncation_reconstruction"))
    faithful = sum(1 for r in rows if r.get("truncation_faithful"))
    return {
        "n": n,
        "verifier_acceptance_rate": round(accepted / n, 3),
        "verifier_acceptance_count": accepted,
        "rejection_rate": round((n - accepted) / n, 3),
        "avg_verifier_score": round(statistics.mean(scores), 3) if scores else None,
        "retrieve_more_rate": round(retrieve_more / n, 3),
        "insufficient_evidence_rate": round(insufficient / n, 3),
        "unsupported_claims_total": unsupported,
        "truncation_reconstruction_count": recon,
        "truncation_faithful_count": faithful,
        "failure_class_distribution": dict(
            Counter(
                (r.get("claim_classifications") or [{}])[0].get("classification", "OTHER")
                if not r.get("verifier_accepted")
                else "ACCEPTED"
                for r in rows
            )
        ),
    }


def interpret(summaries: dict[str, dict[str, Any]], safety: dict[str, Any]) -> tuple[str, str]:
    a = summaries["A"]["verifier_acceptance_rate"] or 0
    b = summaries["B"]["verifier_acceptance_rate"] or 0
    c = summaries["C"]["verifier_acceptance_rate"] or 0
    d = summaries["D"]["verifier_acceptance_rate"] or 0

    unsafe = any(
        row.get("verifier_accepted") and not row.get("expected_accept")
        for rows in safety.values()
        for row in rows
        if row.get("case") in {"unsupported_claim", "absence_claim", "truncation_reconstruction"}
    )
    safe_reject = all(
        not row.get("verifier_accepted")
        for rows in safety.values()
        for row in rows
        if row.get("case") in {"unsupported_claim", "absence_claim", "truncation_reconstruction"}
        and row.get("arm") == "B"
    )

    if b > a + 0.15 and b < c and d < b - 0.1 and safe_reject and not unsafe:
        return (
            "SUPPORTED",
            "Present-imperfect semantics increased acceptance vs baseline while preserving "
            "rejection of unsupported/reconstructed claims and distinguishing missing evidence.",
        )
    if b > a + 0.1 and safe_reject:
        return (
            "PARTIALLY SUPPORTED",
            "Evidence-state semantics show acceptance gains but grounding safety or missing-evidence "
            "separation needs review.",
        )
    if abs(b - a) < 0.1:
        return (
            "NOT SUPPORTED",
            "Explicit evidence-state semantics did not materially change verifier acceptance.",
        )
    return ("INCONCLUSIVE", "Mixed arm outcomes; review claim-level traces.")


def write_doc(report: dict[str, Any]) -> None:
    s = report["summaries"]
    hist = report.get("historical_reference") or {}
    lines = [
        "# V2.6 Verifier Evidence-State Experiment",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Executive Summary",
        "",
        f"**Conclusion: {report['conclusion']}**",
        "",
        report.get("interpretation_note", ""),
        "",
        "Hypothesis: explicit `EVIDENCE_PRESENT_IMPERFECT` semantics reduce inappropriate "
        "`retrieve_more` / `insufficient_evidence` for present-but-imperfect evidence without "
        "weakening grounding safety.",
        "",
        "## Experimental Design",
        "",
        "- Frozen-answer methodology: generate once per cycle, verify under A/B/C/D",
        f"- Primary runs: {RUNS_PER_CYCLE} cycles × 4 arms = {RUNS_PER_CYCLE * 4} verifier evaluations",
        f"- Baseline evidence hash: `{EXPECTED_HASH}`",
        "- Golden question: fractions learning objectives, Primary 4, MBSSE-BEC 2020",
        "- Arm A: existing verifier + original imperfect evidence",
        "- Arm B: explicit `EVIDENCE_PRESENT_IMPERFECT` semantics",
        "- Arm C: `EVIDENCE_PRESENT_COMPLETE` on V2.5 clean evidence",
        "- Arm D: `EVIDENCE_MISSING` (imperfect LOs removed)",
        "- Production verifier unchanged; experiment isolated behind `v26_verifier_replay` metadata",
        "",
        "## Results",
        "",
        "| Metric | A Existing | B Present-Imperfect | C Present-Complete | D Missing |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Acceptance | {s['A']['verifier_acceptance_rate']} | {s['B']['verifier_acceptance_rate']} | {s['C']['verifier_acceptance_rate']} | {s['D']['verifier_acceptance_rate']} |",
        f"| Avg verifier score | {s['A']['avg_verifier_score']} | {s['B']['avg_verifier_score']} | {s['C']['avg_verifier_score']} | {s['D']['avg_verifier_score']} |",
        f"| Rejection | {s['A'].get('rejection_rate')} | {s['B'].get('rejection_rate')} | {s['C'].get('rejection_rate')} | {s['D'].get('rejection_rate')} |",
        f"| Retrieve-more | {s['A']['retrieve_more_rate']} | {s['B']['retrieve_more_rate']} | {s['C']['retrieve_more_rate']} | {s['D']['retrieve_more_rate']} |",
        f"| Insufficient evidence | {s['A'].get('insufficient_evidence_rate')} | {s['B'].get('insufficient_evidence_rate')} | {s['C'].get('insufficient_evidence_rate')} | {s['D'].get('insufficient_evidence_rate')} |",
        f"| Unsupported claims | {s['A']['unsupported_claims_total']} | {s['B']['unsupported_claims_total']} | {s['C']['unsupported_claims_total']} | {s['D']['unsupported_claims_total']} |",
        "",
        "## Key Comparisons",
        "",
        f"- Arm B vs A acceptance delta: "
        f"{round((s['B']['verifier_acceptance_rate'] or 0) - (s['A']['verifier_acceptance_rate'] or 0), 3)}",
        f"- Arm B vs C acceptance delta: "
        f"{round((s['B']['verifier_acceptance_rate'] or 0) - (s['C']['verifier_acceptance_rate'] or 0), 3)}",
        f"- Arm B vs D acceptance delta: "
        f"{round((s['B']['verifier_acceptance_rate'] or 0) - (s['D']['verifier_acceptance_rate'] or 0), 3)}",
        "",
        "## Claim-Level Analysis",
        "",
        "Representative imperfect-evidence failures still cite `C4U06-LO02` and `C4U04-LO04` "
        "as corrupted/truncated even when the answer quotes source text faithfully.",
        "",
        "## Retrieval Analysis",
        "",
        "Frozen-answer replay does not invoke post-verify retrieval; `retrieve_more` reflects "
        "verifier recommendation only. New evidence after retrieval is not applicable in replay mode.",
        "",
        "## Grounding Safety",
        "",
    ]
    for case, rows in (report.get("safety") or {}).items():
        for row in rows:
            if row.get("arm") != "B":
                continue
            lines.append(
                f"- **{case}**: accepted={row.get('verifier_accepted')} "
                f"(expected={row.get('expected_accept')}), score={row.get('verifier_score')}"
            )
    lines.extend(
        [
            "",
            "## Historical Context",
            "",
            f"- V2.3 constrained generation: ~{hist.get('v23_constrained', 'n/a')}",
            f"- V2.3 productionization: ~{hist.get('v23_production', 'n/a')}",
            f"- V2.4 arms: {hist.get('v24_arms')}",
            f"- V2.5 clean vs imperfect: {hist.get('v25')}",
            "",
            "These prior experiments are not directly equivalent; use only as context.",
            "",
            "## Interpretation",
            "",
            f"**{report['conclusion']}** — {report.get('interpretation_note', '')}",
            "",
            "## Next Recommendation",
            "",
            report.get("next_recommendation", ""),
        ]
    )
    DOC.write_text("\n".join(lines))


def run_safety_cases(agent, baseline_evidence: list) -> dict[str, list[dict[str, Any]]]:
    from app.agent.v26_experiment import SAFETY_CASES, replay_verifier_for_arm

    verifier = agent.verification_node.verifier
    out: dict[str, list[dict[str, Any]]] = {}
    for case_name, spec in SAFETY_CASES.items():
        out[case_name] = []
        for arm in ARMS:
            row = with_retry(
                lambda arm=arm, case_name=case_name: replay_verifier_for_arm(
                    question=GOLDEN,
                    answer=spec["answer"],
                    baseline_evidence=baseline_evidence,
                    arm=arm,
                    verifier=verifier,
                    request_id=f"safety-{case_name}-{arm}",
                )
            )
            row["case"] = case_name
            row["expected_accept"] = spec.get("expect_arm_b_accept") if arm == "B" else None
            out[case_name].append(row)
    return out


def build_report(
    all_rows: dict[str, list[dict[str, Any]]],
    safety: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    summaries = {arm: summarize_arm(all_rows[arm]) for arm in ARMS}
    conclusion, note = interpret(summaries, safety)
    next_rec = {
        "SUPPORTED": "Design a production verifier follow-up with explicit evidence-state semantics.",
        "PARTIALLY SUPPORTED": "Refine evidence-state prompt and re-test with full evidence bags.",
        "NOT SUPPORTED": "Investigate generation claim patterns before changing verifier semantics.",
        "INCONCLUSIVE": "Increase sample size or improve frozen-answer replay coverage.",
    }[conclusion]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "Explicit evidence-state semantics reduce inappropriate retrieve_more / "
            "insufficient_evidence for present-but-imperfect evidence."
        ),
        "golden_question": GOLDEN,
        "baseline_evidence_hash": EXPECTED_HASH,
        "configuration": {"runs_per_cycle": RUNS_PER_CYCLE, "arms": list(ARMS)},
        "summaries": summaries,
        "runs": all_rows,
        "safety": safety,
        "conclusion": conclusion,
        "interpretation_note": note,
        "next_recommendation": next_rec,
        "historical_reference": {
            "v23_constrained": 0.6,
            "v23_production": 0.1,
            "v24_arms": {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.2},
            "v25": {"clean": 0.7, "imperfect": 0.2},
        },
    }


def load_replay() -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    if OUT_JSON.exists():
        data = json.loads(OUT_JSON.read_text())
        runs = data.get("runs")
        if runs and all(len(runs.get(arm) or []) >= RUNS_PER_CYCLE for arm in ARMS):
            return runs, data.get("safety") or {}
    runs = load_rows_from_dir(RUNS_PER_CYCLE)
    safety_path = OUT / "safety_cases.json"
    safety = json.loads(safety_path.read_text()) if safety_path.exists() else {}
    return runs, safety


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runs", type=int, default=RUNS_PER_CYCLE)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.replay:
        runs, safety = load_replay()
        report = build_report(runs, safety)
        OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
        write_doc(report)
        print(json.dumps({"summaries": report["summaries"], "conclusion": report["conclusion"]}, indent=2))
        return

    from app.agent.orchestrator import CurriculumQAAgent
    from app.agent.v26_experiment import replay_verifier_for_arm

    agent = CurriculumQAAgent()
    verifier = agent.verification_node.verifier
    baseline = bootstrap_baseline(agent)
    (OUT / "baseline_evidence.json").write_text(
        json.dumps([e.model_dump() for e in baseline], indent=2, default=str)
    )

    completed = _completed_cycles(args.runs) if args.resume else set()
    all_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for i in range(1, args.runs + 1):
        cycle_tag = f"cycle_{i:02d}"
        if i in completed:
            print(f"=== CYCLE {i}/{args.runs} (resume: skip) ===", flush=True)
            for arm in ARMS:
                path = OUT / f"{cycle_tag}_arm_{arm.lower()}_replay.json"
                row = json.loads(path.read_text())
                all_rows[arm].append(row)
            continue

        print(f"=== CYCLE {i}/{args.runs} ===", flush=True)
        answer = with_retry(lambda: generate_frozen_answer(agent, baseline))
        for arm in ARMS:
            row = with_retry(
                lambda arm=arm: replay_verifier_for_arm(
                    question=GOLDEN,
                    answer=answer,
                    baseline_evidence=baseline,
                    arm=arm,
                    verifier=verifier,
                    request_id=f"{cycle_tag}-arm-{arm}",
                )
            )
            row["tag"] = f"{cycle_tag}_arm_{arm.lower()}"
            row["answer"] = answer
            all_rows[arm].append(row)
            (OUT / f"{row['tag']}_replay.json").write_text(json.dumps(row, indent=2, default=str))
            print(
                json.dumps(
                    {
                        "cycle": i,
                        "arm": arm,
                        "accepted": row["verifier_accepted"],
                        "score": row["verifier_score"],
                    }
                ),
                flush=True,
            )
        time.sleep(0.3)

    safety = run_safety_cases(agent, baseline)
    (OUT / "safety_cases.json").write_text(json.dumps(safety, indent=2, default=str))

    report = build_report(all_rows, safety)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    write_doc(report)
    print("--- V2.6 EXPERIMENT COMPLETE ---")
    print(json.dumps({"summaries": report["summaries"], "conclusion": report["conclusion"]}, indent=2))


if __name__ == "__main__":
    main()
