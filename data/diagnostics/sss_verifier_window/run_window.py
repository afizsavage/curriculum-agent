"""Verifier evidence-window experiment.

Uses the frozen 48-case corpus. The only change is the verifier-visible
record limit: every ranked theme and topic is included. Production code,
the prompt, temperature, ranking, and the stored answers are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.agent.answer_generator import AnswerGenerator, select_evidence_for_prompt
from app.agent.verifier import AnswerVerifier
from app.config import get_settings
from app.llm.provider import build_llm_provider

ROOT = Path(__file__).parent
REPEAT = ROOT.parent / "sss_verifier_repeat"
sys.path.insert(0, str(REPEAT))
import run_verifier_repeat as frozen  # noqa: E402

OUT = ROOT / "results.jsonl"
SUMMARY = ROOT / "summary.json"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return round(ordered[index], 1)


def _baseline_verdict(group: list[dict]) -> str:
    verdicts = [row["verifier_passed"] for row in group if row["verifier_passed"] is not None]
    if not verdicts:
        return "no_verdict"
    if True in verdicts and False in verdicts:
        return "flipped"
    return "accepted" if all(verdicts) else "rejected"


def _hash_records(records: list) -> str:
    payload = [
        {
            "entity_type": item.entity_type,
            "name": item.name,
            "grade": item.grade,
            "subject": item.subject,
            "topic": item.topic,
            "source_reference": item.source_reference,
            "stream_name": (item.metadata or {}).get("stream_name"),
        }
        for item in records
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _visible(evidence: list, question: str, limit: int) -> list:
    ranked, _ids = select_evidence_for_prompt(
        evidence, question=question, max_records=limit
    )
    return ranked


class FullTreeVerifier(AnswerVerifier):
    """Same verifier. The evidence limit is the full ranked list."""

    def build_messages(self, state):  # type: ignore[override]
        import app.agent.verifier as verifier_module
        from app.agent.answer_generator import format_evidence_for_prompt

        limit = max(len(state.evidence), 1)
        original = verifier_module.format_evidence_for_prompt
        verifier_module.format_evidence_for_prompt = (
            lambda evidence, question=None, max_records=24: format_evidence_for_prompt(
                evidence, question=question, max_records=limit
            )
        )
        try:
            return super().build_messages(state)
        finally:
            verifier_module.format_evidence_for_prompt = original


def _groups() -> list[dict]:
    rows = _load(frozen.OUT)
    by: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by[row["question"]].append(row)
    cases = frozen._select()
    grouped = []
    for index, case in enumerate(cases, start=1):
        group = by.get(case["question"]) or []
        if not group:
            raise SystemExit(f"missing frozen repeats for {case['question']}")
        first = sorted(group, key=lambda row: row["repeat"])[0]
        grouped.append(
            {
                "case_id": f"V{index:02d}",
                "case": case,
                "group": group,
                "first": first,
                "baseline_verdict": _baseline_verdict(group),
            }
        )
    if len(grouped) != 48:
        raise SystemExit(f"expected 48 frozen cases, found {len(grouped)}")
    return grouped


def main() -> None:
    settings = get_settings()
    llm = build_llm_provider(settings)
    generator = AnswerGenerator(llm)
    verifier = FullTreeVerifier(llm, settings=settings)
    cases = _groups()
    done = {row["case_id"] for row in _load(OUT)}
    print(
        f"WINDOW cases={len(cases)} already_done={len(done)} "
        f"model={llm.model} temperature=0.0 prompt={frozen.PROMPT_ID}",
        flush=True,
    )
    for item in cases:
        if item["case_id"] in done:
            continue
        case = item["case"]
        first = item["first"]
        evidence = frozen._evidence(case["syllabus"], case["stream"])
        evidence_hash = hashlib.sha256(
            json.dumps(
                [row.model_dump() for row in evidence],
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        if evidence_hash != first["evidence_hash"]:
            raise SystemExit(
                f"{item['case_id']} evidence hash drifted from the frozen corpus"
            )
        state = frozen._state(
            case["question"], case["grade"], case["stream"], case["subject"], evidence
        )
        rendered = generator.generate(state)
        answer = rendered.answer or ""
        answer_hash = hashlib.sha256(answer.encode()).hexdigest()
        if answer_hash != first["answer_hash"]:
            raise SystemExit(
                f"{item['case_id']} answer hash drifted from the frozen corpus"
            )
        state.final_answer = answer
        state.draft_answer = answer
        state.answer_evidence = list(rendered.evidence)
        window = _visible(evidence, case["question"], 24)
        expanded = _visible(evidence, case["question"], max(len(evidence), 1))
        window_keys = {(item.entity_type, item.name) for item in window}
        newly_exposed = [
            {"entity_type": row.entity_type, "name": row.name}
            for row in expanded
            if (row.entity_type, row.name) not in window_keys
        ]
        started = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        error = None
        result = None
        try:
            result = verifier.verify(state, request_id=f"window-{item['case_id']}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        if result is None:
            expanded_verdict = "no_verdict"
        elif result.passed:
            expanded_verdict = "accepted"
        else:
            expanded_verdict = "rejected"
        baseline = item["baseline_verdict"]
        changed = baseline in {"accepted", "rejected"} and expanded_verdict != baseline
        omitted = int(first["evidence_facts"]["prompt_omitted"])
        if baseline == "rejected" and omitted > 0 and expanded_verdict == "accepted":
            change_class = "truncated_recovered"
        elif baseline == "rejected" and omitted == 0 and expanded_verdict == "rejected":
            change_class = "complete_evidence_still_rejected"
        elif baseline == "rejected" and omitted > 0 and expanded_verdict == "rejected":
            change_class = "truncated_still_rejected"
        elif baseline == "rejected" and omitted == 0 and expanded_verdict == "accepted":
            change_class = "unexpected_acceptance"
        elif baseline == "accepted" and expanded_verdict == "rejected":
            change_class = "regression"
        elif baseline == "accepted" and expanded_verdict == "accepted":
            change_class = "accepted_unchanged"
        else:
            change_class = "other"
        record = {
            "captured_at": started.isoformat(),
            "case_id": item["case_id"],
            "role": first["role"],
            "grade": case["grade"],
            "stream": case["stream"],
            "subject": case["subject"],
            "question": case["question"],
            "source_document": first["source_document"],
            "baseline_verdict": baseline,
            "expanded_evidence_verdict": expanded_verdict,
            "baseline_evidence_count": len(window),
            "expanded_evidence_count": len(expanded),
            "baseline_evidence_hash": first["evidence_hash"],
            "expanded_evidence_hash": _hash_records(expanded),
            "answer_hash": answer_hash,
            "verifier_reason": list(result.issues) if result else [],
            "verifier_recommendation": result.recommendation.value if result else None,
            "verifier_latency_ms": latency_ms,
            "changed": changed,
            "change_class": change_class,
            "prompt_omitted_baseline": omitted,
            "newly_exposed": newly_exposed,
            "model": llm.model,
            "temperature": 0.0,
            "prompt_id": frozen.PROMPT_ID,
            "error": error,
        }
        with OUT.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"{item['case_id']} {change_class} {case['grade']} {case['subject'][:36]} "
            f"{latency_ms}ms",
            flush=True,
        )
    rows = _load(OUT)
    latencies = [row["verifier_latency_ms"] for row in rows]
    truncated = [row for row in rows if row["prompt_omitted_baseline"] > 0 and row["baseline_verdict"] == "rejected"]
    complete = [row for row in rows if row["prompt_omitted_baseline"] == 0 and row["baseline_verdict"] == "rejected"]
    summary = {
        "cases": len(rows),
        "model": settings.llm_model,
        "temperature": 0.0,
        "prompt_id": frozen.PROMPT_ID,
        "accepted": sum(1 for row in rows if row["expanded_evidence_verdict"] == "accepted"),
        "rejected": sum(1 for row in rows if row["expanded_evidence_verdict"] == "rejected"),
        "no_verdict": sum(1 for row in rows if row["expanded_evidence_verdict"] == "no_verdict"),
        "latency_p50": _percentile(latencies, 50),
        "latency_p95": _percentile(latencies, 95),
        "truncated_rejects": len(truncated),
        "truncated_recovered": sum(1 for row in truncated if row["change_class"] == "truncated_recovered"),
        "truncated_still_rejected": sum(1 for row in truncated if row["change_class"] == "truncated_still_rejected"),
        "complete_still_rejected": sum(1 for row in complete if row["change_class"] == "complete_evidence_still_rejected"),
        "unexpected_acceptance": sum(1 for row in rows if row["change_class"] == "unexpected_acceptance"),
        "regression": sum(1 for row in rows if row["change_class"] == "regression"),
        "classes": dict(Counter(row["change_class"] for row in rows)),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print("WINDOW_DONE", flush=True)


if __name__ == "__main__":
    main()
