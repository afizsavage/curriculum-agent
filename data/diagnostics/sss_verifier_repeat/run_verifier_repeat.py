"""Repeat the live verifier on frozen SSS evidence. No retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.agent.answer_generator import (
    AnswerGenerator,
    format_evidence_for_prompt,
    select_evidence_for_prompt,
)
from app.agent.state import CurriculumQAState
from app.agent.verifier import VERIFIER_SYSTEM_PROMPT, AnswerVerifier
from app.config import get_settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.llm.provider import build_llm_provider

ROOT = Path(__file__).parent
DIAG = ROOT.parent / "sss_grade_stream_experiment"
MANIFEST = DIAG / "manifest.json"
OUT = ROOT / "results.jsonl"
SUMMARY = ROOT / "summary.json"
REPEATS = 5
PROMPT_ID = hashlib.sha256(VERIFIER_SYSTEM_PROMPT.encode()).hexdigest()[:16]


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


def _evidence(syllabus: dict, stream: str) -> list[CurriculumEvidence]:
    grade = syllabus["grade"]
    subject = syllabus["subject_name"]
    source = syllabus.get("source_reference")
    rows = [
        CurriculumEvidence(
            entity_type="sss_stream",
            name=stream,
            level="senior_secondary",
            metadata={"stream_name": stream},
        ),
        CurriculumEvidence(
            entity_type="subject",
            name=subject,
            grade=grade,
            subject=subject,
            level="senior_secondary",
            metadata={"stream_name": stream},
        ),
    ]
    for item in syllabus.get("content") or []:
        kind = str(item.get("type") or "").lower()
        name = item.get("name")
        if kind not in {"theme", "topic", "learning_outcome"} or not name:
            continue
        parent = item.get("parent")
        rows.append(
            CurriculumEvidence(
                entity_type=kind,
                name=str(name),
                grade=grade,
                subject=subject,
                level="senior_secondary",
                topic=None if kind == "theme" else (str(parent) if parent else None),
                content=str(name),
                metadata={
                    "source_type": "grade_curriculum_content",
                    "content_type": kind.upper(),
                    "stream_name": stream,
                    "parent_name": parent,
                },
                source_reference=source,
            )
        )
    return rows


def _state(question: str, grade: str, stream: str, subject: str, evidence: list) -> CurriculumQAState:
    return CurriculumQAState(
        question=question,
        intent="SSS_STREAM_SUBJECTS",
        level="senior_secondary",
        grade=grade,
        subject=None,
        evidence=evidence,
        evidence_status=EvidenceStatus.FOUND,
        metadata={
            "sss_stream_resolution": "found",
            "resolved_stream_name": stream,
            "subject_name": subject,
            "grade": grade,
            "sss_focus": "coverage",
            "focus": "coverage",
            "stream_name": stream,
        },
    )


def _classify(issues: list[str], facts: dict, answer: str) -> list[str]:
    text = " ".join(issues).casefold()
    answer_cf = answer.casefold()
    tags = []
    names = facts["content_names"]
    cited = [
        name
        for name in names
        if name.casefold() in text
    ]
    if any(word in text for word in ("learning outcome", "learning-outcome", "objectives")) and facts["outcome_count"] == 0:
        tags.append("expected_learning_outcomes_absent_from_syllabus")
    if cited and any(name.casefold() not in answer_cf for name in cited) and any(
        name in facts["prompt_names"] for name in cited
    ):
        tags.append("answer_wording_omits_evidence")
    if cited and any(name not in facts["prompt_names"] for name in cited):
        tags.append("evidence_format_or_context")
    if cited and all(name.casefold() in answer_cf for name in cited) and all(
        name in facts["prompt_names"] for name in cited
    ):
        tags.append("evidence_present_but_not_recognized")
    if any(word in text for word in ("wrong grade", "different grade", "sss 2", "sss 3", "sss 1")) and facts["grades_match"]:
        if facts["requested_grade"].replace("_", " ").casefold() not in text:
            tags.append("grade_or_stream_metadata")
    theme_as_topic = False
    for name, kind in facts["name_kinds"].items():
        if name.casefold() not in text:
            continue
        if kind == "topic" and "theme" in text and "topic" not in text:
            theme_as_topic = True
        if kind == "theme" and "topic" in text and "theme" not in text:
            theme_as_topic = True
    if theme_as_topic:
        tags.append("theme_topic_confusion")
    if not tags and any(word in text for word in ("unsupported", "not supported", "hallucin", "not in the evidence", "does not contain")):
        if not cited:
            tags.append("genuinely_unsupported_claim")
    if not tags:
        tags.append("other")
    return tags


def _facts(evidence: list[CurriculumEvidence], question: str, grade: str, stream: str) -> dict:
    content = [
        item
        for item in evidence
        if (item.entity_type or "").lower() in {"theme", "topic", "learning_outcome"}
    ]
    prompt_rows, _ids = select_evidence_for_prompt(evidence, question=question, max_records=24)
    return {
        "evidence_count": len(content),
        "prompt_record_count": len(prompt_rows),
        "prompt_omitted": max(0, len(evidence) - len(prompt_rows)),
        "theme_count": sum(1 for item in content if item.entity_type == "theme"),
        "topic_count": sum(1 for item in content if item.entity_type == "topic"),
        "outcome_count": sum(1 for item in content if item.entity_type == "learning_outcome"),
        "content_names": [item.name for item in content if item.name],
        "prompt_names": [item.name for item in prompt_rows if item.name],
        "name_kinds": {item.name: item.entity_type for item in content if item.name},
        "grades_match": all((item.grade in {None, grade}) for item in content),
        "streams_match": all(
            (item.metadata or {}).get("stream_name") in {None, stream} for item in content
        ),
        "requested_grade": grade,
        "requested_stream": stream,
    }


def _select() -> list[dict]:
    import sys

    sys.path.insert(0, str(DIAG))
    from replay_identity_retry import rescore, _themes

    manifest = json.loads(MANIFEST.read_text())
    themes = _themes(manifest)
    syllabi = {
        (row["subject_name"], row["grade"]): row
        for row in manifest["syllabi"]
    }
    rows_a = _load(DIAG / "results_A.jsonl")
    rows_b = _load(DIAG / "results_B.jsonl")
    rows_r = _load(DIAG / "results_identity_retry.jsonl")
    official = [
        row
        for row in rows_a
        if row.get("kind") == "valid" and row.get("formulation") == "official"
    ]
    for row in official:
        row["rescore"] = rescore(row, themes)
    by_q = {row["question"]: row for row in official}
    b_by_q = {}
    for row in rows_b:
        if row.get("kind") == "valid":
            row["rescore"] = rescore(row, themes)
            b_by_q[row["question"]] = row

    selected: list[dict] = []
    seen: set[str] = set()

    def add(row: dict, role: str) -> None:
        if row["question"] in seen:
            return
        syllabus = syllabi.get((row["subject"], row["grade"]))
        if not syllabus or not syllabus.get("content"):
            return
        seen.add(row["question"])
        selected.append({**row, "role": role, "syllabus": syllabus})

    for row in official:
        if row["rescore"] == "fail_verification":
            add(row, "official_verifier_failure")
    for row in rows_r:
        if row.get("decision") == "fail_verification" and row.get("cohort") != "control":
            base = by_q.get(row["question"])
            if base:
                add(base, "retry_exposed_verifier_failure")
    for question, brow in b_by_q.items():
        if brow.get("rescore") != "fail_verification":
            continue
        base = by_q.get(question)
        if base is None:
            matches = [
                row
                for row in official
                if row["subject"] == brow["subject"]
                and row["grade"] == brow["grade"]
                and row["stream"] == brow["stream"]
                and row["rescore"] == "grounded"
            ]
            base = matches[0] if matches else None
            if base is not None:
                base = {**base, "question": brow["question"]}
        if base and base.get("rescore", "grounded") == "grounded":
            add(base, "run_b_regression")

    controls = [
        row
        for row in official
        if row["rescore"] == "grounded" and row["question"] not in seen
    ]
    for grade in ("SSS_1", "SSS_2", "SSS_3"):
        for row in controls:
            if row["subject"] == "Biology" and row["grade"] == grade:
                add(row, "control")
    streams = [
        "Sciences & Technologies",
        "Mathematics & Numeracy",
        "Languages & Literatures",
        "Social & Cultural Studies",
        "Economics, Business & Entrepreneurship",
    ]
    for stream in streams:
        added = 0
        for row in controls:
            if row["stream"] == stream and row["question"] not in seen:
                add(row, "control")
                added += 1
            if added == 2:
                break
    return selected


def main() -> None:
    settings = get_settings()
    llm = build_llm_provider(settings)
    generator = AnswerGenerator(llm)
    verifier = AnswerVerifier(llm, settings=settings)
    cases = _select()
    done = Counter()
    if OUT.exists():
        for row in _load(OUT):
            done[row["question"]] += 1
    print(
        f"VERIFIER cases={len(cases)} model={llm.model} temperature=0.0 prompt={PROMPT_ID}",
        flush=True,
    )
    for index, case in enumerate(cases, start=1):
        evidence = _evidence(case["syllabus"], case["stream"])
        state = _state(case["question"], case["grade"], case["stream"], case["subject"], evidence)
        rendered = generator.generate(state)
        answer = rendered.answer or ""
        state.final_answer = answer
        state.draft_answer = answer
        state.answer_evidence = list(rendered.evidence)
        facts = _facts(evidence, case["question"], case["grade"], case["stream"])
        evidence_hash = hashlib.sha256(
            json.dumps(
                [item.model_dump() for item in evidence],
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        answer_hash = hashlib.sha256(answer.encode()).hexdigest()
        already = done[case["question"]]
        for repeat in range(already + 1, REPEATS + 1):
            started = datetime.now(timezone.utc)
            import time

            t0 = time.perf_counter()
            error = None
            result = None
            try:
                result = verifier.verify(state, request_id=f"verifier-{index}-{repeat}")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            issues = list(result.issues) if result else []
            missing = []
            if result:
                for item in result.missing_evidence:
                    missing.append(item if isinstance(item, str) else item.model_dump())
            record = {
                "captured_at": started.isoformat(),
                "repeat": repeat,
                "role": case["role"],
                "question": case["question"],
                "requested_grade": case["grade"],
                "requested_stream": case["stream"],
                "requested_subject": case["subject"],
                "resolved_grade": case["grade"],
                "resolved_stream": case["stream"],
                "resolved_subject": case["subject"],
                "source_document": case["syllabus"].get("source_reference"),
                "source_grade": case["grade"],
                "source_subject": case["subject"],
                "source_stream": case["stream"],
                "model": llm.model,
                "temperature": 0.0,
                "prompt_id": PROMPT_ID,
                "evidence_hash": evidence_hash,
                "answer_hash": answer_hash,
                "verifier_passed": bool(result.passed) if result else None,
                "verifier_score": result.score if result else None,
                "verifier_recommendation": result.recommendation.value if result else None,
                "verifier_issues": issues,
                "verifier_missing_evidence": missing,
                "verifier_latency_ms": latency_ms,
                "error": error,
                "answer": answer,
                "evidence_facts": {
                    key: facts[key]
                    for key in (
                        "evidence_count",
                        "prompt_record_count",
                        "prompt_omitted",
                        "theme_count",
                        "topic_count",
                        "outcome_count",
                    )
                },
                "categories": _classify(issues, facts, answer) if result and not result.passed else [],
                "prompt_excerpt_records": facts["prompt_record_count"],
            }
            # Drop the large name lists from the stored facts copy already sliced.
            with OUT.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"{index}/{len(cases)} r{repeat} {case['role']} "
                f"passed={record['verifier_passed']} {case['grade']} {case['subject'][:32]} "
                f"{latency_ms}ms",
                flush=True,
            )
    rows = _load(OUT)
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        by_case.setdefault(row["question"], []).append(row)
    flips = 0
    accepted_cases = 0
    rejected_cases = 0
    mixed_cases = 0
    categories = Counter()
    latencies = []
    for group in by_case.values():
        verdicts = [row["verifier_passed"] for row in group if row["verifier_passed"] is not None]
        if not verdicts:
            continue
        latencies.extend(row["verifier_latency_ms"] for row in group)
        if any(row["verifier_passed"] is True for row in group) and any(
            row["verifier_passed"] is False for row in group
        ):
            flips += 1
            mixed_cases += 1
        elif all(verdicts):
            accepted_cases += 1
        else:
            rejected_cases += 1
        for row in group:
            if row["verifier_passed"] is False and row["repeat"] == 1:
                for tag in row["categories"]:
                    categories[tag] += 1
    summary = {
        "cases": len(by_case),
        "evaluations": len(rows),
        "repeats": REPEATS,
        "model": settings.llm_model,
        "temperature": 0.0,
        "prompt_id": PROMPT_ID,
        "roles": dict(Counter(row["role"] for row in rows if row["repeat"] == 1)),
        "cases_always_accepted": accepted_cases,
        "cases_always_rejected": rejected_cases,
        "cases_flipped": flips,
        "flip_rate": round(flips / len(by_case), 3) if by_case else None,
        "acceptance_rate": round(
            sum(1 for row in rows if row["verifier_passed"] is True) / len(rows), 3
        )
        if rows
        else None,
        "rejection_rate": round(
            sum(1 for row in rows if row["verifier_passed"] is False) / len(rows), 3
        )
        if rows
        else None,
        "categories_first_rejection": dict(categories),
        "latency_p50": _percentile(latencies, 50),
        "latency_p95": _percentile(latencies, 95),
        "outcome_absent_rejections": sum(
            1
            for row in rows
            if row["repeat"] == 1
            and row["verifier_passed"] is False
            and row["evidence_facts"]["outcome_count"] == 0
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print("VERIFIER_DONE", flush=True)


if __name__ == "__main__":
    main()
