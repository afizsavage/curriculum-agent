#!/usr/bin/env python3
"""V2.13C controlled hybrid retrieval + curriculum QA evaluation."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "diagnostics" / "v213c_curriculum_qa"
DOC = ROOT / "docs" / "V2_13C_CURRICULUM_QA_EVALUATION.md"
OUT_JSON = ROOT / "data" / "diagnostics" / "v213c_curriculum_qa.json"
DATASET = ROOT / "data" / "evals" / "v213c_curriculum_qa.json"


def write_dataset() -> list:
    from app.agent.v213c_dataset import DATASET_VERSION, build_v213c_dataset
    from app.agent.v213c_experiment import dataset_hash

    items = build_v213c_dataset()
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_version": DATASET_VERSION,
        "dataset_hash": dataset_hash(items),
        "questions": items,
    }
    DATASET.write_text(json.dumps(payload, indent=2))
    return items


def write_doc(report: dict) -> None:
    aq = report.get("answer_quality", {})
    ret = report.get("retrieval_metrics", {})
    lat = report.get("latency_metrics", {})
    newly = report.get("newly_answerable", {})
    cats = report.get("category_breakdown", {})
    lines = [
        "# V2.13C — Controlled Hybrid Retrieval + Real Curriculum QA Evaluation",
        "",
        f"Generated: `{report.get('generated_at')}`",
        "",
        f"**Conclusion: {report['conclusion']}**",
        "",
        report.get("interpretation_note", ""),
        "",
        "## Executive Summary",
        "",
        "Hypothesis: adding grounded MBSSE document evidence via V2.13B context-hybrid retrieval "
        "increases correctly answerable narrative curriculum questions without increasing unsupported "
        "or wrong-context answers.",
        "",
        f"- Dataset: **{report.get('dataset_size')}** questions (`{report.get('dataset_version')}`)",
        f"- Dataset hash: `{report.get('dataset_hash')}`",
        f"- Corpus: V2.13A fixtures, hashes `{json.dumps(report.get('document_hashes', {}))}`",
        "- Control: structured Curriculum API evidence only (frozen catalog)",
        "- Experiment: structured + V2.13B `context_hybrid` document retrieval",
        f"- Newly answerable: **{newly.get('count')}** ({newly.get('rate', 0):.3f}); "
        f"document-only: **{newly.get('document_only_count')}**",
        "",
        "## Dataset Breakdown",
        "",
        "| Category | Count |",
        "|----------|------:|",
    ]
    for key, count in sorted(cats.items()):
        lines.append(f"| {key} | {count} |")
    lines.extend(
        [
            "",
            "## Retrieval Results",
            "",
            "```json",
            json.dumps(ret, indent=2),
            "```",
            "",
            "## Answer Quality",
            "",
            "| Metric | Control | Experiment |",
            "|--------|--------:|-----------:|",
            f"| Grounded correct | {aq.get('control_grounded_correct', 0):.3f} | {aq.get('experiment_grounded_correct', 0):.3f} |",
            f"| Verifier/mapper accept | {aq.get('control_final_accepted', 0):.3f} | {aq.get('experiment_final_accepted', 0):.3f} |",
            f"| Structured-fact delta | | {aq.get('structured_fact_delta', 0):.3f} |",
            "",
            f"Paired: improved {report.get('paired', {}).get('improved')}, "
            f"unchanged {report.get('paired', {}).get('unchanged')}, "
            f"regressed {report.get('paired', {}).get('regressed')}.",
            "",
            "```json",
            json.dumps(report.get("paired", {}).get("mcnemar", {}), indent=2),
            "```",
            "",
            "McNemar statistic is reported as an observed paired discordance measure, not a significance claim.",
            "",
            "## Safety Results",
            "",
            "```json",
            json.dumps(report.get("safety_metrics", {}), indent=2),
            "```",
            "",
            "## Latency",
            "",
            "```json",
            json.dumps(lat, indent=2),
            "```",
            "",
            "## Biggest Improvements",
            "",
            "```json",
            json.dumps(report.get("wins", []), indent=2),
            "```",
            "",
            "## Regressions",
            "",
            "```json",
            json.dumps(report.get("regressions", []), indent=2),
            "```",
            "",
            "## Failure Analysis",
            "",
            "```json",
            json.dumps(report.get("difference_counts", {}), indent=2),
            "```",
            "",
            "Document-only questions lack structured LOs, so control is typically insufficient. "
            "Experiment succeeds when context-hybrid retrieves gold fragments and the evidence-quoting "
            "synthesizer stays within retrieved text. Adversarial structured poison is present in both "
            "arms; V2.11 + V2.8 mapper must keep false acceptance at zero.",
            "",
            "## Production integrity",
            "",
            "- LangGraph path unchanged",
            "- Verifier unchanged",
            "- V2.8 mapper unchanged",
            "- V2.11 guard unchanged",
            "- `/api/v1` unchanged",
            "- Production document retrieval disabled",
            "- `v213c_experiment=false`, `v213c_document_retrieval=false`",
            "",
            "## Recommendation",
            "",
            report.get("v213d_recommendation", ""),
            "",
            "Do **not** automatically promote document retrieval to production.",
        ]
    )
    DOC.write_text("\n".join(lines))


def main() -> int:
    from app.agent.v213c_experiment import V213CEvaluationHarness

    items = write_dataset()
    OUT.mkdir(parents=True, exist_ok=True)
    harness = V213CEvaluationHarness(store_root=OUT / "documents", index_root=OUT / "index")
    report = harness.evaluate(items)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    slim = copy_without_answers(report)
    OUT_JSON.write_text(json.dumps(slim, indent=2))
    (OUT / "pairs.json").write_text(json.dumps(report["pairs"], indent=2))
    write_doc(report)
    print(f"Conclusion: {report['conclusion']}")
    print(f"Dataset: {DATASET} ({report['dataset_size']} questions)")
    print(f"Report: {DOC}")
    return 0


def copy_without_answers(report: dict) -> dict:
    """Keep diagnostics smaller: drop full answers from the summary JSON."""
    out = dict(report)
    pairs = []
    for p in report.get("pairs", []):
        row = {
            "id": p["id"],
            "category": p["category"],
            "question": p["question"],
            "expected_answerability": p.get("expected_answerability"),
            "difference": p["difference"],
            "control": {k: v for k, v in p["control"].items() if k != "answer"},
            "experiment": {k: v for k, v in p["experiment"].items() if k != "answer"},
        }
        pairs.append(row)
    out["pairs"] = pairs
    return out


if __name__ == "__main__":
    raise SystemExit(main())
