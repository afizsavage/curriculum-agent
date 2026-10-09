"""Replay D1–D4 and M1/M2 through the isolated coverage validator.

Does not change the agent, the renderer text, or the verifier.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.agent.answer_generator import (  # noqa: E402
    _sss_parentless_topics,
    _sss_theme_occurrences,
)
from app.tools.curriculum import _content_evidence  # noqa: E402
from validator import (  # noqa: E402
    AnswerBlock,
    combined_decision,
    faithful_blocks,
    records_from_tree,
    validate,
)

CASES = {
    "D1": {
        "grade_curriculum_id": "4d4c4e92-aae9-4c81-98f0-2cc570700a60",
        "grade": "SSS_3",
        "subject": "African Literature",
        "stream": "Languages & Literatures",
        "expected": "accept",
    },
    "D2": {
        "grade_curriculum_id": "9e701a83-0188-491d-8afd-c5adb14684ea",
        "grade": "SSS_3",
        "subject": "Environmental Science",
        "stream": "Sciences & Technologies",
        "expected": "accept",
    },
    "D3": {
        "grade_curriculum_id": "5763760b-deea-4908-a0b6-268ea78994cb",
        "grade": "SSS_1",
        "subject": "Derivatives of Religious and Moral Education",
        "stream": "Social & Cultural Studies",
        "expected": "accept",
    },
    "D4": {
        "grade_curriculum_id": "3a9c8471-f18a-4d38-b1f7-c87fe160596a",
        "grade": "SSS_1",
        "subject": "Fula",
        "stream": "Languages & Literatures",
        "expected": "accept",
    },
}


def fetch(grade_curriculum_id: str):
    url = f"http://127.0.0.1:8000/api/v1/grade-curricula/{grade_curriculum_id}/content"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode())


def load_trees() -> dict:
    frozen = Path("/tmp/sss_hierarchy_trees.json")
    trees = json.loads(frozen.read_text()) if frozen.exists() else {}
    for case_id, case in CASES.items():
        if case_id not in trees:
            trees[case_id] = fetch(case["grade_curriculum_id"])
    return trees


def renderer_blocks(tree, case) -> list[AnswerBlock]:
    content = _content_evidence(
        tree,
        grade=case["grade"],
        subject=case["subject"],
        stream_name=case["stream"],
        source_reference=None,
    )
    blocks = [
        AnswerBlock(
            theme.entity_id,
            "theme",
            tuple(
                AnswerBlock(child.entity_id, (child.entity_type or "topic").lower())
                for child in children
                if child.entity_id
            ),
        )
        for theme, children in _sss_theme_occurrences(content)
        if theme.entity_id
    ]
    blocks.extend(
        AnswerBlock(topic.entity_id, "topic")
        for topic in _sss_parentless_topics(content)
        if topic.entity_id
    )
    return blocks


def mutate_m1(blocks: list[AnswerBlock]) -> list[AnswerBlock]:
    themes = [block for block in blocks if block.entity_type == "theme"]
    rest = [block for block in blocks if block.entity_type != "theme"]
    first, second, third = themes[:3]
    merged = AnswerBlock(first.entity_id, first.entity_type, first.children + third.children)
    return [merged, second, *themes[3:], *rest]


def mutate_m2(blocks: list[AnswerBlock]) -> list[AnswerBlock]:
    themes = [block for block in blocks if block.entity_type == "theme"]
    rest = [block for block in blocks if block.entity_type != "theme"]
    revision = next(block for block in rest if block.entity_id == "878307d7-fe08-45af-8085-978600b40227")
    nuclear = next(block for block in themes if block.entity_id == "d3d5d312-e768-44b0-ad60-9d638c7211ab")
    nested = AnswerBlock(
        nuclear.entity_id,
        nuclear.entity_type,
        nuclear.children + (revision,),
    )
    return [
        nested if block.entity_id == nuclear.entity_id else block
        for block in themes
    ] + [block for block in rest if block.entity_id != revision.entity_id]


def pairs(blocks: list[AnswerBlock]) -> list[tuple[str, str | None]]:
    found: list[tuple[str, str | None]] = []

    def walk(nodes, parent):
        for node in nodes:
            found.append((node.entity_id, parent))
            walk(node.children, node.entity_id)

    walk(blocks, None)
    return sorted(found)


def dump(result) -> dict:
    return {
        "passed": result.passed,
        "errors": [
            {"code": error.code, "entity_id": error.entity_id, "message": error.message}
            for error in result.errors
        ],
    }


def main() -> None:
    trees = load_trees()
    llm = {
        row["case"]: row
        for row in json.loads(Path("/tmp/sss_hierarchy_after.json").read_text())["results"]
    }
    report = []
    for case_id, case in CASES.items():
        records = records_from_tree(trees[case_id])
        started = time.perf_counter()
        blocks = faithful_blocks(records)
        structural = validate(records, blocks)
        elapsed_ms = (time.perf_counter() - started) * 1000
        rendered = renderer_blocks(trees[case_id], case)
        renderer_result = validate(records, rendered)
        llm_row = llm[case_id]
        llm_passed = all(repeat["passed"] for repeat in llm_row["repeats"])
        report.append(
            {
                "case": case_id,
                "expected": case["expected"],
                "source_records": len(records),
                "in_scope": len(blocks) + sum(len(block.children) for block in blocks),
                "structural": dump(structural),
                "structural_ms": round(elapsed_ms, 3),
                "renderer_matches_parent_ids": pairs(rendered) == pairs(blocks),
                "renderer_structural": dump(renderer_result),
                "llm_passed": llm_passed,
                "llm_scores": [repeat["score"] for repeat in llm_row["repeats"]],
                "llm_recommendations": [repeat["recommendation"] for repeat in llm_row["repeats"]],
                "combined": combined_decision(structural, llm_passed),
            }
        )
        print(
            case_id,
            "structural",
            structural.passed,
            "renderer_match",
            pairs(rendered) == pairs(blocks),
            "combined",
            combined_decision(structural, llm_passed),
            f"{elapsed_ms:.3f}ms",
        )

    for label, base, mutate in (
        ("M1", "D1", mutate_m1),
        ("M2", "D2", mutate_m2),
    ):
        records = records_from_tree(trees[base])
        started = time.perf_counter()
        structural = validate(records, mutate(faithful_blocks(records)))
        elapsed_ms = (time.perf_counter() - started) * 1000
        llm_row = llm[label]
        llm_passed = all(repeat["passed"] for repeat in llm_row["repeats"])
        report.append(
            {
                "case": label,
                "expected": "reject",
                "structural": dump(structural),
                "structural_ms": round(elapsed_ms, 3),
                "llm_passed": llm_passed,
                "llm_scores": [repeat["score"] for repeat in llm_row["repeats"]],
                "llm_recommendations": [repeat["recommendation"] for repeat in llm_row["repeats"]],
                "llm_issues": [repeat["issues"] for repeat in llm_row["repeats"]],
                "combined": combined_decision(structural, llm_passed),
            }
        )
        print(label, "structural", structural.passed, "errors", len(structural.errors), f"{elapsed_ms:.3f}ms")
        for error in structural.errors:
            print(f"  {error.code} {error.entity_id}: {error.message}")

    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
