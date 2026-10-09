"""Compare 14 frozen SSS answers with their frozen syllabus trees.

No verifier call, no retrieval, and no change to generation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
WINDOW = ROOT.parent / "sss_verifier_window" / "results.jsonl"
REPEAT = ROOT.parent / "sss_verifier_repeat" / "results.jsonl"
MANIFEST = ROOT.parent / "sss_grade_stream_experiment" / "manifest.json"
OUT = ROOT / "diagnostic.json"

CASES = [
    ("SSS_3", "French as a Foreign Language"),
    ("SSS_3", "Mining Industry and the Environment"),
    ("SSS_2", "Business Accounting"),
    ("SSS_1", "Calculus"),
    ("SSS_1", "Fundamentals of Mathematics"),
    ("SSS_3", "Further Mathematics"),
    ("SSS_2", "Maths for Other Disciplines"),
    ("SSS_2", "Statistics and Probability"),
    ("SSS_2", "Integrated Science"),
    ("SSS_3", "Creative Writing"),
    ("SSS_1", "Music"),
    ("SSS_1", "Mathematics for STEAMM"),
    ("SSS_2", "Mathematics for STEAMM"),
    ("SSS_2", "Engineering Science"),
]


def _norm(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").casefold()).strip()


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _bullets(answer: str) -> list[tuple[int, str]]:
    found = []
    for line in answer.splitlines():
        match = re.match(r"^(\s*)\*\s+(.+?)\s*$", line)
        if match:
            found.append((len(match.group(1)), match.group(2)))
    return found


def _sections(bullets: list[tuple[int, str]]) -> list[dict]:
    sections = []
    current = None
    for indent, text in bullets:
        if indent == 0:
            current = {"theme": text, "topics": []}
            sections.append(current)
        elif current is not None:
            current["topics"].append(text)
        else:
            sections.append({"theme": None, "topics": [text]})
    return sections


def _tree(content: list[dict]) -> dict:
    themes = []
    topics = []
    for index, item in enumerate(content):
        kind = str(item.get("type") or "").lower()
        name = item.get("name")
        if not name or kind not in {"theme", "topic", "learning_outcome"}:
            continue
        row = {
            "order": index,
            "type": kind,
            "name": name,
            "parent": item.get("parent"),
        }
        if kind == "theme":
            themes.append(row)
        elif kind == "topic":
            topics.append(row)
    return {"themes": themes, "topics": topics}


def _ordered_blocks(tree: dict) -> list[dict]:
    """Group topics under the theme row that precedes them in the frozen tree."""
    blocks: list[dict] = []
    current: dict | None = None
    for theme in tree["themes"]:
        current = {"theme": theme["name"], "topics": []}
        blocks.append(current)
    if not blocks:
        return blocks
    theme_orders = [theme["order"] for theme in tree["themes"]]
    for topic in tree["topics"]:
        prior = [order for order in theme_orders if order < topic["order"]]
        if not prior:
            continue
        index = theme_orders.index(prior[-1])
        blocks[index]["topics"].append(topic["name"])
    return blocks


def _source_repeats(tree: dict) -> list[dict]:
    blocks = _ordered_blocks(tree)
    by_name: dict[str, list[dict]] = {}
    for block in blocks:
        by_name.setdefault(_norm(block["theme"]), []).append(block)
    repeats = []
    for name, rows in by_name.items():
        if len(rows) < 2:
            continue
        signatures = [" | ".join(_norm(topic) for topic in row["topics"]) for row in rows]
        repeats.append(
            {
                "theme": rows[0]["theme"],
                "occurrences": len(rows),
                "child_counts": [len(row["topics"]) for row in rows],
                "identical_children": len(set(signatures)) == 1,
                "overlap": "identical" if len(set(signatures)) == 1 else "distinct",
                "child_topics": [topic for row in rows for topic in row["topics"]],
            }
        )
    topic_counts = Counter(_norm(topic["name"]) for topic in tree["topics"])
    repeated_topics = [
        next(topic["name"] for topic in tree["topics"] if _norm(topic["name"]) == key)
        for key, count in topic_counts.items()
        if count > 1
    ]
    return repeats, repeated_topics


def _answer_repeats(sections: list[dict]) -> list[dict]:
    seen: dict[str, int] = {}
    repeats = []
    for section in sections:
        if not section["theme"]:
            continue
        key = _norm(section["theme"]) + "|" + "|".join(_norm(topic) for topic in section["topics"])
        seen[key] = seen.get(key, 0) + 1
    counted = Counter(_norm(section["theme"]) for section in sections if section["theme"])
    for section in sections:
        theme = section["theme"]
        if theme and counted[_norm(theme)] > 1:
            repeats.append({"theme": theme, "topics": section["topics"], "times": counted[_norm(theme)]})
    # unique by theme
    unique = []
    used = set()
    for item in repeats:
        if _norm(item["theme"]) in used:
            continue
        used.add(_norm(item["theme"]))
        unique.append(item)
    return unique


def _present(name: str, bullets: list[tuple[int, str]]) -> bool:
    target = _norm(name)
    return any(_norm(text) == target for _indent, text in bullets)


def _classify(row: dict) -> str:
    """One primary class from the answer text and the frozen tree.

    Parentless topics that the answer never lists are omissions. Repeated
    theme names are not classed as source duplication unless the answer
    copies those blocks without merging distinct children. Merging distinct
    same-named blocks is OTHER: it is not one theme standing for the course,
    and it is not a faithful copy of the source.
    """
    if row["omitted_topics"]:
        return "ANSWER_OMITS_VALID_TOPICS"
    if row["source_repeated_themes"] and row["answer_repeated_themes"]:
        faithful = all(
            item.get("identical_children") for item in row["source_repeated_themes"]
        )
        if faithful:
            return "DUPLICATE_SYLLABUS_STRUCTURE"
        return "OTHER"
    return "OTHER"


def main() -> None:
    window = _load(WINDOW)
    repeat = _load(REPEAT)
    manifest = json.loads(MANIFEST.read_text())
    syllabi = {
        (row["subject_name"], row["grade"]): row for row in manifest["syllabi"]
    }
    answers = {}
    for row in repeat:
        if row["repeat"] != 1:
            continue
        answers[(row["requested_subject"], row["requested_grade"])] = row
    windows = {(row["subject"], row["grade"]): row for row in window}
    records = []
    for grade, subject in CASES:
        syllabus = syllabi[(subject, grade)]
        frozen = answers[(subject, grade)]
        prior = windows[(subject, grade)]
        answer = frozen["answer"]
        content = syllabus.get("content") or []
        tree = _tree(content)
        bullets = _bullets(answer)
        sections = _sections(bullets)
        source_repeats, repeated_topics = _source_repeats(tree)
        answer_repeats = _answer_repeats(sections)
        omitted_topics = [
            {"name": topic["name"], "parent": topic.get("parent")}
            for topic in tree["topics"]
            if not _present(topic["name"], bullets)
        ]
        # unique omitted by normalized name
        seen = set()
        unique_omitted = []
        for topic in omitted_topics:
            key = _norm(topic["name"])
            if key in seen:
                continue
            seen.add(key)
            unique_omitted.append(topic)
        omitted_themes = [
            theme["name"]
            for theme in tree["themes"]
            if not _present(theme["name"], bullets)
        ]
        seen_themes = set()
        unique_omitted_themes = []
        for name in omitted_themes:
            key = _norm(name)
            if key in seen_themes:
                continue
            seen_themes.add(key)
            unique_omitted_themes.append(name)
        presented_themes = [text for indent, text in bullets if indent == 0]
        claims = "covers:" in answer.casefold()
        answer_hash = hashlib.sha256(answer.encode()).hexdigest()
        syllabus_hash = hashlib.sha256(
            json.dumps(content, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        row = {
            "grade": grade,
            "subject": subject,
            "stream": frozen["requested_stream"],
            "source_document": syllabus.get("source_reference"),
            "question": frozen["question"],
            "answer_hash": answer_hash,
            "frozen_answer_hash": frozen["answer_hash"],
            "answer_hash_matches": answer_hash == frozen["answer_hash"],
            "syllabus_sha256": syllabus_hash,
            "prior_verifier_reason": prior.get("verifier_reason"),
            "source_theme_count": len({_norm(theme["name"]) for theme in tree["themes"]}),
            "source_theme_rows": len(tree["themes"]),
            "source_topic_count": len({_norm(topic["name"]) for topic in tree["topics"]}),
            "source_topic_rows": len(tree["topics"]),
            "source_repeated_themes": source_repeats,
            "source_repeated_topic_names": repeated_topics,
            "answer_themes": presented_themes,
            "answer_repeated_themes": answer_repeats,
            "omitted_topics": unique_omitted,
            "omitted_themes": unique_omitted_themes,
            "claims_coverage": claims,
            "coverage_sentence": next(
                (line for line in answer.splitlines() if "covers:" in line.casefold()),
                None,
            ),
            "closing_sentence": next(
                (line for line in answer.splitlines() if line.startswith("These areas are based on")),
                None,
            ),
            "answer_excerpt": "\n".join(answer.splitlines()[:8]),
        }
        row["primary_classification"] = _classify(row)
        records.append(row)
    counts = Counter(row["primary_classification"] for row in records)
    summary = {
        "cases": len(records),
        "classifications": dict(counts),
        "source_duplication": sum(1 for row in records if row["source_repeated_themes"] or row["source_repeated_topic_names"]),
        "answer_omissions": sum(1 for row in records if row["omitted_topics"] or row["omitted_themes"]),
        "records": records,
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({key: summary[key] for key in summary if key != "records"}, indent=2))
    for row in records:
        print(
            f"{row['primary_classification']:42} {row['grade']} {row['subject'][:36]:36} "
            f"omit_topics={len(row['omitted_topics']):3} omit_themes={len(row['omitted_themes']):2} "
            f"src_theme_rows={row['source_theme_rows']:3} unique_themes={row['source_theme_count']:3} "
            f"src_repeat_themes={len(row['source_repeated_themes']):2} "
            f"ans_repeat={len(row['answer_repeated_themes']):2}"
        )


if __name__ == "__main__":
    main()
