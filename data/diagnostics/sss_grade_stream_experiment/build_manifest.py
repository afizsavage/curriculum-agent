"""Freeze the live SSS stream, subject, and grade-syllabus universe.

This script only reads the Curriculum Structure API. It does not change the
QA agent.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.curriculum.client import CurriculumAPIClient
from app.curriculum.errors import CurriculumTimeoutError

OUT = Path(__file__).with_name("manifest.json")
GRADES = ("SSS_1", "SSS_2", "SSS_3")


def _pages(fetch) -> list[dict]:
    items: list[dict] = []
    offset = 0
    while True:
        page = _call(lambda: fetch(limit=200, offset=offset))
        batch = [row for row in (page.get("items") or []) if isinstance(row, dict)]
        items.extend(batch)
        total = page.get("total")
        if not batch:
            break
        offset += len(batch)
        if isinstance(total, int) and offset >= total:
            break
        if not isinstance(total, int) and len(batch) < 200:
            break
    return items


def _call(fn, attempts: int = 3):
    last = None
    for _ in range(attempts):
        try:
            return fn()
        except CurriculumTimeoutError as exc:
            last = exc
            time.sleep(1)
    raise last


def _walk(nodes, parent: str | None, rows: list[dict], counts: dict) -> None:
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        kind = str(node.get("content_type") or "").upper()
        name = node.get("name") or node.get("statement") or node.get("description")
        next_parent = parent
        if kind in {"THEME", "TOPIC", "LEARNING_OUTCOME"} and name:
            rows.append({"type": kind.lower(), "name": str(name), "parent": parent})
            counts[kind] = counts.get(kind, 0) + 1
            if kind == "THEME":
                next_parent = str(name)
        _walk(node.get("children") or [], next_parent, rows, counts)


def main() -> None:
    settings = get_settings()
    client = CurriculumAPIClient(settings=settings)
    started = time.perf_counter()
    curriculum_id = _call(
        lambda: client.resolve_curriculum_id(code="MBSSE-SSC", version="2021")
    )
    if not curriculum_id:
        raise SystemExit("MBSSE-SSC 2021 curriculum was not found")
    streams_raw = _pages(lambda **params: client.list_sss_streams(curriculum_id, **params))
    streams = []
    for stream in streams_raw:
        stream_id = str(stream["id"])
        detail = _call(lambda sid=stream_id: client.get_sss_stream(sid))
        assignments = _pages(
            lambda sid=stream_id, **params: client.list_sss_stream_subjects(sid, **params)
        )
        subjects = []
        for row in assignments:
            subject = row.get("subject") if isinstance(row.get("subject"), dict) else {}
            if not subject.get("id") or not subject.get("name"):
                continue
            subjects.append(
                {
                    "id": str(subject["id"]),
                    "name": subject["name"],
                    "code": subject.get("code"),
                    "subject_type": row.get("subject_type"),
                }
            )
        streams.append(
            {
                "id": stream_id,
                "name": detail.get("name") or stream.get("name"),
                "code": detail.get("code") or stream.get("code"),
                "subjects": subjects,
            }
        )
    grade_rows = _pages(
        lambda **params: client.list_curriculum_grade_curricula(curriculum_id, **params)
    )
    syllabi = []
    errors = []
    for index, row in enumerate(grade_rows, start=1):
        grade = str((row.get("grade") or {}).get("code") or "")
        if grade not in GRADES:
            continue
        subject = row.get("subject") if isinstance(row.get("subject"), dict) else {}
        subject_id = str(subject.get("id") or row.get("subject_id") or "")
        row_id = str(row.get("id") or "")
        record = {
            "grade_curriculum_id": row_id,
            "grade": grade,
            "subject_id": subject_id,
            "subject_name": subject.get("name"),
            "source_reference": row.get("source_reference"),
            "content": [],
            "counts": {},
            "content_error": None,
        }
        try:
            tree = _call(lambda rid=row_id: client.get_grade_curriculum_content(rid))
            rows: list[dict] = []
            counts: dict = {}
            _walk(tree, None, rows, counts)
            record["content"] = rows
            record["counts"] = counts
            digest = hashlib.sha256(
                json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            record["evidence_sha256"] = digest
        except Exception as exc:  # oracle capture; the agent is unchanged
            record["content_error"] = type(exc).__name__
            errors.append({"grade_curriculum_id": row_id, "error": type(exc).__name__})
        syllabi.append(record)
        if index % 25 == 0:
            print(f"syllabi {index}/{len(grade_rows)}", flush=True)
    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "curriculum_code": "MBSSE-SSC",
        "curriculum_version": "2021",
        "curriculum_id": curriculum_id,
        "api_base_url": settings.resolved_curriculum_api_url(),
        "streams": streams,
        "syllabi": syllabi,
        "content_errors": errors,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(
        f"MANIFEST_DONE streams={len(streams)} syllabi={len(syllabi)} "
        f"errors={len(errors)} seconds={payload['elapsed_seconds']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
