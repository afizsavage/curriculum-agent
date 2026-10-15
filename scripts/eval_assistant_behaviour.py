#!/usr/bin/env python3
"""Run the teacher-assistant behaviour evaluation.

The default mode is deterministic and uses mocked QA results. It does not call
a model provider. Live evaluation is opt-in:

    ASSISTANT_EVAL_LIVE=1 python scripts/eval_assistant_behaviour.py --live

Answer text is written only when ``--include-answers`` is set. The assistant
HTTP route is not enabled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.assistant.evaluation import (  # noqa: E402
    LiveEvaluationRefused,
    format_summary,
    run_deterministic,
    run_live,
    write_report,
)
from app.config import Settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate teacher assistant behaviour.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Opt-in run against the configured provider and QA agent.",
    )
    parser.add_argument(
        "--include-answers",
        action="store_true",
        help="Store answer text in the report file. Omit this for the summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the machine-readable report to this path.",
    )
    args = parser.parse_args(argv)
    try:
        if args.live:
            report = run_live(Settings(), include_answers=args.include_answers)
        else:
            report = run_deterministic(include_answers=args.include_answers)
    except LiveEvaluationRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(format_summary(report))
    if args.output is not None:
        write_report(report, args.output)
        print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
