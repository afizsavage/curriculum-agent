#!/usr/bin/env python3
"""Score the frozen curriculum gold set.

The default mode audits the dataset and does not call a model. Live scoring
is opt-in and separate from CI:

    CURRICULUM_GOLD_EVAL_LIVE=1 python scripts/eval_curriculum_gold.py --live

The report records the dataset version and hash. Answer text is included only
with ``--include-answers``. Question text is not written to the process log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.assistant.curriculum_gold import (  # noqa: E402
    GoldEvaluationRefused,
    format_summary,
    run_audit,
    run_live,
    write_report,
)
from app.config import Settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate curriculum QA gold facts.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Opt-in run through the current QA agent.",
    )
    parser.add_argument(
        "--include-answers",
        action="store_true",
        help="Store answer text in the report file.",
    )
    parser.add_argument("--output", type=Path, help="Write the JSON report here.")
    args = parser.parse_args(argv)
    try:
        if args.live:
            report = run_live(Settings(), include_answers=args.include_answers)
        else:
            report = run_audit()
    except GoldEvaluationRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(format_summary(report))
    if args.output is not None:
        write_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
