#!/usr/bin/env python3
"""Alias for scripts/eval_v213c_curriculum_qa.py."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("eval_v213c_curriculum_qa.py")), run_name="__main__")
