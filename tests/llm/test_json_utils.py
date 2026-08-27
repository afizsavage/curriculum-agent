"""Tests for resilient LLM JSON parsing."""

import json

import pytest

from app.llm.json_utils import parse_llm_json


def test_parse_plain_object():
    assert parse_llm_json('{"answer":"ok","confidence":"high"}')["answer"] == "ok"


def test_parse_markdown_fenced_json():
    text = """Here you go:
```json
{
  "answer": "Fractions outcomes",
  "evidence": [],
  "limitations": [],
  "confidence": "medium"
}
```
"""
    parsed = parse_llm_json(text)
    assert parsed["answer"] == "Fractions outcomes"
    assert parsed["confidence"] == "medium"


def test_parse_prose_wrapped_json():
    text = 'Sure. {"passed": true, "score": 0.9, "issues": [] } thanks'
    assert parse_llm_json(text)["passed"] is True


def test_parse_truncated_object_repairs():
    text = '{"answer": "Partial", "evidence": [{"entity_id": "1", "entity_type": "topic", "claim": "x"}]'
    parsed = parse_llm_json(text)
    assert parsed["answer"] == "Partial"
    assert parsed["evidence"][0]["entity_id"] == "1"


def test_parse_empty_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json("   ")
