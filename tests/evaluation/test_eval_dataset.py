import json
from pathlib import Path

import pytest


EVAL_PATH = Path(__file__).resolve().parents[2] / "data/evaluation/curriculum_qa_eval_v1.json"


@pytest.fixture
def eval_records() -> list[dict]:
    return json.loads(EVAL_PATH.read_text())


def test_evaluation_dataset_has_minimum_questions(eval_records):
    assert len(eval_records) >= 30


def test_evaluation_records_have_required_fields(eval_records):
    required = {"id", "category", "question", "expected_answer_traits"}
    for record in eval_records:
        missing = required - set(record)
        assert not missing, f"{record.get('id')} missing {missing}"


def test_evaluation_categories_are_diverse(eval_records):
    categories = {r["category"] for r in eval_records}
    expected = {
        "curriculum_structure",
        "subject_lookup",
        "topic_lookup",
        "learning_objectives",
        "natural_language_search",
        "grade_progression",
        "follow_up",
        "ambiguous",
        "insufficient_evidence",
    }
    assert expected.issubset(categories)
