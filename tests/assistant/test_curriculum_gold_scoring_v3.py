"""Opt-in scoring v3. These tests do not call a model or edit the gold set."""

from __future__ import annotations

import inspect

from app.assistant.curriculum_gold import (
    SCORING_VERSION,
    SCORING_VERSION_V2,
    SCORING_VERSION_V3,
    file_sha256,
    load_dataset,
    run_audit,
    score_observation,
)

_GOLD_SHA = "f16fc259720bfd9ff19550f9c6e8ac6c4303003a2502aa39f3745f35c5549cf5"
_EN_DASH = "Living and Non \u2013 living Things"


def _case(case_id: str) -> dict:
    dataset = load_dataset()
    return next(case for case in dataset["cases"] if case["id"] == case_id)


def _score(case_id: str, answer: str, version: str) -> dict:
    return score_observation(
        _case(case_id),
        {"answer": answer, "status": "completed", "evidence_status": "found"},
        scoring_version=version,
    )


def _recalled(case_id: str, fact_id: str, answer: str, version: str) -> bool:
    return fact_id in _score(case_id, answer, version)["recalled_fact_ids"]


def test_article_and_terminal_mark_pairs_pass_only_under_v3_when_v2_cannot():
    pairs = [
        (
            "gold-c4-math-u01-outcomes-complete",
            "c4-u01-c4u01-lo03",
            "Compare numbers up to 10,000 using a number line.",
            True,
        ),
        (
            "gold-c4-math-u01-outcomes-complete",
            "c4-u01-c4u01-lo05",
            "Use the place value system up to 10,000,000.",
            True,
        ),
        (
            "gold-c3-math-u15-outcomes-complete",
            "c3-u15-c3u15-lo04",
            "Write the multiplication table of 3 and 6.",
            False,
        ),
        (
            "gold-c3-math-u15-outcomes-complete",
            "c3-u15-c3u15-lo05",
            "Write the division table of 3 and 6.",
            False,
        ),
        (
            "gold-c4-science-solar-outcomes-complete",
            "c4-solar-c4t14-lo03",
            "Describe the movement of the moon round the earth.",
            True,
        ),
        (
            "gold-c4-science-solar-outcomes-complete",
            "c4-solar-c4t14-lo04",
            "State and describe the members of the solar system.",
            True,
        ),
        (
            "gold-c4-science-solar-outcomes-complete",
            "c4-solar-c4t14-lo05",
            "Explain whether the moon is a planet or a star.",
            True,
        ),
    ]
    for case_id, fact_id, answer, v2_accepts in pairs:
        assert _recalled(case_id, fact_id, answer, SCORING_VERSION) is False
        assert _recalled(case_id, fact_id, answer, SCORING_VERSION_V2) is v2_accepts
        assert _recalled(case_id, fact_id, answer, SCORING_VERSION_V3) is True
        metrics = _score(case_id, answer, SCORING_VERSION_V3)
        assert metrics["scoring_version"] == SCORING_VERSION_V3


def test_approved_dash_title_matches_without_requiring_the_code():
    case = _case("gold-c4-science-not-integrated")
    fact = next(
        item for item in case["required_facts"] if item["fact_id"] == "c4-sci-first-c4-u01"
    )
    assert fact["match_any"] == ["C4-U01", _EN_DASH]
    hyphen = "The first published unit is Living and Non-living Things."
    assert _recalled(
        "gold-c4-science-not-integrated", "c4-sci-first-c4-u01", hyphen, SCORING_VERSION
    ) is False
    assert _recalled(
        "gold-c4-science-not-integrated", "c4-sci-first-c4-u01", hyphen, SCORING_VERSION_V2
    ) is False
    assert _recalled(
        "gold-c4-science-not-integrated", "c4-sci-first-c4-u01", hyphen, SCORING_VERSION_V3
    ) is True
    exact = f"The first published unit is {_EN_DASH}."
    for version in (SCORING_VERSION, SCORING_VERSION_V2, SCORING_VERSION_V3):
        assert _recalled(
            "gold-c4-science-not-integrated", "c4-sci-first-c4-u01", exact, version
        ) is True
    code_only = "The first published unit is C4-U01."
    for version in (SCORING_VERSION, SCORING_VERSION_V2, SCORING_VERSION_V3):
        assert _recalled(
            "gold-c4-science-not-integrated", "c4-sci-first-c4-u01", code_only, version
        ) is True


def test_v3_rejects_meaning_changes_and_other_dashes():
    rejected = [
        (
            "gold-c4-math-u01-outcomes-complete",
            "c4-u01-c4u01-lo03",
            "Order numbers up to 10,000 using a number line.",
        ),
        (
            "gold-c4-math-u01-outcomes-complete",
            "c4-u01-c4u01-lo03",
            "Compare numbers up to 100,000 using a number line.",
        ),
        (
            "gold-c4-math-u01-outcomes-complete",
            "c4-u01-c4u01-lo03",
            "Compare numbers up to 10,000 using a number chart.",
        ),
        (
            "gold-c4-math-u01-outcomes-complete",
            "c4-u01-c4u01-lo03",
            "Compare numbers up to 10.000 using a number line.",
        ),
        (
            "gold-c3-math-u15-outcomes-complete",
            "c3-u15-c3u15-lo04",
            "Write the multiplication table of 3. and 6.",
        ),
        (
            "gold-c4-science-solar-outcomes-complete",
            "c4-solar-c4t14-lo05",
            "Explain whether the moon is not a planet or a star.",
        ),
        (
            "gold-c4-science-solar-outcomes-complete",
            "c4-solar-c4t14-lo03",
            "Describe the movement of the moon around the earth.",
        ),
        (
            "gold-c4-science-not-integrated",
            "c4-sci-first-c4-u01",
            "Living and Non living Things",
        ),
        (
            "gold-c4-science-not-integrated",
            "c4-sci-first-c4-u01",
            "Living and Non\u2014living Things",
        ),
        (
            "gold-c4-science-not-integrated",
            "c4-sci-first-c4-u01",
            "Living and Non - living Things",
        ),
        (
            "gold-c4-science-not-integrated",
            "c4-sci-first-c4-u01",
            "Living and Non-living Animals",
        ),
    ]
    for case_id, fact_id, answer in rejected:
        assert _recalled(case_id, fact_id, answer, SCORING_VERSION_V3) is False


def test_versions_are_stable_and_v1_stays_the_default():
    answer = "Write the multiplication table of 3 and 6."
    first = _score("gold-c3-math-u15-outcomes-complete", answer, SCORING_VERSION_V3)
    second = _score("gold-c3-math-u15-outcomes-complete", answer, SCORING_VERSION_V3)
    assert first["recalled_fact_ids"] == second["recalled_fact_ids"]
    assert first["missed_fact_ids"] == second["missed_fact_ids"]
    default = score_observation(
        _case("gold-c3-math-u15-outcomes-complete"),
        {"answer": answer, "status": "completed", "evidence_status": "found"},
    )
    assert default["scoring_version"] == SCORING_VERSION
    assert (
        inspect.signature(score_observation).parameters["scoring_version"].default
        == SCORING_VERSION
    )
    assert run_audit()["scoring_version"] == SCORING_VERSION
    assert file_sha256() == _GOLD_SHA
    assert load_dataset()["scoring_version"] == SCORING_VERSION
