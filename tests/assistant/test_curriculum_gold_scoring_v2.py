"""Article-tolerant gold scoring. These tests do not call a model."""

from __future__ import annotations

from app.assistant.curriculum_gold import (
    SCORING_VERSION,
    SCORING_VERSION_V2,
    load_dataset,
    score_observation,
)

_SOLAR = "gold-c4-science-solar-outcomes-complete"
_LO03 = "c4-solar-c4t14-lo03"
_LO04 = "c4-solar-c4t14-lo04"
_LO05 = "c4-solar-c4t14-lo05"
_SOLAR_ANSWER = {
    _LO03: "Describe the movement of the moon round the earth.",
    _LO04: "State and describe the members of the solar system.",
    _LO05: "Explain whether the moon is a planet or a star.",
}


def _case(case_id: str = _SOLAR) -> dict:
    dataset = load_dataset()
    return next(case for case in dataset["cases"] if case["id"] == case_id)


def _fact(fact_id: str) -> dict:
    return next(fact for fact in _case()["required_facts"] if fact["fact_id"] == fact_id)


def _text(fact_id: str) -> str:
    return _fact(fact_id)["match_any"][0]


def _score(answer: str, version: str) -> dict:
    return score_observation(
        _case(),
        {"answer": answer, "status": "completed", "evidence_status": "found"},
        scoring_version=version,
    )


def test_v1_rejects_the_three_article_insertions_and_v2_accepts_them():
    for fact_id, answer in _SOLAR_ANSWER.items():
        v1 = _score(answer, SCORING_VERSION)
        v2 = _score(answer, SCORING_VERSION_V2)
        assert fact_id in v1["missed_fact_ids"]
        assert fact_id not in v1["recalled_fact_ids"]
        assert fact_id in v2["recalled_fact_ids"]
        assert v1["scoring_version"] == "curriculum-qa-gold-scoring-v1"
        assert v2["scoring_version"] == "curriculum-qa-gold-scoring-v2"


def test_verbatim_solar_facts_pass_under_both_scorers():
    answer = " ".join(_text(fact_id) for fact_id in _SOLAR_ANSWER)
    for version in (SCORING_VERSION, SCORING_VERSION_V2):
        metrics = _score(answer, version)
        assert set(_SOLAR_ANSWER) <= set(metrics["recalled_fact_ids"])


def test_article_insertion_and_omission_pass_only_when_other_tokens_stay():
    inserted = "Describe the movement of the moon round the earth."
    omitted = "Describe movement of moon round earth."
    assert _LO03 in _score(inserted, SCORING_VERSION_V2)["recalled_fact_ids"]
    assert _LO03 in _score(omitted, SCORING_VERSION_V2)["recalled_fact_ids"]
    assert _LO03 not in _score(inserted, SCORING_VERSION)["recalled_fact_ids"]


def test_case_and_whitespace_stay_compatible_with_the_exact_contract():
    answer = "  DESCRIBE   movement of the moon round the earth.  "
    for version in (SCORING_VERSION, SCORING_VERSION_V2):
        assert _LO03 in _score(answer, version)["recalled_fact_ids"]


def test_changed_verb_object_order_and_substitutions_fail():
    rejected = [
        "Identify movement of the moon round the earth.",
        "Describe movement of the sun round the earth.",
        "Describe movement of the earth round the moon.",
        "Describe orbit of the moon round the earth.",
        "Describe movement of the moon around the earth.",
    ]
    for answer in rejected:
        assert _LO03 not in _score(answer, SCORING_VERSION_V2)["recalled_fact_ids"]


def test_added_or_removed_negation_fails():
    added = "Do not describe movement of the moon round the earth."
    inserted = "Describe movement of the moon not round the earth."
    assert _LO03 not in _score(added, SCORING_VERSION_V2)["recalled_fact_ids"]
    assert _LO03 not in _score(inserted, SCORING_VERSION_V2)["recalled_fact_ids"]
    case = {
        "id": "negation-control",
        "factual_scoring": True,
        "support": "answerable",
        "subject_code": "SCIENCE",
        "grade_code": "CLASS_4",
        "must_be_complete": False,
        "required_facts": [
            {
                "fact_id": "negated",
                "match_any": ["State that the moon is not a star."],
                "source_ids": ["source"],
            }
        ],
        "forbidden_substitutions": [],
        "grade_labels": ["Class 4"],
        "subject_labels": ["Science"],
    }
    removed = score_observation(
        case,
        {
            "answer": "State that the moon is a star.",
            "status": "completed",
            "evidence_status": "found",
        },
        scoring_version=SCORING_VERSION_V2,
    )
    kept = score_observation(
        case,
        {
            "answer": "State that the moon is not a star.",
            "status": "completed",
            "evidence_status": "found",
        },
        scoring_version=SCORING_VERSION_V2,
    )
    assert removed["recalled_fact_ids"] == []
    assert kept["recalled_fact_ids"] == ["negated"]


def test_partial_and_unrelated_wording_does_not_recall_the_fact():
    scattered = (
        "Describe movement in general. The moon is visible. "
        "Pupils walk round the earth model."
    )
    assert _LO03 not in _score(scattered, SCORING_VERSION_V2)["recalled_fact_ids"]


def test_article_letters_inside_longer_words_are_kept():
    thematic = "Describe the thematic movement of the moon round the earth."
    theater = "Describe theater movement of the moon round the earth."
    another = "Describe another movement of the moon round the earth."
    for answer in (thematic, theater, another):
        assert _LO03 not in _score(answer, SCORING_VERSION_V2)["recalled_fact_ids"]
    assert _LO03 in _score(
        "Describe the movement of the moon round the earth. Pupils visit the theater.",
        SCORING_VERSION_V2,
    )["recalled_fact_ids"]
