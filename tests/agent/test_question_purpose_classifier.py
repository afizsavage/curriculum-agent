"""Deterministic question-purpose classification."""

from __future__ import annotations

from app.agent.graph_nodes import GraphNodes
from app.agent.question_purpose import (
    PURPOSE_LEARNING_OUTCOME_LIST,
    PURPOSE_TOPIC_LIST,
    PURPOSE_UNKNOWN,
    classify_question_purpose,
    coerce_question_purpose,
)
from app.agent.state import CurriculumQAState
from app.curriculum.sss_stream_intent import (
    FOCUS_OUTCOMES,
    INTENT_SSS_STREAM_SUBJECTS,
)

_OUTCOME_QUESTION = "What outcomes are published under the Alpha theme?"
_SSS_OUTCOMES = (
    "What are the learning outcomes for Biology in SSS 1 Science and Technology stream?"
)
_TOPIC_QUESTION = "What topics are listed under the Alpha theme?"
_TITLE_QUESTION = "What is the title of the Alpha theme?"
_AMBIGUOUS_QUESTION = "Tell me about Alpha."
_INCIDENTAL_QUESTION = (
    "Does the Alpha unit include a learning outcome about fractions?"
)


def _understand(question: str) -> CurriculumQAState:
    state = CurriculumQAState.initial(question=question)
    GraphNodes.understand(None, {"qa": state, "visited_nodes": []})  # type: ignore[arg-type]
    return state


def test_published_outcomes_are_an_outcome_list():
    assert classify_question_purpose(_OUTCOME_QUESTION) == PURPOSE_LEARNING_OUTCOME_LIST
    assert _understand(_OUTCOME_QUESTION).metadata["question_purpose"] == (
        PURPOSE_LEARNING_OUTCOME_LIST
    )


def test_sss_outcome_question_keeps_stream_routing():
    assert classify_question_purpose(_SSS_OUTCOMES) == PURPOSE_LEARNING_OUTCOME_LIST
    state = _understand(_SSS_OUTCOMES)
    assert state.metadata["question_purpose"] == PURPOSE_LEARNING_OUTCOME_LIST
    assert state.intent == INTENT_SSS_STREAM_SUBJECTS
    assert state.metadata["sss_focus"] == FOCUS_OUTCOMES
    assert state.subject is None


def test_listed_topics_are_a_topic_list():
    assert classify_question_purpose(_TOPIC_QUESTION) == PURPOSE_TOPIC_LIST
    assert _understand(_TOPIC_QUESTION).metadata["question_purpose"] == PURPOSE_TOPIC_LIST
    assert _understand(_TOPIC_QUESTION).intent == "retrieve_curriculum"


def test_theme_title_and_open_wording_are_unknown():
    assert classify_question_purpose(_TITLE_QUESTION) == PURPOSE_UNKNOWN
    assert classify_question_purpose(_AMBIGUOUS_QUESTION) == PURPOSE_UNKNOWN
    assert _understand(_TITLE_QUESTION).metadata["question_purpose"] == PURPOSE_UNKNOWN
    assert _understand(_AMBIGUOUS_QUESTION).metadata["question_purpose"] == PURPOSE_UNKNOWN


def test_incidental_outcome_mention_is_unknown():
    assert classify_question_purpose(_INCIDENTAL_QUESTION) == PURPOSE_UNKNOWN
    state = _understand(_INCIDENTAL_QUESTION)
    assert state.metadata["question_purpose"] == PURPOSE_UNKNOWN
    assert state.intent == "retrieve_curriculum"


def test_french_subject_filter_stays_on_the_catalogue_route():
    state = _understand("What units does the Class 4 French subject cover?")
    assert state.subject == "FRENCH"
    assert state.intent == "retrieve_curriculum"
    assert state.metadata["question_purpose"] == PURPOSE_UNKNOWN
    assert "sss_focus" not in state.metadata


def test_missing_and_invalid_purposes_resolve_to_unknown():
    assert coerce_question_purpose(None) == PURPOSE_UNKNOWN
    assert coerce_question_purpose("") == PURPOSE_UNKNOWN
    assert coerce_question_purpose("SSS_STREAM_SUBJECTS") == PURPOSE_UNKNOWN
    assert coerce_question_purpose("outcomes") == PURPOSE_UNKNOWN
    assert coerce_question_purpose("topic_list ") == PURPOSE_UNKNOWN
    assert coerce_question_purpose(PURPOSE_TOPIC_LIST) == PURPOSE_TOPIC_LIST
    assert classify_question_purpose(None) == PURPOSE_UNKNOWN
    assert classify_question_purpose("   ") == PURPOSE_UNKNOWN


def test_classification_is_stable_across_repeats():
    questions = (
        _OUTCOME_QUESTION,
        _SSS_OUTCOMES,
        _TOPIC_QUESTION,
        _TITLE_QUESTION,
        _AMBIGUOUS_QUESTION,
        _INCIDENTAL_QUESTION,
        "What topics and learning outcomes are listed under the Alpha theme?",
    )
    for question in questions:
        assert classify_question_purpose(question) == classify_question_purpose(question)
        first = _understand(question).metadata["question_purpose"]
        second = _understand(question).metadata["question_purpose"]
        assert first == second


def test_mixed_list_requests_stay_unknown():
    mixed = (
        "What topics and learning outcomes are listed under the Alpha theme?",
        "What are the learning outcomes and the topics for Alpha?",
        "List the topics and the learning outcomes for Alpha.",
    )
    for question in mixed:
        assert classify_question_purpose(question) == PURPOSE_UNKNOWN
        assert _understand(question).metadata["question_purpose"] == PURPOSE_UNKNOWN


def test_gold_outcome_list_phrasings_are_recognized():
    recognized = (
        "What are all the learning outcomes for Class 4 Mathematics unit C4-U01, Number and Numeration?",
        "What are all the learning outcomes for Class 3 Mathematics unit C3-U15?",
        "What learning outcomes are published for the Class 4 Science theme The Earth, The Solar System and the Moon?",
    )
    for question in recognized:
        assert classify_question_purpose(question) == PURPOSE_LEARNING_OUTCOME_LIST
        assert _understand(question).metadata["question_purpose"] == (
            PURPOSE_LEARNING_OUTCOME_LIST
        )
        assert _understand(question).intent == "retrieve_curriculum"


def test_near_miss_outcome_wording_stays_unknown():
    near_misses = (
        "What are the Class 4 Mathematics outcomes?",
        "What is the first published unit of JSS 1 Integrated Science?",
        "What are all the topics under the Alpha theme?",
        "What are all the main points of this learning outcome?",
        "What learning outcome is published for the Alpha theme?",
        "What are all the learning outcomes and the topics for Alpha?",
        "What learning outcomes and topics are published under the Alpha theme?",
        "Tell me about all the learning outcomes for Alpha.",
    )
    for question in near_misses:
        assert classify_question_purpose(question) == PURPOSE_UNKNOWN
        assert _understand(question).metadata["question_purpose"] == PURPOSE_UNKNOWN


def test_location_words_do_not_override_a_clear_list_request():
    assert (
        classify_question_purpose("What outcomes are published under the Alpha topic?")
        == PURPOSE_LEARNING_OUTCOME_LIST
    )
    assert (
        classify_question_purpose(
            "What topics are listed under the theme that introduces one outcome?"
        )
        == PURPOSE_TOPIC_LIST
    )
