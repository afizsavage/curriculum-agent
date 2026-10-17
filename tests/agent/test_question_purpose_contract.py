"""Default evidence selection stays on the lexical window.

These tests call the selector without a purpose. That path keeps the
pre-integration window.
"""

from __future__ import annotations

import inspect

from app.agent.answer_generator import select_evidence_for_prompt
from app.agent.graph_nodes import GraphNodes
from app.agent.state import CurriculumQAState
from app.agent.verifier import AnswerVerifier, question_requires_outcome_list
from app.curriculum.codes import extract_filters_from_question
from app.curriculum.evidence import CurriculumEvidence
from app.curriculum.sss_stream_intent import (
    FOCUS_OUTCOMES,
    INTENT_SSS_STREAM_SUBJECTS,
    detect_sss_stream_subjects,
)
from app.llm.provider import StubLLMProvider

_TOPIC_QUESTION = "What topics are listed under the Alpha theme?"
_TITLE_QUESTION = "What is the title of the Alpha theme?"
_OUTCOME_QUESTION = "What outcomes are published under the Alpha theme?"
_AMBIGUOUS_QUESTION = "Tell me about Alpha."
_SSS_OUTCOMES = (
    "What are the learning outcomes for Biology in SSS 1 Science and Technology stream?"
)


def _parent(parent_id: str, name: str) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="theme",
        entity_id=parent_id,
        name=name,
        content=name,
    )


def _outcome(parent_id: str, number: int, prefix: str = "S1-T1-01") -> CurriculumEvidence:
    code = f"{prefix}-LO{number:02d}"
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id=f"{parent_id}-lo-{number:02d}",
        name=code,
        topic=parent_id,
        content="plain statement",
        metadata={"code": code},
    )


def _filler(index: int) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="theme",
        entity_id=f"filler-{index:02d}",
        name=f"Other theme {index}",
        content=f"Other theme {index}",
    )


def _family(parent_id: str, name: str, count: int, prefix: str) -> list[CurriculumEvidence]:
    return [
        _parent(parent_id, name),
        *[_outcome(parent_id, number, prefix) for number in range(1, count + 1)],
    ]


def _select(evidence: list[CurriculumEvidence], question: str):
    return select_evidence_for_prompt(evidence, question=question, max_records=24)


def _outcome_ids(evidence: list[CurriculumEvidence]) -> set[str]:
    return {
        item.entity_id
        for item in evidence
        if (item.entity_type or "") == "learning_outcome" and item.entity_id
    }


def _understand(question: str) -> CurriculumQAState:
    state = CurriculumQAState.initial(question=question)
    GraphNodes.understand(None, {"qa": state, "visited_nodes": []})  # type: ignore[arg-type]
    return state


def test_selector_accepts_only_question_and_cap():
    parameters = inspect.signature(select_evidence_for_prompt).parameters
    assert list(parameters) == ["evidence", "question", "max_records", "question_purpose"]
    assert parameters["max_records"].default == 24
    assert parameters["question_purpose"].default is None
    source = inspect.getsource(select_evidence_for_prompt)
    assert "classify_question_purpose" not in source


def test_ordinary_questions_keep_intent_separate_from_purpose():
    expected = {
        _TOPIC_QUESTION: "topic_list",
        _TITLE_QUESTION: "unknown",
        _OUTCOME_QUESTION: "learning_outcome_list",
        _AMBIGUOUS_QUESTION: "unknown",
    }
    for question, purpose in expected.items():
        filters = extract_filters_from_question(question)
        assert set(filters) == {"grade", "subject", "level", "topic", "classification"}
        assert detect_sss_stream_subjects(question) is None
        state = _understand(question)
        assert state.intent == "retrieve_curriculum"
        assert "sss_focus" not in state.metadata
        assert state.metadata["question_purpose"] == purpose


def test_sss_stream_focus_is_not_an_evidence_window_input():
    detected = detect_sss_stream_subjects(_SSS_OUTCOMES)
    assert detected is not None
    assert detected.intent == INTENT_SSS_STREAM_SUBJECTS
    assert detected.focus == FOCUS_OUTCOMES
    state = _understand(_SSS_OUTCOMES)
    assert state.intent == INTENT_SSS_STREAM_SUBJECTS
    assert state.metadata["sss_focus"] == FOCUS_OUTCOMES
    assert state.metadata["question_purpose"] == "learning_outcome_list"
    assert "sss_focus" not in inspect.signature(select_evidence_for_prompt).parameters
    evidence = [*_family("parent-a", "Alpha", 5, "S1-T1-01"), *[_filler(index) for index in range(23)]]
    _selected, ids = _select(evidence, _OUTCOME_QUESTION)
    assert _outcome_ids(evidence).isdisjoint(ids)


def test_verifier_keyword_check_does_not_change_selection():
    assert question_requires_outcome_list(_OUTCOME_QUESTION) is True
    evidence = [*_family("parent-a", "Alpha", 5, "S1-T1-01"), *[_filler(index) for index in range(23)]]
    _selected, ids = _select(evidence, _OUTCOME_QUESTION)
    assert "parent-a" in ids
    assert _outcome_ids(evidence).isdisjoint(ids)


def test_topic_title_request_keeps_matching_topics():
    companions = [
        CurriculumEvidence(
            entity_type="topic",
            entity_id=f"alpha-title-{index:02d}",
            name=f"Alpha title record {index}",
            content=f"Alpha title record {index}",
            metadata={"parent_id": "parent-a"},
        )
        for index in range(23)
    ]
    evidence = [*_family("parent-a", "Alpha", 5, "S1-T1-01"), *companions]
    _selected, ids = _select(evidence, _TITLE_QUESTION)
    assert len(ids) == 24
    assert _outcome_ids(evidence).isdisjoint(ids)
    assert sum(entity_id.startswith("alpha-title-") for entity_id in ids) == 23


def test_general_topic_question_does_not_expand_children():
    evidence = [*_family("parent-a", "Alpha", 5, "S1-T1-01"), *[_filler(index) for index in range(23)]]
    _selected, ids = _select(evidence, _TOPIC_QUESTION)
    assert len(ids) == 24
    assert "parent-a" in ids
    assert _outcome_ids(evidence).isdisjoint(ids)


def test_ambiguous_wording_keeps_the_current_window():
    mentions = [
        CurriculumEvidence(
            entity_type="theme",
            entity_id=f"about-{index:02d}",
            name=f"About catalog {index}",
            content=f"About catalog {index}",
        )
        for index in range(23)
    ]
    evidence = [*_family("parent-a", "Alpha", 5, "S1-T1-01"), *mentions]
    _selected, ids = _select(evidence, _AMBIGUOUS_QUESTION)
    assert len(ids) == 24
    assert "parent-a" in ids
    assert _outcome_ids(evidence).isdisjoint(ids)


def test_two_parents_and_an_oversized_family_stay_inside_the_cap():
    evidence = [
        *_family("parent-a", "Alpha", 20, "S1-T1-01"),
        *_family("parent-b", "Beta", 20, "S1-T1-02"),
        *[_filler(index) for index in range(22)],
    ]
    question = "What topics are listed under the Alpha theme and the Beta theme?"
    first, first_ids = _select(evidence, question)
    _, second_ids = _select(evidence, question)
    assert len(first) == 24
    assert first_ids == second_ids
    assert "parent-a" in first_ids and "parent-b" in first_ids
    assert _outcome_ids(evidence).isdisjoint(first_ids)


def test_french_and_stream_routing_are_separate_from_the_window():
    french = _understand("What units does the Class 4 French subject cover?")
    assert french.intent == "retrieve_curriculum"
    assert french.subject == "FRENCH"
    assert french.metadata["question_purpose"] == "unknown"
    assert detect_sss_stream_subjects(french.question) is None
    stream = _understand("Which subjects are taught in the Science and Technology stream?")
    assert stream.intent == INTENT_SSS_STREAM_SUBJECTS
    assert stream.metadata["sss_focus"] == "subjects"
    assert stream.metadata["question_purpose"] == "unknown"


def test_selected_ids_match_the_verifier_prompt():
    evidence = [*_family("parent-a", "Alpha", 5, "S1-T1-01"), *[_filler(index) for index in range(23)]]
    state = CurriculumQAState.initial(question=_TOPIC_QUESTION)
    state.evidence = evidence
    state.draft_answer = "Alpha is the theme."
    messages = AnswerVerifier(StubLLMProvider()).build_messages(state)
    prompt_ids = state.metadata["verification_evidence_ids"]
    rendered = [
        line.removeprefix("Entity ID: ").strip()
        for line in (messages[-1].content or "").splitlines()
        if line.startswith("Entity ID: ")
    ]
    assert prompt_ids == rendered
    assert len(prompt_ids) == 24
    assert _outcome_ids(evidence).isdisjoint(prompt_ids)
