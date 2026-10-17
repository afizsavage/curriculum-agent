"""Purpose-aware evidence selection for an explicit outcome list."""

from __future__ import annotations

from app.agent.answer_generator import AnswerGenerator, select_evidence_for_prompt
from app.agent.graph_nodes import GraphNodes
from app.agent.question_purpose import (
    PURPOSE_LEARNING_OUTCOME_LIST,
    PURPOSE_TOPIC_LIST,
    PURPOSE_UNKNOWN,
    classify_question_purpose,
)
from app.agent.state import CurriculumQAState
from app.agent.verifier import AnswerVerifier
from app.curriculum.evidence import CurriculumEvidence
from app.curriculum.sss_stream_intent import INTENT_SSS_STREAM_SUBJECTS
from app.llm.provider import StubLLMProvider

_OUTCOME_QUESTION = "What outcomes are published under the Alpha theme?"
_TOPIC_QUESTION = "What topics are listed under the Alpha theme?"
_TITLE_QUESTION = "What is the title of the Alpha theme?"
_CLASS4_QUESTION = "What are the Class 4 Mathematics outcomes?"
_TIE_QUESTION = (
    "What outcomes are published under the Alpha theme and the Beta theme?"
)


def _parent(parent_id: str, name: str) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="theme",
        entity_id=parent_id,
        name=name,
        content=name,
    )


def _outcome(
    parent_id: str,
    number: int,
    *,
    prefix: str = "S1-T1-01",
    topic: str | None = None,
    metadata: dict | None = None,
) -> CurriculumEvidence:
    code = f"{prefix}-LO{number:02d}"
    return CurriculumEvidence(
        entity_type="learning_outcome",
        entity_id=f"{parent_id}-lo-{number:02d}",
        name=code,
        topic=parent_id if topic is None else topic,
        content="plain statement",
        metadata={"code": code} if metadata is None else metadata,
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
        *[_outcome(parent_id, number, prefix=prefix) for number in range(1, count + 1)],
    ]


def _select(evidence, question, purpose=None):
    return select_evidence_for_prompt(
        evidence,
        question=question,
        max_records=24,
        question_purpose=purpose,
    )


def _ids(evidence) -> list[str]:
    return [item.entity_id for item in evidence if item.entity_id]


def _outcome_ids(evidence, parent_id: str | None = None) -> set[str]:
    found = set()
    for item in evidence:
        if (item.entity_type or "") != "learning_outcome" or not item.entity_id:
            continue
        if parent_id is not None and item.topic != parent_id:
            continue
        found.add(item.entity_id)
    return found


def _understand(question: str) -> CurriculumQAState:
    state = CurriculumQAState.initial(question=question)
    GraphNodes.understand(None, {"qa": state, "visited_nodes": []})  # type: ignore[arg-type]
    return state


def _prompt_ids(content: str) -> list[str]:
    return [
        line.removeprefix("Entity ID: ").strip()
        for line in (content or "").splitlines()
        if line.startswith("Entity ID: ")
    ]


def test_outcome_list_admits_children_of_the_relevant_parent():
    evidence = [*_family("parent-a", "Alpha", 5, "S1-T1-01"), *[_filler(index) for index in range(23)]]
    selected, ids = _select(evidence, _OUTCOME_QUESTION, PURPOSE_LEARNING_OUTCOME_LIST)
    assert len(selected) == 24
    assert "parent-a" in ids
    assert _outcome_ids(evidence, "parent-a") <= set(ids)
    assert ids == _ids(selected)


def test_topic_title_and_topic_list_do_not_spend_the_window_on_outcomes():
    titles = [
        CurriculumEvidence(
            entity_type="topic",
            entity_id=f"alpha-title-{index:02d}",
            name=f"Alpha title record {index}",
            content=f"Alpha title record {index}",
            metadata={"parent_id": "parent-a"},
        )
        for index in range(23)
    ]
    titled = [*_family("parent-a", "Alpha", 5, "S1-T1-01"), *titles]
    _selected, title_ids = _select(titled, _TITLE_QUESTION, PURPOSE_UNKNOWN)
    assert _outcome_ids(titled).isdisjoint(title_ids)
    assert sum(entity_id.startswith("alpha-title-") for entity_id in title_ids) == 23

    topical = [*_family("parent-a", "Alpha", 5, "S1-T1-01"), *[_filler(index) for index in range(23)]]
    _selected, topic_ids = _select(topical, _TOPIC_QUESTION, PURPOSE_TOPIC_LIST)
    assert len(topic_ids) == 24
    assert "parent-a" in topic_ids
    assert _outcome_ids(topical).isdisjoint(topic_ids)


def test_class4_outcome_wording_stays_on_the_original_policy():
    evidence = [*_family("parent-a", "Alpha", 5, "S1-T1-01"), *[_filler(index) for index in range(23)]]
    assert classify_question_purpose(_CLASS4_QUESTION) == PURPOSE_UNKNOWN
    baseline, base_ids = _select(evidence, _CLASS4_QUESTION)
    _selected, ids = _select(
        evidence,
        _CLASS4_QUESTION,
        classify_question_purpose(_CLASS4_QUESTION),
    )
    assert ids == base_ids


def test_missing_blank_and_invalid_purposes_match_the_omitted_argument():
    evidence = [*_family("parent-a", "Alpha", 5, "S1-T1-01"), *[_filler(index) for index in range(23)]]
    _baseline, base_ids = _select(evidence, _OUTCOME_QUESTION)
    for purpose in (None, "", "   ", "unknown", "not-a-purpose", PURPOSE_TOPIC_LIST):
        _selected, ids = _select(evidence, _OUTCOME_QUESTION, purpose)
        assert ids == base_ids
        assert _outcome_ids(evidence).isdisjoint(ids)


def test_a_second_parent_does_not_contribute_its_outcomes():
    evidence = [
        *_family("parent-a", "Alpha", 5, "S1-T1-01"),
        *_family("parent-b", "Beta", 10, "S1-T1-02"),
        *[_filler(index) for index in range(23)],
    ]
    _selected, ids = _select(evidence, _OUTCOME_QUESTION, PURPOSE_LEARNING_OUTCOME_LIST)
    assert len(ids) == 24
    assert "parent-a" in ids
    assert _outcome_ids(evidence, "parent-a") <= set(ids)
    assert _outcome_ids(evidence, "parent-b").isdisjoint(ids)


def test_tied_parents_keep_the_existing_window():
    evidence = [
        *_family("parent-a", "Alpha", 5, "S1-T1-01"),
        *_family("parent-b", "Beta", 5, "S1-T1-02"),
        *[_filler(index) for index in range(22)],
    ]
    _baseline, base_ids = _select(evidence, _TIE_QUESTION)
    _selected, ids = _select(evidence, _TIE_QUESTION, PURPOSE_LEARNING_OUTCOME_LIST)
    assert ids == base_ids
    assert _outcome_ids(evidence).isdisjoint(ids)


def test_oversized_family_stays_inside_the_cap_in_code_order():
    evidence = [
        *_family("parent-a", "Alpha", 30, "S1-T1-01"),
        *[_filler(index) for index in range(23)],
    ]
    first, first_ids = _select(evidence, _OUTCOME_QUESTION, PURPOSE_LEARNING_OUTCOME_LIST)
    _, second_ids = _select(evidence, _OUTCOME_QUESTION, PURPOSE_LEARNING_OUTCOME_LIST)
    assert len(first) == 24
    assert first_ids == second_ids
    assert "parent-a" in first_ids
    kept_codes = {
        item.metadata["code"]
        for item in first
        if (item.entity_type or "") == "learning_outcome"
    }
    assert kept_codes == {f"S1-T1-01-LO{number:02d}" for number in range(1, 24)}
    assert "parent-a-lo-24" not in first_ids
    assert "parent-a-lo-30" not in first_ids


def test_missing_relationships_are_not_invented():
    loose = [
        _outcome("loose", number, topic="", metadata={"code": f"S1-T1-09-LO{number:02d}"})
        for number in range(1, 4)
    ]
    for item in loose:
        item.topic = None
        item.metadata = {"code": item.metadata["code"]}
    evidence = [_parent("parent-a", "Alpha"), *loose, *[_filler(index) for index in range(23)]]
    baseline, base_ids = _select(evidence, _OUTCOME_QUESTION)
    _selected, ids = _select(evidence, _OUTCOME_QUESTION, PURPOSE_LEARNING_OUTCOME_LIST)
    assert ids == base_ids
    assert _outcome_ids(evidence).isdisjoint(ids)


def test_unknown_purpose_keeps_sibling_fill():
    seeded = _outcome("parent-a", 1)
    seeded.content = "zeta detail"
    hidden = _outcome("parent-a", 2)
    evidence = [
        _parent("parent-a", "Alpha"),
        seeded,
        hidden,
        *[_filler(index) for index in range(23)],
    ]
    question = "What is zeta published under the Alpha theme?"
    _baseline, base_ids = _select(evidence, question)
    _selected, ids = _select(evidence, question, PURPOSE_UNKNOWN)
    assert ids == base_ids
    assert seeded.entity_id in ids
    assert hidden.entity_id in ids


def test_generation_and_verification_use_the_same_ids():
    evidence = [*_family("parent-a", "Alpha", 5, "S1-T1-01"), *[_filler(index) for index in range(23)]]
    state = CurriculumQAState.initial(question=_OUTCOME_QUESTION)
    state.evidence = evidence
    state.draft_answer = "The Alpha outcomes are published."
    state.metadata["question_purpose"] = PURPOSE_LEARNING_OUTCOME_LIST
    generation = AnswerGenerator(StubLLMProvider()).build_messages(state)
    verification = AnswerVerifier(StubLLMProvider()).build_messages(state)
    generated_ids = state.metadata["generation_evidence_ids"]
    verified_ids = state.metadata["verification_evidence_ids"]
    assert generated_ids == verified_ids
    assert _prompt_ids(generation[-1].content) == generated_ids
    assert _prompt_ids(verification[-1].content) == verified_ids
    assert len(generated_ids) == 24
    assert _outcome_ids(evidence, "parent-a") <= set(generated_ids)


def test_stream_routing_and_french_subject_are_unchanged():
    french = _understand("What units does the Class 4 French subject cover?")
    assert french.intent == "retrieve_curriculum"
    assert french.subject == "FRENCH"
    assert french.metadata["question_purpose"] == PURPOSE_UNKNOWN
    stream = _understand(
        "What are the learning outcomes for Biology in SSS 1 Science and Technology stream?"
    )
    assert stream.intent == INTENT_SSS_STREAM_SUBJECTS
    assert stream.metadata["sss_focus"] == "outcomes"
    assert stream.metadata["question_purpose"] == PURPOSE_LEARNING_OUTCOME_LIST
    assert stream.subject is None
