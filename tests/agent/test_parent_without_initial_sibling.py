"""Parent-only windows do not infer an outcome-list request.

Sibling fill starts only after one outcome from that parent is already inside
the lexical window. These tests record that boundary. They do not call a model.
"""

from __future__ import annotations

from app.agent.answer_generator import select_evidence_for_prompt
from app.agent.state import CurriculumQAState
from app.agent.verifier import AnswerVerifier
from app.curriculum.evidence import CurriculumEvidence
from app.llm.provider import StubLLMProvider

_TOPIC_QUESTION = "What topics are listed under the Alpha theme?"
_OUTCOME_QUESTION = "What outcomes are published under the Alpha theme?"
_LEARNING_QUESTION = "What are the learning outcomes under the Alpha theme?"
_TITLE_QUESTION = "What is the title of the Alpha theme?"


def _parent(parent_id: str, name: str) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="theme",
        entity_id=parent_id,
        name=name,
        content=name,
    )


def _outcome(parent_id: str, number: int) -> CurriculumEvidence:
    code = f"S1-T1-01-LO{number:02d}"
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


def _alpha_family(outcome_count: int = 5) -> list[CurriculumEvidence]:
    return [_parent("parent-a", "Alpha"), *[_outcome("parent-a", n) for n in range(1, outcome_count + 1)]]


def _select(evidence: list[CurriculumEvidence], question: str):
    return select_evidence_for_prompt(evidence, question=question, max_records=24)


def _outcome_ids(evidence: list[CurriculumEvidence]) -> set[str]:
    return {
        item.entity_id
        for item in evidence
        if (item.entity_type or "") == "learning_outcome" and item.entity_id
    }


def test_selected_parent_does_not_pull_outcomes_that_lost_the_lexical_window():
    evidence = [*_alpha_family(), *[_filler(index) for index in range(23)]]
    selected, ids = _select(evidence, _TOPIC_QUESTION)
    assert len(selected) == 24
    assert "parent-a" in ids
    assert _outcome_ids(evidence).isdisjoint(ids)


def test_learning_token_can_rank_outcomes_without_parent_fill():
    """'learning' matches the entity type. That is lexical overlap, not parent fill."""
    evidence = [*_alpha_family(), *[_filler(index) for index in range(23)]]
    _selected, ids = _select(evidence, _LEARNING_QUESTION)
    assert "parent-a" in ids
    assert _outcome_ids(evidence) <= set(ids)


def test_outcome_question_still_omits_siblings_when_none_ranked_in():
    evidence = [*_alpha_family(), *[_filler(index) for index in range(23)]]
    selected, ids = _select(evidence, _OUTCOME_QUESTION)
    assert len(selected) == 24
    assert "parent-a" in ids
    assert _outcome_ids(evidence).isdisjoint(ids)


def test_topic_title_question_does_not_spend_the_window_on_outcomes():
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
    evidence = [*_alpha_family(), *companions]
    _selected, ids = _select(evidence, _TITLE_QUESTION)
    assert len(ids) == 24
    assert "parent-a" in ids
    assert _outcome_ids(evidence).isdisjoint(ids)
    assert sum(1 for entity_id in ids if entity_id.startswith("alpha-title-")) == 23


def test_two_parents_compete_without_pulling_unranked_outcomes():
    evidence = [
        _parent("parent-a", "Alpha"),
        _parent("parent-b", "Beta"),
        *_alpha_family(8)[1:],
        *[
            CurriculumEvidence(
                entity_type="learning_outcome",
                entity_id=f"parent-b-lo-{number:02d}",
                name=f"S1-T1-02-LO{number:02d}",
                topic="parent-b",
                content="plain statement",
                metadata={"code": f"S1-T1-02-LO{number:02d}"},
            )
            for number in range(1, 9)
        ],
        *[_filler(index) for index in range(22)],
    ]
    question = "What topics are listed under the Alpha theme and the Beta theme?"
    selected, ids = _select(evidence, question)
    assert len(selected) == 24
    assert ids[:2] == ["parent-a", "parent-b"] or set(ids[:2]) == {"parent-a", "parent-b"}
    assert "parent-a" in ids and "parent-b" in ids
    assert _outcome_ids(evidence).isdisjoint(ids)


def test_oversized_family_stays_outside_when_no_sibling_ranked_in():
    evidence = [*_alpha_family(30), *[_filler(index) for index in range(23)]]
    selected, ids = _select(evidence, _OUTCOME_QUESTION)
    assert len(selected) == 24
    assert "parent-a" in ids
    assert _outcome_ids(evidence).isdisjoint(ids)


def test_irrelevant_family_does_not_displace_matching_alpha_records():
    alpha_topics = [
        CurriculumEvidence(
            entity_type="topic",
            entity_id=f"alpha-topic-{index:02d}",
            name=f"Alpha topic {index}",
            content=f"Alpha topic {index}",
            metadata={"parent_id": "parent-a"},
        )
        for index in range(8)
    ]
    gamma_outcomes = [
        CurriculumEvidence(
            entity_type="learning_outcome",
            entity_id=f"parent-g-lo-{number:02d}",
            name=f"S9-T9-09-LO{number:02d}",
            topic="parent-g",
            content="plain statement",
            metadata={"code": f"S9-T9-09-LO{number:02d}"},
        )
        for number in range(1, 21)
    ]
    evidence = [
        *alpha_topics,
        _parent("parent-a", "Alpha"),
        _parent("parent-g", "Gamma"),
        *gamma_outcomes,
        *[_filler(index) for index in range(20)],
    ]
    _selected, ids = _select(evidence, _TOPIC_QUESTION)
    assert len(ids) == 24
    assert "parent-a" in ids
    for index in range(8):
        assert f"alpha-topic-{index:02d}" in ids
    assert _outcome_ids(gamma_outcomes).isdisjoint(ids)


def test_identical_inputs_keep_the_same_ids_and_order():
    evidence = [*_alpha_family(12), *[_filler(index) for index in range(23)]]
    _, first = _select(evidence, _OUTCOME_QUESTION)
    _, second = _select(evidence, _OUTCOME_QUESTION)
    assert first == second
    assert len(first) == 24


def test_verifier_prompt_ids_match_the_records_in_the_prompt():
    evidence = [*_alpha_family(), *[_filler(index) for index in range(23)]]
    state = CurriculumQAState.initial(question=_TOPIC_QUESTION)
    state.evidence = evidence
    state.draft_answer = "Alpha is the theme."
    messages = AnswerVerifier(StubLLMProvider()).build_messages(state)
    prompt_ids = state.metadata["verification_evidence_ids"]
    prompt = messages[-1].content or ""
    rendered = [
        line.removeprefix("Entity ID: ").strip()
        for line in prompt.splitlines()
        if line.startswith("Entity ID: ")
    ]
    assert prompt_ids == rendered
    assert len(prompt_ids) == 24
    assert "parent-a" in prompt_ids
    assert _outcome_ids(evidence).isdisjoint(prompt_ids)
    assert select_evidence_for_prompt.__kwdefaults__["max_records"] == 24
