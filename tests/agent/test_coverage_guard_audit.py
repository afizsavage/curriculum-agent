"""Frozen adversarial checks for the coverage guard.

Supported representation is a grade heading plus a `*` or `-` bullet, or one
cited bold title, whose text matches one retrieved record. Numbered lists,
tables, and multiline items are not accepted as proof. A paraphrase is not
treated as the same learning outcome.
"""

import pytest

from app.agent.answer_generator import AnswerGenerator
from app.agent.coverage_guard import apply_coverage_guard, assess_coverage
from app.agent.verifier import AnswerVerifier
from app.exceptions import LLMProviderError
from app.agent.state import CurriculumQAState
from app.agent.verification_checks import run_deterministic_checks
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus, ToolCallRecord
from app.llm.provider import StubLLMProvider
from app.schemas.answer import AnswerEvidenceRef, GroundedAnswer

QUESTION = "What does Mathematics cover in classes 1, 2, and 3?"


def _unit(
    entity_id: str,
    name: str,
    grade: str,
    *,
    content: str | None = None,
    entity_type: str = "unit",
    subject: str = "MATHEMATICS",
    parent: str | None = None,
    curriculum_code: str | None = None,
) -> CurriculumEvidence:
    metadata: dict[str, str] = {}
    if parent:
        metadata["parent_display_name"] = parent
        metadata["parent_id"] = f"parent-{parent}"
    if curriculum_code:
        metadata["curriculum_code"] = curriculum_code
    return CurriculumEvidence(
        entity_type=entity_type,
        entity_id=entity_id,
        name=name,
        grade=grade,
        subject=subject,
        topic=name,
        content=name if content is None else content,
        metadata=metadata,
        source_reference="synthetic",
    )


def _state(
    evidence: list[CurriculumEvidence],
    *,
    question: str = QUESTION,
    status: EvidenceStatus = EvidenceStatus.FOUND,
    subject: str = "MATHEMATICS",
    history: list[ToolCallRecord] | None = None,
    prompt_ids: list[str] | None = None,
    curriculum_code: str | None = None,
) -> CurriculumQAState:
    state = CurriculumQAState.initial(question=question)
    state.subject = subject
    state.evidence = evidence
    state.evidence_status = status
    state.retrieval_history = history or []
    if prompt_ids is not None:
        state.metadata["generation_evidence_ids"] = prompt_ids
    if curriculum_code:
        state.metadata["curriculum_code"] = curriculum_code
    return state


def _refs(items: list[CurriculumEvidence]) -> list[AnswerEvidenceRef]:
    return [
        AnswerEvidenceRef(
            entity_id=item.entity_id or "",
            entity_type=item.entity_type,
            claim=item.name or "",
            name=item.name,
            grade=item.grade,
            subject=item.subject,
        )
        for item in items
        if item.entity_id
    ]


def _catalogue(items: list[CurriculumEvidence], *, claim: str = "") -> str:
    lines = [claim] if claim else []
    current = None
    for item in items:
        if item.grade != current:
            current = item.grade
            number = (item.grade or "CLASS_1").split("_")[1]
            lines.append(f"### Primary {number}")
        lines.append(f"* {item.name}")
    return "\n".join(lines)


def _assert_invariant(result) -> None:
    represented = set(result.represented_ids)
    omitted = set(result.omitted_ids)
    assert represented.isdisjoint(omitted)
    assert len(represented) + len(omitted) == result.distinct_retrieved_record_count
    for bucket in result.by_grade.values():
        assert bucket["retrieved"] == bucket["represented"] + bucket["omitted"]


def _guard(state: CurriculumQAState, answer: str, refs: list[AnswerEvidenceRef]):
    return apply_coverage_guard(
        state,
        GroundedAnswer(answer=answer, evidence=refs, limitations=[]),
    )


def test_completeness_phrases_are_rejected_only_when_they_claim_the_catalogue():
    evidence = [
        _unit("c1", "Counting", "CLASS_1"),
        _unit("c2", "Time", "CLASS_2"),
        _unit("c3", "Shapes", "CLASS_3"),
    ]
    shown = evidence[:2]
    claims = {
        "All the topics are listed below.": True,
        "Here are all the topics.": True,
        "The subject covers the following topics.": True,
        "These are the complete topics for the three classes.": True,
        "The full curriculum consists of these units.": True,
        "The following is a comprehensive list.": True,
        "This is the complete list.": True,
    }
    scoped_ok = {
        "Here are the main topics.": False,
        "Here are some examples.": False,
        "This is a partial list, not a complete catalogue.": False,
    }
    for phrase, expects_claim in claims.items():
        result = assess_coverage(
            _state(evidence),
            answer=_catalogue(shown, claim=phrase),
            refs=_refs(shown),
        )
        assert result.claims_complete_coverage is expects_claim, phrase
        assert result.decision == "reject", phrase
        assert any("Completeness claim" in reason for reason in result.reasons), phrase
        _assert_invariant(result)
    for phrase, expects_claim in scoped_ok.items():
        result = assess_coverage(
            _state(evidence),
            answer=_catalogue(shown, claim=phrase),
            refs=_refs(shown),
        )
        assert result.claims_complete_coverage is expects_claim, phrase
        assert result.decision == "pass_partial", phrase
        assert result.discloses_partial_scope is True, phrase


def test_grade_scoped_complete_list_does_not_claim_the_other_grades():
    evidence = [
        _unit("c1", "Counting", "CLASS_1"),
        _unit("c2", "Time", "CLASS_2"),
        _unit("c3", "Shapes", "CLASS_3"),
    ]
    answer = "This is the complete list for Class 3.\n### Primary 3\n* Shapes"
    result = assess_coverage(_state(evidence), answer=answer, refs=_refs(evidence[2:]))
    assert result.claims_complete_coverage is False
    assert result.grade_scope_ok is True
    assert result.decision == "pass_partial"
    assert result.by_grade["CLASS_3"]["omitted"] == 0
    assert result.by_grade["CLASS_1"]["omitted"] == 1

    short = "This is the complete list for Class 3.\n### Primary 3\n* Other"
    rejected = assess_coverage(_state(evidence), answer=short, refs=_refs(evidence[2:]))
    assert rejected.decision == "reject"
    assert rejected.grade_scope_ok is False
    assert rejected.claims_complete_coverage is False


def test_supported_list_formats_count_and_unsupported_formats_do_not():
    evidence = [
        _unit("c1", "Counting", "CLASS_1"),
        _unit("c2", "Time", "CLASS_2"),
    ]
    accepted = [
        "### Primary 1\n* Counting\n### Primary 2\n* Time",
        "### Primary 1\n- Counting\n### Primary 2\n- Time",
        "### Primary 1\n  - Counting\n### Primary 2\n  - Time",
        "### Primary 1\nThe curriculum includes **Counting**.\n### Primary 2\nThe curriculum includes **Time**.",
        "# Primary 1\n## Topics\n* Counting\n# Primary 2\n* Time",
        "### Primary 1\n* **Counting**\n### Primary 2\n* **Time**",
        "### Primary 1\n* Counting\n### Primary 2\n* Time  ",
    ]
    for answer in accepted:
        result = assess_coverage(_state(evidence), answer=answer, refs=_refs(evidence))
        assert result.decision == "pass", answer
        assert result.represented_record_count == 2

    spaced = _unit("space", "Number and Numeration", "CLASS_1")
    spaced_result = assess_coverage(
        _state([spaced]),
        answer="### Primary 1\n* Number   and\tNumeration",
        refs=_refs([spaced]),
    )
    assert spaced_result.represented_ids == ["space"]

    unproven = [
        "Here are all the topics.\n### Primary 1\n1. Counting\n### Primary 2\n2. Time",
        "Here are all the topics.\n| Grade | Title |\n| Primary 1 | Counting |\n| Primary 2 | Time |",
    ]
    for answer in unproven:
        result = assess_coverage(_state(evidence), answer=answer, refs=_refs(evidence))
        assert result.decision == "reject", answer
        assert result.represented_record_count == 0
        guarded = _guard(_state(evidence), answer, _refs(evidence))
        after = assess_coverage(_state(evidence), answer=guarded.answer, refs=list(guarded.evidence))
        assert after.decision == "pass", guarded.answer
        assert "Counting" in guarded.answer and "Time" in guarded.answer
        assert "Here are all the topics" not in guarded.answer
    multiline = "Here are all the topics.\n### Primary 1\n* Number and\nNumeration\n### Primary 2\n* Time"
    split_title = assess_coverage(_state(evidence), answer=multiline, refs=_refs(evidence))
    assert split_title.decision == "reject"
    assert split_title.represented_ids == ["c2"]


def test_punctuation_case_parents_and_grades_stay_distinct():
    evidence = [
        _unit("count-upper", "Number and Numeration Counting", "CLASS_1"),
        _unit("count-lower", "Number and Numeration counting", "CLASS_1"),
        _unit("fraction", "Fraction", "CLASS_3"),
        _unit("fraction-dot", "Fraction.", "CLASS_3"),
        _unit("patterns-upper", "NUMBER PATTERNS", "CLASS_2"),
        _unit("patterns-lower", "Number patterns", "CLASS_2"),
        _unit("time-number", "Time", "CLASS_1", parent="Number"),
        _unit("time-measure", "Time", "CLASS_1", parent="Measurement"),
        _unit("apostrophe", "Pupils' work", "CLASS_2"),
        _unit("colon", "Time: hours", "CLASS_2"),
        _unit("hyphen", "Co-ordinates", "CLASS_2"),
        _unit("slash", "km/h", "CLASS_2"),
    ]
    lines = [
        "### Primary 1",
        "**Number**",
        "* Number and Numeration Counting",
        "* Number and Numeration counting",
        "**Measurement**",
        "* Time",
        "### Primary 2",
        "* NUMBER PATTERNS",
        "* Number patterns",
        "* Pupils' work",
        "* Time: hours",
        "* Co-ordinates",
        "* km/h",
        "### Primary 3",
        "* Fraction",
        "* Fraction.",
    ]
    # The second Class 1 Time is under Number as well as Measurement.
    lines.insert(3, "* Time")
    result = assess_coverage(_state(evidence), answer="\n".join(lines), refs=_refs(evidence))
    assert result.decision == "pass", result.reasons
    assert result.represented_record_count == len(evidence)
    merged = assess_coverage(
        _state(evidence),
        answer="Here are all the topics.\n### Primary 3\n* Fraction",
        refs=_refs([evidence[2]]),
    )
    assert "fraction-dot" in merged.omitted_ids
    assert merged.decision == "reject"


def test_citation_attacks_do_not_create_coverage():
    counting = _unit("counting", "Counting", "CLASS_1")
    time_1 = _unit("time-1", "Time", "CLASS_1")
    time_2 = _unit("time-2", "Time", "CLASS_2")
    shapes_a = _unit("shapes-a", "Shapes", "CLASS_1")
    shapes_b = _unit("shapes-b", "Shapes", "CLASS_1")
    english = _unit("english-shapes", "Shapes", "CLASS_1", subject="ENGLISH")
    other_curriculum = _unit(
        "ssc-shapes",
        "Shapes",
        "CLASS_1",
        curriculum_code="MBSSE-SSC@2021",
    )
    evidence = [counting, time_1, time_2, shapes_a, shapes_b, english, other_curriculum]
    state = _state(evidence, curriculum_code="MBSSE-BEC@2020")

    wrong_record = assess_coverage(
        state,
        answer="Here are all the topics.\n### Primary 1\n* Time",
        refs=_refs([counting]),
    )
    assert wrong_record.represented_record_count == 0
    assert wrong_record.decision == "reject"

    wrong_grade = assess_coverage(
        state,
        answer="### Primary 1\n* Time\n### Primary 2\n* Time",
        refs=_refs([time_1]),
    )
    assert wrong_grade.represented_ids == ["time-1"]
    assert "time-2" in wrong_grade.omitted_ids

    duplicated = _refs([counting]) + _refs([counting]) + _refs([counting])
    once = assess_coverage(
        state,
        answer="### Primary 1\n* Counting\n* Counting\n* Counting",
        refs=duplicated,
    )
    assert once.represented_ids == ["counting"]

    unknown = assess_coverage(
        state,
        answer="### Primary 1\n* Counting",
        refs=[
            AnswerEvidenceRef(
                entity_id="missing-id",
                entity_type="unit",
                claim="Counting",
                name="Counting",
                grade="CLASS_1",
            )
        ],
    )
    assert unknown.represented_record_count == 0

    prose_only = assess_coverage(
        state,
        answer="Here are all the topics. Counting is important.",
        refs=_refs([counting]),
    )
    assert prose_only.represented_record_count == 0
    assert prose_only.decision == "reject"

    wrong_ids = assess_coverage(
        state,
        answer=(
            "### Primary 1\n* Shapes (2 separate records share this exact title "
            "and have no further description)"
        ),
        refs=_refs([counting, time_1]),
    )
    assert wrong_ids.decision == "reject"
    assert wrong_ids.multiplicity_ok is False
    assert any("do not match that title" in reason for reason in wrong_ids.reasons)

    foreign = assess_coverage(
        state,
        answer="### Primary 1\n* Shapes",
        refs=_refs([english, other_curriculum]),
    )
    assert "english-shapes" not in foreign.represented_ids
    assert "ssc-shapes" not in foreign.represented_ids
    assert "shapes-a" not in foreign.represented_ids


def test_grade_counts_obey_the_accounting_invariant():
    evidence = [
        _unit("a1", "Alpha", "CLASS_1"),
        _unit("a2", "Beta", "CLASS_1"),
        _unit("a3", "Gamma", "CLASS_1"),
        _unit("b1", "Delta", "CLASS_2"),
        _unit("c1", "Epsilon", "CLASS_3"),
        _unit("c2", "Zeta", "CLASS_3"),
        _unit("c3", "Eta", "CLASS_3"),
        _unit("c4", "Theta", "CLASS_3"),
        _unit("c5", "Iota", "CLASS_3"),
    ]
    full = _catalogue(evidence)
    hidden_grade = "\n".join(
        [
            full,
            "This answer includes 9 of 9 retrieved records "
            "(Primary 1: 4 of 3; Primary 2: 1 of 1; Primary 3: 4 of 5). "
            "It does not include every retrieved record.",
        ]
    )
    wrong_grades = assess_coverage(_state(evidence), answer=hidden_grade, refs=_refs(evidence))
    assert wrong_grades.represented_record_count == 9
    assert wrong_grades.stated_counts_match is False
    assert wrong_grades.decision == "reject"
    _assert_invariant(wrong_grades)

    wrong_total = "\n".join(
        [
            full,
            "This answer includes 8 of 9 retrieved records "
            "(Primary 1: 3 of 3; Primary 2: 1 of 1; Primary 3: 5 of 5). "
            "It does not include every retrieved record.",
        ]
    )
    total = assess_coverage(_state(evidence), answer=wrong_total, refs=_refs(evidence))
    assert total.decision == "reject"
    assert total.by_grade["CLASS_1"]["represented"] == 3

    omitted = [item for item in evidence if item.entity_id != "c5"]
    honest = "\n".join(
        [
            _catalogue(omitted),
            "This answer includes 8 of 9 retrieved records "
            "(Primary 1: 3 of 3; Primary 2: 1 of 1; Primary 3: 4 of 5). "
            "It does not include every retrieved record.",
        ]
    )
    partial = assess_coverage(_state(evidence), answer=honest, refs=_refs(omitted))
    assert partial.decision == "pass_partial"
    _assert_invariant(partial)

    misplaced = assess_coverage(
        _state(evidence),
        answer="Here are all the topics.\n### Primary 1\n* Delta",
        refs=_refs([evidence[3]]),
    )
    assert misplaced.represented_record_count == 0
    assert misplaced.by_grade["CLASS_2"]["represented"] == 0
    assert misplaced.decision == "reject"

    duplicated = assess_coverage(
        _state(evidence[:1]),
        answer="### Primary 1\n* Alpha\n* Alpha",
        refs=_refs(evidence[:1]),
    )
    assert duplicated.represented_ids == ["a1"]
    assert duplicated.decision == "pass"

    copies = [_unit("s1", "Shapes", "CLASS_1"), _unit("s2", "Shapes", "CLASS_1")]
    grouped = (
        "### Primary 1\n* Shapes (2 separate records share this exact title "
        "and have no further description)"
    )
    exact = assess_coverage(_state(copies), answer=grouped, refs=_refs(copies))
    assert exact.decision == "pass"
    assert exact.represented_ids == ["s1", "s2"]
    _assert_invariant(exact)

    low = assess_coverage(
        _state(copies),
        answer=(
            "### Primary 1\n* Shapes (1 separate records share this exact title "
            "and have no further description)"
        ),
        refs=_refs(copies[:1]),
    )
    high = assess_coverage(
        _state(copies),
        answer=(
            "### Primary 1\n* Shapes (3 separate records share this exact title "
            "and have no further description)"
        ),
        refs=_refs(copies),
    )
    assert low.decision == "reject" and high.decision == "reject"

    similar = [
        _unit("f1", "Fraction", "CLASS_3"),
        _unit("f2", "Fraction.", "CLASS_3"),
    ]
    collapsed = assess_coverage(
        _state(similar),
        answer="Here are all the topics.\n### Primary 3\n* Fraction",
        refs=_refs(similar[:1]),
    )
    assert collapsed.represented_ids == ["f1"]
    assert "f2" in collapsed.omitted_ids
    assert collapsed.decision == "reject"

    scope_lie = "\n".join(
        [
            full,
            "This answer includes 3 of 3 retrieved records "
            "(Primary 1: 3 of 3; Primary 2: 0 of 0; Primary 3: 0 of 0). "
            "It does not include every retrieved record.",
        ]
    )
    lied = assess_coverage(_state(evidence), answer=scope_lie, refs=_refs(evidence))
    assert lied.represented_record_count == 9
    assert lied.stated_counts_match is False
    assert lied.decision == "reject"


def test_titles_and_learning_outcomes_are_counted_separately():
    titles = [
        _unit("count-upper", "Number and Numeration Counting", "CLASS_1"),
        _unit("count-lower", "Number and Numeration counting", "CLASS_1"),
        _unit("time-number", "Time", "CLASS_1", parent="Number"),
        _unit("time-measure", "Time", "CLASS_1", parent="Measurement"),
    ]
    title_only = assess_coverage(
        _state(titles),
        answer=(
            "### Primary 1\n**Number**\n* Number and Numeration Counting\n* Time\n"
            "**Measurement**\n* Time\n* Number and Numeration counting\n"
            "Pupils learn to:\n* Pupils learn to count."
        ),
        refs=_refs(titles),
    )
    assert title_only.title_represented_count == 4
    assert title_only.outcome_represented_count == 0
    assert title_only.decision == "reject"
    assert title_only.unsupported_outcomes

    outcomes = [
        _unit(
            "lo-1",
            "count from 1 to 10",
            "CLASS_1",
            content="count from 1 to 10",
            entity_type="learning_outcome",
        ),
        _unit(
            "lo-2",
            "count from 1 to 10",
            "CLASS_2",
            content="count from 1 to 10",
            entity_type="learning_outcome",
        ),
        _unit(
            "lo-3",
            "extend a pattern",
            "CLASS_1",
            content="extend a pattern",
            entity_type="learning_outcome",
        ),
    ]
    exact = assess_coverage(
        _state(outcomes[:1]),
        answer="### Primary 1\nPupils learn to:\n* Count from 1 to 10.",
        refs=_refs(outcomes[:1]),
    )
    assert exact.outcome_represented_count == 1
    assert exact.title_represented_count == 0
    assert exact.decision == "pass"

    paraphrase = assess_coverage(
        _state(outcomes[:1]),
        answer=(
            "Here are all the topics.\n### Primary 1\nPupils learn to:\n"
            "* Pupils learn to count objects up to ten."
        ),
        refs=_refs(outcomes[:1]),
    )
    assert paraphrase.outcome_represented_count == 0
    assert paraphrase.decision == "reject"

    changed = assess_coverage(
        _state(outcomes[:1]),
        answer="### Primary 1\nPupils learn to:\n* Skip count by fives.",
        refs=_refs(outcomes[:1]),
    )
    assert changed.outcome_represented_count == 0
    assert changed.decision == "reject"

    invented = assess_coverage(
        _state(titles[:1]),
        answer=(
            "### Primary 1\n* Number and Numeration Counting\n"
            "Pupils learn to:\n* Invent a fraction skill."
        ),
        refs=_refs(titles[:1]),
    )
    assert invented.title_represented_count == 1
    assert invented.outcome_represented_count == 0
    assert invented.decision == "reject"

    wrong_source = assess_coverage(
        _state(outcomes),
        answer="### Primary 1\nPupils learn to:\n* Extend a pattern.",
        refs=_refs(outcomes[:1]),
    )
    assert wrong_source.outcome_represented_count == 0

    both = assess_coverage(
        _state(outcomes[1:]),
        answer=(
            "### Primary 1\nPupils learn to:\n* Extend a pattern.\n"
            "### Primary 2\nPupils learn to:\n* Count from 1 to 10."
        ),
        refs=_refs([outcomes[2], outcomes[1]]),
    )
    assert both.outcome_represented_count == 2
    assert both.decision == "pass"

    identical = assess_coverage(
        _state(outcomes[:2]),
        answer=(
            "### Primary 1\nPupils learn to:\n* Count from 1 to 10.\n"
            "### Primary 2\nPupils learn to:\n* Count from 1 to 10."
        ),
        refs=_refs(outcomes[:2]),
    )
    assert identical.outcome_represented_count == 2

    same_grade_copies = [
        outcomes[0],
        _unit(
            "lo-1b",
            "count from 1 to 10",
            "CLASS_1",
            content="count from 1 to 10",
            entity_type="learning_outcome",
        ),
    ]
    ambiguous = assess_coverage(
        _state(same_grade_copies),
        answer="Here are all the topics.\n### Primary 1\nPupils learn to:\n* Count from 1 to 10.",
        refs=_refs(same_grade_copies),
    )
    assert ambiguous.outcome_represented_count == 0
    assert ambiguous.decision == "reject"


def test_fallback_matches_the_source_inventory_and_not_the_prompt_window():
    evidence = []
    for grade, count in (("CLASS_1", 13), ("CLASS_2", 12), ("CLASS_3", 13)):
        for index in range(count):
            evidence.append(_unit(f"{grade}-{index}", f"{grade} topic {index}", grade))
    prompt_ids = []
    for grade in ("CLASS_1", "CLASS_2", "CLASS_3"):
        prompt_ids.extend(
            [item.entity_id for item in evidence if item.grade == grade][:8]
        )
    state = _state(evidence, prompt_ids=prompt_ids)
    guarded = _guard(state, "Here are all the topics.\n1. Something else", [])
    after = assess_coverage(state, answer=guarded.answer, refs=list(guarded.evidence))
    assert after.decision == "pass_partial"
    assert after.represented_record_count == 24
    assert after.retrieved_record_count == 38
    assert "24 of 38" in guarded.answer
    assert "does not include every retrieved record" in guarded.answer
    assert "Here are all the topics" not in guarded.answer
    assert after.claims_complete_coverage is False
    _assert_invariant(after)
    assert {ref.entity_id for ref in guarded.evidence} == set(prompt_ids)

    wide = []
    for grade, count in (("CLASS_1", 32), ("CLASS_2", 30), ("CLASS_3", 39)):
        for index in range(count):
            wide.append(_unit(f"w-{grade}-{index}", f"{grade} unit {index}", grade))
    wide_ids = []
    for grade in ("CLASS_1", "CLASS_2", "CLASS_3"):
        wide_ids.extend([item.entity_id for item in wide if item.grade == grade][:8])
    wide_state = _state(wide, prompt_ids=wide_ids)
    wide_guard = _guard(wide_state, "The following is a comprehensive list.", [])
    wide_after = assess_coverage(
        wide_state, answer=wide_guard.answer, refs=list(wide_guard.evidence)
    )
    assert wide_after.represented_record_count == 24
    assert wide_after.retrieved_record_count == 101
    assert "24 of 101" in wide_guard.answer
    assert "comprehensive list" not in wide_guard.answer.lower()

    copies = [_unit("s1", "Shapes", "CLASS_1"), _unit("s2", "Shapes", "CLASS_1")]
    similar = [
        _unit("f1", "Fraction", "CLASS_3"),
        _unit("f2", "Fraction.", "CLASS_3"),
        _unit("u1", "NUMBER PATTERNS", "CLASS_2"),
        _unit("u2", "Number patterns", "CLASS_2"),
    ]
    parents = [
        _unit("t1", "Time", "CLASS_1", parent="Number"),
        _unit("t2", "Time", "CLASS_1", parent="Measurement"),
        _unit("plain", "Counting", "CLASS_2"),
    ]
    long_text = "extend a pattern " * 40
    assert len(long_text) > 500
    long_row = _unit("long", "Patterns", "CLASS_3", content=long_text.strip())
    outcome = _unit(
        "lo",
        "count from 1 to 10",
        "CLASS_1",
        content="count from 1 to 10",
        entity_type="learning_outcome",
    )
    mixed = copies + similar + parents + [long_row, outcome]
    mixed_state = _state(mixed)
    mixed_guard = _guard(mixed_state, "Here are all the topics.", [])
    mixed_after = assess_coverage(
        mixed_state, answer=mixed_guard.answer, refs=list(mixed_guard.evidence)
    )
    assert mixed_after.decision == "pass", mixed_after.reasons
    assert mixed_after.represented_record_count == len(mixed)
    assert "2 separate records" in mixed_guard.answer
    assert "* Fraction\n" in mixed_guard.answer or mixed_guard.answer.endswith("* Fraction")
    assert "* Fraction." in mixed_guard.answer
    assert "**Number**" in mixed_guard.answer
    assert "**Measurement**" in mixed_guard.answer
    assert long_text.strip() in mixed_guard.answer
    assert "Pupils learn to:" in mixed_guard.answer
    assert "Count from 1 to 10." in mixed_guard.answer
    assert "Pupils learn to count" not in mixed_guard.answer
    counting_section = mixed_guard.answer.split("### Primary 2")[1].split("###")[0]
    assert "**" not in counting_section


def test_retrieval_states_are_not_turned_into_a_catalogue():
    counting = _unit("c1", "Counting", "CLASS_1")
    found = _guard(_state([counting]), "### Primary 1\n* Counting", _refs([counting]))
    assert "Counting" in found.answer
    assert assess_coverage(_state([counting]), answer=found.answer, refs=list(found.evidence)).decision == "pass"

    history_error = [
        ToolCallRecord(
            tool="resolve_curriculum_context",
            arguments={"grade": "CLASS_2"},
            status="error",
            error="timed out",
        )
    ]
    partial_state = _state(
        [counting],
        status=EvidenceStatus.PARTIAL,
        history=history_error,
    )
    partial = _guard(
        partial_state,
        "Here are all the topics.\n### Primary 1\n* Counting",
        _refs([counting]),
    )
    assert "could not be completed" in partial.answer
    assert "No syllabus was found" not in partial.answer
    assert "Here are all the topics" not in partial.answer
    partial_after = assess_coverage(
        partial_state, answer=partial.answer, refs=list(partial.evidence)
    )
    assert partial_after.decision == "pass"
    assert partial_after.claims_complete_coverage is False

    absent_history = [
        ToolCallRecord(
            tool="get_curriculum_structure",
            arguments={"grade": "CLASS_2"},
            status="not_found",
            error="No syllabus found",
        )
    ]
    absent_state = _state([counting], history=absent_history)
    absent = _guard(absent_state, "### Primary 1\n* Counting", _refs([counting]))
    assert "No syllabus was found for CLASS_2" in absent.answer
    assert "could not be checked" not in absent.answer.lower()

    error_state = _state([], status=EvidenceStatus.ERROR, history=history_error)
    error = _guard(error_state, "Here are all the topics.\n* Counting", [])
    assert "Counting" not in error.answer
    assert "could not be checked" in error.answer.lower() or "could not be completed" in error.answer
    assert "no syllabus was found" not in error.answer.lower()

    absence_as_timeout = _guard(
        _state([], status=EvidenceStatus.NOT_FOUND),
        "The curriculum service timed out or was unavailable.",
        [],
    )
    assert "no syllabus was found" in absence_as_timeout.answer.lower()
    assert "timed out" not in absence_as_timeout.answer.lower()

    timeout_as_absence = _guard(
        _state([], status=EvidenceStatus.ERROR),
        "No syllabus was found for Class 1.\n* Counting",
        [],
    )
    assert "could not be checked" in timeout_as_absence.answer.lower()
    assert "no syllabus was found" not in timeout_as_absence.answer.lower()
    assert "Counting" not in timeout_as_absence.answer


def test_direct_renderer_still_accounts_for_title_only_records():
    evidence = [
        _unit("c1", "Counting", "CLASS_1"),
        _unit("c2", "Time", "CLASS_2"),
        _unit("c3", "Shapes", "CLASS_3"),
    ]
    state = _state(evidence)
    result = AnswerGenerator(StubLLMProvider()).generate(state)
    coverage = assess_coverage(state, answer=result.answer, refs=list(result.evidence))
    assert coverage.decision == "pass"
    assert coverage.represented_record_count == 3
    assert coverage.title_represented_count == 3
    assert coverage.outcome_represented_count == 0
    state.final_answer = result.answer
    state.answer_evidence = list(result.evidence)
    verification = run_deterministic_checks(state)
    assert verification.passed is True


class _TimeoutLLM:
    name = "timeout-harness"
    model = "timeout-harness"

    def generate_structured(self, messages, *, schema, temperature=0.0):
        raise LLMProviderError("request timed out")


def _ict_units(count: int = 24) -> list[CurriculumEvidence]:
    return [
        _unit(f"ict-{index:02d}", f"ICT topic {index:02d}", "CLASS_1", subject="ICT_LITERACY")
        for index in range(1, count + 1)
    ]


def test_subject_completeness_wording_is_not_returned_unchanged():
    evidence = _ict_units(24)
    shown = evidence[:23]
    ict = {"subject": "ICT_LITERACY"}
    bullets = _catalogue(shown)
    claim = "All subjects are listed below.\n" + bullets
    before = assess_coverage(_state(evidence, **ict), answer=claim, refs=_refs(shown))
    assert before.decision == "reject"
    assert before.claims_complete_coverage is True
    guarded = _guard(_state(evidence, **ict), claim, _refs(shown))
    assert guarded.answer != claim
    assert "All subjects are listed below" not in guarded.answer
    after = assess_coverage(_state(evidence, **ict), answer=guarded.answer, refs=list(guarded.evidence))
    assert after.decision == "pass_partial"
    assert after.represented_record_count == 23
    assert "23 of 24" in guarded.answer
    _assert_invariant(after)

    complete = "All subjects are listed below.\n" + _catalogue(evidence)
    accepted = assess_coverage(_state(evidence, **ict), answer=complete, refs=_refs(evidence))
    assert accepted.decision == "pass"
    assert accepted.represented_record_count == 24

    negated = assess_coverage(
        _state(evidence, **ict),
        answer="Not all subjects are listed below.\n" + bullets,
        refs=_refs(shown),
    )
    assert negated.claims_complete_coverage is False


def test_numbered_and_count_mismatches_do_not_pass_as_complete():
    evidence = _ict_units(24)
    shown = evidence[:23]
    ict = {"subject": "ICT_LITERACY"}
    numbered = "### Primary 1\n" + "\n".join(
        f"{index}. {item.name}" for index, item in enumerate(shown, start=1)
    )
    silent = assess_coverage(_state(evidence, **ict), answer=numbered, refs=_refs(shown))
    assert silent.decision == "reject"
    assert silent.represented_record_count == 0
    assert silent.silent_catalogue_omission is True
    guarded = _guard(_state(evidence, **ict), numbered, _refs(shown))
    assert guarded.answer != numbered
    after = assess_coverage(_state(evidence, **ict), answer=guarded.answer, refs=list(guarded.evidence))
    assert after.decision == "pass"
    assert after.represented_record_count == 24
    assert "ICT topic 24" in guarded.answer
    assert {ref.entity_id for ref in guarded.evidence} == {item.entity_id for item in evidence}

    claimed = (
        "The complete list contains 24 subjects.\n"
        + _catalogue(shown)
        + "\nThis answer includes 24 of 24 retrieved records (Primary 1: 24 of 24). "
        "It does not include every retrieved record."
    )
    counts = assess_coverage(_state(evidence, **ict), answer=claimed, refs=_refs(shown))
    assert counts.decision == "reject"
    assert counts.represented_record_count == 23
    assert counts.stated_counts_match is False
    fixed = _guard(_state(evidence, **ict), claimed, _refs(shown))
    fixed_after = assess_coverage(_state(evidence, **ict), answer=fixed.answer, refs=list(fixed.evidence))
    assert fixed_after.decision == "pass_partial"
    assert "24 of 24" not in fixed.answer
    assert "23 of 24" in fixed.answer
    assert "The complete list contains 24 subjects" not in fixed.answer

    duplicated = _catalogue(evidence) + "\n* ICT topic 01"
    reordered = list(reversed(evidence))
    same = assess_coverage(_state(evidence, **ict), answer=_catalogue(reordered), refs=_refs(evidence))
    assert same.decision == "pass"
    assert same.represented_record_count == 24
    again = assess_coverage(_state(evidence, **ict), answer=duplicated, refs=_refs(evidence))
    assert again.represented_record_count == 24
    _assert_invariant(again)


def test_truthful_partial_list_is_kept():
    evidence = _ict_units(24)
    shown = evidence[:23]
    ict = {"subject": "ICT_LITERACY"}
    answer = (
        _catalogue(shown)
        + "\nThis is a partial list, not a complete catalogue.\n"
        "This answer includes 23 of 24 retrieved records (Primary 1: 23 of 24). "
        "It does not include every retrieved record."
    )
    result = assess_coverage(_state(evidence, **ict), answer=answer, refs=_refs(shown))
    assert result.decision == "pass_partial"
    guarded = _guard(_state(evidence, **ict), answer, _refs(shown))
    assert guarded.answer == answer


def test_a_title_citation_does_not_prove_an_outcome():
    title = _unit("upper", "Number and Numeration Counting", "CLASS_1")
    similar = _unit("lower", "Number and Numeration counting", "CLASS_1")
    outcome = _unit(
        "outcome",
        "count from 1 to 10",
        "CLASS_1",
        content="count from 1 to 10",
        entity_type="learning_outcome",
    )
    evidence = [title, similar, outcome]
    answer = (
        "### Primary 1\n* Number and Numeration Counting\n"
        "Pupils learn to:\n* Pupils learn to count."
    )
    before = assess_coverage(_state(evidence), answer=answer, refs=_refs([title]))
    assert before.title_represented_count == 1
    assert before.outcome_represented_count == 0
    assert before.decision == "reject"
    guarded = _guard(_state(evidence), answer, _refs([title]))
    assert "Pupils learn to count" not in guarded.answer
    assert "count from 1 to 10" not in guarded.answer.lower()
    after = assess_coverage(_state(evidence), answer=guarded.answer, refs=list(guarded.evidence))
    assert after.outcome_represented_count == 0
    assert after.decision == "pass_partial"

    wrong = (
        "### Primary 1\n* Number and Numeration counting\n"
        "Pupils learn to:\n* Count from 1 to 10."
    )
    mismatched = assess_coverage(_state(evidence), answer=wrong, refs=_refs([similar]))
    assert mismatched.represented_ids == ["lower"]
    assert "outcome" in mismatched.omitted_ids


def test_subject_lists_are_not_replaced_by_the_unit_fallback():
    subjects = [
        CurriculumEvidence(
            entity_type="subject",
            entity_id="subj-1",
            name="ICT Literacy",
            grade="CLASS_1",
            subject="ICT_LITERACY",
            content="ICT Literacy",
        )
    ]
    answer = "All subjects are listed below.\n* ICT Literacy"
    guarded = _guard(_state(subjects), answer, _refs(subjects))
    assert guarded.answer == answer


def test_generation_timeout_does_not_fabricate_a_catalogue():
    evidence = [
        _unit("ict-1", "ICT topic 01", "CLASS_1", subject="ICT_LITERACY"),
        _unit(
            "ict-lo",
            "identify a device",
            "CLASS_1",
            subject="ICT_LITERACY",
            content="identify a device",
            entity_type="learning_outcome",
        ),
    ]
    with pytest.raises(LLMProviderError):
        AnswerGenerator(_TimeoutLLM()).generate(_state(evidence))


def test_verifier_cannot_accept_a_silent_catalogue_omission():
    evidence = _ict_units(24)
    shown = evidence[:23]
    state = _state(evidence, subject="ICT_LITERACY")
    state.final_answer = _catalogue(shown)
    state.answer_evidence = _refs(shown)
    verification = run_deterministic_checks(state)
    assert verification.passed is False
    guarded = _guard(state, state.final_answer, state.answer_evidence)
    state.final_answer = guarded.answer
    state.answer_evidence = list(guarded.evidence)
    accepted = run_deterministic_checks(state)
    assert accepted.passed is True
    verifier = AnswerVerifier(_TimeoutLLM())
    merged = verifier._merge(verification, accepted)
    assert merged.passed is False
