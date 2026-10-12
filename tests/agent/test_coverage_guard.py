"""Coverage claims are checked against cited records, not the verifier score."""

from app.agent.answer_generator import AnswerGenerator
from app.agent.coverage_guard import assess_coverage
from app.agent.state import CurriculumQAState
from app.agent.verification_checks import run_deterministic_checks
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.schemas.answer import AnswerEvidenceRef


def _unit(entity_id: str, name: str, grade: str, *, content: str | None = None) -> CurriculumEvidence:
    return CurriculumEvidence(
        entity_type="unit",
        entity_id=entity_id,
        name=name,
        grade=grade,
        subject="MATHEMATICS",
        topic=name,
        content=name if content is None else content,
        source_reference="synthetic",
    )


def _state(question: str, evidence: list[CurriculumEvidence], *, status: EvidenceStatus = EvidenceStatus.FOUND) -> CurriculumQAState:
    state = CurriculumQAState.initial(question=question)
    state.subject = "MATHEMATICS"
    state.evidence = evidence
    state.evidence_status = status
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


QUESTION = "What does Mathematics cover in classes 1, 2, and 3?"


def _inventory() -> list[CurriculumEvidence]:
    evidence = []
    for grade, count in (("CLASS_1", 13), ("CLASS_2", 12), ("CLASS_3", 13)):
        for index in range(count):
            evidence.append(_unit(f"{grade}-{index}", f"{grade} topic {index}", grade))
    return evidence


def _subset(evidence: list[CurriculumEvidence]) -> list[CurriculumEvidence]:
    chosen = []
    for grade in ("CLASS_1", "CLASS_2", "CLASS_3"):
        chosen.extend([item for item in evidence if item.grade == grade][:8])
    return chosen


def test_complete_listing_can_claim_full_coverage():
    evidence = [
        _unit("c1", "Counting", "CLASS_1"),
        _unit("c2", "Time", "CLASS_2"),
        _unit("c3", "Shapes", "CLASS_3"),
    ]
    answer = _catalogue(evidence, claim="Here are all the topics.")
    result = assess_coverage(_state(QUESTION, evidence), answer=answer, refs=_refs(evidence))
    assert result.decision == "pass"
    assert result.represented_record_count == 3
    assert result.omitted_record_count == 0
    assert result.claims_complete_coverage is True


def test_honest_partial_answer_passes_without_a_completeness_claim():
    evidence = _inventory()
    shown = _subset(evidence)
    answer = "\n".join(
        [
            _catalogue(shown),
            "This answer includes 24 of 38 retrieved records "
            "(Primary 1: 8 of 13; Primary 2: 8 of 12; Primary 3: 8 of 13). "
            "It does not include every retrieved record.",
        ]
    )
    result = assess_coverage(_state(QUESTION, evidence), answer=answer, refs=_refs(shown))
    assert result.decision == "pass_partial"
    assert result.represented_record_count == 24
    assert result.omitted_record_count == 14
    assert result.claims_complete_coverage is False
    assert result.by_grade["CLASS_3"]["represented"] == 8


def test_false_completeness_claim_fails_when_records_are_missing():
    evidence = _inventory()
    shown = _subset(evidence)
    answer = _catalogue(shown, claim="Here are all the topics.")
    result = assess_coverage(_state(QUESTION, evidence), answer=answer, refs=_refs(shown))
    assert result.decision == "reject"
    assert result.claims_complete_coverage is True
    state = _state(QUESTION, evidence)
    state.final_answer = answer
    state.answer_evidence = _refs(shown)
    verification = run_deterministic_checks(state)
    assert verification.passed is False


def test_catalogue_phrases_do_not_pass_for_a_partial_list():
    evidence = _inventory()
    shown = _subset(evidence)
    phrases = [
        "Here are all the topics.",
        "The subject covers the following topics.",
        "These are the complete topics for the three classes.",
        "Pupils learn the following.",
        "The full curriculum consists of these units.",
    ]
    for phrase in phrases:
        result = assess_coverage(
            _state(QUESTION, evidence),
            answer=_catalogue(shown, claim=phrase),
            refs=_refs(shown),
        )
        assert result.decision == "reject", phrase
        assert result.claims_complete_coverage is True


def test_sample_wording_passes_when_it_does_not_claim_completeness():
    evidence = _inventory()
    shown = _subset(evidence)
    answer = "\n".join(
        [
            "This is a sample of the retrieved topics, not a complete catalogue.",
            _catalogue(shown),
        ]
    )
    result = assess_coverage(_state(QUESTION, evidence), answer=answer, refs=_refs(shown))
    assert result.decision == "pass_partial"
    assert result.claims_complete_coverage is False
    assert result.discloses_partial_scope is True


def test_a_grade_cannot_be_hidden_by_an_overall_count():
    evidence = _inventory()
    shown = [item for item in _subset(evidence) if item.grade != "CLASS_3"]
    shown.append(_unit("CLASS_3-0", "CLASS_3 topic 0", "CLASS_3"))
    answer = "\n".join(
        [
            _catalogue(shown),
            "This answer includes 17 of 38 retrieved records "
            "(Primary 1: 8 of 13; Primary 2: 8 of 12; Primary 3: 8 of 13). "
            "It does not include every retrieved record.",
        ]
    )
    result = assess_coverage(_state(QUESTION, evidence), answer=answer, refs=_refs(shown))
    assert result.decision == "reject"
    assert result.by_grade["CLASS_3"]["represented"] == 1
    assert result.stated_counts_match is False


def test_incorrect_counts_fail_even_when_every_record_is_listed():
    evidence = [
        _unit("c1", "Counting", "CLASS_1"),
        _unit("c2", "Time", "CLASS_2"),
    ]
    answer = "\n".join(
        [
            _catalogue(evidence),
            "This answer includes 1 of 2 retrieved records "
            "(Primary 1: 1 of 1; Primary 2: 0 of 1). "
            "It does not include every retrieved record.",
        ]
    )
    result = assess_coverage(_state(QUESTION, evidence), answer=answer, refs=_refs(evidence))
    assert result.represented_record_count == 2
    assert result.decision == "reject"
    assert result.stated_counts_match is False


def test_invented_outcome_fails_even_when_every_title_is_present():
    evidence = [
        _unit("c1", "Counting", "CLASS_1"),
        _unit("c2", "Time", "CLASS_2"),
    ]
    answer = "\n".join(
        [
            _catalogue(evidence),
            "Pupils learn to:",
            "* Invent a fraction skill.",
        ]
    )
    result = assess_coverage(_state(QUESTION, evidence), answer=answer, refs=_refs(evidence))
    assert result.decision == "reject"
    assert result.unsupported_outcomes


def test_retrieval_failures_stay_distinct_from_a_missing_syllabus():
    evidence = [_unit("c1", "Counting", "CLASS_1")]
    answer = "### Primary 1\n* Counting"
    found = assess_coverage(
        _state(QUESTION, evidence, status=EvidenceStatus.FOUND),
        answer=answer,
        refs=_refs(evidence),
    )
    partial = assess_coverage(
        _state(QUESTION, evidence, status=EvidenceStatus.PARTIAL),
        answer=answer,
        refs=_refs(evidence),
    )
    absent = assess_coverage(
        _state(QUESTION, [], status=EvidenceStatus.NOT_FOUND),
        answer="No syllabus was found for CLASS_2.",
        refs=[],
    )
    failed = assess_coverage(
        _state(QUESTION, [], status=EvidenceStatus.ERROR),
        answer="Curriculum retrieval could not be completed for CLASS_2.",
        refs=[],
    )
    assert found.retrieval_status == "found"
    assert partial.retrieval_status == "partial"
    assert absent.retrieval_status == "not_found"
    assert failed.retrieval_status == "error"
    assert absent.decision != "reject"
    assert failed.decision != "reject"


def test_one_bullet_can_represent_several_records_only_with_the_right_count():
    evidence = [
        _unit("a", "Shapes", "CLASS_1"),
        _unit("b", "Shapes", "CLASS_1"),
    ]
    answer = (
        "### Primary 1\n"
        "* Shapes (2 separate records share this exact title and have no further description)"
    )
    accepted = assess_coverage(_state(QUESTION, evidence), answer=answer, refs=_refs(evidence))
    assert accepted.decision == "pass"
    assert accepted.represented_ids == ["a", "b"]
    wrong = assess_coverage(
        _state(QUESTION, evidence),
        answer="### Primary 1\n* Shapes (1 separate records share this exact title and have no further description)",
        refs=_refs(evidence[:1]),
    )
    assert wrong.decision == "reject"
    assert wrong.multiplicity_ok is False


class _ClaimLLM:
    name = "echo-harness"
    model = "echo-harness"

    def generate_structured(self, messages, *, schema, temperature=0.0):
        return {
            "answer": "Here are all the topics.\n### Primary 1\n* Counting",
            "summary": "incomplete",
            "refs": ["c1"],
            "limitations": [],
            "confidence": "high",
        }


def test_generation_removes_a_false_completeness_claim():
    evidence = [
        _unit("c1", "Counting", "CLASS_1"),
        CurriculumEvidence(
            entity_type="learning_outcome",
            entity_id="lo-half",
            name="identify a half",
            grade="CLASS_1",
            subject="MATHEMATICS",
            content="identify a half",
        ),
        _unit("c2", "Time", "CLASS_2"),
    ]
    state = _state("What does Mathematics cover in classes 1 and 2?", evidence)
    result = AnswerGenerator(_ClaimLLM()).generate(state)
    assert "all the topics" not in result.answer.lower()
    assert "1 of 3" in result.answer
    assert "does not include every retrieved record" in result.answer.lower()
    assert state.metadata["coverage_guard_before"]["decision"] == "reject"
    assert state.metadata["coverage_guard"]["decision"] == "pass_partial"
    assert {ref.entity_id for ref in result.evidence} == {"c1"}
