"""Status follows the verifier route. No-progress is not treated as acceptance."""

from __future__ import annotations

from app.agent.absence_evidence import (
    format_absence_prompt,
    select_records_for_absence_verification,
    serialized_entity_ids,
)
from app.agent.answer_generator import format_evidence_for_prompt, select_evidence_for_prompt
from app.agent.graph_nodes import GraphNodes, apply_fallback
from app.agent.graph_routing import route_after_verification
from app.agent.retrieval_state import targeted_tool_calls_from_missing, tool_fingerprint
from app.agent.state import CurriculumQAState
from app.agent.verifier import AnswerVerifier
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence
from app.curriculum.normalize import evidence_from_hit, iter_content_nodes, node_to_search_hit
from app.curriculum.structure_scope import should_stamp_grade_curriculum_content
from app.enums import AgentStatus
from app.llm.provider import StubLLMProvider
from app.schemas.verification import (
    MissingEvidenceRequest,
    VerificationRecommendation,
    VerificationResult,
)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        curriculum_api_base_url="http://curriculum.test",
        agent_max_iterations=3,
        agent_max_retrieval_rounds=3,
        agent_max_tool_calls=10,
    )


_TOOLS = {
    "resolve_curriculum_context",
    "search_curriculum",
    "get_curriculum_structure",
    "get_topic",
    "get_learning_objectives",
    "get_subject",
}


def _route(qa: CurriculumQAState) -> str:
    return route_after_verification({"qa": qa}, settings=_settings())


def _exhaust_targeted_path(qa: CurriculumQAState) -> None:
    saw_call = False
    for _ in range(6):
        calls = targeted_tool_calls_from_missing(
            qa.pending_missing_evidence,
            available_tools=_TOOLS,
            grade=qa.grade,
            subject=qa.subject,
            topic=qa.topic,
            retrieval_state=qa.retrieval_state,
        )
        if not calls:
            break
        saw_call = True
        for call in calls:
            qa.retrieval_state.remember_fingerprint(
                tool_fingerprint(call.name, call.arguments),
                1,
            )
    assert saw_call


def test_published_parent_id_survives_structure_evidence():
    section_id = "6eb64218-8429-4217-94e2-48b5ad72fb48"
    unit_id = "f583efc3-228b-44ff-a096-792f5ec85788"
    tree = [
        {
            "id": section_id,
            "content_type": "SECTION",
            "code": "C4-T1",
            "name": "Class 4 — 1st Term",
            "children": [
                {
                    "id": unit_id,
                    "content_type": "UNIT",
                    "code": "C4-U01",
                    "name": "My family",
                    "parent_id": section_id,
                    "children": [],
                }
            ],
        }
    ]
    evidence = []
    for node, parent_id in iter_content_nodes(tree):
        hit = node_to_search_hit(
            node,
            parent_id=parent_id,
            grade="CLASS_4",
            subject="ENGLISH_LANGUAGE",
            level="primary",
        )
        evidence.append(
            evidence_from_hit(hit, source_reference="grade_curriculum.content")
        )
    unit = next(item for item in evidence if item.entity_id == unit_id)
    assert unit.metadata["parent_id"] == section_id
    block = format_evidence_for_prompt(evidence, question="first term units")
    assert f"Parent ID: {section_id}" in block


def test_ancillary_parent_absence_does_not_block_a_supported_name():
    """A parent link is ancillary when the requested claim does not need it.

    The router does not invent that distinction. It follows the verifier.
    When the verifier accepts and records no missing evidence, finalization
    completes even if the unit row has no parent id.
    """
    evidence = [
        CurriculumEvidence(
            entity_type="unit",
            entity_id="f583efc3-228b-44ff-a096-792f5ec85788",
            name="My family",
            grade="CLASS_4",
            subject="ENGLISH_LANGUAGE",
            metadata={"code": "C4-U01"},
        )
    ]
    question = "What is the published name of Class 4 English Language unit C4-U01?"
    selected, _ids = select_evidence_for_prompt(evidence, question=question)
    assert selected[0].metadata.get("parent_id") is None
    qa = CurriculumQAState.initial(question=question)
    qa.evidence = evidence
    qa.draft_answer = "My family — C4-U01"
    qa.verification = VerificationResult(
        passed=True,
        score=1.0,
        recommendation=VerificationRecommendation.ACCEPT,
        missing_evidence=[],
    )
    assert _route(qa) == "finish"
    finished = GraphNodes.finish(object.__new__(GraphNodes), {"qa": qa, "visited_nodes": []})
    assert finished["qa"].status == AgentStatus.COMPLETED


def test_missing_parent_link_does_not_finish():
    qa = CurriculumQAState.initial(
        question="What units are published for the Class 4 English Language first term?"
    )
    qa.grade = "CLASS_4"
    qa.subject = "ENGLISH_LANGUAGE"
    qa.draft_answer = "My family — C4-U01"
    qa.verification = VerificationResult(
        passed=False,
        score=0.4,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        missing_evidence=[
            MissingEvidenceRequest(
                type="unit_term_mapping",
                subject="ENGLISH_LANGUAGE",
                topic="first term units",
                detail="No evidence record links any unit to the first-term section.",
            )
        ],
    )
    qa.pending_missing_evidence = list(qa.verification.missing_evidence)
    assert _route(qa) == "retrieve_more"
    _exhaust_targeted_path(qa)
    assert _route(qa) == "fallback"
    out = apply_fallback(qa, reason="no_retrieval_progress")
    assert out.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert "My family" in (out.final_answer or "")


def test_absent_code_request_without_a_retrieval_path_stays_insufficient():
    qa = CurriculumQAState.initial(
        question="Give the content code and name of every published Class 4 Mathematics unit."
    )
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.draft_answer = "The evidence does not include a unit with the code C4-U12."
    qa.verification = VerificationResult(
        passed=False,
        score=0.8,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        issues=[
            "The answer correctly states this absence. This is not an error in the answer."
        ],
        missing_evidence=[
            MissingEvidenceRequest(
                type="unit",
                subject="MATHEMATICS",
                topic="C4-U12",
                detail="No evidence record exists for C4-U12.",
            )
        ],
    )
    qa.pending_missing_evidence = list(qa.verification.missing_evidence)
    assert _route(qa) == "retrieve_more"
    _exhaust_targeted_path(qa)
    assert _route(qa) == "fallback"
    out = apply_fallback(qa, reason="no_retrieval_progress")
    assert out.status == AgentStatus.INSUFFICIENT_EVIDENCE


def test_complete_list_missing_one_record_does_not_finish():
    evidence = [
        CurriculumEvidence(
            entity_type="unit",
            entity_id=f"unit-{code}",
            name=name,
            grade="CLASS_4",
            subject="SCIENCE",
            metadata={"code": code},
        )
        for code, name in (
            ("C4-U01", "Living things"),
            ("C4-U02", "Animals"),
            ("C4-U03", "Plants"),
        )
    ]
    qa = CurriculumQAState.initial(
        question="List every published Class 4 Science unit, with its content code."
    )
    qa.evidence = evidence
    qa.grade = "CLASS_4"
    qa.subject = "SCIENCE"
    qa.draft_answer = "C4-U01 Living things. C4-U02 Animals."
    assert "C4-U03" not in qa.draft_answer
    assert any((item.metadata or {}).get("code") == "C4-U03" for item in qa.evidence)
    qa.verification = VerificationResult(
        passed=False,
        score=0.6,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        missing_evidence=[
            MissingEvidenceRequest(
                type="unit",
                subject="SCIENCE",
                topic="C4-U03",
                detail="The complete-list answer omits published unit C4-U03.",
            )
        ],
    )
    qa.pending_missing_evidence = list(qa.verification.missing_evidence)
    assert _route(qa) == "retrieve_more"
    _exhaust_targeted_path(qa)
    assert _route(qa) == "fallback"
    out = apply_fallback(qa, reason="no_retrieval_progress")
    assert out.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert qa.pending_missing_evidence


def test_supported_claims_with_unproven_source_completeness_stay_insufficient():
    qa = CurriculumQAState.initial(
        question="List every published Class 4 Science unit."
    )
    qa.grade = "CLASS_4"
    qa.subject = "SCIENCE"
    qa.draft_answer = "Living things. Animals."
    qa.verification = VerificationResult(
        passed=False,
        score=0.7,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        issues=["Each named unit is supported. The evidence does not show that the set is exhaustive."],
        missing_evidence=[
            MissingEvidenceRequest(
                type="curriculum_structure",
                subject="SCIENCE",
                grade="CLASS_4",
                detail="No retrieved record establishes that these units are the complete published set.",
            )
        ],
    )
    qa.pending_missing_evidence = list(qa.verification.missing_evidence)
    assert _route(qa) == "retrieve_more"
    _exhaust_targeted_path(qa)
    assert _route(qa) == "fallback"
    out = apply_fallback(qa, reason="no_retrieval_progress")
    assert out.status == AgentStatus.INSUFFICIENT_EVIDENCE


def test_unsupported_claim_without_a_retrieval_gap_does_not_finish():
    qa = CurriculumQAState.initial(
        question="What topics are listed under the SSS 1 Biology theme Introduction to Biology?"
    )
    qa.grade = "SSS_1"
    qa.subject = "BIOLOGY"
    qa.draft_answer = "Diversity of living things. Structure and functions of plant and animal cells."
    qa.verification = VerificationResult(
        passed=False,
        score=0.5,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        issues=["The answer presents learning outcomes as if they were topics."],
        missing_evidence=[],
    )
    qa.pending_missing_evidence = []
    assert _route(qa) == "fallback"
    out = apply_fallback(qa, reason="no_retrieval_progress")
    assert out.status == AgentStatus.INSUFFICIENT_EVIDENCE


def test_accept_with_missing_evidence_does_not_finish():
    verifier = AnswerVerifier(llm=StubLLMProvider(), settings=_settings())
    qa = CurriculumQAState.initial(question="Is astronomy part of Class 4 Mathematics?")
    result = verifier._normalize_recommendation(
        qa,
        VerificationResult(
            passed=True,
            score=0.9,
            recommendation=VerificationRecommendation.ACCEPT,
            missing_evidence=[
                MissingEvidenceRequest(
                    type="unit",
                    detail="Nine retrieved records were omitted, so absence is not confirmed.",
                )
            ],
        ),
    )
    qa.verification = result
    qa.pending_missing_evidence = list(result.missing_evidence)
    assert result.passed is False
    assert result.recommendation == VerificationRecommendation.RETRIEVE_MORE
    assert _route(qa) != "finish"


def test_supported_accept_finishes_completed():
    qa = CurriculumQAState.initial(question="What topics are taught in Primary 4 Mathematics?")
    qa.draft_answer = "Number and numeration."
    qa.verification = VerificationResult(
        passed=True,
        score=0.95,
        recommendation=VerificationRecommendation.ACCEPT,
        missing_evidence=[],
    )
    assert _route(qa) == "finish"
    finished = GraphNodes.finish(object.__new__(GraphNodes), {"qa": qa, "visited_nodes": []})
    assert finished["qa"].status == AgentStatus.COMPLETED


def test_retry_exhausted_keeps_the_draft_and_stays_insufficient():
    qa = CurriculumQAState.initial(question="List the units.")
    qa.retrieval_rounds = 3
    qa.draft_answer = "Number and numeration."
    qa.verification = VerificationResult(
        passed=False,
        score=0.4,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        missing_evidence=[
            MissingEvidenceRequest(type="unit", detail="Need another unit record.")
        ],
    )
    assert _route(qa) == "fallback"
    out = apply_fallback(qa, reason="max_retrieval_rounds")
    assert out.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert "Number and numeration." in (out.final_answer or "")


def test_correct_absence_and_the_same_code_request_do_not_finish():
    """The verifier may call an absence correct and still request that code.

    Normalization refuses to accept while the request is open. Exhausting
    the targeted path stays insufficient. The published tree, not this
    route, is what shows whether the code exists.
    """
    verifier = AnswerVerifier(llm=StubLLMProvider(), settings=_settings())
    qa = CurriculumQAState.initial(
        question="Give the content code and name of every published Class 4 Mathematics unit."
    )
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.draft_answer = "The evidence does not include a unit with the code C4-U12."
    result = verifier._normalize_recommendation(
        qa,
        VerificationResult(
            passed=True,
            score=0.95,
            recommendation=VerificationRecommendation.ACCEPT,
            issues=[
                "The answer correctly notes that C4-U12 is absent from the evidence; this is not an error in the answer."
            ],
            missing_evidence=[
                MissingEvidenceRequest(
                    type="unit",
                    subject="MATHEMATICS",
                    grade="CLASS_4",
                    topic="C4-U12",
                    detail="No evidence record exists for a Class 4 Mathematics unit with code C4-U12.",
                )
            ],
        ),
    )
    assert result.passed is False
    assert result.recommendation == VerificationRecommendation.RETRIEVE_MORE
    assert result.missing_evidence[0].topic == "C4-U12"
    qa.verification = result
    qa.pending_missing_evidence = list(result.missing_evidence)
    assert _route(qa) == "retrieve_more"
    _exhaust_targeted_path(qa)
    assert _route(qa) == "fallback"
    out = apply_fallback(qa, reason="no_retrieval_progress")
    assert out.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert "C4-U12" in (out.final_answer or "")


def test_material_unsupported_claims_do_not_finish_when_nothing_more_can_be_retrieved():
    """A populated unsupported-claim list with no retrieval path does not finish.

    A later accept that records only a stylistic issue, and no unsupported
    claim, still finishes. The two results are injected; the test does not
    call the model.
    """
    qa = CurriculumQAState.initial(
        question="What topics are listed under the SSS 1 Biology theme Introduction to Biology?"
    )
    qa.grade = "SSS_1"
    qa.subject = "BIOLOGY"
    qa.draft_answer = (
        "Living and non-living things. Pupils learn to explore the diversity "
        "of living things, including plants and animals."
    )
    qa.verification = VerificationResult(
        passed=False,
        score=0.4,
        recommendation=VerificationRecommendation.RETRIEVE_MORE,
        issues=["The answer presents learning outcomes as if they were topics."],
        unsupported_claims=[
            "explore the diversity of living things, including plants and animals"
        ],
        missing_evidence=[],
    )
    qa.pending_missing_evidence = []
    assert _route(qa) == "fallback"
    rejected = apply_fallback(qa, reason="no_retrieval_progress")
    assert rejected.status == AgentStatus.INSUFFICIENT_EVIDENCE

    qa.verification = VerificationResult(
        passed=True,
        score=0.9,
        recommendation=VerificationRecommendation.ACCEPT,
        issues=["The answer uses 'Pupils' for senior secondary learners."],
        unsupported_claims=[],
        missing_evidence=[],
    )
    assert _route(qa) == "finish"


def test_same_name_dedupe_can_still_be_accepted_when_missing_evidence_is_empty():
    """Ordinary dedupe still drops same-name units. An absence accept does not.

    The generation window may omit those units. Certification requires the
    verifier's checked ids to include them. ACCEPT with an empty missing
    list is not that check.
    """
    evidence = [
        CurriculumEvidence(
            entity_type="unit",
            entity_id=f"c4-u{index:02d}",
            name="Number and Numeration",
            grade="CLASS_4",
            subject="MATHEMATICS",
            metadata={"code": f"C4-U{index:02d}"},
        )
        for index in range(1, 6)
    ]
    question = "What does the Class 4 Mathematics syllabus say about astronomy?"
    selected, _ids = select_evidence_for_prompt(evidence, question=question)
    assert len(selected) == 1
    assert len(selected) < len(evidence)
    prompt = format_evidence_for_prompt(evidence, question=question)
    assert "omitted 4 additional evidence records" in prompt
    assert "Code: C4-U01" in prompt
    assert "Code: C4-U05" not in prompt
    qa = CurriculumQAState.initial(question=question)
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.evidence = evidence
    qa.draft_answer = "Astronomy is not among the topics shown in this evidence."
    partial_prompt = format_absence_prompt(selected)
    qa.metadata["verification_evidence_ids"] = [item.entity_id for item in evidence]
    qa.metadata["verification_prompt_ids"] = serialized_entity_ids(partial_prompt)
    assert "c4-u05" not in qa.metadata["verification_prompt_ids"]
    qa.verification = VerificationResult(
        passed=True,
        score=0.7,
        recommendation=VerificationRecommendation.ACCEPT,
        issues=[
            "Evidence set appears partial; absence is inferred from the retrieved units only."
        ],
        missing_evidence=[],
    )
    assert _route(qa) == "fallback"
    assert qa.metadata["fallback_reason"] == "absence_records_unchecked"
    out = apply_fallback(qa, reason="absence_records_unchecked")
    assert out.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert "not certified" in (out.final_answer or "")


def _exhaustive_tree(
    subject: str,
    grade: str,
    version: str,
    units: list[tuple[str, str]],
    *,
    source_reference: str = "grade_curriculum.content",
    exhaustive: bool = True,
    curriculum_code: str = "MBSSE-BEC",
) -> list[CurriculumEvidence]:
    return [
        CurriculumEvidence(
            entity_type="unit",
            entity_id=f"{subject}-{code}".lower(),
            name=name,
            grade=grade,
            subject=subject,
            source_reference=source_reference,
            metadata={
                "code": code,
                "curriculum_code": curriculum_code,
                "curriculum_version": version,
                "grade_curriculum_id": f"gc-{subject}-{grade}-{version}",
                "structure_scope": "grade_curriculum_content",
                "structure_exhaustive": exhaustive,
            },
        )
        for code, name in units
    ]


def _normalize(qa: CurriculumQAState, result: VerificationResult) -> VerificationResult:
    verifier = AnswerVerifier(llm=StubLLMProvider(), settings=_settings())
    return verifier._normalize_recommendation(qa, result)


def test_grade_curriculum_content_is_the_only_exhaustive_stamp():
    coded = [
        CurriculumEvidence(
            entity_type="unit",
            entity_id="u1",
            name="Plants",
            metadata={"code": "AG-U01"},
        )
    ]
    assert should_stamp_grade_curriculum_content(
        {"source": "grade_curriculum", "id": "gc-1"}, coded
    )
    assert not should_stamp_grade_curriculum_content(
        {"source": "syllabus", "id": "sy-1"}, coded
    )
    assert not should_stamp_grade_curriculum_content(
        {"source": "grade_curriculum", "id": "gc-1"},
        [CurriculumEvidence(entity_type="unit", entity_id="u1", name="Plants")],
    )


def test_exhaustive_tree_does_not_keep_requesting_an_absent_code():
    qa = CurriculumQAState.initial(
        question="Give the content code and name of every published Class 4 Mathematics unit."
    )
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.metadata["curriculum_version"] = "2020"
    qa.evidence = _exhaustive_tree(
        "MATHEMATICS",
        "CLASS_4",
        "2020",
        [
            ("C4-U01", "Number and Numeration"),
            ("C4-U11", "Everyday Arithmetic"),
            ("C4-U13", "Everyday Arithmetic Multiplication"),
        ],
    )
    qa.draft_answer = "The evidence does not include a unit with the code C4-U12."
    result = _normalize(
        qa,
        VerificationResult(
            passed=False,
            score=0.95,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            issues=[
                "The answer correctly notes that C4-U12 is absent from the evidence; this is not an error in the answer."
            ],
            missing_evidence=[
                MissingEvidenceRequest(
                    type="unit",
                    subject="MATHEMATICS",
                    grade="CLASS_4",
                    topic="C4-U12",
                    detail="No evidence record exists for a Class 4 Mathematics unit with code C4-U12.",
                )
            ],
        ),
    )
    assert result.missing_evidence == []
    assert result.metadata["resolved_absent_codes"] == ["C4-U12"]
    assert result.passed is False
    qa.verification = result
    qa.pending_missing_evidence = list(result.missing_evidence)
    assert _route(qa) == "fallback"
    assert (
        targeted_tool_calls_from_missing(
            qa.pending_missing_evidence,
            available_tools=_TOOLS,
            grade=qa.grade,
            subject=qa.subject,
            topic=qa.topic,
            retrieval_state=qa.retrieval_state,
        )
        == []
    )
    out = apply_fallback(qa, reason="no_retrieval_progress")
    assert out.status == AgentStatus.INSUFFICIENT_EVIDENCE


def test_absent_code_rule_applies_to_another_subject_grade_and_version():
    qa = CurriculumQAState.initial(question="List every published Class 2 Agriculture unit.")
    qa.grade = "CLASS_2"
    qa.subject = "AGRICULTURE"
    qa.metadata["curriculum_version"] = "2019"
    qa.evidence = _exhaustive_tree(
        "AGRICULTURE",
        "CLASS_2",
        "2019",
        [("AG-U01", "Soils"), ("AG-U02", "Crops")],
    )
    qa.draft_answer = "The Class 2 Agriculture tree does not include AG-U09."
    result = _normalize(
        qa,
        VerificationResult(
            passed=False,
            score=0.9,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            missing_evidence=[
                MissingEvidenceRequest(
                    type="unit",
                    subject="AGRICULTURE",
                    grade="CLASS_2",
                    topic="AG-U09",
                )
            ],
        ),
    )
    assert result.missing_evidence == []
    assert result.metadata["resolved_absent_codes"] == ["AG-U09"]


def test_supported_answer_completes_after_the_absent_code_request_is_resolved():
    qa = CurriculumQAState.initial(question="List every published Class 2 Agriculture unit.")
    qa.grade = "CLASS_2"
    qa.subject = "AGRICULTURE"
    qa.metadata["curriculum_version"] = "2019"
    qa.evidence = _exhaustive_tree(
        "AGRICULTURE",
        "CLASS_2",
        "2019",
        [("AG-U01", "Soils"), ("AG-U02", "Crops")],
    )
    qa.draft_answer = "AG-U01 Soils. AG-U02 Crops. The tree does not include AG-U09."
    result = _normalize(
        qa,
        VerificationResult(
            passed=True,
            score=1.0,
            recommendation=VerificationRecommendation.ACCEPT,
            missing_evidence=[
                MissingEvidenceRequest(
                    type="unit",
                    subject="AGRICULTURE",
                    grade="CLASS_2",
                    topic="AG-U09",
                )
            ],
        ),
    )
    assert result.passed is True
    assert result.recommendation == VerificationRecommendation.ACCEPT
    assert result.missing_evidence == []
    qa.verification = result
    assert _route(qa) == "finish"
    finished = GraphNodes.finish(
        object.__new__(GraphNodes), {"qa": qa, "visited_nodes": []}
    )
    assert finished["qa"].status == AgentStatus.COMPLETED


def test_failed_lookup_without_an_exhaustive_tree_does_not_establish_absence():
    qa = CurriculumQAState.initial(question="List the Class 4 Mathematics units.")
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.draft_answer = "The evidence does not include C4-U12."
    result = _normalize(
        qa,
        VerificationResult(
            passed=True,
            score=0.8,
            recommendation=VerificationRecommendation.ACCEPT,
            missing_evidence=[
                MissingEvidenceRequest(
                    type="unit",
                    subject="MATHEMATICS",
                    grade="CLASS_4",
                    topic="C4-U12",
                )
            ],
        ),
    )
    assert result.passed is False
    assert result.recommendation == VerificationRecommendation.RETRIEVE_MORE
    assert result.missing_evidence[0].topic == "C4-U12"


def test_wrong_version_or_non_authoritative_tree_does_not_resolve_absence():
    request = MissingEvidenceRequest(
        type="unit",
        subject="MATHEMATICS",
        grade="CLASS_4",
        topic="C4-U12",
    )
    qa = CurriculumQAState.initial(question="List the Class 4 Mathematics units.")
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.metadata["curriculum_version"] = "2020"
    qa.draft_answer = "The evidence does not include C4-U12."
    qa.evidence = _exhaustive_tree(
        "MATHEMATICS",
        "CLASS_4",
        "2018",
        [("C4-U01", "Number and Numeration")],
    )
    wrong_version = _normalize(
        qa,
        VerificationResult(
            passed=True,
            recommendation=VerificationRecommendation.ACCEPT,
            missing_evidence=[request],
        ),
    )
    assert wrong_version.missing_evidence[0].topic == "C4-U12"

    qa.evidence = _exhaustive_tree(
        "MATHEMATICS",
        "CLASS_4",
        "2020",
        [("C4-U01", "Number and Numeration")],
        source_reference="syllabus.content.tree",
    )
    syllabus = _normalize(
        qa,
        VerificationResult(
            passed=True,
            recommendation=VerificationRecommendation.ACCEPT,
            missing_evidence=[request],
        ),
    )
    assert syllabus.missing_evidence[0].topic == "C4-U12"

    qa.evidence = _exhaustive_tree(
        "MATHEMATICS",
        "CLASS_4",
        "2020",
        [("C4-U01", "Number and Numeration")],
        exhaustive=False,
    )
    incomplete = _normalize(
        qa,
        VerificationResult(
            passed=True,
            recommendation=VerificationRecommendation.ACCEPT,
            missing_evidence=[request],
        ),
    )
    assert incomplete.missing_evidence[0].topic == "C4-U12"


def test_science_code_is_not_imported_as_mathematics_evidence():
    qa = CurriculumQAState.initial(question="List every Class 4 Mathematics unit.")
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.metadata["curriculum_version"] = "2020"
    qa.evidence = _exhaustive_tree(
        "MATHEMATICS",
        "CLASS_4",
        "2020",
        [
            ("C4-U01", "Number and Numeration"),
            ("C4-U13", "Everyday Arithmetic Multiplication"),
        ],
    ) + _exhaustive_tree(
        "SCIENCE",
        "CLASS_4",
        "2020",
        [("C4-U12", "Feeding in Humans")],
    )
    qa.draft_answer = "The Mathematics evidence does not include C4-U12."
    result = _normalize(
        qa,
        VerificationResult(
            passed=False,
            recommendation=VerificationRecommendation.RETRIEVE_MORE,
            missing_evidence=[
                MissingEvidenceRequest(
                    type="unit",
                    subject="MATHEMATICS",
                    grade="CLASS_4",
                    topic="C4-U12",
                )
            ],
        ),
    )
    assert result.missing_evidence == []
    assert result.metadata["resolved_absent_codes"] == ["C4-U12"]
    science = next(item for item in qa.evidence if item.subject == "SCIENCE")
    assert science.metadata["code"] == "C4-U12"
    assert not any(
        item.subject == "MATHEMATICS" and item.metadata.get("code") == "C4-U12"
        for item in qa.evidence
    )


def test_asserting_the_absent_code_is_not_cleared_by_the_tree():
    qa = CurriculumQAState.initial(question="List every Class 4 Mathematics unit.")
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.metadata["curriculum_version"] = "2020"
    qa.evidence = _exhaustive_tree(
        "MATHEMATICS",
        "CLASS_4",
        "2020",
        [
            ("C4-U01", "Number and Numeration"),
            ("C4-U13", "Everyday Arithmetic Multiplication"),
        ],
    )
    qa.draft_answer = "C4-U12 is a Class 4 Mathematics unit named Feeding in Humans."
    result = _normalize(
        qa,
        VerificationResult(
            passed=True,
            score=0.4,
            recommendation=VerificationRecommendation.ACCEPT,
            missing_evidence=[
                MissingEvidenceRequest(
                    type="unit",
                    subject="MATHEMATICS",
                    grade="CLASS_4",
                    topic="C4-U12",
                )
            ],
        ),
    )
    assert result.passed is False
    assert result.recommendation == VerificationRecommendation.RETRIEVE_MORE
    assert result.missing_evidence[0].topic == "C4-U12"
    qa.verification = result
    qa.pending_missing_evidence = list(result.missing_evidence)
    assert _route(qa) != "finish"


def test_absent_unit_code_does_not_suppress_outcome_or_parent_requests():
    qa = CurriculumQAState.initial(question="What are the Class 4 Mathematics outcomes?")
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.metadata["curriculum_version"] = "2020"
    qa.evidence = _exhaustive_tree(
        "MATHEMATICS",
        "CLASS_4",
        "2020",
        [("C4-U01", "Number and Numeration")],
    )
    qa.draft_answer = "Count up to 5000 using a number line."
    result = _normalize(
        qa,
        VerificationResult(
            passed=True,
            recommendation=VerificationRecommendation.ACCEPT,
            missing_evidence=[
                MissingEvidenceRequest(
                    type="learning_outcome",
                    subject="MATHEMATICS",
                    grade="CLASS_4",
                    topic="C4U01-LO01",
                    detail="The outcome text is not in the answer.",
                ),
                MissingEvidenceRequest(
                    type="unit_term_mapping",
                    subject="MATHEMATICS",
                    grade="CLASS_4",
                    topic="C4-T1",
                    detail="No record links the unit to its parent section.",
                ),
            ],
        ),
    )
    assert [item.topic for item in result.missing_evidence] == ["C4U01-LO01", "C4-T1"]
    assert result.recommendation == VerificationRecommendation.RETRIEVE_MORE


def test_unrelated_missing_evidence_remains_after_an_absent_code_is_resolved():
    qa = CurriculumQAState.initial(question="List every Class 2 Agriculture unit.")
    qa.grade = "CLASS_2"
    qa.subject = "AGRICULTURE"
    qa.metadata["curriculum_version"] = "2019"
    qa.evidence = _exhaustive_tree(
        "AGRICULTURE",
        "CLASS_2",
        "2019",
        [("AG-U01", "Soils")],
    )
    qa.draft_answer = "AG-U01 Soils. The tree does not include AG-U09."
    result = _normalize(
        qa,
        VerificationResult(
            passed=True,
            recommendation=VerificationRecommendation.ACCEPT,
            missing_evidence=[
                MissingEvidenceRequest(
                    type="unit",
                    subject="AGRICULTURE",
                    grade="CLASS_2",
                    topic="AG-U09",
                ),
                MissingEvidenceRequest(
                    type="learning_outcome",
                    subject="AGRICULTURE",
                    grade="CLASS_2",
                    topic="soil types",
                    detail="Need the learning outcome for soil types.",
                ),
            ],
        ),
    )
    assert result.passed is False
    assert result.recommendation == VerificationRecommendation.RETRIEVE_MORE
    assert [item.topic for item in result.missing_evidence] == ["soil types"]
    qa.verification = result
    qa.pending_missing_evidence = list(result.missing_evidence)
    assert _route(qa) == "retrieve_more"


def test_unsupported_claims_still_block_when_an_absent_code_is_also_requested():
    qa = CurriculumQAState.initial(question="List every Class 2 Agriculture unit.")
    qa.grade = "CLASS_2"
    qa.subject = "AGRICULTURE"
    qa.metadata["curriculum_version"] = "2019"
    qa.evidence = _exhaustive_tree(
        "AGRICULTURE",
        "CLASS_2",
        "2019",
        [("AG-U01", "Soils")],
    )
    qa.draft_answer = "AG-U01 Soils. Pupils explore crops that are not in the evidence."
    result = _normalize(
        qa,
        VerificationResult(
            passed=True,
            recommendation=VerificationRecommendation.ACCEPT,
            unsupported_claims=["Pupils explore crops that are not in the evidence."],
            missing_evidence=[
                MissingEvidenceRequest(
                    type="unit",
                    subject="AGRICULTURE",
                    grade="CLASS_2",
                    topic="AG-U09",
                )
            ],
        ),
    )
    assert result.passed is False
    assert result.recommendation == VerificationRecommendation.RETRIEVE_MORE
    assert result.missing_evidence[0].topic == "AG-U09"
    qa.verification = result
    qa.pending_missing_evidence = list(result.missing_evidence)
    assert _route(qa) != "finish"


def _same_name_units_with_outcomes() -> list[CurriculumEvidence]:
    units = [
        CurriculumEvidence(
            entity_type="unit",
            entity_id=f"unit-{index}",
            name="Number and Numeration",
            grade="CLASS_4",
            subject="MATHEMATICS",
            content="Number and Numeration",
            metadata={"code": f"C4-U{index:02d}"},
        )
        for index in (1, 7)
    ]
    outcomes = [
        CurriculumEvidence(
            entity_type="learning_outcome",
            entity_id="outcome-1",
            name="C4U01-LO01",
            grade="CLASS_4",
            subject="MATHEMATICS",
            topic="unit-1",
            content="Count up to 5000 using a number line.",
            metadata={"parent_id": "unit-1"},
        ),
        CurriculumEvidence(
            entity_type="learning_outcome",
            entity_id="outcome-7",
            name="C4U07-LO01",
            grade="CLASS_4",
            subject="MATHEMATICS",
            topic="unit-7",
            content="Describe how sailors used the stars to navigate.",
            metadata={"parent_id": "unit-7"},
        ),
    ]
    return units + outcomes


def test_checked_absence_records_can_finish_when_the_claim_is_scoped():
    evidence = _same_name_units_with_outcomes()
    question = "What does the Class 4 Mathematics syllabus say about astronomy?"
    checked = select_records_for_absence_verification(
        evidence, grade="CLASS_4", subject="MATHEMATICS"
    )
    assert {item.entity_id for item in checked} == {
        "unit-1",
        "unit-7",
        "outcome-1",
        "outcome-7",
    }
    prompt = format_absence_prompt(checked)
    assert "Count up to 5000 using a number line." in prompt
    assert "Describe how sailors used the stars to navigate." in prompt
    prompt_ids = serialized_entity_ids(prompt)
    assert {"unit-1", "unit-7", "outcome-1", "outcome-7"} <= set(prompt_ids)
    qa = CurriculumQAState.initial(question=question)
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.evidence = evidence
    qa.draft_answer = (
        "Astronomy is not among the Class 4 Mathematics records that were checked."
    )
    qa.metadata["verification_evidence_ids"] = [item.entity_id for item in checked]
    qa.metadata["verification_prompt_ids"] = prompt_ids
    qa.verification = VerificationResult(
        passed=True,
        score=0.9,
        recommendation=VerificationRecommendation.ACCEPT,
        missing_evidence=[],
    )
    assert _route(qa) == "finish"
    finished = GraphNodes.finish(
        object.__new__(GraphNodes), {"qa": qa, "visited_nodes": []}
    )
    assert finished["qa"].status == AgentStatus.COMPLETED


def test_ordinary_questions_still_dedupe_same_name_units():
    evidence = _same_name_units_with_outcomes()
    question = "What topics are taught in Class 4 Mathematics?"
    selected, _ids = select_evidence_for_prompt(evidence, question=question)
    unit_ids = [item.entity_id for item in selected if item.entity_type == "unit"]
    assert unit_ids == ["unit-1"]
    assert question_needs_absence_context(question) is False
    absence = select_records_for_absence_verification(
        evidence, grade="CLASS_4", subject="MATHEMATICS"
    )
    assert {item.entity_id for item in absence if item.entity_type == "unit"} == {
        "unit-1",
        "unit-7",
    }


def question_needs_absence_context(question: str) -> bool:
    from app.agent.absence_evidence import question_probes_topic_presence

    return question_probes_topic_presence(question)


def test_different_outcome_text_is_unchecked_when_the_unit_is_deduped():
    evidence = _same_name_units_with_outcomes()
    qa = CurriculumQAState.initial(
        question="What does the Class 4 Mathematics syllabus say about astronomy?"
    )
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.evidence = evidence
    qa.draft_answer = "Astronomy is not among the records shown in this evidence."
    partial = [item for item in evidence if item.entity_id in {"unit-1", "outcome-1"}]
    prompt = format_absence_prompt(partial)
    assert "Describe how sailors used the stars to navigate." not in prompt
    qa.metadata["verification_evidence_ids"] = [item.entity_id for item in evidence]
    qa.metadata["verification_prompt_ids"] = serialized_entity_ids(prompt)
    assert "outcome-7" not in qa.metadata["verification_prompt_ids"]
    qa.verification = VerificationResult(
        passed=True,
        recommendation=VerificationRecommendation.ACCEPT,
        missing_evidence=[],
    )
    assert _route(qa) == "fallback"
    assert qa.metadata["fallback_reason"] == "absence_records_unchecked"


def test_failed_topic_lookup_does_not_certify_global_absence():
    qa = CurriculumQAState.initial(
        question="What does the Class 4 Mathematics syllabus say about astronomy?"
    )
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.draft_answer = "Astronomy is absent from the entire MBSSE curriculum."
    qa.metadata["verification_evidence_ids"] = []
    qa.verification = VerificationResult(
        passed=True,
        recommendation=VerificationRecommendation.ACCEPT,
        missing_evidence=[],
    )
    assert _route(qa) == "fallback"
    out = apply_fallback(qa, reason="absence_records_unchecked")
    assert out.status == AgentStatus.INSUFFICIENT_EVIDENCE


def test_scoped_absence_is_not_certified_from_the_mathematics_tree_as_global():
    evidence = _same_name_units_with_outcomes()
    qa = CurriculumQAState.initial(
        question="What does the Class 4 Mathematics syllabus say about astronomy?"
    )
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.evidence = evidence
    qa.draft_answer = "Astronomy is absent from the entire MBSSE curriculum."
    prompt = format_absence_prompt(evidence)
    qa.metadata["verification_evidence_ids"] = [item.entity_id for item in evidence]
    qa.metadata["verification_prompt_ids"] = serialized_entity_ids(prompt)
    qa.verification = VerificationResult(
        passed=True,
        recommendation=VerificationRecommendation.ACCEPT,
        missing_evidence=[],
    )
    assert _route(qa) == "fallback"


def test_failed_topic_lookup_does_not_establish_a_scoped_absence():
    qa = CurriculumQAState.initial(
        question="What does the Class 4 Mathematics syllabus say about astronomy?"
    )
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.evidence = []
    qa.draft_answer = "Astronomy is not among the Class 4 Mathematics records."
    qa.metadata["verification_prompt_ids"] = serialized_entity_ids(format_absence_prompt([]))
    qa.verification = VerificationResult(
        passed=True,
        recommendation=VerificationRecommendation.ACCEPT,
        missing_evidence=[],
    )
    assert _route(qa) == "fallback"
    assert qa.metadata["fallback_reason"] == "absence_records_unchecked"


def test_embedded_outcomes_in_the_verifier_prompt_can_support_a_scoped_claim():
    unit = CurriculumEvidence(
        entity_type="unit",
        entity_id="unit-1",
        name="Number and Numeration",
        grade="CLASS_4",
        subject="MATHEMATICS",
        metadata={
            "code": "C4-U01",
            "structure_exhaustive": True,
            "curriculum_version": "2020",
            "learning_outcomes": [
                {
                    "id": "lo-1",
                    "code": "C4U01-LO01",
                    "description": "Count up to 5000 using a number line.",
                },
                {
                    "id": "lo-stars",
                    "code": "C4U01-LO09",
                    "description": "Name the stars used in navigation.",
                },
            ],
        },
    )
    qa = CurriculumQAState.initial(
        question="What does the Class 4 Mathematics syllabus say about astronomy?"
    )
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.evidence = [unit]
    qa.draft_answer = (
        "Astronomy is not among the Class 4 Mathematics records that were checked."
    )
    messages = AnswerVerifier(StubLLMProvider(), settings=_settings()).build_messages(qa)
    user = messages[-1].content
    assert "Name the stars used in navigation." in user
    assert qa.metadata["verification_prompt_ids"] == serialized_entity_ids(user)
    assert {"unit-1", "lo-1", "lo-stars"} <= set(qa.metadata["verification_prompt_ids"])
    audit = qa.metadata["absence_evidence"]
    assert audit["omitted"] == []
    assert audit["claim_scope"] == "curriculum_content"
    assert audit["source_exhaustive"] is True
    assert audit["in_scope_outcome_ids"] == ["lo-1", "lo-stars"]
    qa.verification = VerificationResult(
        passed=True,
        recommendation=VerificationRecommendation.ACCEPT,
        missing_evidence=[],
    )
    assert _route(qa) == "finish"


def test_outcome_text_left_out_of_the_prompt_cannot_certify_content_absence():
    unit = CurriculumEvidence(
        entity_type="unit",
        entity_id="unit-1",
        name="Number and Numeration",
        grade="CLASS_4",
        subject="MATHEMATICS",
        metadata={
            "code": "C4-U01",
            "learning_outcomes": [
                {
                    "id": "lo-stars",
                    "code": "C4U01-LO09",
                    "description": "Name the stars used in navigation.",
                }
            ],
        },
    )
    titles_only = CurriculumEvidence(
        entity_type="unit",
        entity_id="unit-1",
        name="Number and Numeration",
        grade="CLASS_4",
        subject="MATHEMATICS",
        metadata={"code": "C4-U01"},
    )
    prompt = format_absence_prompt([titles_only])
    assert "lo-stars" not in prompt
    assert "Name the stars used in navigation." not in prompt
    qa = CurriculumQAState.initial(
        question="What does the Class 4 Mathematics syllabus say about astronomy?"
    )
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.evidence = [unit]
    qa.draft_answer = (
        "The evidence does not include a curriculum statement about astronomy."
    )
    qa.metadata["verification_evidence_ids"] = ["unit-1", "lo-stars"]
    qa.metadata["verification_prompt_ids"] = serialized_entity_ids(prompt)
    qa.verification = VerificationResult(
        passed=True,
        recommendation=VerificationRecommendation.ACCEPT,
        missing_evidence=[],
    )
    assert "lo-stars" not in qa.metadata["verification_prompt_ids"]
    assert _route(qa) == "fallback"
    assert qa.metadata["fallback_reason"] == "absence_records_unchecked"


def test_title_level_claim_can_finish_when_outcome_text_is_unavailable():
    evidence = _exhaustive_tree(
        "MATHEMATICS", "CLASS_4", "2020", [("C4-U01", "Number and Numeration")]
    )
    prompt = format_absence_prompt(evidence)
    qa = CurriculumQAState.initial(
        question="What does the Class 4 Mathematics syllabus say about astronomy?"
    )
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.evidence = evidence
    qa.draft_answer = "No unit title matches astronomy."
    qa.metadata["verification_prompt_ids"] = serialized_entity_ids(prompt)
    qa.verification = VerificationResult(
        passed=True,
        recommendation=VerificationRecommendation.ACCEPT,
        missing_evidence=[],
    )
    assert _route(qa) == "finish"


def test_content_claim_abstains_when_outcome_text_is_unavailable():
    evidence = _exhaustive_tree(
        "MATHEMATICS", "CLASS_4", "2020", [("C4-U01", "Number and Numeration")]
    )
    prompt = format_absence_prompt(evidence)
    qa = CurriculumQAState.initial(
        question="What does the Class 4 Mathematics syllabus say about astronomy?"
    )
    qa.grade = "CLASS_4"
    qa.subject = "MATHEMATICS"
    qa.evidence = evidence
    qa.draft_answer = (
        "The evidence does not include a curriculum statement about astronomy."
    )
    qa.metadata["verification_prompt_ids"] = serialized_entity_ids(prompt)
    qa.verification = VerificationResult(
        passed=True,
        recommendation=VerificationRecommendation.ACCEPT,
        missing_evidence=[],
    )
    assert _route(qa) == "fallback"
    assert qa.metadata["fallback_reason"] == "absence_records_unchecked"
