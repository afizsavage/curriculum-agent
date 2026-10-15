"""Status follows the verifier route. No-progress is not treated as acceptance."""

from __future__ import annotations

from app.agent.answer_generator import format_evidence_for_prompt, select_evidence_for_prompt
from app.agent.graph_nodes import GraphNodes, apply_fallback
from app.agent.graph_routing import route_after_verification
from app.agent.retrieval_state import targeted_tool_calls_from_missing, tool_fingerprint
from app.agent.state import CurriculumQAState
from app.agent.verifier import AnswerVerifier
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence
from app.curriculum.normalize import evidence_from_hit, iter_content_nodes, node_to_search_hit
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
    """Current contract: an accept with no missing-evidence list finishes.

    Same-name deduplication can remove retrieved units from the generation
    context. An accept in that state is not proof that absence was established.
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
    qa.evidence = evidence
    qa.draft_answer = "Astronomy is not among the topics shown in this evidence."
    qa.verification = VerificationResult(
        passed=True,
        score=0.7,
        recommendation=VerificationRecommendation.ACCEPT,
        issues=[
            "Evidence set appears partial; absence is inferred from the retrieved units only."
        ],
        missing_evidence=[],
    )
    assert _route(qa) == "finish"
