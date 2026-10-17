"""A missing subject syllabus falls back to that grade's subject list."""

from __future__ import annotations

import httpx

from app.agent.answer_generator import format_evidence_for_prompt
from app.agent.retrieve import RetrievalNode
from app.agent.state import CurriculumQAState
from app.agent.verification_checks import run_deterministic_checks
from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.curriculum.codes import extract_filters_from_question
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.llm.base import ToolCallRequest
from app.llm.provider import StubLLMProvider
from app.tools.curriculum import GetCurriculumStructureTool
from app.tools.registry import build_default_registry

GOLD_QUESTION = "What units does the Class 4 French subject cover?"
BEC_ID = "11111111-1111-4111-8111-111111111111"
SSC_ID = "66666666-6666-4666-8666-666666666666"
GRADE_4_ID = "g4"
JSS_1_ID = "jss1"
SSS_1_ID = "sss1"
APPRECIATION_ID = "44c8b694-1111-4111-8111-111111111111"
SCIENCE_ID = "77777777-7777-4777-8777-777777777777"
MATH_ID = "22222222-2222-4222-8222-222222222222"
INTEGRATED_ID = "88888888-8888-4888-8888-888888888888"
FOREIGN_ID = "99999999-9999-4999-8999-999999999999"

CLASS_4_SUBJECTS = (
    ("MATHEMATICS", "Mathematics", MATH_ID),
    ("SCIENCE", "Science", SCIENCE_ID),
    ("FRENCH_APPRECIATION", "French Appreciation", APPRECIATION_ID),
)


def _grade_row(grade_id: str, code: str, subjects: list[dict]) -> dict:
    return {"id": grade_id, "code": code, "name": code, "subjects": subjects}


def _subject(code: str, name: str, subject_id: str) -> dict:
    return {"id": subject_id, "code": code, "name": name}


def _grade_subject(grade_id: str, code: str, name: str, subject_id: str) -> dict:
    return {
        "id": f"gs-{subject_id}",
        "grade_id": grade_id,
        "subject_id": subject_id,
        "status": "ACTIVE",
        "classification": "CORE",
        "subject": _subject(code, name, subject_id),
    }


def _grade_curriculum(grade_id: str, grade_code: str, code: str, name: str, subject_id: str) -> dict:
    return {
        "id": f"gc-{subject_id}",
        "status": "ACTIVE",
        "grade": {"id": grade_id, "code": grade_code},
        "subject": _subject(code, name, subject_id),
    }


class _API:
    def __init__(self, *, grade_subjects: str = "ok") -> None:
        self.grade_subjects = grade_subjects
        self.paths: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)
        if path.endswith("/api/v1/curricula"):
            items = [
                {"id": BEC_ID, "code": "MBSSE-BEC", "version": "2020", "name": "BEC"},
                {"id": SSC_ID, "code": "MBSSE-SSC", "version": "2021", "name": "SSC"},
            ]
            code = request.url.params.get("code")
            if code:
                items = [item for item in items if item["code"] == code]
            return httpx.Response(200, json={"items": items, "total": len(items)})
        if path.endswith("/structure"):
            curriculum_id = BEC_ID if BEC_ID in path else SSC_ID
            if curriculum_id == SSC_ID:
                grades = [
                    _grade_row(
                        SSS_1_ID,
                        "SSS_1",
                        [_subject("FRENCH_FOREIGN_LANGUAGE", "French as a Foreign Language", FOREIGN_ID)],
                    )
                ]
            else:
                grades = [
                    _grade_row(
                        GRADE_4_ID,
                        "CLASS_4",
                        [_subject(code, name, subject_id) for code, name, subject_id in CLASS_4_SUBJECTS],
                    ),
                    _grade_row(
                        JSS_1_ID,
                        "JSS_1",
                        [_subject("INTEGRATED_SCIENCE", "Integrated Science", INTEGRATED_ID)],
                    ),
                ]
            return httpx.Response(
                200,
                json={"id": curriculum_id, "education_levels": [{"grades": grades}]},
            )
        if path.endswith("/grade-subjects"):
            if self.grade_subjects == "empty":
                return httpx.Response(200, json={"items": [], "total": 0})
            if self.grade_subjects == "unavailable":
                return httpx.Response(503, json={"detail": "unavailable"})
            grade_id = request.url.params.get("grade_id")
            rows = {
                GRADE_4_ID: [
                    _grade_subject(GRADE_4_ID, code, name, subject_id)
                    for code, name, subject_id in CLASS_4_SUBJECTS
                ],
                JSS_1_ID: [
                    _grade_subject(JSS_1_ID, "INTEGRATED_SCIENCE", "Integrated Science", INTEGRATED_ID)
                ],
                SSS_1_ID: [
                    _grade_subject(
                        SSS_1_ID,
                        "FRENCH_FOREIGN_LANGUAGE",
                        "French as a Foreign Language",
                        FOREIGN_ID,
                    )
                ],
            }
            items = rows.get(grade_id, [])
            return httpx.Response(200, json={"items": items, "total": len(items)})
        if path.endswith("/grade-curricula"):
            curriculum_id = SSC_ID if SSC_ID in path else BEC_ID
            if curriculum_id == SSC_ID:
                items = [
                    _grade_curriculum(
                        SSS_1_ID,
                        "SSS_1",
                        "FRENCH_FOREIGN_LANGUAGE",
                        "French as a Foreign Language",
                        FOREIGN_ID,
                    )
                ]
            else:
                items = [
                    _grade_curriculum(GRADE_4_ID, "CLASS_4", code, name, subject_id)
                    for code, name, subject_id in CLASS_4_SUBJECTS
                ]
                items.append(
                    _grade_curriculum(
                        JSS_1_ID, "JSS_1", "INTEGRATED_SCIENCE", "Integrated Science", INTEGRATED_ID
                    )
                )
            return httpx.Response(200, json={"items": items, "total": len(items)})
        if path.endswith("/content"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": f"unit-{path.split('/')[-2]}",
                        "content_type": "UNIT",
                        "name": "Published unit",
                        "code": "U01",
                        "children": [],
                        "learning_outcomes": [],
                    }
                ],
            )
        if path.endswith("/api/v1/syllabuses"):
            return httpx.Response(200, json={"items": [], "total": 0})
        if path.endswith("/subjects"):
            return httpx.Response(200, json={"items": [], "total": 0})
        return httpx.Response(404, json={"detail": f"unhandled {path}"})


def _client(api: _API) -> CurriculumAPIClient:
    return CurriculumAPIClient(
        settings=Settings(curriculum_api_base_url="http://curriculum.test"),
        transport=httpx.MockTransport(api.handle),
    )


def _codes(result) -> set[str]:
    return {row["subject"] for row in result.data["evidence"]}


def test_missing_french_syllabus_requests_the_class_4_subject_list():
    api = _API()
    result = GetCurriculumStructureTool(_client(api)).execute(
        grade="CLASS_4", subject="FRENCH", level="primary"
    )
    assert any(path.endswith("/grade-curricula") for path in api.paths)
    assert any(path.endswith("/grade-subjects") for path in api.paths)
    assert not any(path.endswith("/subjects") for path in api.paths)
    assert result.success is True
    assert result.data["requested_subject_syllabus"] == "not_found"
    assert result.data["subject"] == "FRENCH"
    assert result.data["grade"] == "CLASS_4"


def test_recovered_subject_list_keeps_identifiers_for_later_stages():
    api = _API()
    result = GetCurriculumStructureTool(_client(api)).execute(
        grade="CLASS_4", subject="FRENCH"
    )
    appreciation = next(
        row for row in result.data["evidence"] if row["entity_id"] == APPRECIATION_ID
    )
    assert appreciation["source_reference"] == "grade_subjects"
    assert appreciation["metadata"]["grade_subject_id"] == f"gs-{APPRECIATION_ID}"
    assert appreciation["metadata"]["code"] == "FRENCH_APPRECIATION"
    assert appreciation["grade"] == "CLASS_4"

    filters = extract_filters_from_question(GOLD_QUESTION)
    state = CurriculumQAState.initial(question=GOLD_QUESTION)
    state.grade = filters["grade"]
    state.subject = filters["subject"]
    state.evidence = [CurriculumEvidence.model_validate(row) for row in result.data["evidence"]]
    prompt = format_evidence_for_prompt(state.evidence, question=state.question)
    assert APPRECIATION_ID in prompt
    assert "French Appreciation" in prompt


def test_class_4_list_distinguishes_french_from_french_appreciation():
    api = _API()
    result = GetCurriculumStructureTool(_client(api)).execute(
        grade="Class 4", subject="French"
    )
    assert "FRENCH_APPRECIATION" in _codes(result)
    assert "FRENCH" not in _codes(result)
    assert "nodes" not in result.data
    state = CurriculumQAState.initial(question=GOLD_QUESTION)
    state.grade = "CLASS_4"
    state.subject = "FRENCH"
    state.evidence = [CurriculumEvidence.model_validate(row) for row in result.data["evidence"]]
    state.evidence_status = EvidenceStatus.FOUND
    state.draft_answer = "French Appreciation is a published Class 4 subject."
    checked = run_deterministic_checks(state)
    assert "Subject mismatch for FRENCH." in checked.incorrect_claims


def test_a_published_syllabus_does_not_request_the_subject_list():
    api = _API()
    result = GetCurriculumStructureTool(_client(api)).execute(
        grade="CLASS_4", subject="MATHEMATICS"
    )
    assert result.success is True
    assert result.data["grade_curriculum_id"] == f"gc-{MATH_ID}"
    assert "requested_subject_syllabus" not in result.data
    assert any(row["entity_id"].startswith("unit-") for row in result.data["evidence"])
    assert not any(path.endswith("/grade-subjects") for path in api.paths)

    integrated = _API()
    jss = GetCurriculumStructureTool(_client(integrated)).execute(
        grade="JSS_1", subject="INTEGRATED_SCIENCE"
    )
    assert jss.success is True
    assert jss.data["subject"] == "INTEGRATED_SCIENCE"
    assert not any(path.endswith("/grade-subjects") for path in integrated.paths)


def test_failed_subject_list_does_not_establish_that_french_is_absent():
    for mode in ("empty", "unavailable"):
        api = _API(grade_subjects=mode)
        result = GetCurriculumStructureTool(_client(api)).execute(
            grade="CLASS_4", subject="FRENCH"
        )
        assert result.success is False
        assert result.data["error_code"] == "CURRICULUM_NOT_FOUND"
        assert result.error == "No syllabus found for the given subject/grade filters"
        assert "absent" not in (result.error or "").lower()
        assert not any(path.endswith("/subjects") for path in api.paths)


def test_fallback_keeps_the_extracted_subject_grade_and_stream():
    api = _API()
    client = _client(api)
    settings = Settings(
        llm_provider="stub",
        curriculum_api_base_url="http://curriculum.test",
        agent_max_tool_calls=4,
    )
    node = RetrievalNode(
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        settings=settings,
    )
    state = CurriculumQAState.initial(question=GOLD_QUESTION)
    filters = extract_filters_from_question(GOLD_QUESTION)
    state.grade = filters["grade"]
    state.subject = filters["subject"]
    state.level = filters["level"]
    node._execute_call(
        state,
        ToolCallRequest(
            id="call-1",
            name="get_curriculum_structure",
            arguments={"grade": "CLASS_4", "subject": "FRENCH", "level": "primary"},
        ),
        request_id="req-french",
    )
    assert state.grade == "CLASS_4"
    assert state.subject == "FRENCH"
    assert state.level == "primary"
    assert {item.entity_id for item in state.evidence} >= {
        MATH_ID,
        SCIENCE_ID,
        APPRECIATION_ID,
    }
    node._finalize_evidence_status(state)
    assert state.evidence_status == EvidenceStatus.FOUND
    assert state.retrieval_history[-1].arguments["subject"] == "FRENCH"

    sss = _API()
    foreign = GetCurriculumStructureTool(_client(sss)).execute(
        grade="SSS_1", subject="French as a Foreign Language"
    )
    assert foreign.success is True
    assert foreign.data["subject"] == "FRENCH_FOREIGN_LANGUAGE"
    assert "requested_subject_syllabus" not in foreign.data
    assert not any(path.endswith("/grade-subjects") for path in sss.paths)
