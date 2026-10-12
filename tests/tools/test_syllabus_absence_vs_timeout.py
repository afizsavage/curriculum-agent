"""Timeouts stay distinct from a completed lookup that finds no syllabus."""

from __future__ import annotations

import httpx

from app.agent.answer_generator import AnswerGenerator
from app.agent.retrieve import RetrievalNode
from app.agent.state import CurriculumQAState
from app.agent.verification_checks import run_deterministic_checks
from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.curriculum.evidence import EvidenceStatus
from app.llm.base import LLMProvider, ToolCallRequest
from app.llm.provider import StubLLMProvider
from app.schemas.verification import VerificationRecommendation
from app.tools.registry import build_default_registry

CURRICULUM_ID = "curr-bec"


class _SilentLLM(LLMProvider):
    @property
    def name(self) -> str:
        return "silent"

    @property
    def model(self) -> str:
        return "silent"

    def generate(self, messages, *, temperature=0.0, max_tokens=None):
        raise AssertionError("generation must not run in these tests")

    def generate_structured(self, messages, *, schema, temperature=0.0):
        raise AssertionError("generation must not run in these tests")

    def generate_with_tools(self, messages, *, tools, temperature=0.0):
        raise AssertionError("generation must not run in these tests")


def _units(count: int, prefix: str) -> list[dict]:
    return [
        {
            "id": f"{prefix}-{index}",
            "content_type": "UNIT",
            "name": f"Unit {index}",
            "children": [],
            "learning_outcomes": [],
        }
        for index in range(count)
    ]


def _row(grade: str, row_id: str) -> dict:
    return {
        "id": row_id,
        "grade": {"code": grade, "name": grade},
        "subject": {"code": "ICT_LITERACY", "name": "ICT Literacy"},
    }


class _API:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.grade_list_calls = 0
        self.syllabus_calls = 0
        self.content_calls: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/api/v1/curricula"):
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": CURRICULUM_ID,
                            "code": "MBSSE-BEC",
                            "version": "2020",
                        }
                    ]
                },
            )
        if path.endswith("/grade-curricula") and "/content" not in path:
            self.grade_list_calls += 1
            if self.mode == "timeout" or (
                self.mode == "timeout_third" and self.grade_list_calls >= 3
            ):
                raise httpx.TimeoutException("grade-curricula timed out")
            if self.mode == "timeout_then_downstream":
                raise httpx.TimeoutException("grade-curricula timed out")
            if self.mode == "class3_absent":
                items = [_row("CLASS_1", "gc-1"), _row("CLASS_2", "gc-2")]
            elif self.mode == "all_absent":
                items = []
            else:
                items = [
                    _row("CLASS_1", "gc-1"),
                    _row("CLASS_2", "gc-2"),
                    _row("CLASS_3", "gc-3"),
                ]
            return httpx.Response(
                200, json={"items": items, "total": len(items)}
            )
        if path.endswith("/content"):
            row_id = path.rstrip("/").split("/")[-2]
            self.content_calls.append(row_id)
            count = {"gc-1": 7, "gc-2": 7, "gc-3": 8}[row_id]
            return httpx.Response(200, json=_units(count, row_id))
        if path.endswith("/api/v1/syllabuses"):
            self.syllabus_calls += 1
            if self.mode == "timeout_then_downstream":
                raise RuntimeError("syllabus lookup must not run after a timeout")
            return httpx.Response(200, json={"items": [], "total": 0})
        return httpx.Response(404, json={"detail": f"unhandled {path}"})


def _node(mode: str) -> tuple[RetrievalNode, _API]:
    api = _API(mode)
    settings = Settings(
        llm_provider="stub",
        curriculum_api_base_url="http://curriculum.test",
        agent_checkpointing_enabled=False,
    )
    client = CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(api)
    )
    tools = build_default_registry(settings=settings, include_echo=False, client=client)
    return RetrievalNode(llm=StubLLMProvider(), tools=tools, settings=settings), api


def _call(grade: str) -> ToolCallRequest:
    return ToolCallRequest(
        id=f"call-{grade}",
        name="get_curriculum_structure",
        arguments={"grade": grade, "subject": "ICT_LITERACY"},
    )


def _run(mode: str, grades: list[str]) -> tuple[CurriculumQAState, _API]:
    node, api = _node(mode)
    state = CurriculumQAState(
        question="What does ICT Literacy cover in classes 1, 2, and 3?",
        subject="ICT_LITERACY",
    )
    for grade in grades:
        node._execute_call(state, _call(grade), request_id="test")
    node._finalize_evidence_status(state)
    return state, api


def _grade_states(state: CurriculumQAState) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for record in state.retrieval_history:
        grade = str(record.arguments.get("grade"))
        rows = [item for item in state.evidence if item.grade == grade]
        found[grade] = {
            "status": record.status,
            "api_status": record.curriculum_api_status,
            "error": record.error,
            "records": len(rows),
            "absence_established": record.status == "not_found",
            "infrastructure_invalid": record.status == "error",
        }
    return found


def test_successful_class3_lookup_returns_eight_units():
    state, api = _run("success", ["CLASS_3"])
    assert state.evidence_status == EvidenceStatus.FOUND
    assert len(state.evidence) == 8
    grade = _grade_states(state)["CLASS_3"]
    assert grade["status"] == "success"
    assert grade["api_status"] == 200
    assert grade["absence_established"] is False
    assert api.syllabus_calls == 0
    assert all(item.subject == "ICT_LITERACY" for item in state.evidence)
    assert all(item.grade == "CLASS_3" for item in state.evidence)


def test_class3_timeout_is_not_reported_as_missing_syllabus():
    state, api = _run("timeout", ["CLASS_3"])
    grade = _grade_states(state)["CLASS_3"]
    assert grade["status"] == "error"
    assert grade["api_status"] == 504
    assert grade["records"] == 0
    assert grade["absence_established"] is False
    assert grade["infrastructure_invalid"] is True
    assert "No syllabus found" not in (grade["error"] or "")
    assert state.evidence_status == EvidenceStatus.ERROR
    assert api.syllabus_calls == 0
    answer = AnswerGenerator(_SilentLLM()).generate(state)
    assert "could not be checked" in (answer.answer or "").lower()
    assert "couldn't find sufficient" not in (answer.answer or "").lower()
    state.draft_answer = answer.answer
    verification = run_deterministic_checks(state)
    assert verification.metadata.get("absence_established") is False
    assert verification.metadata.get("retrieval_error") is True
    assert verification.recommendation == VerificationRecommendation.FALLBACK
    assert "No curriculum evidence supports the answer." not in verification.issues


def test_completed_lookup_can_establish_that_no_syllabus_exists():
    state, api = _run("all_absent", ["CLASS_3"])
    grade = _grade_states(state)["CLASS_3"]
    assert grade["status"] == "not_found"
    assert grade["api_status"] == 404
    assert grade["absence_established"] is True
    assert "No syllabus found" in (grade["error"] or "")
    assert state.evidence_status == EvidenceStatus.NOT_FOUND
    assert api.syllabus_calls == 1
    answer = AnswerGenerator(_SilentLLM()).generate(state)
    assert "couldn't find sufficient" in (answer.answer or "").lower()


def test_class3_timeout_preserves_class1_and_class2_evidence():
    state, api = _run("timeout_third", ["CLASS_1", "CLASS_2", "CLASS_3"])
    grades = _grade_states(state)
    assert grades["CLASS_1"]["status"] == "success"
    assert grades["CLASS_1"]["records"] == 7
    assert grades["CLASS_2"]["status"] == "success"
    assert grades["CLASS_2"]["records"] == 7
    assert grades["CLASS_3"]["status"] == "error"
    assert grades["CLASS_3"]["absence_established"] is False
    assert grades["CLASS_3"]["records"] == 0
    assert len(state.evidence) == 14
    assert state.evidence_status == EvidenceStatus.PARTIAL
    assert api.syllabus_calls == 0
    assert {item.grade for item in state.evidence} == {"CLASS_1", "CLASS_2"}
    answer = AnswerGenerator(StubLLMProvider()).generate(state)
    text = " ".join(answer.limitations)
    assert "CLASS_3" in text
    assert "could not be completed" in text
    assert "No syllabus was found for CLASS_3" not in text


def test_genuine_class3_absence_keeps_other_grades_and_records_the_miss():
    state, _api = _run("class3_absent", ["CLASS_1", "CLASS_2", "CLASS_3"])
    grades = _grade_states(state)
    assert grades["CLASS_1"]["records"] == 7
    assert grades["CLASS_2"]["records"] == 7
    assert grades["CLASS_3"]["status"] == "not_found"
    assert grades["CLASS_3"]["absence_established"] is True
    assert grades["CLASS_3"]["records"] == 0
    assert len(state.evidence) == 14
    assert state.evidence_status == EvidenceStatus.FOUND
    answer = AnswerGenerator(StubLLMProvider()).generate(state)
    assert "No syllabus was found for CLASS_3" in " ".join(answer.limitations)


def test_all_three_grades_timing_out_does_not_establish_absence():
    state, api = _run("timeout", ["CLASS_1", "CLASS_2", "CLASS_3"])
    assert state.evidence_status == EvidenceStatus.ERROR
    assert state.evidence == []
    grades = _grade_states(state)
    assert all(grades[grade]["infrastructure_invalid"] for grade in grades)
    assert all(not grades[grade]["absence_established"] for grade in grades)
    assert api.syllabus_calls == 0
    answer = AnswerGenerator(_SilentLLM()).generate(state)
    assert "could not be checked" in (answer.answer or "").lower()


def test_timeout_does_not_continue_into_a_downstream_syllabus_exception():
    state, api = _run("timeout_then_downstream", ["CLASS_3"])
    grade = _grade_states(state)["CLASS_3"]
    assert grade["status"] == "error"
    assert grade["api_status"] == 504
    assert api.syllabus_calls == 0
    assert "syllabus lookup must not run" not in (grade["error"] or "")
    assert state.evidence_status == EvidenceStatus.ERROR
