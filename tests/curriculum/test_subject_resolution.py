"""Canonical subject codes are fixed before the first structure call."""

from __future__ import annotations

from app.agent.retrieve import RetrievalNode, _arguments_for_call
from app.agent.state import CurriculumQAState
from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.curriculum.codes import extract_filters_from_question, normalize_subject_code
from app.llm.base import LLMMessage, ToolCallRequest
from app.llm.provider import StubLLMProvider
from app.llm.tool_selection import select_tool_calls
from app.tools.base import ToolResult
from app.tools.curriculum import GetCurriculumStructureTool

ELA_QUESTION = "What units are published for the Class 4 English Language first term?"
JSS_QUESTION = "What is the first published unit of JSS 1 Integrated Science?"

ELA_UNIT_IDS = {
    "f583efc3-228b-44ff-a096-792f5ec85788",
    "1b80e24a-36f3-44ac-b10c-9c41f8b98af4",
    "8e463e73-cb89-4703-9108-cfe7e52997b1",
    "3b79e89d-6113-4e62-9ac0-e8052c9b1074",
    "adfdc60b-d380-4348-8268-d9b9022bf07d",
    "1c031ed1-d524-4664-b089-0e3caf102a5f",
    "7574b891-99ff-4f24-a917-c013245e2931",
    "ce770023-d448-42e5-81ff-644acb9d6f97",
    "7619175c-5402-46ae-ae71-252bd58db056",
    "7544d1d6-7128-4b0f-91fa-7274010be3d7",
}
JSS_UNIT_ID = "83b85707-2bfc-429e-bbff-9c2be76dd28a"

_STRUCTURE_TOOL = {
    "name": "get_curriculum_structure",
    "description": "",
    "parameters": {},
}


def test_english_language_is_resolved_before_structure_selection():
    filters = extract_filters_from_question(ELA_QUESTION)
    assert filters["grade"] == "CLASS_4"
    assert filters["subject"] == "ENGLISH_LANGUAGE"
    calls = select_tool_calls(
        [LLMMessage(role="user", content=f"Question: {ELA_QUESTION}")],
        [_STRUCTURE_TOOL],
    )
    assert calls[0].name == "get_curriculum_structure"
    assert calls[0].arguments["subject"] == "ENGLISH_LANGUAGE"
    assert calls[0].arguments["grade"] == "CLASS_4"


def test_integrated_science_is_resolved_before_structure_selection():
    filters = extract_filters_from_question(JSS_QUESTION)
    assert filters["grade"] == "JSS_1"
    assert filters["subject"] == "INTEGRATED_SCIENCE"
    calls = select_tool_calls(
        [LLMMessage(role="user", content=f"Question: {JSS_QUESTION}")],
        [_STRUCTURE_TOOL],
    )
    assert calls[0].name == "get_curriculum_structure"
    assert calls[0].arguments["subject"] == "INTEGRATED_SCIENCE"
    assert calls[0].arguments["grade"] == "JSS_1"


def test_structure_call_keeps_grade_and_replaces_short_subject():
    state = CurriculumQAState.initial(question=ELA_QUESTION)
    state.grade = "CLASS_4"
    state.level = "primary"
    state.subject = "ENGLISH_LANGUAGE"
    args = _arguments_for_call(
        state,
        "get_curriculum_structure",
        {"grade": "CLASS_4", "subject": "ENGLISH", "level": "primary"},
    )
    assert args["grade"] == "CLASS_4"
    assert args["level"] == "primary"
    assert args["subject"] == "ENGLISH_LANGUAGE"

    jss = CurriculumQAState.initial(question=JSS_QUESTION)
    jss.grade = "JSS_1"
    jss.level = "junior_secondary"
    jss.subject = "INTEGRATED_SCIENCE"
    jss_args = _arguments_for_call(
        jss,
        "get_curriculum_structure",
        {"grade": "JSS_1", "subject": "SCIENCE", "level": "junior_secondary"},
    )
    assert jss_args["grade"] == "JSS_1"
    assert jss_args["level"] == "junior_secondary"
    assert jss_args["subject"] == "INTEGRATED_SCIENCE"


def test_unresolved_subject_is_not_replaced_on_structure_call():
    state = CurriculumQAState.initial(question="What is basket weaving in Class 4?")
    state.grade = "CLASS_4"
    state.subject = None
    args = _arguments_for_call(
        state,
        "get_curriculum_structure",
        {"grade": "CLASS_4", "subject": "SCIENCE"},
    )
    assert args["subject"] == "SCIENCE"
    assert args["grade"] == "CLASS_4"
    assert normalize_subject_code("Basket Weaving") not in {
        "ENGLISH",
        "ENGLISH_LANGUAGE",
        "SCIENCE",
        "INTEGRATED_SCIENCE",
        "MATHEMATICS",
        "BIOLOGY",
    }


def test_execute_call_sends_resolved_subject_to_the_tool():
    class _Capture:
        def __init__(self) -> None:
            self.kwargs: dict | None = None

        def execute(self, name: str, **kwargs):
            self.kwargs = {"name": name, **kwargs}
            return ToolResult(success=True, data={"evidence": []})

    tools = _Capture()
    node = RetrievalNode(
        llm=StubLLMProvider(),
        tools=tools,  # type: ignore[arg-type]
        settings=Settings(
            _env_file=None,
            curriculum_api_base_url="http://curriculum.test",
        ),
    )
    state = CurriculumQAState.initial(question=JSS_QUESTION)
    state.grade = "JSS_1"
    state.subject = "INTEGRATED_SCIENCE"
    state.level = "junior_secondary"
    node._execute_call(
        state,
        ToolCallRequest(
            id="1",
            name="get_curriculum_structure",
            arguments={"grade": "JSS_1", "subject": "SCIENCE", "level": "junior_secondary"},
        ),
        request_id=None,
    )
    assert tools.kwargs is not None
    assert tools.kwargs["name"] == "get_curriculum_structure"
    assert tools.kwargs["subject"] == "INTEGRATED_SCIENCE"
    assert tools.kwargs["grade"] == "JSS_1"
    assert tools.kwargs["level"] == "junior_secondary"


def test_mathematics_science_and_biology_routing_stay_unchanged():
    math = extract_filters_from_question(
        "What topics are taught in Primary 4 Mathematics?"
    )
    science = extract_filters_from_question(
        "What is the first published unit of Class 4 Science?"
    )
    biology = extract_filters_from_question(
        "What topics are listed under the SSS 1 Biology theme Introduction to Biology?"
    )
    assert math["grade"] == "CLASS_4"
    assert math["subject"] == "MATHEMATICS"
    assert science["grade"] == "CLASS_4"
    assert science["subject"] == "SCIENCE"
    assert biology["grade"] == "SSS_1"
    assert biology["subject"] is None
    assert normalize_subject_code("Biology") == "BIOLOGY"


def test_live_structure_retrieves_english_language_and_integrated_science_units():
    tool = GetCurriculumStructureTool(CurriculumAPIClient(settings=Settings()))
    ela = tool.execute(grade="Class 4", subject="English Language")
    assert ela.success
    assert ela.data["grade"] == "CLASS_4"
    assert ela.data["subject"] == "ENGLISH_LANGUAGE"
    assert ela.data["curriculum"]["code"] == "MBSSE-BEC"
    assert ela.data["curriculum"]["version"] == "2020"
    ela_ids = {node["id"] for node in ela.data["nodes"]}
    assert ELA_UNIT_IDS <= ela_ids

    catalogue = tool.execute(grade="CLASS_4")
    assert catalogue.success
    catalogue_ids = {row.get("id") for row in catalogue.data["subjects"]}
    assert ELA_UNIT_IDS.isdisjoint(catalogue_ids)
    assert JSS_UNIT_ID not in catalogue_ids

    jss = tool.execute(grade="JSS 1", subject="Integrated Science")
    assert jss.success
    assert jss.data["grade"] == "JSS_1"
    assert jss.data["subject"] == "INTEGRATED_SCIENCE"
    assert jss.data["curriculum"]["code"] == "MBSSE-BEC"
    assert jss.data["curriculum"]["version"] == "2020"
    jss_ids = {node["id"] for node in jss.data["nodes"]}
    assert JSS_UNIT_ID in jss_ids
    assert JSS_UNIT_ID not in ela_ids
