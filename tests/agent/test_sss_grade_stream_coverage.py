"""Grade, stream, and subject stay together for SSS coverage questions."""

from __future__ import annotations

import re

import httpx
import pytest

from app.agent.context import ConversationStore
from app.agent.orchestrator import CurriculumQAAgent
from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.curriculum.sss_stream_intent import (
    detect_sss_stream_subjects,
    match_sss_stream,
    match_stream_subject,
)
from app.enums import AgentStatus
from app.llm.provider import StubLLMProvider
from app.tools.registry import build_default_registry

CURRICULUM_ID = "curr-ssc"
SCIENCE_STREAM = "stream-science"
BUSINESS_STREAM = "stream-business"
BIO = "subject-bio"
ICT = "subject-ict"
ICT_SSS1 = "grade-ict-s1"
LITERACY = "subject-literacy"
ACCOUNTING = "subject-accounting"
BIO_SSS1 = "grade-bio-s1"
BIO_SSS2 = "grade-bio-s2"
LITERACY_SSS2 = "grade-literacy-s2"

_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.I,
)


def _streams() -> dict:
    return {
        "items": [
            {
                "id": SCIENCE_STREAM,
                "name": "Sciences & Technologies",
                "code": "SCIENCES_TECHNOLOGIES",
                "curriculum_id": CURRICULUM_ID,
            },
            {
                "id": BUSINESS_STREAM,
                "name": "Economics, Business & Entrepreneurship",
                "code": "ECONOMICS_BUSINESS_ENTREPRENEURSHIP",
                "curriculum_id": CURRICULUM_ID,
            },
        ],
        "total": 2,
        "limit": 200,
        "offset": 0,
    }


def _assignment(subject_id: str, name: str, code: str, order: int) -> dict:
    return {
        "id": f"assign-{subject_id}",
        "subject_id": subject_id,
        "subject_type": "CORE",
        "display_order": order,
        "subject": {"id": subject_id, "name": name, "code": code, "status": "ACTIVE"},
    }


SCIENCE_SUBJECTS = [
    _assignment(BIO, "Biology", "BIOLOGY", 1),
    _assignment(ICT, "Information & Communication Technology (ICT)", "ICT", 2),
    _assignment(LITERACY, "Computer Literacy", "COMPUTER_LITERACY", 3),
]
BUSINESS_SUBJECTS = [
    _assignment(ACCOUNTING, "Principles of Accounting", "PRINCIPLES_ACCOUNTING", 1),
]


def _grade_row(row_id: str, grade: str, subject_id: str, name: str, source: str) -> dict:
    return {
        "id": row_id,
        "grade": {"code": grade, "name": grade.replace("_", " ")},
        "subject": {"id": subject_id, "name": name},
        "subject_id": subject_id,
        "source_reference": source,
    }


def _tree(theme: str, topics: list[str]) -> list[dict]:
    return [
        {
            "id": f"theme-{theme}",
            "content_type": "THEME",
            "name": theme,
            "children": [
                {"id": f"topic-{topic}", "content_type": "TOPIC", "name": topic, "children": []}
                for topic in topics
            ],
        }
    ]


GRADE_ROWS = [
    _grade_row(BIO_SSS1, "SSS_1", BIO, "Biology", "SSS-Syllabus-Biology.pdf"),
    _grade_row(
        ICT_SSS1,
        "SSS_1",
        ICT,
        "Information & Communication Technology (ICT)",
        "SSS-Syllabus-ICT.pdf",
    ),
    _grade_row(BIO_SSS2, "SSS_2", BIO, "Biology", "SSS-Syllabus-Biology.pdf"),
    _grade_row(
        LITERACY_SSS2,
        "SSS_2",
        LITERACY,
        "Computer Literacy",
        "SSS-Syllabus-Computer-Literacy.pdf",
    ),
    _grade_row(
        "grade-accounting-s1",
        "SSS_1",
        ACCOUNTING,
        "Principles of Accounting",
        "SSS-Syllabus-Accounting.pdf",
    ),
]
CONTENT = {
    BIO_SSS1: _tree("Introduction to Biology", ["Living and non-living things"]),
    ICT_SSS1: _tree("Data Representation", ["Number systems"]),
    BIO_SSS2: _tree("Transport System", ["Transport in animals"]),
    LITERACY_SSS2: _tree("Advance Word Processing", ["Find and replace"]),
    "grade-accounting-s1": _tree("Source documents", ["Invoices and receipts"]),
}


def _router() -> tuple[list[str], httpx.MockTransport]:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        if path.endswith("/api/v1/curricula"):
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": CURRICULUM_ID,
                            "code": "MBSSE-SSC",
                            "version": "2021",
                            "name": "SSC",
                        }
                    ],
                    "total": 1,
                },
            )
        if path.endswith(f"/api/v1/curricula/{CURRICULUM_ID}/sss-streams"):
            return httpx.Response(200, json=_streams())
        if path.endswith(f"/api/v1/sss-streams/{SCIENCE_STREAM}"):
            return httpx.Response(
                200,
                json={"id": SCIENCE_STREAM, "name": "Sciences & Technologies"},
            )
        if path.endswith(f"/api/v1/sss-streams/{BUSINESS_STREAM}"):
            return httpx.Response(
                200,
                json={
                    "id": BUSINESS_STREAM,
                    "name": "Economics, Business & Entrepreneurship",
                },
            )
        if path.endswith(f"/api/v1/sss-streams/{SCIENCE_STREAM}/subjects"):
            return httpx.Response(
                200,
                json={
                    "items": SCIENCE_SUBJECTS,
                    "total": len(SCIENCE_SUBJECTS),
                    "limit": 200,
                    "offset": 0,
                },
            )
        if path.endswith(f"/api/v1/sss-streams/{BUSINESS_STREAM}/subjects"):
            return httpx.Response(
                200,
                json={
                    "items": BUSINESS_SUBJECTS,
                    "total": len(BUSINESS_SUBJECTS),
                    "limit": 200,
                    "offset": 0,
                },
            )
        if path.endswith(f"/api/v1/curricula/{CURRICULUM_ID}/grade-curricula"):
            return httpx.Response(
                200,
                json={"items": GRADE_ROWS, "total": len(GRADE_ROWS), "limit": 200, "offset": 0},
            )
        prefix = "/api/v1/grade-curricula/"
        if path.startswith(prefix) and path.endswith("/content"):
            row_id = path[len(prefix) : -len("/content")]
            return httpx.Response(200, json=CONTENT.get(row_id, []))
        return httpx.Response(404, json={"detail": f"unhandled {path}"})

    return calls, httpx.MockTransport(handler)


def _ask(question: str) -> tuple[list[str], object]:
    calls, transport = _router()
    settings = Settings(
        llm_provider="stub",
        curriculum_api_base_url="http://curriculum.test",
        agent_max_iterations=3,
        agent_max_tool_calls=5,
    )
    client = CurriculumAPIClient(settings=settings, transport=transport)
    state = CurriculumQAAgent(
        settings=settings,
        llm=StubLLMProvider(),
        tools=build_default_registry(settings=settings, client=client),
        conversations=ConversationStore(),
    ).ask(question)
    return calls, state


def test_ict_literacy_question_keeps_grade_stream_and_subject():
    detected = detect_sss_stream_subjects(
        "What does ICT Literacy cover for SSS1 Science and Technology stream?"
    )
    assert detected is not None
    assert detected.grade == "SSS_1"
    assert detected.subject_name == "ICT Literacy"
    assert detected.stream_name == "Science and Technology"
    assert detected.focus == "coverage"


def test_ict_literacy_does_not_borrow_another_subject_or_grade():
    calls, state = _ask(
        "What does ICT literacy cover for SSS1 Science and Technology?"
    )
    answer = state.final_answer or ""
    assert state.grade == "SSS_1"
    assert state.metadata["sss_stream_resolution"] == "ambiguous_subject"
    assert state.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert "does not list" in answer
    assert "alternative name" in answer
    assert "If you mean" in answer
    assert "couldn't find sufficient" not in answer.lower()
    assert "Computer Literacy" in answer
    assert "Information & Communication Technology (ICT)" in answer
    assert "()" not in answer
    assert "Find and replace" not in answer
    assert "Living and non-living things" not in answer
    assert "Advance Word Processing" not in answer
    assert _UUID.search(answer) is None
    assert not any(path.endswith("/content") for path in calls)


def test_biology_sss1_uses_only_sss1_topics():
    calls, state = _ask(
        "What does Biology cover for SSS1 Sciences & Technologies?"
    )
    answer = state.final_answer or ""
    assert state.status == AgentStatus.COMPLETED
    assert state.grade == "SSS_1"
    assert "Living and non-living things" in answer
    assert "Transport in animals" not in answer
    assert "SSS-Syllabus-Biology.pdf" in answer
    assert _UUID.search(answer) is None
    assert any(path.endswith(f"/{BIO_SSS1}/content") for path in calls)
    assert not any(BIO_SSS2 in path for path in calls)


def test_biology_sss2_does_not_reuse_sss1_topics():
    _calls, state = _ask(
        "What does Biology cover for SSS2 Science and Technology stream?"
    )
    answer = state.final_answer or ""
    assert state.status == AgentStatus.COMPLETED
    assert state.grade == "SSS_2"
    assert "Transport in animals" in answer
    assert "Living and non-living things" not in answer


def test_accounting_is_not_answered_from_the_science_stream():
    calls, state = _ask(
        "What does Principles of Accounting cover for SSS1 Sciences & Technologies?"
    )
    answer = state.final_answer or ""
    assert state.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert "Invoices and receipts" not in answer
    assert "not assign" in answer.lower()
    assert not any(path.endswith("/content") for path in calls)


def test_accounting_sss1_uses_the_business_stream():
    calls, state = _ask(
        "What does Principles of Accounting cover for SSS1 "
        "Economics, Business & Entrepreneurship?"
    )
    answer = state.final_answer or ""
    assert state.status == AgentStatus.COMPLETED
    assert "Invoices and receipts" in answer
    assert "Living and non-living things" not in answer
    assert any("grade-accounting-s1/content" in path for path in calls)
    assert not any(SCIENCE_STREAM in path and path.endswith("/subjects") for path in calls)


def test_grade_stream_subject_list_uses_that_grades_records():
    _calls, state = _ask(
        "What subjects are offered in SSS1 Science and Technology?"
    )
    answer = state.final_answer or ""
    assert state.status == AgentStatus.COMPLETED
    assert state.grade == "SSS_1"
    assert "Biology" in answer
    assert "Computer Literacy" not in answer
    assert "Principles of Accounting" not in answer


@pytest.mark.parametrize(
    "question",
    [
        "What topics are covered in Biology for SSS1 Sciences & Technologies?",
        "What are the learning outcomes for Biology in SSS1 Science and Technology stream?",
        "What should students learn in Biology in SSS1 Sciences & Technologies?",
    ],
)
def test_other_question_shapes_stay_on_the_sss1_syllabus(question: str):
    _calls, state = _ask(question)
    answer = state.final_answer or ""
    assert state.grade == "SSS_1"
    assert "Living and non-living things" in answer
    assert "Transport in animals" not in answer


def test_membership_confirms_the_stream_without_another_grades_topics():
    calls, state = _ask(
        "Is Biology part of the SSS1 Science and Technology stream?"
    )
    answer = state.final_answer or ""
    assert state.status == AgentStatus.COMPLETED
    assert "Biology" in answer
    assert "Sciences & Technologies" in answer
    assert "Living and non-living things" not in answer
    assert not any(path.endswith("/content") for path in calls)


def test_stream_list_is_not_grade_specific():
    _calls, state = _ask("What streams are available at SSS level?")
    answer = state.final_answer or ""
    assert state.status == AgentStatus.COMPLETED
    assert "Sciences & Technologies" in answer
    assert "Economics, Business & Entrepreneurship" in answer
    assert "not define a different set of streams" in answer


def test_stream_cover_question_lists_subjects_for_that_grade():
    _calls, state = _ask(
        "What does the Science and Technology stream cover in SSS2?"
    )
    answer = state.final_answer or ""
    assert state.status == AgentStatus.COMPLETED
    assert state.grade == "SSS_2"
    assert "Biology" in answer
    assert "Computer Literacy" in answer
    assert "Information & Communication Technology (ICT)" not in answer
    assert "Find and replace" not in answer


OFFICIAL_STREAMS = [
    {"name": "Sciences & Technologies"},
    {"name": "Mathematics & Numeracy"},
    {"name": "Languages & Literatures"},
    {"name": "Social & Cultural Studies"},
    {"name": "Economics, Business & Entrepreneurship"},
]


@pytest.mark.parametrize(
    ("spoken", "official"),
    [
        ("Science and Technology", "Sciences & Technologies"),
        ("Sciences and Technologies", "Sciences & Technologies"),
        ("Mathematics and Numeracy", "Mathematics & Numeracy"),
        ("Language and Literature", "Languages & Literatures"),
        ("Languages and Literatures", "Languages & Literatures"),
        ("Social and Cultural Studies", "Social & Cultural Studies"),
        (
            "Economics, Business and Entrepreneurship",
            "Economics, Business & Entrepreneurship",
        ),
    ],
)
def test_spoken_stream_titles_match_the_official_name(spoken: str, official: str):
    matched = match_sss_stream(OFFICIAL_STREAMS, spoken)
    assert matched is not None
    assert matched["name"] == official


@pytest.mark.parametrize("spoken", ["Science", "Business", "Humanities", "Social Science"])
def test_short_stream_words_do_not_match_an_official_stream(spoken: str):
    assert match_sss_stream(OFFICIAL_STREAMS, spoken) is None


def test_official_acronym_resolves_without_using_another_subject():
    calls, state = _ask("What does ICT cover for SSS1 Sciences & Technologies?")
    answer = state.final_answer or ""
    assert state.status == AgentStatus.COMPLETED
    assert state.grade == "SSS_1"
    assert "Number systems" in answer
    assert "Information & Communication Technology (ICT)" in answer
    assert "Find and replace" not in answer
    assert "Living and non-living things" not in answer
    assert any(path.endswith(f"/{ICT_SSS1}/content") for path in calls)
    assert not any(LITERACY_SSS2 in path for path in calls)


def test_science_and_technology_uses_the_official_stream_syllabus():
    calls, state = _ask(
        "What does Biology cover for SSS1 Science and Technology?"
    )
    answer = state.final_answer or ""
    assert state.status == AgentStatus.COMPLETED
    assert state.grade == "SSS_1"
    assert "Sciences & Technologies" in answer
    assert "Living and non-living things" in answer
    assert not any(BIO_SSS2 in path for path in calls)


def test_hyphenated_grade_keeps_the_sss1_syllabus():
    calls, state = _ask("What does Biology cover for SSS-1 Science and Technology?")
    answer = state.final_answer or ""
    assert state.grade == "SSS_1"
    assert "Living and non-living things" in answer
    assert "Transport in animals" not in answer
    assert not any(BIO_SSS2 in path for path in calls)


@pytest.mark.parametrize(
    ("question", "grade", "stream"),
    [
        (
            "What does Biology cover for SSS 1 Sciences & Technologies?",
            "SSS_1",
            "Sciences & Technologies",
        ),
        (
            "What does Biology cover for Senior Secondary 2 Science and Technology?",
            "SSS_2",
            "Science and Technology",
        ),
        (
            "What does Biology cover for Senior Secondary School 3 Sciences & Technologies?",
            "SSS_3",
            "Sciences & Technologies",
        ),
    ],
)
def test_grade_wording_keeps_grade_and_stream(question: str, grade: str, stream: str):
    detected = detect_sss_stream_subjects(question)
    assert detected is not None
    assert detected.grade == grade
    assert detected.stream_name == stream
    assert detected.subject_name == "Biology"


@pytest.mark.parametrize(
    "question",
    [
        "What does Biology cover for SSS1 Science?",
        "What does Accounting cover for SSS2 Business?",
        "What does History cover for SSS1 Humanities?",
    ],
)
def test_single_word_stream_hints_are_not_inferred(question: str):
    assert detect_sss_stream_subjects(question) is None


def test_unknown_subject_is_not_replaced_by_a_stream_subject():
    calls, state = _ask(
        "What does Basket Weaving cover for SSS1 Sciences & Technologies?"
    )
    answer = state.final_answer or ""
    assert state.metadata["sss_stream_resolution"] == "subject_not_in_stream"
    assert state.status == AgentStatus.INSUFFICIENT_EVIDENCE
    assert "Basket Weaving" in answer
    assert "Computer Literacy" not in answer
    assert "Biology" not in answer
    assert "If you mean" not in answer
    assert not any(path.endswith("/content") for path in calls)


def test_ict_literacy_is_not_an_alias_for_either_official_subject():
    subject, candidates, kind = match_stream_subject(
        SCIENCE_SUBJECTS, "ICT Literacy"
    )
    assert subject is None
    assert kind == "ambiguous"
    names = {
        row["subject"]["name"]
        for row in candidates
    }
    assert names == {
        "Information & Communication Technology (ICT)",
        "Computer Literacy",
    }
    subject, _candidates, kind = match_stream_subject(SCIENCE_SUBJECTS, "ICT")
    assert kind == "acronym"
    assert subject is not None
    assert subject["subject"]["name"] == "Information & Communication Technology (ICT)"
