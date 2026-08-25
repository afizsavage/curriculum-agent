import httpx
import pytest

from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.tools.curriculum import (
    GetCurriculumStructureTool,
    GetLearningObjectivesTool,
    GetSubjectTool,
    GetTopicTool,
    SearchCurriculumTool,
)


CURRICULUM_ID = "11111111-1111-1111-1111-111111111111"
SUBJECT_ID = "22222222-2222-2222-2222-222222222222"
SYLLABUS_ID = "33333333-3333-3333-3333-333333333333"
TOPIC_ID = "44444444-4444-4444-4444-444444444444"


def _router(request: httpx.Request) -> httpx.Response:
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
                        "name": "BEC",
                    }
                ],
                "total": 1,
                "limit": 20,
                "offset": 0,
            },
        )
    if path.endswith(f"/api/v1/curricula/{CURRICULUM_ID}/structure"):
        return httpx.Response(
            200,
            json={
                "id": CURRICULUM_ID,
                "code": "MBSSE-BEC",
                "education_levels": [
                    {
                        "id": "lvl",
                        "name": "Primary",
                        "grades": [
                            {
                                "id": "g4",
                                "code": "CLASS_4",
                                "name": "Class 4",
                                "subjects": [
                                    {
                                        "id": SUBJECT_ID,
                                        "code": "MATHEMATICS",
                                        "name": "Mathematics",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
        )
    if path.endswith(f"/api/v1/curricula/{CURRICULUM_ID}/subjects"):
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": SUBJECT_ID,
                        "code": "MATHEMATICS",
                        "name": "Mathematics",
                        "description": "Primary mathematics",
                    }
                ],
                "total": 1,
            },
        )
    if path.endswith(f"/api/v1/subjects/{SUBJECT_ID}"):
        return httpx.Response(
            200,
            json={
                "id": SUBJECT_ID,
                "code": "MATHEMATICS",
                "name": "Mathematics",
                "description": "Primary mathematics",
            },
        )
    if path.endswith("/api/v1/syllabuses"):
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": SYLLABUS_ID,
                        "code": "MBSSE-MATH-P4",
                        "subject_id": SUBJECT_ID,
                        "version": "2020",
                    }
                ],
                "total": 1,
            },
        )
    if path.endswith(f"/api/v1/syllabuses/{SYLLABUS_ID}/content/tree"):
        return httpx.Response(
            200,
            json=[
                {
                    "id": "strand-1",
                    "content_type": "STRAND",
                    "name": "Numbers",
                    "children": [
                        {
                            "id": TOPIC_ID,
                            "content_type": "TOPIC",
                            "name": "Fractions",
                            "code": "C4-U03",
                            "description": "Fractions topic",
                            "children": [],
                            "learning_outcomes": [
                                {
                                    "id": "lo-1",
                                    "code": "LO1",
                                    "description": "Identify fractions",
                                    "display_order": 1,
                                }
                            ],
                        }
                    ],
                }
            ],
        )
    if "curriculum-context" in path:
        return httpx.Response(
            200,
            json={
                "authoritative": {
                    "topic": {
                        "id": TOPIC_ID,
                        "name": "Fractions",
                        "code": "C4-U03",
                        "description": "Fractions topic",
                    },
                    "learning_outcomes": [
                        {
                            "id": "lo-1",
                            "code": "LO1",
                            "description": "Identify fractions",
                            "display_order": 1,
                        }
                    ],
                },
                "instructional_references": {"lesson_plans": []},
                "source": None,
            },
        )
    if path.endswith(f"/api/v1/topics/{TOPIC_ID}"):
        return httpx.Response(
            200,
            json={"id": TOPIC_ID, "name": "Fractions", "code": "C4-U03"},
        )
    if path.endswith(f"/api/v1/topics/{TOPIC_ID}/learning-outcomes"):
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "lo-1",
                        "description": "Identify fractions",
                        "display_order": 1,
                    }
                ],
                "total": 1,
            },
        )
    return httpx.Response(404, json={"detail": f"unhandled {path}"})


@pytest.fixture
def client() -> CurriculumAPIClient:
    settings = Settings(curriculum_api_base_url="http://curriculum.test")
    return CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_router)
    )


def test_search_curriculum_valid(client):
    tool = SearchCurriculumTool(client)
    result = tool.execute(
        query="fractions", grade="Primary 4", subject="Mathematics"
    )
    assert result.success
    assert result.data["count"] >= 1
    assert result.data["results"][0]["name"] == "Fractions"
    assert result.data["evidence"]


def test_search_curriculum_missing_query(client):
    tool = SearchCurriculumTool(client)
    result = tool.execute(query="")
    assert result.success is False
    assert result.data["error_code"] == "CURRICULUM_INVALID_QUERY"


def test_get_curriculum_structure_topics(client):
    tool = GetCurriculumStructureTool(client)
    result = tool.execute(grade="Primary 4", subject="Mathematics")
    assert result.success
    names = [n["name"] for n in result.data["nodes"]]
    assert "Fractions" in names


def test_get_curriculum_structure_subjects_only(client):
    tool = GetCurriculumStructureTool(client)
    result = tool.execute(grade="Primary 4")
    assert result.success
    assert result.data["subjects"][0]["code"] == "MATHEMATICS"


def test_get_subject(client):
    tool = GetSubjectTool(client)
    result = tool.execute(grade="Primary 4", subject="Mathematics")
    assert result.success
    assert result.data["subject"]["code"] == "MATHEMATICS"


def test_get_topic_by_id(client):
    tool = GetTopicTool(client)
    result = tool.execute(topic_id=TOPIC_ID)
    assert result.success
    assert result.data["topic"]["name"] == "Fractions"


def test_get_topic_by_name(client):
    tool = GetTopicTool(client)
    result = tool.execute(
        topic="Fractions", grade="Primary 4", subject="Mathematics"
    )
    assert result.success
    assert result.data["topic"]["id"] == TOPIC_ID


def test_get_learning_objectives(client):
    tool = GetLearningObjectivesTool(client)
    result = tool.execute(topic_id=TOPIC_ID)
    assert result.success
    assert result.data["objectives"][0]["text"] == "Identify fractions"


def test_invalid_topic_id(client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v1/curricula"):
            return _router(request)
        return httpx.Response(404, json={"detail": "not found"})

    bad = CurriculumAPIClient(
        settings=Settings(curriculum_api_base_url="http://curriculum.test"),
        transport=httpx.MockTransport(handler),
    )
    result = GetTopicTool(bad).execute(topic_id="00000000-0000-0000-0000-000000000099")
    assert result.success is False
    assert result.data["error_code"] == "CURRICULUM_NOT_FOUND"


def test_api_500_mapped(client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v1/curricula"):
            return httpx.Response(500, json={"detail": "down"})
        return httpx.Response(404, json={"detail": "x"})

    down = CurriculumAPIClient(
        settings=Settings(curriculum_api_base_url="http://curriculum.test"),
        transport=httpx.MockTransport(handler),
    )
    result = SearchCurriculumTool(down).execute(query="fractions", grade="Primary 4")
    assert result.success is False
    assert result.data["error_code"] == "CURRICULUM_UNAVAILABLE"
