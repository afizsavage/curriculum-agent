import httpx
import pytest

from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.curriculum.errors import (
    CurriculumNotFoundError,
    CurriculumTimeoutError,
    CurriculumUnavailableError,
)


def _client(handler) -> CurriculumAPIClient:
    transport = httpx.MockTransport(handler)
    settings = Settings(
        curriculum_api_base_url="http://curriculum.test",
        curriculum_api_timeout=1.0,
    )
    return CurriculumAPIClient(settings=settings, transport=transport)


def test_client_get_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/v1/curricula")
        return httpx.Response(200, json={"items": [{"id": "c1", "code": "MBSSE-BEC"}], "total": 1})

    client = _client(handler)
    data = client.list_curricula(code="MBSSE-BEC")
    assert data["items"][0]["code"] == "MBSSE-BEC"


def test_client_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "missing"})

    client = _client(handler)
    with pytest.raises(CurriculumNotFoundError):
        client.get_subject("missing-id")


def test_client_500():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = _client(handler)
    with pytest.raises(CurriculumUnavailableError):
        client.list_curricula()


def test_client_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    client = _client(handler)
    with pytest.raises(CurriculumTimeoutError):
        client.list_curricula()
