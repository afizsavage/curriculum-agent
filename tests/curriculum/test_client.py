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


def test_client_sss_stream_routes():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("/subjects"):
            return httpx.Response(200, json={"items": [], "total": 0, "limit": 200, "offset": 0})
        if request.url.path.endswith("/sss-streams/stream-1"):
            return httpx.Response(
                200, json={"id": "stream-1", "name": "Sciences & Technologies"}
            )
        return httpx.Response(
            200, json={"items": [], "total": 0, "limit": 200, "offset": 0}
        )

    client = _client(handler)
    client.list_sss_streams("curr-1", limit=200)
    client.get_sss_stream("stream-1")
    client.list_sss_stream_subjects("stream-1", limit=200)
    assert seen[0].endswith("/api/v1/curricula/curr-1/sss-streams")
    assert seen[1].endswith("/api/v1/sss-streams/stream-1")
    assert seen[2].endswith("/api/v1/sss-streams/stream-1/subjects")


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
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"detail": "boom"})

    client = _client(handler)
    with pytest.raises(CurriculumUnavailableError):
        client.list_curricula()
    assert calls["n"] == 2
    assert client.identity_lookups[-1]["status"] == "failed"
    assert client.identity_lookups[-1]["retry_used"] is True
    assert client.identity_lookups[-1]["curriculum_identity_attempts"] == 2


def test_client_timeout():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("slow")

    client = _client(handler)
    with pytest.raises(CurriculumTimeoutError):
        client.list_curricula()
    assert calls["n"] == 2
    assert client.identity_lookups[-1]["retry_used"] is True


def test_curriculum_identity_retries_a_timeout_once():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("slow")
        return httpx.Response(
            200, json={"items": [{"id": "c1", "code": "MBSSE-SSC"}], "total": 1}
        )

    client = _client(handler)
    data = client.list_curricula(code="MBSSE-SSC", limit=20)
    assert data["items"][0]["id"] == "c1"
    assert calls["n"] == 2
    record = client.identity_lookups[-1]
    assert record["curriculum_identity_attempts"] == 2
    assert record["retry_used"] is True
    assert record["status"] == "recovered"
    assert record["retry_duration_ms"] is not None


def test_curriculum_identity_does_not_retry_not_found():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, json={"detail": "missing"})

    client = _client(handler)
    with pytest.raises(CurriculumNotFoundError):
        client.list_curricula()
    assert calls["n"] == 1
    assert client.identity_lookups[-1]["retry_used"] is False
    assert client.identity_lookups[-1]["curriculum_identity_attempts"] == 1
    assert client.identity_lookups[-1]["status"] == "failed"


def test_other_curriculum_gets_do_not_retry_on_timeout():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("slow")

    client = _client(handler)
    with pytest.raises(CurriculumTimeoutError):
        client.list_sss_streams("curr-1")
    assert calls["n"] == 1
    assert client.identity_lookups == []
