"""Assistant HTTP boundary. The route stays dark unless the flag is on."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.assistant.chat import configure_assistant_runtime, reset_assistant_runtime
from app.assistant.identity import (
    TrustedTeacher,
    UnauthenticatedTeacherError,
    VerifiedTeacherClaims,
    authenticate_http_request,
    bind_verified_claims,
    current_identity_gate,
    register_identity_gate,
    reset_identity_gate,
)
from app.assistant.threads import AssistantThreadStore, ThreadStoreError
from app.config import Settings
from app.curriculum.evidence import EvidenceStatus
from app.enums import AgentStatus
from app.main import create_app
from app.schemas.verification import VerificationRecommendation, VerificationStatus
from tests.assistant.test_qa_adapter import FakeQAAgent, _state

_CURRICULUM = "What topics are taught in Primary 4 Mathematics?"
_GENERAL = "How should I settle a noisy class before the lesson starts?"
_MIXED = (
    "What does the Class 4 Mathematics syllabus say about fractions, "
    "and how should I teach it?"
)
_ADVICE = "Use counters, then ask pupils to explain one step."
_GAP = "I couldn't find sufficient MBSSE curriculum evidence to answer this reliably."


class _Gate:
    def __init__(self, teacher_id: str = "teacher-a", school_id: str | None = None) -> None:
        self.teacher_id = teacher_id
        self.school_id = school_id
        self.calls = 0

    def verify(self, request: object) -> VerifiedTeacherClaims:
        self.calls += 1
        return VerifiedTeacherClaims(self.teacher_id, self.school_id)


class _Calls:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def __call__(self, request) -> str:
        self.questions.append(request.question)
        return _ADVICE


@pytest.fixture(autouse=True)
def _reset_boundaries():
    reset_identity_gate()
    reset_assistant_runtime()
    yield
    reset_identity_gate()
    reset_assistant_runtime()


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "_env_file": None,
        "environment": "development",
        "assistant_enabled": False,
        "assistant_local_stub_identity": False,
        "assistant_local_stub_teacher_id": "local-stub-teacher",
        "assistant_thread_sqlite_path": str(tmp_path / "assistant_threads.sqlite"),
        "llm_provider": "stub",
    }
    values.update(overrides)
    return Settings(**values)


def _post(monkeypatch, settings: Settings, payload: dict, headers: dict | None = None):
    monkeypatch.setattr("app.api.v1.assistant.get_settings", lambda: settings)
    application = create_app()
    with TestClient(application, raise_server_exceptions=False) as client:
        return client.post("/api/v1/assistant/chat", json=payload, headers=headers)


def test_startup_leaves_the_assistant_unauthenticated():
    reset_identity_gate()
    application = create_app()
    assert current_identity_gate() is None
    assert Settings.model_fields["assistant_enabled"].default is False
    assert Settings.model_fields["assistant_local_stub_identity"].default is False
    paths = set(application.openapi()["paths"])
    assert "/api/v1/assistant/chat" in paths
    assert "/api/v1/agent/ask" in paths


def test_feature_flag_defaults_off():
    assert Settings.model_fields["assistant_enabled"].default is False
    assert Settings.model_fields["assistant_local_stub_identity"].default is False
    loaded = Settings(_env_file=None)
    assert loaded.assistant_enabled is False
    assert loaded.assistant_local_stub_identity is False


def test_disabled_request_does_not_execute_assistant_work(tmp_path: Path, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("assistant work ran while disabled")

    monkeypatch.setattr("app.api.v1.assistant.execute_chat", refuse)
    monkeypatch.setattr("app.assistant.chat.execute_chat", refuse)
    monkeypatch.setattr("app.assistant.chat.AssistantThreadStore", refuse)
    monkeypatch.setattr("app.assistant.chat.dispatch_turn", refuse)
    monkeypatch.setattr("app.assistant.chat.get_agent", refuse)
    monkeypatch.setattr("app.llm.provider.build_llm_provider", refuse)
    response = _post(
        monkeypatch,
        _settings(tmp_path),
        {
            "question": _CURRICULUM,
            "teacher_id": "forged-teacher",
            "thread_id": "forged-thread",
        },
        headers={"X-Teacher-Id": "forged-teacher"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Not found"
    assert not (tmp_path / "assistant_threads.sqlite").exists()


def test_enabled_without_identity_is_rejected_before_the_store(tmp_path: Path, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("thread store opened without a principal")

    monkeypatch.setattr("app.assistant.chat.AssistantThreadStore", refuse)
    monkeypatch.setattr("app.assistant.chat.dispatch_turn", refuse)
    monkeypatch.setattr("app.assistant.chat.get_agent", refuse)
    response = _post(
        monkeypatch,
        _settings(tmp_path, assistant_enabled=True),
        {"question": _GENERAL, "teacher_id": "forged-teacher"},
        headers={"X-Teacher-Id": "forged-teacher"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Authenticated teacher principal is required"
    assert not (tmp_path / "assistant_threads.sqlite").exists()


def test_untrusted_sources_cannot_authorize(tmp_path: Path, monkeypatch):
    class HeaderGate:
        def verify(self, request: object):
            header = request.headers.get("X-Teacher-Id")
            return {"teacher_id": header or "body"}

    register_identity_gate(HeaderGate())
    monkeypatch.setattr(
        "app.assistant.chat.AssistantThreadStore",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("store opened")),
    )
    response = _post(
        monkeypatch,
        _settings(tmp_path, assistant_enabled=True, environment="production"),
        {"question": _GENERAL, "teacher_id": "forged-teacher"},
        headers={"X-Teacher-Id": "forged-teacher"},
    )
    assert response.status_code == 401
    assert not (tmp_path / "assistant_threads.sqlite").exists()

    other = _Gate("not-registered")
    with pytest.raises(UnauthenticatedTeacherError):
        bind_verified_claims(VerifiedTeacherClaims("not-registered"), gate=other)
    with pytest.raises(UnauthenticatedTeacherError):
        TrustedTeacher()
    assert not hasattr(TrustedTeacher, "from_verified_issuer")


def test_production_refuses_local_stub_identity(tmp_path: Path):
    settings = _settings(
        tmp_path,
        environment="production",
        assistant_enabled=True,
        assistant_local_stub_identity=True,
    )
    with pytest.raises(UnauthenticatedTeacherError):
        authenticate_http_request(object(), settings=settings)


def test_local_stub_ignores_the_request_teacher_id(tmp_path: Path, monkeypatch):
    general = _Calls()
    agent = FakeQAAgent(
        _state(
            answer="should-not-run",
            status=AgentStatus.COMPLETED,
            evidence_status=EvidenceStatus.FOUND,
            verification_status=VerificationStatus.PASSED,
            recommendation=VerificationRecommendation.ACCEPT,
            passed=True,
        )
    )
    configure_assistant_runtime(qa_agent=agent, general=general)
    monkeypatch.setattr(
        "app.assistant.chat.get_agent",
        lambda: (_ for _ in ()).throw(AssertionError("qa agent constructed")),
    )
    response = _post(
        monkeypatch,
        _settings(
            tmp_path,
            assistant_enabled=True,
            assistant_local_stub_identity=True,
            assistant_local_stub_teacher_id="local-stub-teacher",
        ),
        {"question": _GENERAL, "teacher_id": "forged-teacher"},
        headers={"X-Teacher-Id": "forged-teacher"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["route"] == "general"
    assert body["general_text"] == _ADVICE
    assert body["general_label"] == "General AI assistance"
    assert body["curriculum"] is None
    assert agent.calls == []
    raw = sqlite3.connect(tmp_path / "assistant_threads.sqlite")
    try:
        teacher_id = raw.execute("SELECT teacher_id FROM threads").fetchone()[0]
    finally:
        raw.close()
    assert teacher_id == "local-stub-teacher"


def test_registered_gate_is_the_only_non_stub_principal(tmp_path: Path):
    gate = _Gate("verified-teacher", "school-1")
    duplicate = _Gate("verified-teacher", "school-1")
    register_identity_gate(gate)
    with pytest.raises(UnauthenticatedTeacherError):
        bind_verified_claims(VerifiedTeacherClaims("verified-teacher"), gate=duplicate)
    settings = _settings(
        tmp_path,
        assistant_local_stub_identity=True,
        assistant_local_stub_teacher_id="local-stub-teacher",
    )
    request = type("Request", (), {"teacher_id": "forged-teacher"})()
    principal = authenticate_http_request(request, settings=settings)
    assert principal.teacher_id == "verified-teacher"
    assert principal.school_id == "school-1"
    assert gate.calls == 1


def test_cross_teacher_and_unknown_thread_fail_before_handlers(tmp_path: Path, monkeypatch):
    gate = _Gate("teacher-a")
    register_identity_gate(gate)
    agent = FakeQAAgent(
        _state(
            answer="Primary 4 Mathematics includes Number and Numeration.",
            status=AgentStatus.COMPLETED,
            evidence_status=EvidenceStatus.FOUND,
            verification_status=VerificationStatus.PASSED,
            recommendation=VerificationRecommendation.ACCEPT,
            passed=True,
            coverage_decision="accept",
        )
    )
    general = _Calls()
    configure_assistant_runtime(qa_agent=agent, general=general)
    created = _post(
        monkeypatch,
        _settings(tmp_path, assistant_enabled=True),
        {"question": _CURRICULUM},
    )
    assert created.status_code == 200, created.text
    thread_id = created.json()["thread_id"]
    assert agent.calls[0]["conversation_id"] is None
    assert thread_id != created.json()["curriculum"]["conversation_id"]
    agent.calls.clear()
    general.questions.clear()

    gate.teacher_id = "teacher-b"
    denied = _post(
        monkeypatch,
        _settings(tmp_path, assistant_enabled=True),
        {"question": _CURRICULUM, "thread_id": thread_id, "teacher_id": "teacher-b"},
    )
    unknown = _post(
        monkeypatch,
        _settings(tmp_path, assistant_enabled=True),
        {"question": _CURRICULUM, "thread_id": "missing-thread"},
    )
    assert denied.status_code == unknown.status_code == 404
    assert denied.json()["detail"] == unknown.json()["detail"] == "Thread not found"
    assert "Number and Numeration" not in denied.text
    assert agent.calls == []
    assert general.questions == []


def test_general_and_curriculum_use_their_own_handlers(tmp_path: Path, monkeypatch):
    register_identity_gate(_Gate("teacher-a"))
    agent = FakeQAAgent(
        _state(
            answer="Primary 4 Mathematics includes Number and Numeration.",
            status=AgentStatus.COMPLETED,
            evidence_status=EvidenceStatus.FOUND,
            verification_status=VerificationStatus.PASSED,
            recommendation=VerificationRecommendation.ACCEPT,
            passed=True,
            coverage_decision="accept",
        )
    )
    general = _Calls()
    configure_assistant_runtime(qa_agent=agent, general=general)
    monkeypatch.setattr(
        "app.assistant.chat.get_agent",
        lambda: (_ for _ in ()).throw(AssertionError("default qa agent constructed")),
    )
    monkeypatch.setattr(
        "app.llm.provider.build_llm_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("model provider built")),
    )
    settings = _settings(tmp_path, assistant_enabled=True)
    general_response = _post(monkeypatch, settings, {"question": _GENERAL})
    assert general_response.status_code == 200, general_response.text
    assert general_response.json()["route"] == "general"
    assert general_response.json()["curriculum"] is None
    assert agent.calls == []
    assert general.questions == [_GENERAL]

    curriculum_response = _post(monkeypatch, settings, {"question": _CURRICULUM})
    assert curriculum_response.status_code == 200, curriculum_response.text
    body = curriculum_response.json()
    assert body["route"] == "curriculum"
    assert body["general_text"] is None
    assert "Number and Numeration" in body["curriculum"]["answer"]
    assert len(agent.calls) == 1
    assert agent.calls[0]["conversation_id"] is None
    assert general.questions == [_GENERAL]


def test_mixed_request_keeps_curriculum_evidence_restrictions(tmp_path: Path, monkeypatch):
    register_identity_gate(_Gate("teacher-a"))
    gap = FakeQAAgent(
        _state(
            answer=_GAP,
            status=AgentStatus.INSUFFICIENT_EVIDENCE,
            evidence_status=EvidenceStatus.NOT_FOUND,
            verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            recommendation=VerificationRecommendation.FALLBACK,
            passed=False,
        )
    )

    def refuse_general(_request):
        raise AssertionError("general advice ran after insufficient evidence")

    configure_assistant_runtime(qa_agent=gap, general=refuse_general)
    blocked = _post(
        monkeypatch,
        _settings(tmp_path, assistant_enabled=True),
        {"question": _MIXED},
    )
    assert blocked.status_code == 200, blocked.text
    body = blocked.json()
    assert body["route"] == "mixed"
    assert body["general_text"] is None
    assert body["curriculum"]["answer"] == _GAP
    assert _ADVICE not in body["curriculum"]["answer"]
    assert body["curriculum"]["status"] == "insufficient_evidence"

    completed = FakeQAAgent(
        _state(
            answer="The Class 4 syllabus includes fractions of a whole.",
            status=AgentStatus.COMPLETED,
            evidence_status=EvidenceStatus.FOUND,
            verification_status=VerificationStatus.PASSED,
            recommendation=VerificationRecommendation.ACCEPT,
            passed=True,
            coverage_decision="accept",
        )
    )
    general = _Calls()
    configure_assistant_runtime(qa_agent=completed, general=general)
    allowed = _post(
        monkeypatch,
        _settings(tmp_path, assistant_enabled=True),
        {"question": _MIXED, "thread_id": body["thread_id"]},
    )
    assert allowed.status_code == 200, allowed.text
    allowed_body = allowed.json()
    assert allowed_body["general_text"] == _ADVICE
    assert allowed_body["general_label"] == "General AI assistance"
    assert _ADVICE not in allowed_body["curriculum"]["answer"]
    assert "fractions of a whole" in allowed_body["curriculum"]["answer"]
    raw = sqlite3.connect(tmp_path / "assistant_threads.sqlite")
    try:
        messages = raw.execute(
            "SELECT role, content FROM messages WHERE thread_id = ? ORDER BY message_id",
            (body["thread_id"],),
        ).fetchall()
    finally:
        raw.close()
    assert [row[0] for row in messages] == ["user", "assistant", "user", "assistant", "assistant"]
    assert messages[1][1] == _GAP
    assert messages[3][1] == "The Class 4 syllabus includes fractions of a whole."
    assert messages[4][1] == f"General AI assistance: {_ADVICE}"


def test_reply_is_returned_when_assistant_message_persistence_fails(
    tmp_path: Path, monkeypatch
):
    register_identity_gate(_Gate("teacher-a"))
    general = _Calls()
    configure_assistant_runtime(qa_agent=FakeQAAgent(_state(
        answer="unused",
        status=AgentStatus.COMPLETED,
        evidence_status=EvidenceStatus.FOUND,
        verification_status=VerificationStatus.PASSED,
        recommendation=VerificationRecommendation.ACCEPT,
        passed=True,
    )), general=general)
    real = AssistantThreadStore.append_message

    def fail_assistant(self, principal, thread_id, role, content):
        if role == "assistant":
            raise ThreadStoreError("Assistant thread store is unavailable")
        return real(self, principal, thread_id, role, content)

    monkeypatch.setattr(AssistantThreadStore, "append_message", fail_assistant)
    response = _post(
        monkeypatch,
        _settings(tmp_path, assistant_enabled=True),
        {"question": _GENERAL},
    )
    assert response.status_code == 200, response.text
    assert response.json()["persistence_warning"] == (
        "The reply was produced but not fully stored."
    )
    assert response.json()["general_text"] == _ADVICE
    assert general.questions == [_GENERAL]
    raw = sqlite3.connect(tmp_path / "assistant_threads.sqlite")
    try:
        rows = raw.execute("SELECT role, content FROM messages").fetchall()
    finally:
        raw.close()
    assert rows == [("user", _GENERAL)]


def test_ask_validation_is_unchanged_when_assistant_is_mounted():
    application = create_app()
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/agent/ask", json={"question": "  "})
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_REQUEST"
