"""Authenticated recent-chat list and retrieve routes."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.assistant.auth_store import AuthStore
from app.assistant.identity import (
    VerifiedTeacherClaims,
    bind_verified_claims,
    current_identity_gate,
    reset_identity_gate,
)
from app.assistant.session_gate import SessionIdentityGate
from app.assistant.threads import (
    SCHEMA_VERSION,
    AssistantThreadStore,
    QaOutcomeSummary,
)
from app.config import Settings
from app.main import create_app

_PASSWORD = "correct-horse-battery"
_ALPHA = "  What subjects\nare taught in Class 4?  "
_ALPHA_TITLE = "What subjects are taught in Class 4?"
_LATER = "later detail that must stay out of the title"
_BETA = "Private beta plan that teacher one must not see"
_PROVENANCE = "accept-secret-provenance"
_LONG = "L" * 120


@pytest.fixture(autouse=True)
def _reset_gate():
    reset_identity_gate()
    yield
    gate = current_identity_gate()
    if isinstance(gate, SessionIdentityGate):
        gate.close()
    reset_identity_gate()


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "_env_file": None,
        "environment": "development",
        "assistant_enabled": True,
        "assistant_local_stub_identity": False,
        "assistant_auth_enabled": True,
        "assistant_auth_sqlite_path": str(tmp_path / "assistant_auth.sqlite"),
        "assistant_thread_sqlite_path": str(tmp_path / "assistant_threads.sqlite"),
        "assistant_session_cookie_name": "marcus_session",
        "assistant_auth_allowed_origins": "http://localhost:3000,http://127.0.0.1:3000",
        "llm_provider": "stub",
    }
    values.update(overrides)
    return Settings(**values)


def _install(monkeypatch, settings: Settings):
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.api.v1.assistant.get_settings", lambda: settings)
    return create_app()


def _login(client: TestClient, username: str) -> None:
    response = client.post(
        "/api/v1/session/login",
        json={"username": username, "password": _PASSWORD},
    )
    assert response.status_code == 204, response.text


def _principal(account_id: str, school_id: str):
    gate = current_identity_gate()
    assert isinstance(gate, SessionIdentityGate)
    return bind_verified_claims(
        VerifiedTeacherClaims(account_id, school_id),
        gate=gate,
    )


def test_owner_lists_and_reopens_only_their_threads(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    accounts = AuthStore(settings.assistant_auth_sqlite_path)
    teacher_a = accounts.provision_account("teacher.one", _PASSWORD, "school-1")
    teacher_b = accounts.provision_account("teacher.two", _PASSWORD, "school-2")
    accounts.close()
    application = _install(monkeypatch, settings)
    monkeypatch.setattr(
        "app.assistant.chat.dispatch_turn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dispatch ran")),
    )
    monkeypatch.setattr(
        "app.assistant.chat.get_agent",
        lambda: (_ for _ in ()).throw(AssertionError("qa agent constructed")),
    )

    owner = _principal(teacher_a.account_id, "school-1")
    other = _principal(teacher_b.account_id, "school-2")
    store = AssistantThreadStore(settings.assistant_thread_sqlite_path)
    try:
        empty = store.create_thread(owner)
        long_thread = store.create_thread(owner)
        store.append_message(owner, long_thread.thread_id, "user", _LONG)
        beta = store.create_thread(other)
        store.append_message(other, beta.thread_id, "user", _BETA)
        alpha = store.create_thread(owner)
        store.append_message(owner, alpha.thread_id, "user", _ALPHA)
        store.append_message(owner, alpha.thread_id, "assistant", "Class 4 subjects reply")
        store.update_working_context(
            owner,
            alpha.thread_id,
            grade="Class 4",
            subject="Mathematics",
            task="subject-list",
        )
        store.record_qa_outcome(
            owner,
            alpha.thread_id,
            QaOutcomeSummary(
                status="completed",
                verification_recommendation=_PROVENANCE,
            ),
        )
        store.append_message(owner, alpha.thread_id, "user", _LATER)
    finally:
        store.close()

    columns = _columns(Path(settings.assistant_thread_sqlite_path))
    assert "title" not in columns
    assert _schema_version(Path(settings.assistant_thread_sqlite_path)) == SCHEMA_VERSION == 1

    with TestClient(application, raise_server_exceptions=False) as client:
        _login(client, "teacher.one")
        listed = client.get(
            "/api/v1/assistant/threads",
            params={"teacher_id": teacher_b.account_id},
            headers={"X-Teacher-Id": teacher_b.account_id},
        )
        opened = client.get(f"/api/v1/assistant/threads/{alpha.thread_id}")
    assert listed.status_code == 200, listed.text
    body = listed.json()
    assert [row["thread_id"] for row in body["threads"]] == [
        alpha.thread_id,
        long_thread.thread_id,
        empty.thread_id,
    ]
    assert body["threads"][0]["title"] == _ALPHA_TITLE
    assert body["threads"][1]["title"] == ("L" * 79) + "…"
    assert len(body["threads"][1]["title"]) == 80
    assert body["threads"][2]["title"] == "New chat"
    assert body["threads"][0]["grade"] == "Class 4"
    assert body["threads"][0]["subject"] == "Mathematics"
    assert body["threads"][0]["task"] == "subject-list"
    assert body["threads"][0]["updated_at"] >= body["threads"][1]["updated_at"]
    assert set(body["threads"][0]) == {
        "thread_id",
        "title",
        "created_at",
        "updated_at",
        "grade",
        "subject",
        "task",
    }
    assert _BETA not in listed.text
    assert _LATER not in listed.text
    assert _PROVENANCE not in listed.text
    assert teacher_a.account_id not in listed.text
    assert teacher_b.account_id not in listed.text
    assert "school-1" not in listed.text
    assert "school-2" not in listed.text

    assert opened.status_code == 200, opened.text
    detail = opened.json()
    assert detail["thread_id"] == alpha.thread_id
    assert detail["title"] == _ALPHA_TITLE
    assert [message["role"] for message in detail["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert [message["content"] for message in detail["messages"]] == [
        _ALPHA,
        "Class 4 subjects reply",
        _LATER,
    ]
    assert detail["messages"][0]["message_id"] < detail["messages"][1]["message_id"]
    assert detail["messages"][1]["message_id"] < detail["messages"][2]["message_id"]
    assert "qa_outcome" not in detail
    assert _PROVENANCE not in opened.text
    assert _BETA not in opened.text
    assert set(detail["messages"][0]) == {"message_id", "role", "content", "created_at"}

    with TestClient(application, raise_server_exceptions=False) as other_client:
        _login(other_client, "teacher.two")
        own = other_client.get("/api/v1/assistant/threads")
        denied = other_client.get(f"/api/v1/assistant/threads/{alpha.thread_id}")
        missing = other_client.get(f"/api/v1/assistant/threads/{uuid4()}")
        opened_beta = other_client.get(f"/api/v1/assistant/threads/{beta.thread_id}")
    assert own.status_code == 200
    assert [row["thread_id"] for row in own.json()["threads"]] == [beta.thread_id]
    assert own.json()["threads"][0]["title"] == _BETA
    assert denied.status_code == missing.status_code == 404
    assert denied.json() == missing.json() == {"detail": "Thread not found"}
    assert _ALPHA.strip() not in denied.text
    assert _LATER not in denied.text
    assert "Class 4 subjects reply" not in denied.text
    assert teacher_a.account_id not in denied.text
    assert opened_beta.status_code == 200
    assert opened_beta.json()["messages"][0]["content"] == _BETA


def test_empty_thread_list(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    accounts = AuthStore(settings.assistant_auth_sqlite_path)
    accounts.provision_account("teacher.one", _PASSWORD, None)
    accounts.close()
    application = _install(monkeypatch, settings)
    with TestClient(application, raise_server_exceptions=False) as client:
        _login(client, "teacher.one")
        response = client.get("/api/v1/assistant/threads")
    assert response.status_code == 200
    assert response.json() == {"threads": []}


def test_missing_identity_does_not_open_the_thread_store(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    application = _install(monkeypatch, settings)

    def refuse(*_args, **_kwargs):
        raise AssertionError("thread store opened without a principal")

    monkeypatch.setattr("app.api.v1.assistant.AssistantThreadStore", refuse)
    monkeypatch.setattr("app.api.v1.assistant.thread_store_path", refuse)
    with TestClient(application, raise_server_exceptions=False) as client:
        listed = client.get(
            "/api/v1/assistant/threads",
            params={"teacher_id": "forged-teacher"},
            headers={"X-Teacher-Id": "forged-teacher"},
        )
        opened = client.get(
            f"/api/v1/assistant/threads/{uuid4()}",
            params={"teacher_id": "forged-teacher"},
            headers={"X-Teacher-Id": "forged-teacher"},
        )
    assert listed.status_code == opened.status_code == 401
    assert listed.json() == opened.json()
    assert listed.json()["detail"] == "Authenticated teacher principal is required"
    assert not Path(settings.assistant_thread_sqlite_path).exists()


def test_disabled_assistant_returns_404_before_identity_or_store(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, assistant_enabled=False)
    accounts = AuthStore(settings.assistant_auth_sqlite_path)
    accounts.provision_account("teacher.one", _PASSWORD, None)
    accounts.close()
    application = _install(monkeypatch, settings)

    def refuse(*_args, **_kwargs):
        raise AssertionError("assistant work ran while disabled")

    monkeypatch.setattr("app.api.v1.assistant.authenticate_http_request", refuse)
    monkeypatch.setattr("app.api.v1.assistant.AssistantThreadStore", refuse)
    monkeypatch.setattr("app.api.v1.assistant.thread_store_path", refuse)
    with TestClient(application, raise_server_exceptions=False) as client:
        _login(client, "teacher.one")
        listed = client.get("/api/v1/assistant/threads")
        opened = client.get(f"/api/v1/assistant/threads/{uuid4()}")
    assert listed.status_code == opened.status_code == 404
    assert listed.json() == opened.json() == {"detail": "Not found"}
    assert not Path(settings.assistant_thread_sqlite_path).exists()


def _columns(path: Path) -> list[str]:
    connection = sqlite3.connect(path)
    try:
        return [row[1] for row in connection.execute("PRAGMA table_info(threads)")]
    finally:
        connection.close()


def _schema_version(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        return int(connection.execute("SELECT version FROM schema_meta").fetchone()[0])
    finally:
        connection.close()
