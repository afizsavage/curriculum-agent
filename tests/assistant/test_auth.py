"""Provisioned accounts, session cookies, and the session identity gate."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.assistant.auth_startup import AuthConfigurationError, install_configured_identity_gate
from app.assistant.auth_store import (
    SESSION_TTL,
    AuthStore,
    hash_session_token,
)
from app.assistant.chat import configure_assistant_runtime, reset_assistant_runtime
from app.assistant.identity import (
    UnauthenticatedTeacherError,
    authenticate_http_request,
    current_identity_gate,
    reset_identity_gate,
)
from app.assistant.provision import main as provision_main
from app.assistant.session_gate import SessionIdentityGate
from app.config import Settings
from app.main import create_app

_PASSWORD = "correct-horse-battery"
_USERNAME = "teacher.one"
_GENERAL = "How should I settle a noisy class before the lesson starts?"
_LOGIN_FAILURE = {"detail": "Invalid username or password"}


class _Request:
    def __init__(self, cookies: dict[str, str] | None = None) -> None:
        self.cookies = cookies or {}
        self.teacher_id = "forged-teacher"
        self.headers = {"X-Teacher-Id": "forged-teacher", "X-School-Id": "forged-school"}


@pytest.fixture(autouse=True)
def _reset_boundaries():
    reset_identity_gate()
    reset_assistant_runtime()
    yield
    gate = current_identity_gate()
    if isinstance(gate, SessionIdentityGate):
        gate.close()
    reset_identity_gate()
    reset_assistant_runtime()


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "_env_file": None,
        "environment": "development",
        "assistant_enabled": False,
        "assistant_local_stub_identity": False,
        "assistant_local_stub_teacher_id": "local-stub-teacher",
        "assistant_auth_enabled": True,
        "assistant_auth_sqlite_path": str(tmp_path / "assistant_auth.sqlite"),
        "assistant_thread_sqlite_path": str(tmp_path / "assistant_threads.sqlite"),
        "assistant_session_cookie_name": "marcus_session",
        "assistant_auth_allowed_origins": "http://localhost:3000,http://127.0.0.1:3000",
        "llm_provider": "stub",
    }
    values.update(overrides)
    return Settings(**values)


def _auth_path(settings: Settings) -> Path:
    return Path(settings.assistant_auth_sqlite_path)


def _provision(settings: Settings, *, school_id: str | None = "school-7") -> AuthStore:
    store = AuthStore(_auth_path(settings))
    store.provision_account(_USERNAME, _PASSWORD, school_id)
    return store


def _install(monkeypatch, settings: Settings):
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.api.v1.assistant.get_settings", lambda: settings)
    application = create_app()
    return application


def _cookie_header(response) -> str:
    header = response.headers.get("set-cookie")
    assert header
    return header


def _cookie_value(response, name: str = "marcus_session") -> str:
    prefix = f"{name}="
    pair = _cookie_header(response).split(";", 1)[0]
    assert pair.startswith(prefix)
    return pair[len(prefix) :]


def _query(path: Path, sql: str, params: tuple = ()) -> list:
    connection = sqlite3.connect(path)
    try:
        return connection.execute(sql, params).fetchall()
    finally:
        connection.close()


def _write(path: Path, sql: str, params: tuple = ()) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(sql, params)
        connection.commit()
    finally:
        connection.close()


def _stored_bytes(path: Path) -> bytes:
    blob = bytearray()
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.exists():
            blob.extend(candidate.read_bytes())
    return bytes(blob)


def test_auth_and_assistant_default_off():
    assert Settings.model_fields["assistant_auth_enabled"].default is False
    assert Settings.model_fields["assistant_enabled"].default is False
    assert Settings.model_fields["assistant_local_stub_identity"].default is False
    loaded = Settings(_env_file=None)
    assert loaded.assistant_auth_enabled is False
    assert loaded.assistant_enabled is False
    assert loaded.assistant_local_stub_identity is False


def test_disabled_auth_does_not_open_a_store_or_register_a_gate(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, assistant_auth_enabled=False)
    _install(monkeypatch, settings)
    assert current_identity_gate() is None
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/session/login",
            json={"username": _USERNAME, "password": _PASSWORD},
        )
    assert response.status_code == 404
    assert response.json()["detail"] == "Not found"
    assert not _auth_path(settings).exists()


def test_login_and_claims_use_the_account_id(tmp_path: Path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    settings = _settings(tmp_path)
    store = _provision(settings, school_id="school-7")
    store.close()
    application = _install(monkeypatch, settings)
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/session/login",
            json={
                "username": "Teacher.One",
                "password": _PASSWORD,
                "teacher_id": "forged-teacher",
            },
            headers={"X-Teacher-Id": "forged-teacher", "Origin": "http://localhost:3000"},
        )
    assert response.status_code == 204
    header = _cookie_header(response)
    token = _cookie_value(response)
    assert "HttpOnly" in header
    assert "SameSite=lax" in header
    assert "Path=/" in header
    assert "Max-Age=1209600" in header
    assert "Secure" not in header
    assert token not in response.text
    assert _PASSWORD not in caplog.text
    assert token not in caplog.text

    gate = current_identity_gate()
    assert isinstance(gate, SessionIdentityGate)
    claims = gate.verify(_Request({settings.assistant_session_cookie_name: token}))
    assert claims is not None
    account_id = _query(_auth_path(settings), "SELECT account_id FROM accounts")[0][0]
    assert claims.teacher_id == account_id
    assert claims.teacher_id != "forged-teacher"
    assert claims.teacher_id != _USERNAME
    assert claims.school_id == "school-7"
    principal = authenticate_http_request(
        _Request({settings.assistant_session_cookie_name: token}),
        settings=settings,
    )
    assert principal.teacher_id == account_id
    assert principal.school_id == "school-7"


def test_wrong_password_and_unknown_account_look_the_same(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    store = _provision(settings, school_id=None)
    store.close()
    application = _install(monkeypatch, settings)
    with TestClient(application, raise_server_exceptions=False) as client:
        wrong = client.post(
            "/api/v1/session/login",
            json={"username": _USERNAME, "password": "not-the-real-password"},
        )
        unknown = client.post(
            "/api/v1/session/login",
            json={"username": "missing.teacher", "password": _PASSWORD},
        )
        malformed = client.post(
            "/api/v1/session/login",
            data={"username": _USERNAME, "password": _PASSWORD},
        )
    assert wrong.status_code == unknown.status_code == malformed.status_code == 401
    assert wrong.json() == unknown.json() == malformed.json() == _LOGIN_FAILURE
    assert "set-cookie" not in wrong.headers
    assert _PASSWORD not in wrong.text
    assert _USERNAME not in wrong.text
    assert "missing.teacher" not in unknown.text
    assert _query(_auth_path(settings), "SELECT COUNT(*) FROM sessions")[0][0] == 0


def test_password_hashes_and_session_tokens_are_not_stored_in_plaintext(tmp_path: Path):
    settings = _settings(tmp_path)
    store = AuthStore(_auth_path(settings))
    first = store.provision_account("teacher.one", _PASSWORD, None)
    second = store.provision_account("teacher.two", _PASSWORD, None)
    token = store.create_session(first.account_id)
    other = store.create_session(second.account_id)
    hashes = _query(
        _auth_path(settings),
        "SELECT username, password_hash FROM accounts ORDER BY username",
    )
    sessions = _query(_auth_path(settings), "SELECT token_hash, expires_at FROM sessions")
    store.close()
    assert hashes[0][1] != hashes[1][1]
    for _username, password_hash in hashes:
        assert password_hash.startswith(f"scrypt${2**14}$8$1$")
        assert _PASSWORD not in password_hash
    salts = [row[1].split("$")[4] for row in hashes]
    assert salts[0] != salts[1]
    digest = {row[0] for row in sessions}
    assert hash_session_token(token) in digest
    assert hash_session_token(other) in digest
    assert token not in digest
    assert other not in digest
    expires = datetime.fromisoformat(sessions[0][1])
    remaining = expires - datetime.now(timezone.utc)
    assert SESSION_TTL - timedelta(minutes=1) <= remaining <= SESSION_TTL
    blob = _stored_bytes(_auth_path(settings))
    assert _PASSWORD.encode() not in blob
    assert token.encode() not in blob
    assert other.encode() not in blob


def test_missing_malformed_expired_and_revoked_sessions_are_rejected(
    tmp_path: Path, monkeypatch
):
    settings = _settings(tmp_path, assistant_enabled=True)
    store = _provision(settings)
    store.close()
    application = _install(monkeypatch, settings)

    def refuse_store(*_args, **_kwargs):
        raise AssertionError("thread store opened without a live session")

    monkeypatch.setattr("app.assistant.chat.AssistantThreadStore", refuse_store)
    with TestClient(application, raise_server_exceptions=False) as client:
        logged_in = client.post(
            "/api/v1/session/login",
            json={"username": _USERNAME, "password": _PASSWORD},
        )
        token = _cookie_value(logged_in)
        client.cookies.clear()
        missing = client.post("/api/v1/assistant/chat", json={"question": _GENERAL})
        malformed = client.post(
            "/api/v1/assistant/chat",
            json={"question": _GENERAL},
            cookies={"marcus_session": "not-a-session"},
        )
        _write(
            _auth_path(settings),
            "UPDATE sessions SET expires_at = ?",
            ((datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(),),
        )
        expired = client.post(
            "/api/v1/assistant/chat",
            json={"question": _GENERAL, "teacher_id": "forged-teacher"},
            cookies={"marcus_session": token},
            headers={"X-Teacher-Id": "forged-teacher"},
        )
    assert missing.status_code == malformed.status_code == expired.status_code == 401
    assert missing.json()["detail"] == "Authenticated teacher principal is required"
    gate = current_identity_gate()
    assert isinstance(gate, SessionIdentityGate)
    assert gate.verify(_Request()) is None
    assert gate.verify(_Request({"marcus_session": "not-a-session"})) is None
    assert gate.verify(_Request({"marcus_session": token})) is None
    assert not (tmp_path / "assistant_threads.sqlite").exists()


def test_logout_revokes_the_session_and_clears_the_cookie(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, assistant_enabled=True)
    store = _provision(settings)
    store.close()
    application = _install(monkeypatch, settings)
    monkeypatch.setattr(
        "app.assistant.chat.AssistantThreadStore",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("store opened")),
    )
    with TestClient(application, raise_server_exceptions=False) as client:
        logged_in = client.post(
            "/api/v1/session/login",
            json={"username": _USERNAME, "password": _PASSWORD},
        )
        token = _cookie_value(logged_in)
        logged_out = client.post("/api/v1/session/logout")
    assert logged_out.status_code == 204
    clearing = _cookie_header(logged_out)
    assert "Max-Age=0" in clearing
    assert "HttpOnly" in clearing
    assert _query(
        _auth_path(settings),
        "SELECT revoked_at FROM sessions WHERE token_hash = ?",
        (hash_session_token(token),),
    )[0][0] is not None

    with TestClient(application, raise_server_exceptions=False) as client:
        logged_in = client.post(
            "/api/v1/session/login",
            json={"username": _USERNAME, "password": _PASSWORD},
        )
        live = _cookie_value(logged_in)
        method = client.get("/api/v1/session/logout")
        assert method.status_code == 405
        gate = current_identity_gate()
        assert isinstance(gate, SessionIdentityGate)
        assert gate.verify(_Request({"marcus_session": live})) is not None
        blocked = client.post(
            "/api/v1/session/logout",
            headers={"Origin": "https://evil.example"},
        )
        assert blocked.status_code == 403
        assert gate.verify(_Request({"marcus_session": live})) is not None
        client.cookies.clear()
        empty = client.post("/api/v1/session/logout")
        assert empty.status_code == 204
        assert gate.verify(_Request({"marcus_session": live})) is not None
        assert _query(_auth_path(settings), "SELECT COUNT(*) FROM accounts")[0][0] == 1
        logged_out = client.post(
            "/api/v1/session/logout",
            cookies={"marcus_session": live},
        )
        denied = client.post(
            "/api/v1/assistant/chat",
            json={"question": _GENERAL},
            cookies={"marcus_session": token},
        )
        replay = client.post(
            "/api/v1/assistant/chat",
            json={"question": _GENERAL},
            cookies={"marcus_session": live},
        )
    assert logged_out.status_code == 204
    assert denied.status_code == replay.status_code == 401


def test_forged_teacher_id_cannot_authorize(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, assistant_enabled=True)
    store = _provision(settings, school_id="school-9")
    store.close()
    application = _install(monkeypatch, settings)
    configure_assistant_runtime(general=lambda _request: "Start with a quiet signal.")
    monkeypatch.setattr(
        "app.assistant.chat.get_agent",
        lambda: (_ for _ in ()).throw(AssertionError("qa agent constructed")),
    )
    monkeypatch.setattr(
        "app.llm.provider.build_llm_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("model provider built")),
    )
    with TestClient(application, raise_server_exceptions=False) as client:
        forged = client.post(
            "/api/v1/assistant/chat",
            json={"question": _GENERAL, "teacher_id": "forged-teacher"},
            headers={"X-Teacher-Id": "forged-teacher"},
        )
        logged_in = client.post(
            "/api/v1/session/login",
            json={"username": _USERNAME, "password": _PASSWORD, "teacher_id": "forged-teacher"},
        )
        accepted = client.post(
            "/api/v1/assistant/chat",
            json={"question": _GENERAL, "teacher_id": "forged-teacher"},
            headers={"X-Teacher-Id": "forged-teacher", "X-School-Id": "forged-school"},
        )
    assert forged.status_code == 401
    assert logged_in.status_code == 204
    assert accepted.status_code == 200, accepted.text
    assert "forged-teacher" not in accepted.text
    account_id = _query(_auth_path(settings), "SELECT account_id FROM accounts")[0][0]
    owner = _query(
        tmp_path / "assistant_threads.sqlite",
        "SELECT teacher_id, school_id FROM threads",
    )
    assert owner == [(account_id, "school-9")]


def test_disabled_assistant_returns_404_before_identity_or_threads(
    tmp_path: Path, monkeypatch
):
    settings = _settings(tmp_path, assistant_enabled=False)
    store = _provision(settings)
    store.close()
    application = _install(monkeypatch, settings)

    def refuse(*_args, **_kwargs):
        raise AssertionError("assistant work ran while disabled")

    monkeypatch.setattr("app.api.v1.assistant.execute_chat", refuse)
    monkeypatch.setattr("app.assistant.chat.execute_chat", refuse)
    monkeypatch.setattr("app.assistant.chat.AssistantThreadStore", refuse)
    with TestClient(application, raise_server_exceptions=False) as client:
        logged_in = client.post(
            "/api/v1/session/login",
            json={"username": _USERNAME, "password": _PASSWORD},
        )
        response = client.post(
            "/api/v1/assistant/chat",
            json={"question": _GENERAL, "teacher_id": "forged-teacher"},
            headers={"X-Teacher-Id": "forged-teacher"},
        )
    assert logged_in.status_code == 204
    assert response.status_code == 404
    assert response.json()["detail"] == "Not found"
    assert not (tmp_path / "assistant_threads.sqlite").exists()


def test_production_refuses_local_stub_identity(tmp_path: Path):
    settings = _settings(
        tmp_path,
        environment="production",
        assistant_local_stub_identity=True,
        assistant_auth_enabled=False,
    )
    with pytest.raises(UnauthenticatedTeacherError, match="Local stub identity is disabled"):
        authenticate_http_request(_Request(), settings=settings)


def test_startup_fails_when_authentication_cannot_be_registered(tmp_path: Path, monkeypatch):
    combined = _settings(
        tmp_path,
        environment="production",
        assistant_auth_enabled=True,
        assistant_local_stub_identity=True,
    )
    with pytest.raises(AuthConfigurationError, match="cannot be combined"):
        install_configured_identity_gate(combined)
    assert current_identity_gate() is None

    development = _settings(
        tmp_path,
        assistant_auth_enabled=True,
        assistant_local_stub_identity=True,
    )
    with pytest.raises(AuthConfigurationError):
        install_configured_identity_gate(development)
    assert current_identity_gate() is None

    reserved = _settings(
        tmp_path,
        assistant_auth_sqlite_path=str(tmp_path / "checkpoints.sqlite"),
    )
    with pytest.raises(AuthConfigurationError, match="could not be registered"):
        install_configured_identity_gate(reserved)
    assert current_identity_gate() is None
    assert not (tmp_path / "checkpoints.sqlite").exists()

    monkeypatch.setattr("app.config.get_settings", lambda: combined)
    with pytest.raises(AuthConfigurationError):
        create_app()
    assert current_identity_gate() is None


def test_enabled_auth_without_a_gate_does_not_use_the_stub(tmp_path: Path, monkeypatch):
    settings = _settings(
        tmp_path,
        assistant_auth_enabled=True,
        assistant_local_stub_identity=True,
    )
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/session/login",
            json={"username": _USERNAME, "password": _PASSWORD, "teacher_id": "forged-teacher"},
            headers={"X-Teacher-Id": "forged-teacher"},
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "Authentication is unavailable"
    assert "set-cookie" not in response.headers
    assert current_identity_gate() is None
    assert not _auth_path(settings).exists()


def test_registered_gate_does_not_fall_back_to_the_stub(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, assistant_local_stub_identity=False)
    store = _provision(settings)
    store.close()
    _install(monkeypatch, settings)
    stub_settings = _settings(tmp_path, assistant_local_stub_identity=True)
    with pytest.raises(UnauthenticatedTeacherError):
        authenticate_http_request(_Request(), settings=stub_settings)
    assert current_identity_gate() is not None


def test_production_cookie_is_secure(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, environment="production")
    store = _provision(settings)
    store.close()
    application = _install(monkeypatch, settings)
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/session/login",
            json={"username": _USERNAME, "password": _PASSWORD},
        )
    assert response.status_code == 204
    header = _cookie_header(response)
    assert "Secure" in header
    assert "HttpOnly" in header
    assert "SameSite=lax" in header


def test_cross_site_login_does_not_create_a_session(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    store = _provision(settings)
    store.close()
    application = _install(monkeypatch, settings)
    with TestClient(application, raise_server_exceptions=False) as client:
        origin = client.post(
            "/api/v1/session/login",
            json={"username": _USERNAME, "password": _PASSWORD},
            headers={"Origin": "https://evil.example"},
        )
        fetched = client.post(
            "/api/v1/session/login",
            json={"username": _USERNAME, "password": _PASSWORD},
            headers={"Sec-Fetch-Site": "cross-site"},
        )
    assert origin.status_code == fetched.status_code == 403
    assert origin.json()["detail"] == "Cross-origin request rejected"
    assert "set-cookie" not in origin.headers
    assert _query(_auth_path(settings), "SELECT COUNT(*) FROM sessions")[0][0] == 0


def test_cli_provisions_without_printing_secrets(tmp_path: Path, monkeypatch, capsys):
    database = tmp_path / "operator_auth.sqlite"
    monkeypatch.setenv("ASSISTANT_ACCOUNT_PASSWORD", _PASSWORD)
    code = provision_main(
        ["--username", "Teacher.One", "--school-id", "school-4", "--database", str(database)]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "Provisioned teacher account 'teacher.one'."
    assert _PASSWORD not in captured.out
    assert _PASSWORD not in captured.err
    assert "scrypt$" not in captured.out
    assert "scrypt$" not in captured.err
    stored = _query(database, "SELECT username, school_id, password_hash FROM accounts")
    assert stored[0][0] == "teacher.one"
    assert stored[0][1] == "school-4"
    assert _PASSWORD not in stored[0][2]

    again = provision_main(
        ["--username", "teacher.one", "--database", str(database)]
    )
    duplicate = capsys.readouterr()
    assert again == 2
    assert "already exists" in duplicate.err
    assert _PASSWORD not in duplicate.err
    assert _query(database, "SELECT COUNT(*) FROM accounts")[0][0] == 1


def test_cli_refuses_a_non_interactive_run_without_a_password(tmp_path: Path, monkeypatch, capsys):
    database = tmp_path / "operator_auth.sqlite"
    monkeypatch.delenv("ASSISTANT_ACCOUNT_PASSWORD", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    code = provision_main(["--username", "teacher.one", "--database", str(database)])
    captured = capsys.readouterr()
    assert code == 2
    assert "no default password" in captured.err
    assert not database.exists()

    monkeypatch.setenv("ASSISTANT_ACCOUNT_PASSWORD", "too-short")
    short = provision_main(["--username", "teacher.one", "--database", str(database)])
    short_output = capsys.readouterr()
    assert short == 2
    assert "too-short" not in short_output.err
    assert "12" in short_output.err
    assert not database.exists()

    monkeypatch.delenv("ASSISTANT_ACCOUNT_PASSWORD", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def mismatched(prompt: str) -> str:
        if prompt.startswith("Confirm"):
            return "other-password-value"
        return _PASSWORD

    monkeypatch.setattr("getpass.getpass", mismatched)
    mismatch = provision_main(["--username", "teacher.one", "--database", str(database)])
    mismatch_output = capsys.readouterr()
    assert mismatch == 2
    assert "did not match" in mismatch_output.err
    assert _PASSWORD not in mismatch_output.out
    assert _PASSWORD not in mismatch_output.err
    assert not database.exists()

    monkeypatch.setenv("ASSISTANT_ACCOUNT_PASSWORD", _PASSWORD)
    reserved = provision_main(
        ["--username", "teacher.one", "--database", str(tmp_path / "checkpoints.sqlite")]
    )
    reserved_output = capsys.readouterr()
    assert reserved == 2
    assert _PASSWORD not in reserved_output.err
    assert not (tmp_path / "checkpoints.sqlite").exists()


def test_session_status_follows_the_cookie_and_stays_empty(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, assistant_enabled=False)
    store = _provision(settings, school_id="school-7")
    store.close()
    application = _install(monkeypatch, settings)
    account_id = _query(_auth_path(settings), "SELECT account_id FROM accounts")[0][0]
    with TestClient(application, raise_server_exceptions=False) as client:
        missing = client.get(
            "/api/v1/session",
            params={"teacher_id": "forged-teacher", "school_id": "forged-school"},
            headers={"X-Teacher-Id": "forged-teacher", "X-School-Id": "forged-school"},
        )
        assert missing.status_code == 401
        assert "forged" not in missing.text
        assert account_id not in missing.text

        logged_in = client.post(
            "/api/v1/session/login",
            json={"username": _USERNAME, "password": _PASSWORD, "teacher_id": "forged-teacher"},
            headers={"Origin": "http://localhost:3000"},
        )
        assert logged_in.status_code == 204
        token = _cookie_value(logged_in)

        status = client.get(
            "/api/v1/session",
            params={"teacher_id": "forged-teacher"},
            headers={"X-Teacher-Id": "forged-teacher", "Origin": "http://localhost:3000"},
        )
        assert status.status_code == 204
        assert status.content == b""
        assert token not in status.text
        assert account_id not in status.text
        assert "school-7" not in status.text
        assert status.headers["cache-control"] == "no-store"
        assert "set-cookie" not in status.headers

        rejected = client.get(
            "/api/v1/session",
            headers={"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"},
        )
        assert rejected.status_code == 403
        assert token not in rejected.text

        still_valid = client.get(
            "/api/v1/session",
            headers={"Origin": "http://localhost:3000"},
        )
        assert still_valid.status_code == 204
        assert still_valid.content == b""


def test_session_status_is_hidden_when_authentication_is_disabled(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, assistant_auth_enabled=False, assistant_enabled=False)
    application = _install(monkeypatch, settings)
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.get(
            "/api/v1/session",
            headers={"X-Teacher-Id": "forged-teacher"},
            cookies={"marcus_session": "not-a-session"},
        )
    assert response.status_code == 404
    assert response.json()["detail"] == "Not found"
    assert current_identity_gate() is None
    assert not _auth_path(settings).exists()
