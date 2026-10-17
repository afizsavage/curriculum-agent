"""Legacy QA conversation ownership at the HTTP boundary."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Barrier, Thread
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.agent.context import ConversationStore
from app.agent.conversation_owners import ConversationOwnerStore, OwnerStoreError
from app.agent.state import CurriculumQAState
from app.assistant.identity import (
    VerifiedTeacherClaims,
    current_identity_gate,
    register_identity_gate,
    reset_identity_gate,
)
from app.assistant.session_gate import SessionIdentityGate
from app.assistant.threads import SCHEMA_VERSION, AssistantThreadStore
from app.config import Settings
from app.deps import get_agent
from app.enums import AgentStatus
from app.exceptions import AgentExecutionError
from app.llm.provider import StubLLMProvider
from app.main import create_app
from app.schemas.agent import AskResponse
from app.schemas.answer import AnswerConfidence

_QUESTION = "What topics are taught in Primary 4 Mathematics?"
_NOT_FOUND = {"detail": "Conversation not found"}


class CookieGate:
    def __init__(self, sessions: dict[str, str]) -> None:
        self.sessions = sessions

    def verify(self, request: object) -> VerifiedTeacherClaims | None:
        cookies = getattr(request, "cookies", None) or {}
        teacher_id = self.sessions.get(str(cookies.get("marcus_session", "")))
        if not teacher_id:
            return None
        return VerifiedTeacherClaims(teacher_id=teacher_id)


class RecordingAgent:
    def __init__(self) -> None:
        self.llm = StubLLMProvider()
        self.calls: list[str | None] = []
        self.store = ConversationStore()
        self.checkpoints: dict[str, int] = {}
        self.fail_once = False

    def ask(self, question: str, *, conversation_id: str | None = None, request_id=None, **kwargs):
        self.calls.append(conversation_id)
        if self.fail_once:
            self.fail_once = False
            raise AgentExecutionError("Agent failed while processing the question")
        if conversation_id:
            context = self.store.get_or_create(conversation_id)
            context.append_user(question)
            self.store.save(context)
            self.checkpoints[conversation_id] = self.checkpoints.get(conversation_id, 0) + 1
        state = CurriculumQAState.initial(question=question, conversation_id=conversation_id)
        state.final_answer = "Owned answer"
        state.status = AgentStatus.COMPLETED
        state.answer_confidence = AnswerConfidence.HIGH
        return state


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
        "assistant_enabled": False,
        "assistant_local_stub_identity": False,
        "assistant_auth_enabled": False,
        "assistant_auth_sqlite_path": str(tmp_path / "assistant_auth.sqlite"),
        "assistant_thread_sqlite_path": str(tmp_path / "assistant_threads.sqlite"),
        "assistant_session_cookie_name": "marcus_session",
        "assistant_auth_allowed_origins": "http://localhost:3000,http://127.0.0.1:3000",
        "agent_checkpointing_enabled": True,
        "agent_checkpoint_sqlite_path": str(tmp_path / "checkpoints.sqlite"),
        "qa_conversation_owner_sqlite_path": str(tmp_path / "qa_conversation_owners.sqlite"),
        "llm_provider": "stub",
    }
    values.update(overrides)
    return Settings(**values)


def _client(
    monkeypatch,
    settings: Settings,
    agent: RecordingAgent,
    sessions: dict[str, str] | None = None,
) -> TestClient:
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    # The assistant routes bind get_settings at import. Patch that name too,
    # or a disabled test still sees the process settings and their identity gate.
    monkeypatch.setattr("app.api.v1.assistant.get_settings", lambda: settings)
    application = create_app()
    if sessions is not None:
        previous = current_identity_gate()
        if isinstance(previous, SessionIdentityGate):
            previous.close()
        register_identity_gate(CookieGate(sessions))
    application.dependency_overrides[get_agent] = lambda: agent
    return TestClient(application, raise_server_exceptions=False)


def _post(client: TestClient, *, cookie: str | None = None, conversation_id: str | None = None, forged: bool = False):
    payload: dict = {"question": _QUESTION}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    if forged:
        payload["teacher_id"] = "forged-teacher"
        payload["school_id"] = "forged-school"
    headers = {"X-Teacher-Id": "forged-teacher", "X-School-Id": "forged-school"} if forged else {}
    cookies = {"marcus_session": cookie} if cookie else None
    return client.post("/api/v1/agent/ask", json=payload, headers=headers, cookies=cookies)


def test_feature_defaults_stay_off():
    assert Settings.model_fields["assistant_enabled"].default is False
    assert Settings.model_fields["assistant_auth_enabled"].default is False


def test_anonymous_production_does_not_run_the_graph(tmp_path: Path, monkeypatch):
    agent = RecordingAgent()
    settings = _settings(tmp_path, environment="production", assistant_auth_enabled=False)
    with _client(monkeypatch, settings, agent) as client:
        response = _post(client, forged=True)
    assert response.status_code == 503
    assert response.json()["detail"] == "Authentication is unavailable"
    assert "forged-teacher" not in response.text
    assert agent.calls == []
    assert not Path(settings.qa_conversation_owner_sqlite_path).exists()


def test_production_ignores_the_disabled_auth_flag(tmp_path: Path, monkeypatch):
    agent = RecordingAgent()
    settings = _settings(tmp_path, environment="Production", assistant_auth_enabled=False)
    with _client(monkeypatch, settings, agent, {"teacher-a": "teacher-a"}) as client:
        blocked = _post(client, forged=True)
        assert blocked.status_code == 401
        assert blocked.json()["detail"] == "Authenticated teacher principal is required"
        assert agent.calls == []
        allowed = _post(client, cookie="teacher-a")
    assert allowed.status_code == 200
    owner = ConversationOwnerStore(settings.qa_conversation_owner_sqlite_path)
    try:
        assert owner.teacher_for(allowed.json()["conversation_id"]) == "teacher-a"
    finally:
        owner.close()


def test_missing_gate_in_production_fails_closed(tmp_path: Path, monkeypatch):
    agent = RecordingAgent()
    settings = _settings(tmp_path, environment="production", assistant_auth_enabled=True)
    holder = {"settings": _settings(tmp_path, environment="development", assistant_auth_enabled=False)}
    monkeypatch.setattr("app.config.get_settings", lambda: holder["settings"])
    application = create_app()
    application.dependency_overrides[get_agent] = lambda: agent
    holder["settings"] = settings
    reset_identity_gate()
    with TestClient(application, raise_server_exceptions=False) as client:
        response = _post(client)
    assert response.status_code == 503
    assert "sqlite" not in response.text.lower()
    assert str(tmp_path) not in response.text
    assert agent.calls == []


def test_forged_identity_cannot_open_a_conversation(tmp_path: Path, monkeypatch):
    agent = RecordingAgent()
    settings = _settings(tmp_path, environment="development", assistant_auth_enabled=True)
    with _client(monkeypatch, settings, agent, {"teacher-a": "teacher-a"}) as client:
        response = _post(client, forged=True)
    assert response.status_code == 401
    assert agent.calls == []
    assert agent.checkpoints == {}


def test_other_teacher_cannot_continue_or_change_a_checkpoint(tmp_path: Path, monkeypatch):
    agent = RecordingAgent()
    checkpoint = Path(tmp_path / "checkpoints.sqlite")
    checkpoint.write_bytes(b"existing-checkpoint")
    settings = _settings(tmp_path, environment="development", assistant_auth_enabled=True)
    with _client(
        monkeypatch, settings, agent, {"teacher-a": "teacher-a", "teacher-b": "teacher-b"}
    ) as client:
        created = _post(client, cookie="teacher-a")
        conversation_id = created.json()["conversation_id"]
        denied = _post(client, cookie="teacher-b", conversation_id=conversation_id, forged=True)
    assert created.status_code == 200
    assert denied.status_code == 404
    assert denied.json() == _NOT_FOUND
    assert "teacher-a" not in denied.text
    assert agent.checkpoints == {conversation_id: 1}
    assert checkpoint.read_bytes() == b"existing-checkpoint"
    store = ConversationOwnerStore(settings.qa_conversation_owner_sqlite_path)
    try:
        assert store.teacher_for(conversation_id) == "teacher-a"
    finally:
        store.close()


def test_owner_can_continue_and_the_response_shape_stays(tmp_path: Path, monkeypatch):
    agent = RecordingAgent()
    settings = _settings(tmp_path, environment="development", assistant_auth_enabled=True)
    with _client(monkeypatch, settings, agent, {"teacher-a": "teacher-a"}) as client:
        created = _post(client, cookie="teacher-a")
        body = created.json()
        continued = _post(client, cookie="teacher-a", conversation_id=body["conversation_id"])
    assert created.status_code == continued.status_code == 200
    assert continued.json()["conversation_id"] == body["conversation_id"]
    assert set(AskResponse.model_fields).issubset(body.keys())
    assert body["question"] == _QUESTION
    assert body["answer"] == "Owned answer"
    assert body["status"] == "completed"
    assert "teacher_id" not in body
    assert agent.calls == [body["conversation_id"], body["conversation_id"]]
    assert agent.checkpoints[body["conversation_id"]] == 2


def test_omitted_id_is_server_generated(tmp_path: Path, monkeypatch):
    agent = RecordingAgent()
    settings = _settings(tmp_path, environment="development", assistant_auth_enabled=True)
    supplied = str(uuid4())
    with _client(monkeypatch, settings, agent, {"teacher-a": "teacher-a"}) as client:
        created = _post(client, cookie="teacher-a")
    conversation_id = created.json()["conversation_id"]
    assert created.status_code == 200
    assert conversation_id != supplied
    store = ConversationOwnerStore(settings.qa_conversation_owner_sqlite_path)
    try:
        assert store.teacher_for(conversation_id) == "teacher-a"
    finally:
        store.close()


def test_ownership_holds_when_checkpointing_is_disabled(tmp_path: Path, monkeypatch):
    agent = RecordingAgent()
    settings = _settings(
        tmp_path,
        environment="development",
        assistant_auth_enabled=True,
        agent_checkpointing_enabled=False,
    )
    with _client(
        monkeypatch, settings, agent, {"teacher-a": "teacher-a", "teacher-b": "teacher-b"}
    ) as client:
        created = _post(client, cookie="teacher-a")
        conversation_id = created.json()["conversation_id"]
        denied = _post(client, cookie="teacher-b", conversation_id=conversation_id)
        memory = agent.store.get(conversation_id)
    assert denied.status_code == 404
    assert denied.json() == _NOT_FOUND
    assert memory is not None
    assert [message.content for message in memory.messages] == [_QUESTION]
    assert agent.checkpoints == {conversation_id: 1}


def test_preledger_checkpoint_cannot_be_claimed(tmp_path: Path, monkeypatch):
    agent = RecordingAgent()
    checkpoint = tmp_path / "checkpoints.sqlite"
    checkpoint.write_bytes(b"legacy-checkpoint")
    settings = _settings(tmp_path, environment="development", assistant_auth_enabled=True)
    legacy_id = str(uuid4())
    with _client(monkeypatch, settings, agent, {"teacher-a": "teacher-a"}) as client:
        response = _post(client, cookie="teacher-a", conversation_id=legacy_id)
    assert response.status_code == 404
    assert response.json() == _NOT_FOUND
    assert agent.calls == []
    assert checkpoint.read_bytes() == b"legacy-checkpoint"
    store = ConversationOwnerStore(settings.qa_conversation_owner_sqlite_path)
    try:
        assert store.teacher_for(legacy_id) is None
    finally:
        store.close()


def test_graph_failure_keeps_the_owner_for_retry(tmp_path: Path, monkeypatch):
    agent = RecordingAgent()
    agent.fail_once = True
    settings = _settings(tmp_path, environment="development", assistant_auth_enabled=True)
    with _client(
        monkeypatch, settings, agent, {"teacher-a": "teacher-a", "teacher-b": "teacher-b"}
    ) as client:
        failed = _post(client, cookie="teacher-a")
        connection = sqlite3.connect(settings.qa_conversation_owner_sqlite_path)
        try:
            rows = connection.execute(
                "SELECT conversation_id, teacher_id FROM owners"
            ).fetchall()
        finally:
            connection.close()
        assert len(rows) == 1
        conversation_id = rows[0][0]
        assert rows[0][1] == "teacher-a"
        denied = _post(client, cookie="teacher-b", conversation_id=conversation_id)
        retried = _post(client, cookie="teacher-a", conversation_id=conversation_id)
    assert failed.status_code == 500
    assert failed.json()["code"] == "AGENT_EXECUTION_FAILURE"
    assert denied.status_code == 404
    assert retried.status_code == 200
    assert retried.json()["conversation_id"] == conversation_id
    assert agent.checkpoints == {conversation_id: 1}


def test_concurrent_claims_cannot_reassign_an_id(tmp_path: Path):
    path = tmp_path / "qa_conversation_owners.sqlite"
    ready = ConversationOwnerStore(path)
    ready.close()
    conversation_id = str(uuid4())
    barrier = Barrier(2)
    results: list[tuple[str, bool]] = []

    def attempt(teacher_id: str) -> None:
        store = ConversationOwnerStore(path)
        try:
            barrier.wait(timeout=5)
            results.append((teacher_id, store.claim(conversation_id, teacher_id)))
        finally:
            store.close()

    threads = [Thread(target=attempt, args=(teacher,)) for teacher in ("teacher-a", "teacher-b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(won for _, won in results) == [False, True]
    store = ConversationOwnerStore(path)
    try:
        owner = store.teacher_for(conversation_id)
        assert owner in {"teacher-a", "teacher-b"}
        assert store.claim(conversation_id, "teacher-c") is False
        assert store.teacher_for(conversation_id) == owner
    finally:
        store.close()


def test_ledger_failure_does_not_run_the_graph_or_fall_back(tmp_path: Path, monkeypatch):
    agent = RecordingAgent()
    blocked = tmp_path / "qa_conversation_owners.sqlite"
    blocked.mkdir()
    settings = _settings(
        tmp_path,
        environment="development",
        assistant_auth_enabled=True,
        qa_conversation_owner_sqlite_path=str(blocked),
    )
    with _client(monkeypatch, settings, agent, {"teacher-a": "teacher-a"}) as client:
        response = _post(client, cookie="teacher-a")
    assert response.status_code == 503
    assert response.json()["detail"] == "Conversation ownership is unavailable"
    assert str(tmp_path) not in response.text
    assert "sqlite" not in response.text.lower()
    assert agent.calls == []
    with pytest.raises(OwnerStoreError):
        ConversationOwnerStore(tmp_path / "checkpoints.sqlite")


@pytest.mark.parametrize("environment", ["staging", "Staging", "preview", ""])
def test_unlisted_environments_require_authentication(environment: str, tmp_path: Path, monkeypatch):
    agent = RecordingAgent()
    settings = _settings(tmp_path, environment=environment, assistant_auth_enabled=False)
    with _client(monkeypatch, settings, agent) as client:
        response = _post(client, forged=True)
    assert response.status_code == 503
    assert response.json()["detail"] == "Authentication is unavailable"
    assert "forged-teacher" not in response.text
    assert agent.calls == []
    assert not Path(settings.qa_conversation_owner_sqlite_path).exists()


def test_staging_ledger_failure_does_not_run_the_graph(tmp_path: Path, monkeypatch):
    agent = RecordingAgent()
    blocked = tmp_path / "qa_conversation_owners.sqlite"
    blocked.mkdir()
    settings = _settings(
        tmp_path,
        environment="staging",
        assistant_auth_enabled=False,
        qa_conversation_owner_sqlite_path=str(blocked),
    )
    with _client(monkeypatch, settings, agent, {"teacher-a": "teacher-a"}) as client:
        response = _post(client, cookie="teacher-a")
    assert response.status_code == 503
    assert response.json()["detail"] == "Conversation ownership is unavailable"
    assert str(tmp_path) not in response.text
    assert agent.calls == []


def test_staging_owner_can_continue_and_others_cannot(tmp_path: Path, monkeypatch):
    agent = RecordingAgent()
    settings = _settings(tmp_path, environment="staging", assistant_auth_enabled=False)
    with _client(
        monkeypatch, settings, agent, {"teacher-a": "teacher-a", "teacher-b": "teacher-b"}
    ) as client:
        created = _post(client, cookie="teacher-a")
        conversation_id = created.json()["conversation_id"]
        denied = _post(client, cookie="teacher-b", conversation_id=conversation_id, forged=True)
        continued = _post(client, cookie="teacher-a", conversation_id=conversation_id)
    assert created.status_code == continued.status_code == 200
    assert denied.status_code == 404
    assert denied.json() == _NOT_FOUND
    assert agent.calls == [conversation_id, conversation_id]
    assert agent.checkpoints == {conversation_id: 2}


@pytest.mark.parametrize("environment", ["development", "test", "Test"])
def test_allowlisted_environments_stay_anonymous_without_auth(
    environment: str, tmp_path: Path, monkeypatch
):
    agent = RecordingAgent()
    blocked = tmp_path / "qa_conversation_owners.sqlite"
    blocked.mkdir()
    settings = _settings(
        tmp_path,
        environment=environment,
        assistant_auth_enabled=False,
        qa_conversation_owner_sqlite_path=str(blocked),
    )
    with _client(monkeypatch, settings, agent) as client:
        response = _post(client, forged=True)
    assert response.status_code == 200
    assert response.json()["answer"] == "Owned answer"
    assert agent.calls == [None]
    assert not any(blocked.iterdir())


def test_assistant_threads_stay_separate_and_disabled(tmp_path: Path, monkeypatch):
    thread_path = tmp_path / "assistant_threads.sqlite"
    store = AssistantThreadStore(thread_path)
    try:
        tables = {
            row[0]
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        store.close()
    assert SCHEMA_VERSION == 1
    assert "owners" not in tables
    assert "threads" in tables
    agent = RecordingAgent()
    settings = _settings(tmp_path, assistant_enabled=False, assistant_auth_enabled=False)
    with _client(monkeypatch, settings, agent) as client:
        listed = client.get(
            "/api/v1/assistant/threads",
            headers={"X-Teacher-Id": "forged-teacher"},
        )
    assert listed.status_code == 404
    assert listed.json()["detail"] == "Not found"
    connection = sqlite3.connect(thread_path)
    try:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(threads)")]
    finally:
        connection.close()
    assert columns[:2] == ["thread_id", "teacher_id"]
    assert "conversation_id" not in columns
