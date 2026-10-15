"""Teacher-owned assistant thread store. Uses a temporary SQLite file."""

from __future__ import annotations

import inspect
import multiprocessing
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest

from app.agent.orchestrator import CurriculumQAAgent
from app.assistant.identity import (
    TrustedTeacher,
    UnauthenticatedTeacherError,
    VerifiedTeacherClaims,
    bind_verified_claims,
    current_identity_gate,
    register_identity_gate,
    reset_identity_gate,
)
from app.assistant.threads import (
    AssistantThreadStore,
    QaOutcomeSummary,
    ThreadStoreError,
    ThreadUnavailable,
    qa_context_hint,
)


class _Gate:
    def verify(self, request: object):
        return None


@pytest.fixture(autouse=True)
def registered_gate():
    gate = _Gate()
    register_identity_gate(gate)
    yield gate
    reset_identity_gate()


def _teacher(teacher_id: str, school_id: str | None = None) -> TrustedTeacher:
    gate = current_identity_gate()
    assert gate is not None
    return bind_verified_claims(
        VerifiedTeacherClaims(teacher_id, school_id),
        gate=gate,
    )


def _store(tmp_path: Path) -> AssistantThreadStore:
    return AssistantThreadStore(tmp_path / "assistant_threads.sqlite")


def test_client_supplied_teacher_id_is_not_a_principal(tmp_path: Path, registered_gate):
    assert not hasattr(TrustedTeacher, "from_verified_issuer")
    other = _Gate()
    with pytest.raises(UnauthenticatedTeacherError):
        bind_verified_claims(VerifiedTeacherClaims("teacher-a"), gate=other)
    with pytest.raises(UnauthenticatedTeacherError):
        bind_verified_claims("teacher-a", gate=registered_gate)  # type: ignore[arg-type]
    with pytest.raises(UnauthenticatedTeacherError):
        TrustedTeacher()
    store = _store(tmp_path)
    try:
        with pytest.raises(UnauthenticatedTeacherError):
            store.create_thread("teacher-a")  # type: ignore[arg-type]
    finally:
        store.close()


def test_create_get_and_restart_round_trip(tmp_path: Path):
    path = tmp_path / "assistant_threads.sqlite"
    owner = _teacher("teacher-a", "school-1")
    store = AssistantThreadStore(path)
    created = store.create_thread(owner)
    store.append_message(owner, created.thread_id, "user", "What topics are in Class 4?")
    store.append_message(
        owner,
        created.thread_id,
        "assistant",
        "Number and Numeration is one topic.",
    )
    store.update_working_context(
        owner,
        created.thread_id,
        grade="CLASS_4",
        subject="MATHEMATICS",
        task="topic-list",
    )
    store.record_qa_outcome(
        owner,
        created.thread_id,
        QaOutcomeSummary(
            status="completed",
            evidence_status="found",
            verification_passed=True,
            verification_recommendation="accept",
            coverage_decision="accept",
            termination_reason="verification_passed",
        ),
    )
    store.close()

    reopened = AssistantThreadStore(path)
    try:
        loaded = reopened.get_thread(owner, created.thread_id)
    finally:
        reopened.close()

    assert loaded.thread_id == created.thread_id
    assert loaded.teacher_id == "teacher-a"
    assert loaded.school_id == "school-1"
    assert [message.content for message in loaded.messages] == [
        "What topics are in Class 4?",
        "Number and Numeration is one topic.",
    ]
    assert [message.message_id for message in loaded.messages] == [1, 2]
    assert loaded.grade == "CLASS_4"
    assert loaded.subject == "MATHEMATICS"
    assert loaded.task == "topic-list"
    assert loaded.qa_outcome is not None
    assert loaded.qa_outcome.status == "completed"
    assert loaded.qa_outcome.verification_passed is True


def test_unknown_and_other_teacher_look_the_same(tmp_path: Path):
    store = _store(tmp_path)
    owner = _teacher("teacher-a", "school-1")
    other = _teacher("teacher-b", "school-1")
    secret = "private-plan-do-not-leak"
    try:
        created = store.create_thread(owner)
        store.append_message(owner, created.thread_id, "user", secret)
        with pytest.raises(ThreadUnavailable) as missing:
            store.get_thread(owner, str(uuid4()))
        with pytest.raises(ThreadUnavailable) as denied:
            store.get_thread(other, created.thread_id)
        assert str(missing.value) == str(denied.value) == "Thread not found"
        assert secret not in str(denied.value)
        assert "teacher-a" not in str(denied.value)
        assert store.list_threads(other) == []
        assert len(store.list_threads(owner)) == 1
        assert store.list_threads(owner)[0].thread_id == created.thread_id
    finally:
        store.close()


def test_unauthorized_access_does_not_call_qa(tmp_path: Path):
    store = _store(tmp_path)
    owner = _teacher("teacher-a")
    other = _teacher("teacher-b")
    calls: list[str] = []

    def qa(record):
        calls.append(record.thread_id)
        return "should-not-run"

    try:
        created = store.create_thread(owner)
        with pytest.raises(ThreadUnavailable):
            store.with_owned_thread(other, created.thread_id, qa)
        with pytest.raises(ThreadUnavailable):
            store.with_owned_thread(owner, str(uuid4()), qa)
        assert calls == []
        assert store.with_owned_thread(owner, created.thread_id, qa) == "should-not-run"
        assert calls == [created.thread_id]
    finally:
        store.close()


def test_repeated_outcome_updates_replace_one_row(tmp_path: Path):
    store = _store(tmp_path)
    owner = _teacher("teacher-a")
    try:
        created = store.create_thread(owner)
        store.record_qa_outcome(
            owner,
            created.thread_id,
            QaOutcomeSummary(status="insufficient_evidence", evidence_status="partial"),
        )
        store.record_qa_outcome(
            owner,
            created.thread_id,
            QaOutcomeSummary(
                status="completed",
                evidence_status="found",
                verification_passed=True,
                technical_failure=False,
                failure_code=None,
            ),
        )
        loaded = store.get_thread(owner, created.thread_id)
        assert loaded.qa_outcome is not None
        assert loaded.qa_outcome.status == "completed"
        assert loaded.qa_outcome.evidence_status == "found"
        raw = sqlite3.connect(store.path)
        try:
            count = raw.execute("SELECT COUNT(*) FROM qa_outcomes").fetchone()[0]
            columns = {
                row[1] for row in raw.execute("PRAGMA table_info(qa_outcomes)")
            }
        finally:
            raw.close()
        assert count == 1
        assert "answer" not in columns
        assert "content" not in columns
    finally:
        store.close()


def test_database_failure_and_bad_schema_hide_thread_contents(tmp_path: Path):
    path = tmp_path / "assistant_threads.sqlite"
    store = AssistantThreadStore(path)
    owner = _teacher("teacher-a")
    secret = "secret-classroom-note"
    created = store.create_thread(owner)
    store.append_message(owner, created.thread_id, "user", secret)
    store.close()

    with pytest.raises(ThreadStoreError) as closed:
        store.append_message(owner, created.thread_id, "user", "another secret note")
    assert secret not in str(closed.value)
    assert "another secret" not in str(closed.value)
    assert str(closed.value) == "Assistant thread store is unavailable"

    raw = sqlite3.connect(path)
    raw.execute("UPDATE schema_meta SET version = 99")
    raw.commit()
    raw.close()
    with pytest.raises(ThreadStoreError) as mismatched:
        AssistantThreadStore(path)
    assert secret not in str(mismatched.value)


def test_concurrent_appends_keep_every_message(tmp_path: Path):
    store = _store(tmp_path)
    owner = _teacher("teacher-a")
    try:
        created = store.create_thread(owner)

        def write(index: int) -> None:
            store.append_message(owner, created.thread_id, "user", f"note-{index}")

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(write, range(20)))
        loaded = store.get_thread(owner, created.thread_id)
        contents = [message.content for message in loaded.messages]
        assert sorted(contents, key=lambda value: int(value.split("-")[1])) == [
            f"note-{index}" for index in range(20)
        ]
        assert len(contents) == len(set(contents)) == 20
        assert [message.message_id for message in loaded.messages] == list(range(1, 21))
    finally:
        store.close()


def test_store_does_not_touch_langgraph_checkpoints(tmp_path: Path):
    checkpoint = tmp_path / "checkpoints.sqlite"
    store = _store(tmp_path)
    owner = _teacher("teacher-a")
    try:
        store.create_thread(owner)
        with pytest.raises(ThreadStoreError) as refused:
            AssistantThreadStore(checkpoint)
        assert "LangGraph checkpoint" in str(refused.value)
    finally:
        store.close()
    assert not checkpoint.exists()
    params = set(inspect.signature(CurriculumQAAgent.ask).parameters)
    assert {"grade", "subject", "task", "teacher_id"}.isdisjoint(params)


def test_working_context_hint_is_not_sent_as_ask_arguments(tmp_path: Path):
    store = _store(tmp_path)
    owner = _teacher("teacher-a")
    try:
        created = store.create_thread(owner)
        updated = store.update_working_context(
            owner,
            created.thread_id,
            grade="CLASS_4",
            subject="MATHEMATICS",
            task="revision",
        )
        hint = qa_context_hint(updated)
        assert hint.model_dump() == {
            "grade": "CLASS_4",
            "subject": "MATHEMATICS",
            "task": "revision",
        }
        ask_params = set(inspect.signature(CurriculumQAAgent.ask).parameters)
        assert set(hint.model_dump()).isdisjoint(ask_params)
    finally:
        store.close()


def test_other_teacher_cannot_append_or_delete(tmp_path: Path):
    store = _store(tmp_path)
    owner = _teacher("teacher-a")
    other = _teacher("teacher-b")
    try:
        created = store.create_thread(owner)
        store.append_message(owner, created.thread_id, "user", "keep this")
        with pytest.raises(ThreadUnavailable):
            store.append_message(other, created.thread_id, "user", "overwrite")
        with pytest.raises(ThreadUnavailable):
            store.delete_thread(other, created.thread_id)
        store.delete_thread(owner, created.thread_id)
        with pytest.raises(ThreadUnavailable):
            store.get_thread(owner, created.thread_id)
        raw = sqlite3.connect(store.path)
        try:
            assert raw.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        finally:
            raw.close()
    finally:
        store.close()


def _mp_init(path: str) -> None:
    store = AssistantThreadStore(path)
    store.close()


def _mp_append(path: str, teacher_id: str, thread_id: str, notes: list[str]) -> None:
    gate = _Gate()
    register_identity_gate(gate)
    teacher = bind_verified_claims(VerifiedTeacherClaims(teacher_id), gate=gate)
    store = AssistantThreadStore(path)
    try:
        for note in notes:
            store.append_message(teacher, thread_id, "user", note)
    finally:
        store.close()
        reset_identity_gate()


def _mp_probe(path: str, teacher_id: str, thread_id: str, queue) -> None:
    gate = _Gate()
    register_identity_gate(gate)
    teacher = bind_verified_claims(VerifiedTeacherClaims(teacher_id), gate=gate)
    store = AssistantThreadStore(path)
    try:
        try:
            store.get_thread(teacher, thread_id)
            queue.put("allowed")
        except ThreadUnavailable:
            queue.put("unavailable")
    finally:
        store.close()
        reset_identity_gate()


def test_two_instances_isolate_owners_and_see_deletes(tmp_path: Path):
    path = tmp_path / "assistant_threads.sqlite"
    writer = AssistantThreadStore(path)
    owner = _teacher("teacher-a")
    created = writer.create_thread(owner)
    writer.append_message(owner, created.thread_id, "user", "private note")
    writer.close()

    reader = AssistantThreadStore(path)
    try:
        loaded = reader.get_thread(owner, created.thread_id)
        assert [message.content for message in loaded.messages] == ["private note"]
        with pytest.raises(ThreadUnavailable) as other:
            reader.get_thread(_teacher("teacher-b"), created.thread_id)
        with pytest.raises(ThreadUnavailable) as missing:
            reader.get_thread(owner, str(uuid4()))
        assert str(other.value) == str(missing.value) == "Thread not found"
        assert "private note" not in str(other.value)
        reader.delete_thread(owner, created.thread_id)
    finally:
        reader.close()

    follower = AssistantThreadStore(path)
    try:
        with pytest.raises(ThreadUnavailable):
            follower.get_thread(owner, created.thread_id)
        raw = sqlite3.connect(path)
        try:
            assert raw.execute("SELECT COUNT(*) FROM threads").fetchone()[0] == 0
            assert raw.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        finally:
            raw.close()
    finally:
        follower.close()


def test_two_instances_append_concurrently(tmp_path: Path):
    path = tmp_path / "assistant_threads.sqlite"
    setup = AssistantThreadStore(path)
    created = setup.create_thread(_teacher("teacher-a"))
    setup.close()

    def write(index: int) -> None:
        store = AssistantThreadStore(path)
        try:
            store.append_message(
                _teacher("teacher-a"),
                created.thread_id,
                "user",
                f"note-{index}",
            )
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(16)))

    check = AssistantThreadStore(path)
    try:
        loaded = check.get_thread(_teacher("teacher-a"), created.thread_id)
    finally:
        check.close()
    contents = [message.content for message in loaded.messages]
    assert sorted(contents, key=lambda value: int(value.split("-")[1])) == [
        f"note-{index}" for index in range(16)
    ]
    assert len(contents) == len(set(contents)) == 16


def test_separate_processes_initialize_append_and_isolate(tmp_path: Path):
    """Spawned processes, one machine, one SQLite file.

    Not covered: a second host, a network filesystem, or crash recovery
    beyond SQLite's own rollback.
    """
    fresh = tmp_path / "fresh.sqlite"
    ctx = multiprocessing.get_context("spawn")
    initializers = [ctx.Process(target=_mp_init, args=(str(fresh),)) for _ in range(2)]
    for proc in initializers:
        proc.start()
    for proc in initializers:
        proc.join(60)
        assert proc.exitcode == 0

    path = tmp_path / "assistant_threads.sqlite"
    setup = AssistantThreadStore(path)
    created = setup.create_thread(_teacher("teacher-a"))
    thread_id = created.thread_id
    setup.close()

    queue = ctx.Queue()
    workers = [
        ctx.Process(
            target=_mp_append,
            args=(str(path), "teacher-a", thread_id, ["p1-a", "p1-b"]),
        ),
        ctx.Process(
            target=_mp_append,
            args=(str(path), "teacher-a", thread_id, ["p2-a", "p2-b"]),
        ),
        ctx.Process(
            target=_mp_probe,
            args=(str(path), "teacher-b", thread_id, queue),
        ),
    ]
    for proc in workers:
        proc.start()
    for proc in workers:
        proc.join(60)
        assert proc.exitcode == 0
    assert queue.get(timeout=5) == "unavailable"

    check = AssistantThreadStore(path)
    try:
        loaded = check.get_thread(_teacher("teacher-a"), thread_id)
    finally:
        check.close()
    assert sorted(message.content for message in loaded.messages) == [
        "p1-a",
        "p1-b",
        "p2-a",
        "p2-b",
    ]
    raw = sqlite3.connect(fresh)
    try:
        assert raw.execute("SELECT version FROM schema_meta").fetchone()[0] == 1
    finally:
        raw.close()


def test_directory_path_is_a_store_failure(tmp_path: Path):
    target = tmp_path / "not-a-database"
    target.mkdir()
    with pytest.raises(ThreadStoreError) as exc:
        AssistantThreadStore(target)
    assert str(exc.value) == "Assistant thread store is unavailable"
