"""Teacher-owned assistant threads in a dedicated SQLite file.

This store does not open ``data/checkpoints.sqlite`` and does not write
LangGraph checkpoint tables. It is not opened by the FastAPI application.

Schema version 1 is created on first open (``CREATE TABLE IF`` style via
``schema_meta``). A different version is refused. There is no separate
migration runner in this service.

Persisted working context is grade, subject, and task. Those fields are not
parameters of ``CurriculumQAAgent.ask``. A later turn may read them for the
assistant's own prompt. It must not reuse the assistant thread id as the
LangGraph checkpoint id, and it must not pass them as new ``ask`` arguments.

The last QA outcome stores status and provenance only: no answer text, evidence
rows, model reasoning, or provider payloads. Visible turns belong in messages.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.assistant.identity import TrustedTeacher, require_trusted_teacher
from app.logging_utils import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1
_MAX_MESSAGE_CHARS = 20_000
_CHECKPOINT_NAME = "checkpoints.sqlite"
_STORE_FAILURE = "Assistant thread store is unavailable"

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    teacher_id TEXT NOT NULL,
    school_id TEXT,
    grade TEXT,
    subject TEXT,
    task TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS threads_teacher_updated ON threads (teacher_id, updated_at);
CREATE TABLE IF NOT EXISTS messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_thread_order ON messages (thread_id, message_id);
CREATE TABLE IF NOT EXISTS qa_outcomes (
    thread_id TEXT PRIMARY KEY REFERENCES threads(thread_id) ON DELETE CASCADE,
    status TEXT,
    evidence_status TEXT,
    verification_passed INTEGER,
    verification_recommendation TEXT,
    coverage_decision TEXT,
    termination_reason TEXT,
    technical_failure INTEGER NOT NULL DEFAULT 0,
    failure_code TEXT,
    updated_at TEXT NOT NULL
);
"""


class ThreadUnavailable(Exception):
    """The thread is missing or belongs to someone else.

    Both cases use this error so a caller cannot tell them apart.
    """

    def __init__(self) -> None:
        super().__init__("Thread not found")


class ThreadStoreError(Exception):
    """The store failed. The message contains no thread contents."""

    def __init__(self, message: str = _STORE_FAILURE) -> None:
        super().__init__(message)


class ThreadMessage(BaseModel):
    message_id: int
    role: str
    content: str
    created_at: str


class QaOutcomeSummary(BaseModel):
    """Minimum QA provenance needed on the next turn. Not an AskResponse."""

    status: Optional[str] = None
    evidence_status: Optional[str] = None
    verification_passed: Optional[bool] = None
    verification_recommendation: Optional[str] = None
    coverage_decision: Optional[str] = None
    termination_reason: Optional[str] = None
    technical_failure: bool = False
    failure_code: Optional[str] = None


class WorkingContext(BaseModel):
    grade: Optional[str] = None
    subject: Optional[str] = None
    task: Optional[str] = None


class ThreadSummary(BaseModel):
    thread_id: str
    teacher_id: str
    school_id: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    task: Optional[str] = None
    created_at: str
    updated_at: str


class ThreadPreview(BaseModel):
    """Owned thread fields for a sidebar list. No message body and no teacher id."""

    thread_id: str
    grade: Optional[str] = None
    subject: Optional[str] = None
    task: Optional[str] = None
    created_at: str
    updated_at: str
    first_user_message: Optional[str] = None


class ThreadRecord(ThreadSummary):
    messages: list[ThreadMessage] = Field(default_factory=list)
    qa_outcome: Optional[QaOutcomeSummary] = None


_TITLE_LIMIT = 80
_EMPTY_THREAD_TITLE = "New chat"


def thread_title(first_user_message: str | None) -> str:
    """Sidebar label from the first user message. No stored title column."""
    if not isinstance(first_user_message, str):
        return _EMPTY_THREAD_TITLE
    compact = " ".join(first_user_message.split())
    if not compact:
        return _EMPTY_THREAD_TITLE
    if len(compact) <= _TITLE_LIMIT:
        return compact
    return compact[: _TITLE_LIMIT - 1].rstrip() + "…"


def qa_context_hint(record: ThreadSummary) -> WorkingContext:
    """Working context for a later assistant turn.

    ``CurriculumQAAgent.ask`` accepts question, conversation_id, request_id,
    and experiment flags. It does not accept grade, subject, or task. This
    hint must not be expanded into new ``ask`` keyword arguments, and
    ``record.thread_id`` must not be sent as the LangGraph ``conversation_id``.
    """
    return WorkingContext(
        grade=record.grade,
        subject=record.subject,
        task=record.task,
    )


class AssistantThreadStore:
    """SQLite store. Each instance locks its own connection.

    Other instances and processes are serialized with ``BEGIN IMMEDIATE`` and
    SQLite's busy timeout. That cross-process lock is what concurrent tests
    exercise. This store is not opened from application startup.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.name == _CHECKPOINT_NAME:
            raise ThreadStoreError(
                "Assistant thread store cannot use the LangGraph checkpoint database"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        try:
            self._conn = sqlite3.connect(
                str(self.path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA busy_timeout = 5000")
            self._conn.execute("PRAGMA journal_mode = WAL").fetchone()
            self._ensure_schema()
        except ThreadStoreError:
            self.close()
            raise
        except sqlite3.Error:
            self.close()
            raise ThreadStoreError(_STORE_FAILURE) from None

    def close(self) -> None:
        with self._lock:
            conn = self._conn
            self._conn = None
            if conn is not None:
                conn.close()

    def create_thread(self, principal: TrustedTeacher) -> ThreadRecord:
        teacher = require_trusted_teacher(principal)
        now = _now()
        thread_id = str(uuid4())
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO threads (
                    thread_id, teacher_id, school_id, grade, subject, task,
                    created_at, updated_at
                ) VALUES (?, ?, ?, NULL, NULL, NULL, ?, ?)
                """,
                (thread_id, teacher.teacher_id, teacher.school_id, now, now),
            )
        return ThreadRecord(
            thread_id=thread_id,
            teacher_id=teacher.teacher_id,
            school_id=teacher.school_id,
            created_at=now,
            updated_at=now,
        )

    def get_thread(self, principal: TrustedTeacher, thread_id: str) -> ThreadRecord:
        teacher = require_trusted_teacher(principal)
        with self._transaction() as conn:
            row = _owned_thread(conn, teacher.teacher_id, thread_id)
            if row is None:
                raise ThreadUnavailable()
            messages = _load_messages(conn, thread_id)
            outcome = _load_outcome(conn, thread_id)
        return _record(row, messages, outcome)

    def list_threads(self, principal: TrustedTeacher) -> list[ThreadSummary]:
        teacher = require_trusted_teacher(principal)
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT thread_id, teacher_id, school_id, grade, subject, task,
                       created_at, updated_at
                FROM threads
                WHERE teacher_id = ?
                ORDER BY updated_at DESC, thread_id ASC
                """,
                (teacher.teacher_id,),
            ).fetchall()
        return [_summary(row) for row in rows]

    def list_previews(self, principal: TrustedTeacher) -> list[ThreadPreview]:
        """Owned threads in the same order as ``list_threads``.

        ``first_user_message`` is the earliest user row, or None when the
        thread has no user message. Titles are not stored.
        """
        teacher = require_trusted_teacher(principal)
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT t.thread_id, t.grade, t.subject, t.task,
                       t.created_at, t.updated_at,
                       (
                           SELECT m.content
                           FROM messages m
                           WHERE m.thread_id = t.thread_id AND m.role = 'user'
                           ORDER BY m.message_id ASC
                           LIMIT 1
                       ) AS first_user_message
                FROM threads t
                WHERE t.teacher_id = ?
                ORDER BY t.updated_at DESC, t.thread_id ASC
                """,
                (teacher.teacher_id,),
            ).fetchall()
        return [
            ThreadPreview(
                thread_id=row["thread_id"],
                grade=row["grade"],
                subject=row["subject"],
                task=row["task"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                first_user_message=row["first_user_message"],
            )
            for row in rows
        ]

    def append_message(
        self,
        principal: TrustedTeacher,
        thread_id: str,
        role: str,
        content: str,
    ) -> ThreadMessage:
        teacher = require_trusted_teacher(principal)
        if role not in {"user", "assistant"}:
            raise ThreadStoreError("Message role is not storable")
        text = content if isinstance(content, str) else ""
        if not text.strip():
            raise ThreadStoreError("Message content is empty")
        if len(text) > _MAX_MESSAGE_CHARS:
            raise ThreadStoreError("Message exceeds the stored length limit")
        now = _now()
        with self._transaction() as conn:
            if _owned_thread(conn, teacher.teacher_id, thread_id) is None:
                raise ThreadUnavailable()
            cursor = conn.execute(
                """
                INSERT INTO messages (thread_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (thread_id, role, text, now),
            )
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                (now, thread_id),
            )
            message_id = int(cursor.lastrowid)
        return ThreadMessage(
            message_id=message_id,
            role=role,
            content=text,
            created_at=now,
        )

    def update_working_context(
        self,
        principal: TrustedTeacher,
        thread_id: str,
        *,
        grade: str | None = None,
        subject: str | None = None,
        task: str | None = None,
    ) -> ThreadRecord:
        teacher = require_trusted_teacher(principal)
        assignments: list[str] = []
        values: list[str] = []
        for column, raw in (("grade", grade), ("subject", subject), ("task", task)):
            if raw is None:
                continue
            cleaned = raw.strip()
            if len(cleaned) > 128:
                raise ThreadStoreError("Working context value is too long")
            assignments.append(f"{column} = ?")
            values.append(cleaned)
        now = _now()
        with self._transaction() as conn:
            if _owned_thread(conn, teacher.teacher_id, thread_id) is None:
                raise ThreadUnavailable()
            if assignments:
                assignments.append("updated_at = ?")
                values.append(now)
                values.append(thread_id)
                conn.execute(
                    f"UPDATE threads SET {', '.join(assignments)} WHERE thread_id = ?",
                    values,
                )
            row = _owned_thread(conn, teacher.teacher_id, thread_id)
            if row is None:
                raise ThreadUnavailable()
            messages = _load_messages(conn, thread_id)
            outcome = _load_outcome(conn, thread_id)
        return _record(row, messages, outcome)

    def record_qa_outcome(
        self,
        principal: TrustedTeacher,
        thread_id: str,
        outcome: QaOutcomeSummary,
    ) -> QaOutcomeSummary:
        teacher = require_trusted_teacher(principal)
        now = _now()
        passed = outcome.verification_passed
        passed_value = None if passed is None else int(bool(passed))
        with self._transaction() as conn:
            if _owned_thread(conn, teacher.teacher_id, thread_id) is None:
                raise ThreadUnavailable()
            conn.execute(
                """
                INSERT INTO qa_outcomes (
                    thread_id, status, evidence_status, verification_passed,
                    verification_recommendation, coverage_decision,
                    termination_reason, technical_failure, failure_code, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    status = excluded.status,
                    evidence_status = excluded.evidence_status,
                    verification_passed = excluded.verification_passed,
                    verification_recommendation = excluded.verification_recommendation,
                    coverage_decision = excluded.coverage_decision,
                    termination_reason = excluded.termination_reason,
                    technical_failure = excluded.technical_failure,
                    failure_code = excluded.failure_code,
                    updated_at = excluded.updated_at
                """,
                (
                    thread_id,
                    outcome.status,
                    outcome.evidence_status,
                    passed_value,
                    outcome.verification_recommendation,
                    outcome.coverage_decision,
                    outcome.termination_reason,
                    int(bool(outcome.technical_failure)),
                    outcome.failure_code,
                    now,
                ),
            )
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                (now, thread_id),
            )
        return outcome

    def delete_thread(self, principal: TrustedTeacher, thread_id: str) -> None:
        teacher = require_trusted_teacher(principal)
        with self._transaction() as conn:
            if _owned_thread(conn, teacher.teacher_id, thread_id) is None:
                raise ThreadUnavailable()
            conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))

    def with_owned_thread(self, principal: TrustedTeacher, thread_id: str, action):
        """Load an owned thread before invoking the supplied action.

        ``action`` is not called when the principal cannot access the thread.
        """
        record = self.get_thread(principal, thread_id)
        return action(record)

    def _ensure_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(_SCHEMA_V1)
        self._conn.execute(
            "INSERT OR IGNORE INTO schema_meta (id, version) VALUES (1, ?)",
            (SCHEMA_VERSION,),
        )
        self._conn.commit()
        version = _schema_version(self._conn)
        if version != SCHEMA_VERSION:
            raise ThreadStoreError(_STORE_FAILURE)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._conn
            if conn is None:
                raise ThreadStoreError(_STORE_FAILURE)
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except ThreadUnavailable:
                _rollback(conn)
                raise
            except ThreadStoreError:
                _rollback(conn)
                raise
            except sqlite3.Error:
                _rollback(conn)
                logger.warning("assistant.threads.store_failure")
                raise ThreadStoreError(_STORE_FAILURE) from None


def _schema_version(conn: sqlite3.Connection) -> int | None:
    try:
        row = conn.execute(
            "SELECT version FROM schema_meta WHERE id = 1"
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    return int(row["version"])


def _owned_thread(
    conn: sqlite3.Connection,
    teacher_id: str,
    thread_id: str,
) -> sqlite3.Row | None:
    row = conn.execute(
        """
        SELECT thread_id, teacher_id, school_id, grade, subject, task,
               created_at, updated_at
        FROM threads
        WHERE thread_id = ?
        """,
        (thread_id,),
    ).fetchone()
    if row is None or row["teacher_id"] != teacher_id:
        return None
    return row


def _load_messages(conn: sqlite3.Connection, thread_id: str) -> list[ThreadMessage]:
    rows = conn.execute(
        """
        SELECT message_id, role, content, created_at
        FROM messages
        WHERE thread_id = ?
        ORDER BY message_id ASC
        """,
        (thread_id,),
    ).fetchall()
    return [
        ThreadMessage(
            message_id=int(row["message_id"]),
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


def _load_outcome(conn: sqlite3.Connection, thread_id: str) -> QaOutcomeSummary | None:
    row = conn.execute(
        """
        SELECT status, evidence_status, verification_passed,
               verification_recommendation, coverage_decision,
               termination_reason, technical_failure, failure_code
        FROM qa_outcomes
        WHERE thread_id = ?
        """,
        (thread_id,),
    ).fetchone()
    if row is None:
        return None
    passed = row["verification_passed"]
    return QaOutcomeSummary(
        status=row["status"],
        evidence_status=row["evidence_status"],
        verification_passed=None if passed is None else bool(passed),
        verification_recommendation=row["verification_recommendation"],
        coverage_decision=row["coverage_decision"],
        termination_reason=row["termination_reason"],
        technical_failure=bool(row["technical_failure"]),
        failure_code=row["failure_code"],
    )


def _summary(row: sqlite3.Row) -> ThreadSummary:
    return ThreadSummary(
        thread_id=row["thread_id"],
        teacher_id=row["teacher_id"],
        school_id=row["school_id"],
        grade=row["grade"],
        subject=row["subject"],
        task=row["task"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _record(
    row: sqlite3.Row,
    messages: list[ThreadMessage],
    outcome: QaOutcomeSummary | None,
) -> ThreadRecord:
    summary = _summary(row)
    return ThreadRecord(
        **summary.model_dump(),
        messages=messages,
        qa_outcome=outcome,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rollback(conn: sqlite3.Connection) -> None:
    try:
        conn.rollback()
    except sqlite3.Error:
        return
