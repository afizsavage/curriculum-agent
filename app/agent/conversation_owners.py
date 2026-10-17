"""Ownership ledger for legacy QA conversation ids.

The LangGraph checkpoint file is keyed only by conversation id. This database
records which verified teacher may resume that id. It is not the checkpoint
file, the assistant thread file, or the account file.

Anonymous ask remains available only for the explicit environments
``development`` and ``test``, and only while ``assistant_auth_enabled`` is
false. Production, staging, and every other name require the identity gate
and this ledger even when authentication is disabled. A missing ledger in
that configuration fails closed and does not run the graph.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from app.assistant.identity import (
    UnauthenticatedTeacherError,
    authenticate_http_request,
    current_identity_gate,
)
from app.config import Settings

_STORE_FAILURE = "Conversation ownership is unavailable"
_NOT_FOUND = "Conversation not found"
_UNAVAILABLE = "Authentication is unavailable"
_UNAUTHENTICATED = "Authenticated teacher principal is required"
_RESERVED_NAMES = frozenset(
    {
        "checkpoints.sqlite",
        "assistant_threads.sqlite",
        "assistant_auth.sqlite",
    }
)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS owners (
    conversation_id TEXT PRIMARY KEY,
    teacher_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""
SCHEMA_VERSION = 1
# Anonymous legacy ask is not inferred from "not production".
_ANONYMOUS_ENVIRONMENTS = frozenset({"development", "test"})


class OwnerStoreError(Exception):
    """The ledger failed. The message contains no filesystem path."""

    def __init__(self, message: str = _STORE_FAILURE) -> None:
        super().__init__(message)


class OwnerNotFound(Exception):
    """The id is missing or belongs to another teacher. Callers cannot tell."""

    def __init__(self) -> None:
        super().__init__(_NOT_FOUND)


def ownership_required(settings: Settings) -> bool:
    """Require the ledger outside the development and test allowlist.

    ``development`` and ``test`` stay anonymous only while authentication is
    disabled. Production, staging, a blank name, and any other name require
    the identity gate and this ledger even when authentication is disabled.
    """
    name = str(settings.environment or "").strip().lower()
    if name not in _ANONYMOUS_ENVIRONMENTS:
        return True
    return bool(settings.assistant_auth_enabled)


def resolve_owner_database_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        root = Path(__file__).resolve().parents[2]
        candidate = root / candidate
    return candidate


class ConversationOwnerStore:
    """One SQLite file of conversation-id grants. Rows are never reassigned."""

    def __init__(self, path: str | Path) -> None:
        self.path = resolve_owner_database_path(path)
        if self.path.name in _RESERVED_NAMES or self.path.suffix != ".sqlite":
            raise OwnerStoreError(_STORE_FAILURE)
        if self.path.exists() and self.path.is_dir():
            raise OwnerStoreError(_STORE_FAILURE)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA busy_timeout = 5000")
            self._conn.execute("PRAGMA journal_mode = WAL").fetchone()
            self._ensure_schema()
        except OwnerStoreError:
            self.close()
            raise
        except sqlite3.Error:
            self.close()
            raise OwnerStoreError(_STORE_FAILURE) from None

    def close(self) -> None:
        with self._lock:
            conn = self._conn
            self._conn = None
        if conn is not None:
            conn.close()

    def bind_new(self, teacher_id: str) -> str:
        """Insert a new id owned by ``teacher_id`` before the graph runs."""
        for _ in range(3):
            conversation_id = str(uuid4())
            if self.claim(conversation_id, teacher_id):
                return conversation_id
        raise OwnerStoreError(_STORE_FAILURE)

    def claim(self, conversation_id: str, teacher_id: str) -> bool:
        """Insert a grant. False when the id already has an owner."""
        _validate_teacher(teacher_id)
        if not _is_uuid(conversation_id):
            raise OwnerStoreError(_STORE_FAILURE)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._connection()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO owners (conversation_id, teacher_id, created_at) VALUES (?, ?, ?)",
                    (conversation_id, teacher_id, now),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                conn.rollback()
                return False
            except sqlite3.Error:
                conn.rollback()
                raise OwnerStoreError(_STORE_FAILURE) from None

    def require(self, conversation_id: str, teacher_id: str) -> None:
        """Allow the id only for its recorded teacher. Missing and foreign ids match."""
        _validate_teacher(teacher_id)
        if not _is_uuid(conversation_id):
            raise OwnerNotFound()
        with self._lock:
            conn = self._connection()
            try:
                row = conn.execute(
                    "SELECT teacher_id FROM owners WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
            except sqlite3.Error:
                raise OwnerStoreError(_STORE_FAILURE) from None
        if row is None or row["teacher_id"] != teacher_id:
            raise OwnerNotFound()

    def teacher_for(self, conversation_id: str) -> str | None:
        with self._lock:
            conn = self._connection()
            row = conn.execute(
                "SELECT teacher_id FROM owners WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return str(row["teacher_id"])

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise OwnerStoreError(_STORE_FAILURE)
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._connection()
        conn.executescript(_SCHEMA)
        row = conn.execute("SELECT version FROM schema_meta WHERE id = 1").fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO schema_meta (id, version) VALUES (1, ?)",
                (SCHEMA_VERSION,),
            )
            conn.commit()
            return
        if int(row["version"]) != SCHEMA_VERSION:
            raise OwnerStoreError(_STORE_FAILURE)


def resolve_ask_conversation(
    request: object,
    settings: Settings,
    supplied: str | None,
) -> str | None:
    """Return the conversation id to pass into ``CurriculumQAAgent.ask``.

    ``None`` means the anonymous path should keep the caller's id, including
    when the caller omitted one. A string is an id already granted to the
    verified teacher.
    """
    if not ownership_required(settings):
        return supplied
    teacher_id = _authenticated_teacher_id(request, settings)
    try:
        store = ConversationOwnerStore(settings.qa_conversation_owner_sqlite_path)
    except OwnerStoreError:
        raise HTTPException(status_code=503, detail=_STORE_FAILURE) from None
    try:
        if supplied is None:
            return store.bind_new(teacher_id)
        store.require(supplied, teacher_id)
        return supplied
    except OwnerNotFound:
        raise HTTPException(status_code=404, detail=_NOT_FOUND) from None
    except OwnerStoreError:
        raise HTTPException(status_code=503, detail=_STORE_FAILURE) from None
    finally:
        store.close()


def _authenticated_teacher_id(request: object, settings: Settings) -> str:
    if current_identity_gate() is None:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE)
    try:
        principal = authenticate_http_request(request, settings=settings)
    except UnauthenticatedTeacherError:
        raise HTTPException(status_code=401, detail=_UNAUTHENTICATED) from None
    return principal.teacher_id


def _validate_teacher(teacher_id: str) -> None:
    if not isinstance(teacher_id, str) or not teacher_id.strip() or len(teacher_id) > 128:
        raise OwnerStoreError(_STORE_FAILURE)


def _is_uuid(value: str) -> bool:
    from uuid import UUID

    if not isinstance(value, str):
        return False
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return str(parsed) == value.lower()
