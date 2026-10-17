"""Provisioned teacher accounts and server-side sessions.

The database is a dedicated SQLite file. It is not the LangGraph checkpoint
file, the assistant thread file, or the curriculum Postgres database.

Passwords are stored as ``scrypt`` hashes with a random salt. Session cookies
hold an opaque token; only the SHA-256 hash of that token is stored. This
module does not log passwords, tokens, or cookies.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
from base64 import urlsafe_b64decode, urlsafe_b64encode
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from app.logging_utils import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1
SESSION_TTL = timedelta(days=14)
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
_SCRYPT_MAXMEM = 32 * 1024 * 1024
_RESERVED_DATABASE_NAMES = frozenset(
    {"checkpoints.sqlite", "assistant_threads.sqlite"}
)
_USERNAME = re.compile(r"^[A-Za-z0-9._@+-]{1,128}$")
_STORE_FAILURE = "Authentication store is unavailable"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    school_id TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(account_id),
    token_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked_at TEXT
);
CREATE INDEX IF NOT EXISTS sessions_account ON sessions (account_id);
"""

_dummy_lock = threading.Lock()
_dummy_password_hash: str | None = None


class AuthStoreError(Exception):
    """The authentication store failed. The message contains no credentials."""

    def __init__(self, message: str = _STORE_FAILURE) -> None:
        super().__init__(message)


class AccountExists(AuthStoreError):
    """The username is already provisioned."""

    def __init__(self) -> None:
        super().__init__("An account with that username already exists")


@dataclass(frozen=True)
class Account:
    """A provisioned teacher. ``account_id`` is the assistant teacher id."""

    account_id: str
    username: str
    school_id: str | None


def resolve_auth_database_path(path: str | Path) -> Path:
    """Resolve a configured path from the curriculum-agent root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        root = Path(__file__).resolve().parents[2]
        candidate = root / candidate
    return candidate


def canonical_username(value: object) -> str | None:
    """Return the stored username, or None when the value cannot be an account."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if _USERNAME.fullmatch(stripped) is None:
        return None
    return stripped.casefold()


def canonical_school_id(value: object) -> str | None:
    """Return a school id the identity model can carry, or None when absent."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise AuthStoreError("School identity is invalid")
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > 128 or any(ord(char) < 32 for char in cleaned):
        raise AuthStoreError("School identity is invalid")
    return cleaned


def hash_password(password: str) -> str:
    """Hash a password. The result does not contain the password."""
    salt = secrets.token_bytes(16)
    digest = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    return _encode_hash(salt, digest, SCRYPT_N, SCRYPT_R, SCRYPT_P)


def verify_password(password: str, stored: str) -> bool:
    """Return whether ``password`` matches a stored scrypt hash."""
    try:
        n, r, p, salt, expected = _decode_hash(stored)
        digest = _scrypt(password, salt, n, r, p)
    except (AuthStoreError, ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, expected)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthStore:
    """SQLite account and session store. One connection, locked per transaction."""

    def __init__(self, path: str | Path, *, forbidden_path: Path | None = None) -> None:
        self.path = resolve_auth_database_path(path)
        reserved = self.path.name in _RESERVED_DATABASE_NAMES or self.path.suffix != ".sqlite"
        collides = (
            forbidden_path is not None
            and self.path.resolve() == Path(forbidden_path).resolve()
        )
        if reserved or collides:
            raise AuthStoreError(
                "Authentication store cannot use the checkpoint or thread database"
            )
        if self.path.exists() and self.path.is_dir():
            raise AuthStoreError(_STORE_FAILURE)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        try:
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA busy_timeout = 5000")
            self._conn.execute("PRAGMA journal_mode = WAL").fetchone()
            self._ensure_schema()
        except AuthStoreError:
            self.close()
            raise
        except sqlite3.Error:
            self.close()
            raise AuthStoreError(_STORE_FAILURE) from None

    def close(self) -> None:
        with self._lock:
            conn = self._conn
            self._conn = None
            if conn is not None:
                conn.close()

    def provision_account(
        self,
        username: str,
        password: str,
        school_id: str | None = None,
    ) -> Account:
        canonical = canonical_username(username)
        if canonical is None:
            raise AuthStoreError("Username is invalid")
        if not isinstance(password, str) or not 12 <= len(password) <= 128:
            raise AuthStoreError("Password must be 12 to 128 characters")
        school = canonical_school_id(school_id)
        now = _now()
        account_id = str(uuid4())
        password_hash = hash_password(password)
        with self._transaction() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO accounts (
                        account_id, username, password_hash, school_id, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (account_id, canonical, password_hash, school, now),
                )
            except sqlite3.IntegrityError:
                raise AccountExists() from None
        return Account(account_id=account_id, username=canonical, school_id=school)

    def authenticate(self, username: object, password: object) -> Account | None:
        """Return the account, or None for an unknown user or wrong password.

        Both failures take a scrypt verification. The reason is not returned.
        """
        canonical = canonical_username(username)
        if not isinstance(password, str) or not 1 <= len(password) <= 128:
            verify_password("rejected-password", _dummy_hash())
            return None
        stored = None
        account_id = None
        stored_username = None
        school_raw = None
        with self._transaction() as conn:
            row = None
            if canonical is not None:
                row = conn.execute(
                    """
                    SELECT account_id, username, password_hash, school_id
                    FROM accounts
                    WHERE username = ?
                    """,
                    (canonical,),
                ).fetchone()
            if row is not None:
                stored = str(row["password_hash"])
                account_id = str(row["account_id"])
                stored_username = str(row["username"])
                school_raw = row["school_id"]
        if stored is None:
            stored = _dummy_hash()
        if not verify_password(password, stored) or account_id is None or stored_username is None:
            return None
        try:
            school = canonical_school_id(school_raw)
        except AuthStoreError:
            return None
        return Account(
            account_id=account_id,
            username=stored_username,
            school_id=school,
        )

    def create_session(self, account_id: str) -> str:
        """Create a session and return the raw token. The token is not stored."""
        token = secrets.token_urlsafe(32)
        now = _now()
        expires = datetime.now(timezone.utc) + SESSION_TTL
        try:
            with self._transaction() as conn:
                owned = conn.execute(
                    "SELECT account_id FROM accounts WHERE account_id = ?",
                    (account_id,),
                ).fetchone()
                if owned is None:
                    raise AuthStoreError(_STORE_FAILURE)
                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id, account_id, token_hash, created_at, expires_at, revoked_at
                    ) VALUES (?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        str(uuid4()),
                        account_id,
                        hash_session_token(token),
                        now,
                        expires.isoformat(),
                    ),
                )
        except sqlite3.IntegrityError:
            raise AuthStoreError(_STORE_FAILURE) from None
        return token

    def resolve_token(self, token: object) -> Account | None:
        """Return the account for a live session token, otherwise None."""
        if not isinstance(token, str) or not token or len(token) > 512:
            return None
        digest = hash_session_token(token)
        found: dict[str, object] | None = None
        with self._transaction() as conn:
            row = conn.execute(
                """
                SELECT a.account_id, a.username, a.school_id, s.expires_at, s.revoked_at
                FROM sessions s
                JOIN accounts a ON a.account_id = s.account_id
                WHERE s.token_hash = ?
                """,
                (digest,),
            ).fetchone()
            if row is not None:
                found = {
                    "account_id": row["account_id"],
                    "username": row["username"],
                    "school_id": row["school_id"],
                    "expires_at": row["expires_at"],
                    "revoked_at": row["revoked_at"],
                }
        if found is None or found["revoked_at"] is not None:
            return None
        try:
            expires = _parse_time(str(found["expires_at"]))
        except ValueError:
            return None
        if expires <= datetime.now(timezone.utc):
            return None
        try:
            school = canonical_school_id(found["school_id"])
        except AuthStoreError:
            return None
        account_id = found["account_id"]
        if not isinstance(account_id, str) or not account_id or len(account_id) > 128:
            return None
        username = found["username"]
        if not isinstance(username, str):
            return None
        return Account(account_id=account_id, username=username, school_id=school)

    def revoke_token(self, token: object) -> None:
        """Revoke a session if the token matches one. Unknown tokens do nothing."""
        if not isinstance(token, str) or not token or len(token) > 512:
            return
        now = _now()
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET revoked_at = ?
                WHERE token_hash = ? AND revoked_at IS NULL
                """,
                (now, hash_session_token(token)),
            )

    def _ensure_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO schema_meta (id, version) VALUES (1, ?)",
            (SCHEMA_VERSION,),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT version FROM schema_meta WHERE id = 1"
        ).fetchone()
        version = None if row is None else int(row["version"])
        if version != SCHEMA_VERSION:
            raise AuthStoreError(_STORE_FAILURE)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._conn
            if conn is None:
                raise AuthStoreError(_STORE_FAILURE)
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except AuthStoreError:
                _rollback(conn)
                raise
            except sqlite3.IntegrityError:
                _rollback(conn)
                raise
            except sqlite3.Error:
                _rollback(conn)
                logger.warning("assistant.auth.store_failure")
                raise AuthStoreError(_STORE_FAILURE) from None


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    if not isinstance(password, str):
        raise AuthStoreError("Password must be 12 to 128 characters")
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=SCRYPT_DKLEN,
        maxmem=_SCRYPT_MAXMEM,
    )


def _encode_hash(salt: bytes, digest: bytes, n: int, r: int, p: int) -> str:
    return "$".join(
        (
            "scrypt",
            str(n),
            str(r),
            str(p),
            urlsafe_b64encode(salt).decode("ascii"),
            urlsafe_b64encode(digest).decode("ascii"),
        )
    )


def _decode_hash(stored: str) -> tuple[int, int, int, bytes, bytes]:
    parts = stored.split("$")
    if len(parts) != 6 or parts[0] != "scrypt":
        raise AuthStoreError(_STORE_FAILURE)
    n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
    if n != SCRYPT_N or r != SCRYPT_R or p != SCRYPT_P:
        raise AuthStoreError(_STORE_FAILURE)
    salt = urlsafe_b64decode(parts[4].encode("ascii"))
    digest = urlsafe_b64decode(parts[5].encode("ascii"))
    if len(salt) < 16 or len(digest) != SCRYPT_DKLEN:
        raise AuthStoreError(_STORE_FAILURE)
    return n, r, p, salt, digest


def _dummy_hash() -> str:
    """A hash used so an unknown username still costs one scrypt verification."""
    global _dummy_password_hash
    with _dummy_lock:
        if _dummy_password_hash is None:
            _dummy_password_hash = hash_password(secrets.token_urlsafe(24))
        return _dummy_password_hash


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _rollback(conn: sqlite3.Connection) -> None:
    try:
        conn.rollback()
    except sqlite3.Error:
        return
