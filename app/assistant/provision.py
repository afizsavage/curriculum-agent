"""Provision a teacher account from an operator shell.

There is no default password and no public registration route. Create the
first account with::

    python -m app.assistant.provision --username teacher.one

The command prompts for a password twice. When stdin is not a terminal, set
``ASSISTANT_ACCOUNT_PASSWORD`` in that command's environment only. Do not
commit the password. The command does not print the password, the password
hash, or a session token.

The account is written to ``ASSISTANT_AUTH_SQLITE_PATH`` (default
``data/assistant_auth.sqlite``). Session login stays disabled until
``ASSISTANT_AUTH_ENABLED=true``. That flag does not enable
``ASSISTANT_ENABLED``.
"""

from __future__ import annotations

import argparse
import getpass
import hmac
import os
import sys

from app.assistant.auth_startup import AuthConfigurationError
from app.assistant.auth_store import (
    AccountExists,
    AuthStore,
    AuthStoreError,
    canonical_school_id,
    canonical_username,
    resolve_auth_database_path,
)
from app.config import Settings


def main(argv: list[str] | None = None) -> int:
    try:
        _run(argv)
    except (AuthConfigurationError, AuthStoreError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _run(argv: list[str] | None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.assistant.provision",
        description=(
            "Provision a teacher account. There is no default password. "
            "The password is prompted in a terminal, or read from "
            "ASSISTANT_ACCOUNT_PASSWORD when stdin is not a terminal. "
            "Nothing secret is printed."
        ),
    )
    parser.add_argument("--username", required=True)
    parser.add_argument("--school-id", default=None)
    parser.add_argument(
        "--database",
        default=None,
        help="SQLite file. Defaults to ASSISTANT_AUTH_SQLITE_PATH.",
    )
    args = parser.parse_args(argv)
    settings = Settings()
    database = args.database or settings.assistant_auth_sqlite_path
    username = canonical_username(args.username)
    if username is None:
        raise AuthConfigurationError(
            "Username must be 1 to 128 characters and use only letters, "
            "numbers, and . _ @ + -"
        )
    try:
        school_id = canonical_school_id(args.school_id)
    except AuthStoreError as exc:
        raise AuthConfigurationError(str(exc)) from None
    password = _read_password()
    if not 12 <= len(password) <= 128:
        raise AuthConfigurationError("Password must be 12 to 128 characters")
    path = resolve_auth_database_path(database)
    thread_path = resolve_auth_database_path(settings.assistant_thread_sqlite_path)
    store = AuthStore(path, forbidden_path=thread_path)
    try:
        store.provision_account(username, password, school_id)
    except AccountExists:
        raise AuthConfigurationError(
            "An account with that username already exists"
        ) from None
    finally:
        store.close()
    print(f"Provisioned teacher account '{username}'.")


def _read_password() -> str:
    supplied = os.environ.get("ASSISTANT_ACCOUNT_PASSWORD")
    if supplied is not None:
        if supplied == "":
            raise AuthConfigurationError("ASSISTANT_ACCOUNT_PASSWORD is empty")
        return supplied
    if not sys.stdin.isatty():
        raise AuthConfigurationError(
            "Set ASSISTANT_ACCOUNT_PASSWORD for a non-interactive provision, "
            "or run python -m app.assistant.provision in a terminal to be prompted. "
            "There is no default password."
        )
    first = getpass.getpass("Password: ")
    second = getpass.getpass("Confirm password: ")
    if not _same_password(first, second):
        raise AuthConfigurationError("Passwords did not match")
    return first


def _same_password(left: str, right: str) -> bool:
    left_bytes = left.encode("utf-8")
    right_bytes = right.encode("utf-8")
    if len(left_bytes) != len(right_bytes):
        return False
    return hmac.compare_digest(left_bytes, right_bytes)


if __name__ == "__main__":
    raise SystemExit(main())
