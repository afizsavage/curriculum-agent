"""Register the session identity gate when authentication is enabled.

Authentication stays off unless ``ASSISTANT_AUTH_ENABLED`` is true. Enabling it
does not enable ``ASSISTANT_ENABLED`` and does not install the local stub.
Startup fails when authentication is enabled but the gate cannot be registered,
so a misconfigured process does not continue with stub identity.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from app.assistant.auth_store import (
    AuthStore,
    AuthStoreError,
    resolve_auth_database_path,
)
from app.assistant.identity import current_identity_gate, register_identity_gate
from app.assistant.session_gate import SessionIdentityGate
from app.config import Settings

_COOKIE_NAME = re.compile(r"^[A-Za-z0-9_]{1,64}$")


class AuthConfigurationError(RuntimeError):
    """Authentication is enabled, but it cannot be installed safely."""


def install_configured_identity_gate(settings: Settings) -> None:
    """Install the session gate, or do nothing when authentication is off."""
    if not settings.assistant_auth_enabled:
        return
    if settings.assistant_local_stub_identity:
        raise AuthConfigurationError(
            "Local stub identity cannot be combined with authentication"
        )
    path = resolve_auth_database_path(
        settings.assistant_auth_sqlite_path or "data/assistant_auth.sqlite"
    )
    thread_path = resolve_auth_database_path(
        settings.assistant_thread_sqlite_path or "data/assistant_threads.sqlite"
    )
    if path.resolve() == thread_path.resolve():
        raise AuthConfigurationError(
            "Authentication is enabled but the identity gate could not be registered"
        )
    cookie_name = (settings.assistant_session_cookie_name or "").strip()
    if _COOKIE_NAME.fullmatch(cookie_name) is None:
        raise AuthConfigurationError(
            "Authentication is enabled but the identity gate could not be registered"
        )
    try:
        origins = parse_allowed_origins(settings.assistant_auth_allowed_origins)
        store = AuthStore(path, forbidden_path=thread_path)
    except (AuthStoreError, AuthConfigurationError) as exc:
        raise AuthConfigurationError(
            "Authentication is enabled but the identity gate could not be registered"
        ) from exc
    gate = SessionIdentityGate(
        store,
        cookie_name=cookie_name,
        secure=_is_production(settings),
        allowed_origins=origins,
    )
    try:
        previous = current_identity_gate()
        register_identity_gate(gate)
    except Exception:
        store.close()
        raise AuthConfigurationError(
            "Authentication is enabled but the identity gate could not be registered"
        ) from None
    if isinstance(previous, SessionIdentityGate) and previous is not gate:
        previous.close()


def parse_allowed_origins(raw: str) -> frozenset[str]:
    """Parse a comma-separated origin list. An empty list is refused."""
    origins: list[str] = []
    for item in (raw or "").split(","):
        candidate = item.strip().rstrip("/")
        if not candidate:
            continue
        parsed = urlparse(candidate)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or candidate == "null"
        ):
            raise AuthConfigurationError(
                "Authentication is enabled but the identity gate could not be registered"
            )
        origins.append(f"{parsed.scheme}://{parsed.netloc}")
    if not origins:
        raise AuthConfigurationError(
            "Authentication is enabled but the identity gate could not be registered"
        )
    return frozenset(origins)


def _is_production(settings: Settings) -> bool:
    return settings.environment.strip().lower() == "production"
