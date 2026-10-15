"""Teacher principal boundary for assistant threads.

This service does not authenticate teachers. Outbound Authorization headers
authenticate this process to an LLM or the curriculum API.

The only production integration point is ``register_identity_gate``. A gate
must verify a credential and return ``VerifiedTeacherClaims``. An issuer name,
a request body, or a header is not a gate and cannot mint a ``TrustedTeacher``.

When no gate is registered, HTTP authentication fails closed. Local stub
identity is a separate, explicit development switch and is rejected when
``environment`` is production. It does not read a teacher id from the request.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Protocol

_lock = threading.Lock()
_gate: Optional["IdentityGate"] = None


class UnauthenticatedTeacherError(Exception):
    """No verified teacher principal is available."""

    def __init__(self, message: str = "Authenticated teacher principal is required") -> None:
        super().__init__(message)


@dataclass(frozen=True)
class VerifiedTeacherClaims:
    """Claims returned by a registered gate after it has verified a credential."""

    teacher_id: str
    school_id: Optional[str] = None


class IdentityGate(Protocol):
    """Upstream verifier. This repository does not ship an implementation."""

    def verify(self, request: object) -> VerifiedTeacherClaims | None:
        """Return verified claims, or None to reject the request."""


class TrustedTeacher:
    """A teacher id minted only inside this module from verified claims."""

    __slots__ = ("teacher_id", "school_id", "_verified")

    def __init__(self) -> None:
        raise UnauthenticatedTeacherError(
            "Teacher identity must be issued by a verified identity source"
        )


def register_identity_gate(gate: IdentityGate) -> None:
    """Install the single verifier used by assistant HTTP authentication."""
    global _gate
    if gate is None or not callable(getattr(gate, "verify", None)):
        raise UnauthenticatedTeacherError(
            "Authenticated teacher principal is required"
        )
    with _lock:
        _gate = gate


def reset_identity_gate() -> None:
    """Remove the installed gate. Tests use this; the application does not."""
    global _gate
    with _lock:
        _gate = None


def current_identity_gate() -> IdentityGate | None:
    with _lock:
        return _gate


def bind_verified_claims(
    claims: object,
    *,
    gate: IdentityGate,
) -> TrustedTeacher:
    """Mint a principal only for the gate that is currently registered.

    A different object, including one that merely has the same class, cannot
    bind claims. This is how tests obtain a principal without a request body.
    """
    if current_identity_gate() is not gate:
        raise UnauthenticatedTeacherError(
            "Authenticated teacher principal is required"
        )
    if not isinstance(claims, VerifiedTeacherClaims):
        raise UnauthenticatedTeacherError(
            "Authenticated teacher principal is required"
        )
    return _issue(claims.teacher_id, claims.school_id)


def authenticate_http_request(request: object, *, settings: object) -> TrustedTeacher:
    """Resolve the teacher for one HTTP request.

    The request body is not inspected. A registered gate is the only
    non-stub source. Local stub identity ignores ``request`` and is refused
    in production.
    """
    if _is_production(settings) and bool(getattr(settings, "assistant_local_stub_identity", False)):
        raise UnauthenticatedTeacherError(
            "Local stub identity is disabled in production"
        )
    gate = current_identity_gate()
    if gate is not None:
        claims = gate.verify(request)
        if not isinstance(claims, VerifiedTeacherClaims):
            raise UnauthenticatedTeacherError(
                "Authenticated teacher principal is required"
            )
        return _issue(claims.teacher_id, claims.school_id)
    if _local_stub_allowed(settings):
        teacher_id = str(getattr(settings, "assistant_local_stub_teacher_id", "") or "").strip()
        if not teacher_id:
            raise UnauthenticatedTeacherError(
                "Authenticated teacher principal is required"
            )
        return _issue(teacher_id, None)
    raise UnauthenticatedTeacherError(
        "Authenticated teacher principal is required"
    )


def require_trusted_teacher(value: object) -> TrustedTeacher:
    """Reject raw strings and unissued objects before any thread lookup."""
    if (
        isinstance(value, TrustedTeacher)
        and getattr(value, "_verified", False) is True
        and isinstance(getattr(value, "teacher_id", None), str)
        and value.teacher_id
    ):
        return value
    raise UnauthenticatedTeacherError(
        "Authenticated teacher principal is required"
    )


def _issue(teacher_id: str, school_id: str | None) -> TrustedTeacher:
    cleaned = (teacher_id or "").strip()
    if not cleaned or len(cleaned) > 128:
        raise UnauthenticatedTeacherError("Teacher identity is missing")
    school = (school_id or "").strip() or None
    if school is not None and len(school) > 128:
        raise UnauthenticatedTeacherError("School identity is invalid")
    principal = object.__new__(TrustedTeacher)
    principal.teacher_id = cleaned
    principal.school_id = school
    principal._verified = True
    return principal


def _is_production(settings: object) -> bool:
    return str(getattr(settings, "environment", "") or "").strip().lower() == "production"


def _local_stub_allowed(settings: object) -> bool:
    return bool(getattr(settings, "assistant_local_stub_identity", False)) and not _is_production(
        settings
    )
