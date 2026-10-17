"""Identity gate that verifies the Marcus session cookie.

The gate reads only the configured cookie. A teacher id in the body, a query
parameter, or a header is ignored. Claims use the server-side account id.
"""

from __future__ import annotations

from app.assistant.auth_store import Account, AuthStore, AuthStoreError
from app.assistant.identity import VerifiedTeacherClaims
from app.logging_utils import get_logger

logger = get_logger(__name__)


class SessionIdentityGate:
    """``IdentityGate`` backed by the authentication store."""

    def __init__(
        self,
        store: AuthStore,
        *,
        cookie_name: str,
        secure: bool,
        allowed_origins: frozenset[str],
    ) -> None:
        self.store = store
        self.cookie_name = cookie_name
        self.secure = secure
        self.allowed_origins = allowed_origins

    def verify(self, request: object) -> VerifiedTeacherClaims | None:
        token = _cookie(request, self.cookie_name)
        if token is None:
            return None
        try:
            account = self.store.resolve_token(token)
        except AuthStoreError:
            logger.warning("assistant.auth.store_unavailable")
            return None
        if account is None:
            return None
        return _claims(account)

    def close(self) -> None:
        self.store.close()


def _cookie(request: object, name: str) -> str | None:
    cookies = getattr(request, "cookies", None)
    if cookies is None:
        return None
    try:
        raw = cookies.get(name)
    except AttributeError:
        return None
    if not isinstance(raw, str) or not raw:
        return None
    return raw


def _claims(account: Account) -> VerifiedTeacherClaims:
    return VerifiedTeacherClaims(
        teacher_id=account.account_id,
        school_id=account.school_id,
    )
