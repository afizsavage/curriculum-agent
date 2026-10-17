"""Login, logout, and session checks for provisioned teacher accounts.

These routes do not read a teacher id from the body, query string, or headers.
Login issues an HttpOnly session cookie. Logout revokes that session. The
status check reports only whether the identity gate accepts the cookie, and
its response has no body. A missing or invalid session does not create an
identity.

Cross-origin state changes are rejected when the browser sends Origin or
Sec-Fetch-Site. Logout is POST-only so a cross-site navigation cannot revoke
a SameSite=Lax cookie.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from app.assistant.auth_store import SESSION_TTL, AuthStoreError
from app.assistant.identity import VerifiedTeacherClaims, current_identity_gate
from app.assistant.session_gate import SessionIdentityGate
from app.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/session", tags=["session"])

_LOGIN_FAILURE = "Invalid username or password"
_UNAVAILABLE = "Authentication is unavailable"
_CROSS_ORIGIN = "Cross-origin request rejected"
_MAX_LOGIN_BYTES = 4096


@router.post("/login", status_code=204)
async def login(request: Request) -> Response:
    gate = _require_gate()
    _reject_cross_origin(request, gate.allowed_origins)
    username, password = await _credentials(request)
    try:
        account = await run_in_threadpool(gate.store.authenticate, username, password)
        if account is None:
            logger.info("assistant.auth.login_rejected")
            raise HTTPException(status_code=401, detail=_LOGIN_FAILURE)
        token = await run_in_threadpool(gate.store.create_session, account.account_id)
    except HTTPException:
        raise
    except AuthStoreError:
        logger.warning("assistant.auth.store_unavailable")
        raise HTTPException(status_code=503, detail=_UNAVAILABLE) from None
    response = Response(status_code=204)
    _remember(response, gate, token)
    logger.info("assistant.auth.login_accepted")
    return response


@router.get("", status_code=204)
def session_status(request: Request) -> Response:
    """Report whether the session cookie is currently accepted by the identity gate.

    The response has no body. A teacher id in the query string or a header is ignored.
    """
    gate = _require_gate()
    _reject_cross_origin(request, gate.allowed_origins)
    if not isinstance(gate.verify(request), VerifiedTeacherClaims):
        raise HTTPException(
            status_code=401,
            detail="Authenticated teacher principal is required",
        )
    response = Response(status_code=204)
    response.headers["Cache-Control"] = "no-store"
    return response


@router.post("/logout", status_code=204)
def logout(request: Request) -> Response:
    gate = _require_gate()
    _reject_cross_origin(request, gate.allowed_origins)
    raw = request.cookies.get(gate.cookie_name)
    response = Response(status_code=204)
    _clear(response, gate)
    if not isinstance(raw, str) or not raw:
        return response
    try:
        gate.store.revoke_token(raw)
    except AuthStoreError:
        logger.warning("assistant.auth.store_unavailable")
        failed = JSONResponse(status_code=503, content={"detail": _UNAVAILABLE})
        _clear(failed, gate)
        return failed
    return response


def _require_gate() -> SessionIdentityGate:
    gate = current_identity_gate()
    if isinstance(gate, SessionIdentityGate):
        return gate
    from app.config import get_settings

    if get_settings().assistant_auth_enabled:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE)
    raise HTTPException(status_code=404, detail="Not found")


def _reject_cross_origin(request: Request, allowed: frozenset[str]) -> None:
    origin = request.headers.get("origin")
    if origin is not None and origin.strip().rstrip("/") not in allowed:
        raise HTTPException(status_code=403, detail=_CROSS_ORIGIN)
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site.strip().lower() == "cross-site":
        raise HTTPException(status_code=403, detail=_CROSS_ORIGIN)


async def _credentials(request: Request) -> tuple[str, str]:
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > _MAX_LOGIN_BYTES:
        raise HTTPException(status_code=401, detail=_LOGIN_FAILURE)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=401, detail=_LOGIN_FAILURE) from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=401, detail=_LOGIN_FAILURE)
    username = payload.get("username")
    password = payload.get("password")
    if not isinstance(username, str) or not isinstance(password, str):
        raise HTTPException(status_code=401, detail=_LOGIN_FAILURE)
    if len(password) > 128:
        raise HTTPException(status_code=401, detail=_LOGIN_FAILURE)
    return username, password


def _remember(response: Response, gate: SessionIdentityGate, token: str) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.set_cookie(
        key=gate.cookie_name,
        value=token,
        max_age=int(SESSION_TTL.total_seconds()),
        path="/",
        secure=gate.secure,
        httponly=True,
        samesite="lax",
    )


def _clear(response: Response, gate: SessionIdentityGate) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.delete_cookie(
        key=gate.cookie_name,
        path="/",
        secure=gate.secure,
        httponly=True,
        samesite="lax",
    )
