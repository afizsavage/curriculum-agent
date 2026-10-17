"""Assistant HTTP routes.

Disabled unless ASSISTANT_ENABLED is true. Every disabled branch returns 404
before identity resolution, the thread store, the router, the model provider,
and the QA agent.

Enabling a route is not authentication. A principal is accepted only from a
gate installed with register_identity_gate, whose verify method returns
VerifiedTeacherClaims after checking a real credential. These routes do not
read a teacher id from the body, a header, or a query parameter.

Outside production, ASSISTANT_LOCAL_STUB_IDENTITY=true uses
ASSISTANT_LOCAL_STUB_TEACHER_ID and ignores the request. That switch is
refused when ENVIRONMENT is production. When ASSISTANT_AUTH_ENABLED is true,
startup registers the session identity gate and does not fall back to the stub.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.assistant.chat import execute_chat, thread_store_path
from app.assistant.identity import UnauthenticatedTeacherError, authenticate_http_request
from app.assistant.threads import (
    AssistantThreadStore,
    ThreadRecord,
    ThreadStoreError,
    ThreadUnavailable,
    thread_title,
)
from app.config import Settings, get_settings
from app.schemas.assistant import (
    AssistantChatRequest,
    AssistantChatResponse,
    AssistantThreadDetail,
    AssistantThreadListResponse,
    AssistantThreadMessage,
    AssistantThreadSummary,
)

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post(
    "/chat",
    response_model=AssistantChatResponse,
    responses={
        404: {"description": "Assistant disabled, or thread not found"},
        401: {"description": "No verified teacher principal"},
        503: {"description": "Thread store unavailable"},
    },
)
def chat(payload: AssistantChatRequest, request: Request) -> AssistantChatResponse:
    settings = get_settings()
    if not settings.assistant_enabled:
        raise HTTPException(status_code=404, detail="Not found")
    try:
        return execute_chat(payload, request, settings=settings)
    except UnauthenticatedTeacherError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from None
    except ThreadUnavailable:
        raise HTTPException(status_code=404, detail="Thread not found") from None
    except ThreadStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.get(
    "/threads",
    response_model=AssistantThreadListResponse,
    responses={
        404: {"description": "Assistant disabled"},
        401: {"description": "No verified teacher principal"},
        503: {"description": "Thread store unavailable"},
    },
)
def list_threads(request: Request) -> AssistantThreadListResponse:
    settings = get_settings()
    _require_enabled(settings)
    try:
        principal = authenticate_http_request(request, settings=settings)
        store = AssistantThreadStore(thread_store_path(settings))
        try:
            previews = store.list_previews(principal)
        finally:
            store.close()
    except UnauthenticatedTeacherError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from None
    except ThreadStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    return AssistantThreadListResponse(
        threads=[
            AssistantThreadSummary(
                thread_id=preview.thread_id,
                title=thread_title(preview.first_user_message),
                created_at=preview.created_at,
                updated_at=preview.updated_at,
                grade=preview.grade,
                subject=preview.subject,
                task=preview.task,
            )
            for preview in previews
        ]
    )


@router.get(
    "/threads/{thread_id}",
    response_model=AssistantThreadDetail,
    responses={
        404: {"description": "Assistant disabled, or thread not found"},
        401: {"description": "No verified teacher principal"},
        503: {"description": "Thread store unavailable"},
    },
)
def get_thread(thread_id: str, request: Request) -> AssistantThreadDetail:
    settings = get_settings()
    _require_enabled(settings)
    try:
        principal = authenticate_http_request(request, settings=settings)
        store = AssistantThreadStore(thread_store_path(settings))
        try:
            record = store.get_thread(principal, thread_id)
        finally:
            store.close()
    except UnauthenticatedTeacherError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from None
    except ThreadUnavailable:
        raise HTTPException(status_code=404, detail="Thread not found") from None
    except ThreadStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    return _detail(record)


def _require_enabled(settings: Settings) -> None:
    if not settings.assistant_enabled:
        raise HTTPException(status_code=404, detail="Not found")


def _detail(record: ThreadRecord) -> AssistantThreadDetail:
    first_user = next(
        (message.content for message in record.messages if message.role == "user"),
        None,
    )
    return AssistantThreadDetail(
        thread_id=record.thread_id,
        title=thread_title(first_user),
        created_at=record.created_at,
        updated_at=record.updated_at,
        grade=record.grade,
        subject=record.subject,
        task=record.task,
        messages=[
            AssistantThreadMessage(
                message_id=message.message_id,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
            )
            for message in record.messages
        ],
    )
