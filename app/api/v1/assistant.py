"""POST /api/v1/assistant/chat

Disabled unless ASSISTANT_ENABLED is true. The disabled branch returns 404
before identity resolution, the thread store, the router, the model provider,
and the QA agent.

Enabling the route is not authentication. A principal is accepted only from a
gate installed with register_identity_gate, whose verify method returns
VerifiedTeacherClaims after checking a real credential. This service does not
read a teacher id from the body, a header, or a query parameter.

Outside production, ASSISTANT_LOCAL_STUB_IDENTITY=true uses
ASSISTANT_LOCAL_STUB_TEACHER_ID and ignores the request. That switch is
refused when ENVIRONMENT is production. This repository has no production
teacher-identity issuer.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.assistant.chat import execute_chat
from app.assistant.identity import UnauthenticatedTeacherError
from app.assistant.threads import ThreadStoreError, ThreadUnavailable
from app.config import get_settings
from app.schemas.assistant import AssistantChatRequest, AssistantChatResponse

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
