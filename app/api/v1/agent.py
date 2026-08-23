"""POST /api/v1/agent/ask"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.agent.orchestrator import CurriculumQAAgent
from app.deps import get_agent
from app.logging_utils import get_logger, new_request_id, timed_request
from app.schemas.agent import AskMetadata, AskRequest, AskResponse

router = APIRouter(prefix="/agent", tags=["agent"])
logger = get_logger(__name__)


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask the Curriculum Q&A agent",
    description=(
        "Accept a curriculum question and return an agent turn. "
        "Sprint 1 acknowledges the question with status `received` and "
        "`answer: null`. Retrieve / reason / verify arrive in later sprints."
    ),
    responses={
        422: {"description": "Invalid request (validation error)"},
        500: {"description": "Agent execution or configuration failure"},
        502: {"description": "LLM provider or tool failure"},
        504: {"description": "LLM timeout"},
    },
)
def ask(
    payload: AskRequest,
    request: Request,
    agent: CurriculumQAAgent = Depends(get_agent),
) -> AskResponse:
    request_id = getattr(request.state, "request_id", None) or new_request_id()
    conversation_id = (
        str(payload.conversation_id) if payload.conversation_id is not None else None
    )

    with timed_request(
        logger,
        request_id=request_id,
        conversation_id=conversation_id,
        question=payload.question,
        model=agent.llm.model,
    ) as ctx:
        state = agent.ask(
            payload.question,
            conversation_id=conversation_id,
            request_id=request_id,
        )
        ctx["conversation_id"] = state.conversation_id
        ctx["status"] = state.status.value
        ctx["iteration"] = state.iteration
        ctx["tool_calls"] = state.tool_calls

        return AskResponse(
            conversation_id=state.conversation_id or "",
            question=state.question,
            answer=state.draft_answer,
            status=state.status.value,
            metadata=AskMetadata(
                iterations=state.iteration,
                tool_calls=state.tool_calls,
                model=agent.llm.model,
                provider=agent.llm.name,
            ),
            error=state.error,
        )
