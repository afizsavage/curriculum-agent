"""Assistant chat turn. The HTTP route calls this only after the feature flag.

Order, which is also the partial-failure contract:

1. The route returns 404 when ``ASSISTANT_ENABLED`` is false and does not call
   this function.
2. Identity is resolved before the thread store is opened. Failure raises
   ``UnauthenticatedTeacherError`` and does not create a file.
3. An existing thread is loaded with an ownership check before any model or
   QA call. A missing thread and another teacher's thread both raise
   ``ThreadUnavailable``.
4. A new thread is created only for the authenticated principal.
5. The user message is stored before dispatch. If that write fails, dispatch
   does not run.
6. Dispatch may call QA or general generation. The assistant thread id is not
   passed as ``CurriculumQAAgent`` ``conversation_id``.
7. Assistant messages and the QA status summary are stored after dispatch. If
   those writes fail, the reply is still returned with ``persistence_warning``
   and the model is not called again. A user message from step 5 may remain.

The QA outcome row stores status and provenance only, not answer text or
provider payloads.
"""

from __future__ import annotations

from pathlib import Path

from app.assistant.contract import (
    AssistantRoute,
    AssistantTurnRequest,
    AssistantTurnResponse,
)
from app.assistant.dispatch import dispatch_turn
from app.assistant.follow_up import resolve_context_follow_up
from app.assistant.identity import TrustedTeacher, authenticate_http_request
from app.assistant.qa_adapter import CurriculumAskAgent
from app.assistant.router import heuristic_route
from app.assistant.threads import (
    AssistantThreadStore,
    QaOutcomeSummary,
    ThreadStoreError,
)
from app.config import Settings
from app.deps import get_agent
from app.schemas.assistant import AssistantChatRequest, AssistantChatResponse

_PERSISTENCE_WARNING = "The reply was produced but not fully stored."

_qa_agent: CurriculumAskAgent | None = None
_general = None


def configure_assistant_runtime(
    *,
    qa_agent: CurriculumAskAgent | None = None,
    general=None,
) -> None:
    """Test hook. Production leaves both unset until a turn is authorized."""
    global _qa_agent, _general
    _qa_agent = qa_agent
    _general = general


def reset_assistant_runtime() -> None:
    configure_assistant_runtime()


def execute_chat(
    payload: AssistantChatRequest,
    request: object,
    *,
    settings: Settings,
) -> AssistantChatResponse:
    principal = authenticate_http_request(request, settings=settings)
    store = AssistantThreadStore(thread_store_path(settings))
    try:
        return _run_authorized_turn(payload, request, principal, store)
    finally:
        store.close()


def thread_store_path(settings: Settings) -> Path:
    path = Path(settings.assistant_thread_sqlite_path or "data/assistant_threads.sqlite")
    if path.name == "checkpoints.sqlite":
        raise ThreadStoreError(
            "Assistant thread store cannot use the LangGraph checkpoint database"
        )
    if not path.is_absolute():
        root = Path(__file__).resolve().parents[2]
        path = root / path
    return path


def _run_authorized_turn(
    payload: AssistantChatRequest,
    request: object,
    principal: TrustedTeacher,
    store: AssistantThreadStore,
) -> AssistantChatResponse:
    if payload.thread_id:
        record = store.get_thread(principal, payload.thread_id)
        thread_id = payload.thread_id
    else:
        record = None
        thread_id = store.create_thread(principal).thread_id
    resolution = resolve_context_follow_up(
        payload.question,
        [] if record is None else record.messages,
        None if record is None else record.qa_outcome,
    )
    store.append_message(principal, thread_id, "user", payload.question)
    turn = AssistantTurnRequest(
        question=resolution.dispatch_question,
        request_id=getattr(getattr(request, "state", None), "request_id", None),
        conversation_id=None,
    )
    response = dispatch_turn(turn, **_handlers_for(turn))
    warning = _persist_assistant_side(store, principal, thread_id, response)
    return _http_response(thread_id, response, warning)


def _resolve_qa_agent() -> CurriculumAskAgent:
    if _qa_agent is not None:
        return _qa_agent
    return get_agent()


def _resolve_general():
    if _general is not None:
        return _general
    return _default_general_reply


def _handlers_for(turn: AssistantTurnRequest) -> dict:
    """Build only the handlers the heuristic route can call.

    Dispatch classifies again and skips general generation when curriculum
    evidence rules suppress it. Passing the callable does not invoke it.
    """
    route, _reason = heuristic_route(turn.question)
    handlers: dict = {"qa_agent": None, "general": None}
    if route in (AssistantRoute.CURRICULUM, AssistantRoute.MIXED):
        handlers["qa_agent"] = _resolve_qa_agent()
    if route in (AssistantRoute.GENERAL, AssistantRoute.MIXED):
        handlers["general"] = _resolve_general()
    return handlers


def _default_general_reply(request: AssistantTurnRequest) -> str:
    from app.llm.base import LLMMessage
    from app.llm.provider import build_llm_provider

    provider = build_llm_provider()
    result = provider.generate(
        [
            LLMMessage(
                role="system",
                content=(
                    "You help teachers with general classroom practice. "
                    "This is general AI assistance, not an official MBSSE "
                    "curriculum answer."
                ),
            ),
            LLMMessage(role="user", content=request.question),
        ]
    )
    return (result.content or "").strip()


def _persist_assistant_side(
    store: AssistantThreadStore,
    principal: TrustedTeacher,
    thread_id: str,
    response: AssistantTurnResponse,
) -> str | None:
    try:
        for text in _assistant_texts(response):
            store.append_message(principal, thread_id, "assistant", text)
        if response.curriculum is not None:
            store.record_qa_outcome(
                principal,
                thread_id,
                _qa_summary(response),
            )
    except ThreadStoreError:
        return _PERSISTENCE_WARNING
    return None


def _assistant_texts(response: AssistantTurnResponse) -> list[str]:
    texts: list[str] = []
    if response.route == AssistantRoute.CLARIFY and response.clarification:
        texts.append(response.clarification)
        return texts
    official = response.official_curriculum_answer
    if official:
        texts.append(official)
    if response.general is not None and response.general.text:
        texts.append(f"{response.general.label}: {response.general.text}")
    if response.route != AssistantRoute.CLARIFY and response.clarification:
        texts.append(response.clarification)
    return texts


def _qa_summary(response: AssistantTurnResponse) -> QaOutcomeSummary:
    channel = response.curriculum
    if channel is None:
        return QaOutcomeSummary()
    if channel.failure is not None:
        return QaOutcomeSummary(
            status="error",
            technical_failure=True,
            failure_code=channel.failure.code,
            coverage_decision=channel.coverage_decision,
        )
    body = channel.response
    verification = body.verification if body is not None else None
    return QaOutcomeSummary(
        status=body.status if body is not None else None,
        evidence_status=body.metadata.evidence_status if body is not None else None,
        verification_passed=verification.passed if verification is not None else None,
        verification_recommendation=(
            verification.recommendation if verification is not None else None
        ),
        coverage_decision=channel.coverage_decision,
        termination_reason=body.metadata.termination_reason if body is not None else None,
        technical_failure=channel.technical_failure,
        failure_code=None,
    )


def _http_response(
    thread_id: str,
    response: AssistantTurnResponse,
    warning: str | None,
) -> AssistantChatResponse:
    channel = response.curriculum
    return AssistantChatResponse(
        thread_id=thread_id,
        route=response.route.value,
        curriculum=channel.response if channel is not None else None,
        curriculum_technical_failure=bool(channel and channel.technical_failure),
        general_text=response.general.text if response.general is not None else None,
        general_label=response.general.label if response.general is not None else None,
        clarification=response.clarification,
        persistence_warning=warning,
    )
