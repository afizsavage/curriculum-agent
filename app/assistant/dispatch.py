"""Apply a resolved route.

Handlers are injected. Pass qa_agent to call CurriculumQAAgent.ask in process.
Clarify never calls QA or general generation. A mixed turn calls general
generation only after QA returns an outcome that may carry separate advice.
"""

from __future__ import annotations

from collections.abc import Callable

from app.assistant.composer import blocks_general_advice, compose_turn
from app.assistant.contract import (
    AssistantRoute,
    AssistantTurnRequest,
    AssistantTurnResponse,
    CurriculumExecutionFailure,
    CurriculumOutcome,
    GeneralExecutionFailure,
)
from app.assistant.qa_adapter import CurriculumAskAgent, CurriculumQAAdapter
from app.assistant.router import CLARIFICATION_PROMPT, RouteClassifier, resolve_route
from app.curriculum.errors import CurriculumTimeoutError
from app.exceptions import AgentError, LLMProviderError, LLMTimeoutError

QAHandler = Callable[[AssistantTurnRequest], CurriculumOutcome]
GeneralHandler = Callable[[AssistantTurnRequest], str]


class AssistantDispatchError(RuntimeError):
    """A route required a handler this batch did not receive."""


def dispatch_turn(
    request: AssistantTurnRequest,
    *,
    qa: QAHandler | None = None,
    qa_agent: CurriculumAskAgent | None = None,
    general: GeneralHandler | None = None,
    classifier: RouteClassifier | None = None,
) -> AssistantTurnResponse:
    """Resolve a route, call only the handlers that route allows, then compose."""
    if qa is not None and qa_agent is not None:
        raise AssistantDispatchError("pass qa or qa_agent, not both")
    if qa is None and qa_agent is not None:
        qa = CurriculumQAAdapter(qa_agent)
    routing = resolve_route(request, classifier=classifier)
    route = routing.route

    if route == AssistantRoute.CLARIFY:
        return compose_turn(
            routing=routing,
            clarification=CLARIFICATION_PROMPT,
        )

    if route == AssistantRoute.GENERAL:
        if general is None:
            raise AssistantDispatchError("general route requires a general handler")
        text, failure = _invoke_general(general, request)
        return compose_turn(
            routing=routing,
            general_text=text,
            general_failure=failure,
        )

    if qa is None:
        raise AssistantDispatchError(f"{route.value} route requires a QA handler")
    outcome = _invoke_qa(qa, request)

    if route == AssistantRoute.CURRICULUM:
        return compose_turn(routing=routing, qa=outcome)

    if blocks_general_advice(outcome) is not None:
        return compose_turn(routing=routing, qa=outcome)
    if general is None:
        raise AssistantDispatchError("mixed route requires a general handler")
    text, failure = _invoke_general(general, request)
    return compose_turn(
        routing=routing,
        qa=outcome,
        general_text=text,
        general_failure=failure,
    )


def _invoke_qa(handler: QAHandler, request: AssistantTurnRequest) -> CurriculumOutcome:
    try:
        return handler(request)
    except AgentError as exc:
        return CurriculumOutcome(failure=_failure_from_agent_error(exc))


def _invoke_general(
    handler: GeneralHandler,
    request: AssistantTurnRequest,
) -> tuple[str | None, GeneralExecutionFailure | None]:
    try:
        return handler(request), None
    except AgentError as exc:
        return None, GeneralExecutionFailure(
            code=exc.code,
            message=exc.message,
            http_status=exc.status_code,
        )


def _failure_from_agent_error(exc: AgentError) -> CurriculumExecutionFailure:
    if isinstance(exc, (LLMTimeoutError, CurriculumTimeoutError)):
        kind = "timeout"
    elif isinstance(exc, LLMProviderError):
        kind = "provider_error"
    else:
        kind = "execution_failure"
    return CurriculumExecutionFailure(
        kind=kind,
        code=exc.code,
        message=exc.message,
        http_status=exc.status_code,
    )
