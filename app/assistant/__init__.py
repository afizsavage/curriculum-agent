"""Early-release teacher assistant routing and response composition."""

from app.assistant.contract import (
    AssistantRoute,
    AssistantTurnRequest,
    AssistantTurnResponse,
    InvalidAssistantRoute,
)
from app.assistant.dispatch import dispatch_turn
from app.assistant.router import parse_closed_route, resolve_route

__all__ = [
    "AssistantRoute",
    "AssistantTurnRequest",
    "AssistantTurnResponse",
    "InvalidAssistantRoute",
    "dispatch_turn",
    "parse_closed_route",
    "resolve_route",
]
