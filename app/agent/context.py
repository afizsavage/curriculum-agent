"""Conversation context for multi-turn Curriculum Q&A (in-memory for Sprint 1)."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.agent.state import CurriculumQAState
from app.enums import MessageRole


class ConversationMessage(BaseModel):
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)


class ConversationContext(BaseModel):
    """Holds history and the current agent state for one conversation."""

    conversation_id: str
    messages: list[ConversationMessage] = Field(default_factory=list)
    current_question: Optional[str] = None
    current_state: Optional[CurriculumQAState] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def append_user(self, content: str) -> None:
        self.messages.append(ConversationMessage(role=MessageRole.USER, content=content))
        self.current_question = content
        self.updated_at = datetime.now(timezone.utc)

    def append_assistant(self, content: str) -> None:
        self.messages.append(
            ConversationMessage(role=MessageRole.ASSISTANT, content=content)
        )
        self.updated_at = datetime.now(timezone.utc)

    def set_state(self, state: CurriculumQAState) -> None:
        self.current_state = state
        self.updated_at = datetime.now(timezone.utc)


class ConversationStore:
    """Simple in-memory conversation store.

    Sufficient for Sprint 1. Persistent/long-term memory is intentionally deferred.
    """

    def __init__(self) -> None:
        self._conversations: dict[str, ConversationContext] = {}
        self._lock = Lock()

    def get_or_create(self, conversation_id: str | None) -> ConversationContext:
        with self._lock:
            if conversation_id and conversation_id in self._conversations:
                return self._conversations[conversation_id]
            cid = conversation_id or str(uuid4())
            ctx = ConversationContext(conversation_id=cid)
            self._conversations[cid] = ctx
            return ctx

    def get(self, conversation_id: str) -> ConversationContext | None:
        with self._lock:
            return self._conversations.get(conversation_id)

    def save(self, context: ConversationContext) -> None:
        with self._lock:
            self._conversations[context.conversation_id] = context

    def clear(self) -> None:
        with self._lock:
            self._conversations.clear()
