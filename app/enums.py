from enum import Enum


class AgentStatus(str, Enum):
    """Lifecycle status for a Curriculum Q&A agent turn."""

    RECEIVED = "received"
    UNDERSTANDING = "understanding"
    RETRIEVING = "retrieving"
    RETRIEVED = "retrieved"
    ANSWERING = "answering"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    NEEDS_CLARIFICATION = "needs_clarification"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"
    ERROR = "error"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
