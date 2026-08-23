from enum import Enum


class AgentStatus(str, Enum):
    """Lifecycle status for a Curriculum Q&A agent turn."""

    RECEIVED = "received"
    UNDERSTANDING = "understanding"
    RETRIEVING = "retrieving"
    ANSWERING = "answering"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
