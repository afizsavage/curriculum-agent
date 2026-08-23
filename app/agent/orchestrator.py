"""Curriculum Q&A agent orchestrator (Sprint 1 skeleton)."""

from __future__ import annotations

from app.agent.context import ConversationStore
from app.agent.state import CurriculumQAState
from app.config import Settings, get_settings
from app.enums import AgentStatus
from app.exceptions import AgentError, AgentExecutionError, InvalidRequestError
from app.llm.base import LLMProvider
from app.llm.provider import build_llm_provider
from app.logging_utils import get_logger, log_agent_event
from app.tools.registry import ToolRegistry, build_default_registry

logger = get_logger(__name__)


class CurriculumQAAgent:
    """Orchestrates Understand → Retrieve → Answer → Verify (stubs in Sprint 1).

    Sprint 1 accepts a question, creates typed state, records conversation context,
    and returns a `received` result. Later sprints fill in the node methods.
    """

    name = "curriculum_qa"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm: LLMProvider | None = None,
        tools: ToolRegistry | None = None,
        conversations: ConversationStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or build_llm_provider(self.settings)
        self.tools = tools if tools is not None else build_default_registry(include_echo=True)
        self.conversations = conversations or ConversationStore()

    def ask(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        request_id: str | None = None,
    ) -> CurriculumQAState:
        cleaned = (question or "").strip()
        if not cleaned:
            raise InvalidRequestError("question must not be blank")

        try:
            context = self.conversations.get_or_create(conversation_id)
            state = CurriculumQAState.initial(
                question=cleaned,
                conversation_id=context.conversation_id,
            )
            context.append_user(cleaned)
            context.set_state(state)
            self.conversations.save(context)

            # Future loop entry points (intentionally not chained yet):
            # state = self.understand(state)
            # state = self.retrieve(state)
            # state = self.answer(state)
            # state = self.verify(state)

            log_agent_event(
                logger,
                "agent.turn.received",
                request_id=request_id,
                conversation_id=state.conversation_id,
                question=state.question,
                status=state.status.value,
                iteration=state.iteration,
                tool_calls=state.tool_calls,
                model=self.llm.model,
                max_iterations=self.settings.agent_max_iterations,
                max_tool_calls=self.settings.agent_max_tool_calls,
            )
            return state
        except AgentError:
            raise
        except Exception as exc:
            raise AgentExecutionError(
                "Agent failed while accepting the question"
            ) from exc

    def understand(self, state: CurriculumQAState) -> CurriculumQAState:
        """Parse intent / grade / subject. Implemented in a later sprint."""
        state.status = AgentStatus.UNDERSTANDING
        state.bump_iteration()
        return state

    def retrieve(self, state: CurriculumQAState) -> CurriculumQAState:
        """Call curriculum tools. Implemented in Sprint 2."""
        state.status = AgentStatus.RETRIEVING
        return state

    def answer(self, state: CurriculumQAState) -> CurriculumQAState:
        """Draft an answer from retrieved context. Later sprint."""
        state.status = AgentStatus.ANSWERING
        return state

    def verify(self, state: CurriculumQAState) -> CurriculumQAState:
        """Check the draft against retrieved curriculum. Later sprint."""
        state.status = AgentStatus.VERIFYING
        return state

    def within_limits(self, state: CurriculumQAState) -> bool:
        return (
            state.iteration < self.settings.agent_max_iterations
            and state.tool_calls <= self.settings.agent_max_tool_calls
        )
