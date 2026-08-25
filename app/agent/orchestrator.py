"""Curriculum Q&A agent orchestrator."""

from __future__ import annotations

from app.agent.context import ConversationStore
from app.agent.retrieve import RetrievalNode
from app.agent.state import CurriculumQAState
from app.config import Settings, get_settings
from app.curriculum.codes import extract_filters_from_question
from app.enums import AgentStatus
from app.exceptions import AgentError, AgentExecutionError, InvalidRequestError
from app.llm.base import LLMProvider
from app.llm.provider import build_llm_provider
from app.logging_utils import get_logger, log_agent_event
from app.tools.registry import ToolRegistry, build_default_registry

logger = get_logger(__name__)


class CurriculumQAAgent:
    """Orchestrates Understand → Retrieve (Phase 2). Answer/Verify arrive in Phase 3."""

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
        self.tools = (
            tools
            if tools is not None
            else build_default_registry(settings=self.settings, include_echo=False)
        )
        self.conversations = conversations or ConversationStore()
        self.retrieval = RetrievalNode(
            llm=self.llm, tools=self.tools, settings=self.settings
        )

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

            state = self.understand(state)
            state = self.retrieve(state, request_id=request_id)

            context.set_state(state)
            self.conversations.save(context)

            log_agent_event(
                logger,
                "agent.turn.complete",
                request_id=request_id,
                conversation_id=state.conversation_id,
                question=state.question,
                status=state.status.value,
                iteration=state.iteration,
                tool_calls=state.tool_calls,
                model=self.llm.model,
                evidence_count=len(state.evidence),
                evidence_status=state.evidence_status.value,
            )
            return state
        except AgentError:
            raise
        except Exception as exc:
            raise AgentExecutionError(
                "Agent failed while processing the question"
            ) from exc

    def understand(self, state: CurriculumQAState) -> CurriculumQAState:
        """Lightweight filter extraction; Phase 3 may replace with structured LLM intent."""
        state.status = AgentStatus.UNDERSTANDING
        state.bump_iteration()
        filters = extract_filters_from_question(state.question)
        state.grade = filters.get("grade") or state.grade
        state.subject = filters.get("subject") or state.subject
        state.level = filters.get("level") or state.level
        state.topic = filters.get("topic") or state.topic
        state.intent = state.intent or "retrieve_curriculum"
        return state

    def retrieve(
        self,
        state: CurriculumQAState,
        *,
        request_id: str | None = None,
    ) -> CurriculumQAState:
        return self.retrieval.run(state, request_id=request_id)

    def answer(self, state: CurriculumQAState) -> CurriculumQAState:
        """Draft an answer from retrieved context. Phase 3."""
        state.status = AgentStatus.ANSWERING
        return state

    def verify(self, state: CurriculumQAState) -> CurriculumQAState:
        """Check the draft against retrieved curriculum. Phase 3."""
        state.status = AgentStatus.VERIFYING
        return state

    def within_limits(self, state: CurriculumQAState) -> bool:
        return (
            state.iteration < self.settings.agent_max_iterations
            and state.tool_calls <= self.settings.agent_max_tool_calls
        )
