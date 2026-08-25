"""Curriculum Q&A agent orchestrator."""

from __future__ import annotations

from app.agent.answer import AnswerGenerationNode
from app.agent.context import ConversationStore
from app.agent.retrieve import RetrievalNode
from app.agent.state import CurriculumQAState
from app.config import Settings, get_settings
from app.curriculum.codes import extract_filters_from_question, normalize_grade_code
from app.enums import AgentStatus
from app.exceptions import AgentError, AgentExecutionError, InvalidRequestError
from app.llm.base import LLMProvider
from app.llm.provider import build_llm_provider
from app.logging_utils import get_logger, log_agent_event
from app.tools.registry import ToolRegistry, build_default_registry

logger = get_logger(__name__)


class CurriculumQAAgent:
    """Orchestrates Understand → Retrieve → Generate Answer."""

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
        self.answer_node = AnswerGenerationNode(llm=self.llm, settings=self.settings)

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
            prior_state = context.current_state
            state = CurriculumQAState.initial(
                question=cleaned,
                conversation_id=context.conversation_id,
            )
            context.append_user(cleaned)

            state = self.understand(state, prior_state=prior_state)
            state = self.retrieve(state, request_id=request_id)
            state = self.answer(
                state,
                conversation=context,
                request_id=request_id,
            )

            context.set_state(state)
            if state.final_answer:
                context.append_assistant(state.final_answer)
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
                confidence=(
                    state.answer_confidence.value if state.answer_confidence else None
                ),
            )
            return state
        except AgentError:
            raise
        except Exception as exc:
            raise AgentExecutionError(
                "Agent failed while processing the question"
            ) from exc

    def understand(
        self,
        state: CurriculumQAState,
        *,
        prior_state: CurriculumQAState | None = None,
    ) -> CurriculumQAState:
        """Extract filters; inherit conversation context when appropriate."""
        state.status = AgentStatus.UNDERSTANDING
        state.bump_iteration()
        filters = extract_filters_from_question(state.question)

        if prior_state:
            for field in ("grade", "subject", "level", "topic"):
                new_value = filters.get(field)
                if new_value:
                    setattr(state, field, new_value)
                elif getattr(prior_state, field, None):
                    setattr(state, field, getattr(prior_state, field))
        else:
            state.grade = filters.get("grade") or state.grade
            state.subject = filters.get("subject") or state.subject
            state.level = filters.get("level") or state.level
            state.topic = filters.get("topic") or state.topic

        # Explicit grade change in follow-up overrides inherited grade.
        explicit_grade = normalize_grade_code(state.question)
        if explicit_grade:
            state.grade = explicit_grade
            state.level = filters.get("level") or state.level

        state.intent = state.intent or "retrieve_curriculum"
        return state

    def retrieve(
        self,
        state: CurriculumQAState,
        *,
        request_id: str | None = None,
    ) -> CurriculumQAState:
        return self.retrieval.run(state, request_id=request_id)

    def answer(
        self,
        state: CurriculumQAState,
        *,
        conversation=None,
        request_id: str | None = None,
    ) -> CurriculumQAState:
        return self.answer_node.run(
            state, conversation=conversation, request_id=request_id
        )

    def verify(self, state: CurriculumQAState) -> CurriculumQAState:
        """Reserved for Sprint 4 verification loop."""
        state.status = AgentStatus.VERIFYING
        return state

    def within_limits(self, state: CurriculumQAState) -> bool:
        return (
            state.iteration < self.settings.agent_max_iterations
            and state.tool_calls <= self.settings.agent_max_tool_calls
        )
