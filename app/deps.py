from functools import lru_cache

from app.agent.context import ConversationStore
from app.agent.orchestrator import CurriculumQAAgent
from app.config import get_settings
from app.curriculum.client import CurriculumAPIClient
from app.llm.base import LLMProvider
from app.llm.provider import build_llm_provider
from app.tools.registry import ToolRegistry, build_default_registry


@lru_cache
def get_conversation_store() -> ConversationStore:
    return ConversationStore()


@lru_cache
def get_curriculum_client() -> CurriculumAPIClient:
    return CurriculumAPIClient(settings=get_settings())


@lru_cache
def get_tool_registry() -> ToolRegistry:
    return build_default_registry(
        settings=get_settings(),
        include_echo=False,
        client=get_curriculum_client(),
    )


def get_llm_provider() -> LLMProvider:
    return build_llm_provider(get_settings())


def get_agent() -> CurriculumQAAgent:
    return CurriculumQAAgent(
        settings=get_settings(),
        llm=get_llm_provider(),
        tools=get_tool_registry(),
        conversations=get_conversation_store(),
    )


def reset_singletons() -> None:
    """Test helper to clear cached deps."""
    get_conversation_store.cache_clear()
    get_tool_registry.cache_clear()
    get_curriculum_client.cache_clear()
    get_settings.cache_clear()
