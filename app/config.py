from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"
    log_level: str = "INFO"

    llm_provider: str = "stub"
    llm_model: str = "stub-model"
    llm_api_key: str = ""
    llm_timeout_seconds: float = Field(default=30.0, gt=0)

    agent_max_iterations: int = Field(default=3, ge=1)
    agent_max_tool_calls: int = Field(default=10, ge=0)
    agent_request_timeout_seconds: float = Field(default=60.0, gt=0)

    # Reserved for Sprint 2 read-only Curriculum Structure API access.
    curriculum_api_base_url: str = "http://127.0.0.1:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
