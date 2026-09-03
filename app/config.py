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
    llm_base_url: str = "https://api.openai.com/v1"
    # Optional separate model for verification; empty → reuse llm_model.
    verifier_llm_model: str = ""

    agent_max_iterations: int = Field(default=3, ge=1)
    agent_max_tool_calls: int = Field(default=10, ge=0)
    agent_max_retrieval_rounds: int = Field(default=3, ge=1)
    agent_request_timeout_seconds: float = Field(default=60.0, gt=0)

    # V2.2 experiment: treat resolved curriculum context as authoritative evidence boundary.
    curriculum_v2_context_boundary_experiment: bool = False

    # V2.3 experiment: frozen resolve-only retrieval + generation/verifier diagnostics.
    v23_generation_verifier_experiment: bool = False

    # V2.4 experiment: routing / verifier isolation (arms A–D).
    v24_routing_verifier_experiment: bool = False

    # V2.5 experiment: verifier handling of imperfect-but-present evidence.
    v25_verifier_evidence_quality_experiment: bool = False

    # V2.6 experiment: verifier evidence-state isolation (replay harness).
    v26_verifier_evidence_state_experiment: bool = False

    # V2.7 experiment: verifier decision-boundary isolation (replay harness).
    v27_verifier_decision_boundary_experiment: bool = False

    # V2.8 experiment: recommendation-mapping isolation (replay harness).
    v28_recommendation_mapping_experiment: bool = False

    # V2.9 experiment: evidence normalization & grounding-boundary (replay harness).
    v29_evidence_normalization_experiment: bool = False

    # V2.10 experiment: integrated normalization + recommendation mapping (replay harness).
    v210_integrated_experiment: bool = False

    # V2.11 experiment: metadata-integrity pre-verifier guard (replay harness).
    v211_metadata_integrity_experiment: bool = False

    # V2.12A experiment: LangChain vs LangGraph behavioral equivalence (replay harness).
    v212_langchain_experiment: bool = False

    # V2.12B experiment: production-shadow retrieval evaluation (observational only).
    v212b_shadow_enabled: bool = False
    v212b_shadow_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    v212b_shadow_timeout_seconds: float = Field(default=120.0, gt=0)

    # V2.13A experiment: curriculum document evidence layer (feature-flagged).
    v213_document_evidence_experiment: bool = False

    # V2.13B experiment: hybrid semantic document retrieval (feature-flagged).
    v213b_semantic_retrieval_experiment: bool = False
    v213b_retrieval_variant: str = "lexical"
    v213b_embedding_provider: str = "feature_hash"
    v213b_embedding_model: str = "feature-hash-v1"
    v213b_embedding_dimension: int = 128

    # V2.13C experiment: controlled hybrid retrieval + curriculum QA eval (harness-only).
    v213c_experiment: bool = False
    v213c_document_retrieval: bool = False
    v213c_retrieval_variant: str = "context_hybrid"

    # V2.13D experiment: production-shadow context-hybrid document evidence (observational).
    v213d_shadow_enabled: bool = False
    v213d_shadow_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    v213d_shadow_document_retrieval: bool = True
    v213d_shadow_retrieval_variant: str = "context_hybrid"
    v213d_shadow_timeout_seconds: float = Field(default=30.0, gt=0)

    # LangGraph short-term memory / checkpointing
    agent_checkpointing_enabled: bool = True
    agent_checkpoint_backend: str = "sqlite"  # memory | sqlite
    agent_checkpoint_sqlite_path: str = "data/checkpoints.sqlite"

    # Prefer CURRICULUM_API_URL; CURRICULUM_API_BASE_URL kept for Phase 1 compatibility.
    curriculum_api_url: str = ""
    curriculum_api_base_url: str = "http://127.0.0.1:8000"
    curriculum_api_timeout: float = Field(default=15.0, gt=0)

    @property
    def curriculum_api_timeout_seconds(self) -> float:
        return self.curriculum_api_timeout

    def resolved_curriculum_api_url(self) -> str:
        return (self.curriculum_api_url or self.curriculum_api_base_url).rstrip("/")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
