from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/agent_workflows"
    redis_url: str = "redis://localhost:6379/0"

    log_level: str = "INFO"
    embedding_dimension: int = 1536
    max_context_docs: int = 5
    cache_ttl: int = 3600

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
