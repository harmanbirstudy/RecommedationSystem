"""
app/config.py

Pydantic-settings BaseSettings for the recommendation ML service.
Reads from the project-root .env (or real environment variables, which win).

The LLM provider is chosen once at startup with LLM_PROVIDER and used
exclusively for the lifetime of the process:
    LLM_PROVIDER=anthropic  -> Claude (ANTHROPIC_MODEL, needs ANTHROPIC_API_KEY)
    LLM_PROVIDER=ollama     -> local Ollama (OLLAMA_MODEL at OLLAMA_BASE_URL)

Usage:
    from app.config import get_settings
    settings = get_settings()
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env from the repository root regardless of working directory
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class LlmProvider(StrEnum):
    """Supported LLM back ends."""

    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Application settings loaded from .env and the environment."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",  # the shared .env also holds settings for the Next.js API
    )

    # PostgreSQL (the shoppingwebsite database)
    postgres_user: str = "shopping"
    postgres_password: SecretStr = SecretStr("shopping")
    postgres_db: str = "shoppingwebsite"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # LLM
    llm_provider: LlmProvider = LlmProvider.ANTHROPIC
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    llm_timeout_seconds: int = 90

    # Recommendations
    recommendation_count: int = Field(default=5, ge=1, le=20)
    max_recommendation_count: int = Field(default=20, ge=1, le=50)
    # How many top XGBoost candidates the LLM chooses from
    candidate_pool_size: int = Field(default=15, ge=1, le=50)
    exclude_purchased_products: bool = True

    # Caching / refresh
    data_refresh_seconds: int = 300  # reload orders so new checkouts count
    cache_ttl_seconds: int = 600  # per-member response cache (LLM calls are slow)

    # Model persistence
    model_dir: Path = Path(__file__).resolve().parents[1] / "artifacts"
    retrain_on_startup: bool = False

    @model_validator(mode="after")
    def _check_provider(self) -> "Settings":
        """Fail fast when the chosen provider is missing its credentials."""
        if self.llm_provider is LlmProvider.ANTHROPIC and not (
            self.anthropic_api_key and self.anthropic_api_key.get_secret_value()
        ):
            raise ValueError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> SecretStr:
        """Compose the libpq connection string."""
        password = self.postgres_password.get_secret_value()
        return SecretStr(
            f"postgresql://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def llm_model_name(self) -> str:
        """Return the model name for the active provider."""
        if self.llm_provider is LlmProvider.ANTHROPIC:
            return self.anthropic_model
        return self.ollama_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()
