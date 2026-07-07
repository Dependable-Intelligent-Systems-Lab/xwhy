"""Application settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from xwhy.config.env import load_environment

load_environment()


class Settings(BaseSettings):
    """Global application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_cache_dir: Path = Field(
        default=Path.home() / ".cache" / "xwhy" / "embeddings",
    )

    anthropic_api_key: str | None = None

    openai_api_key: str | None = None

    gemini_api_key: str | None = None

    huggingface_token: str | None = None
