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

    zai_api_key: str | None = None

    groq_api_key: str | None = None

    cohere_api_key: str | None = None

    fireworks_api_key: str | None = None

    grok_api_key: str | None = None

    openrouter_api_key: str | None = None

    bytedance_api_key: str | None = None

    lmstudio_api_key: str = "lm-studio"

    lmstudio_base_url: str = "http://localhost:1234/v1"

    # Azure OpenAI
    azure_api_key: str | None = None
    azure_api_version: str = "2024-02-01"
    azure_endpoint: str | None = None

    # Google Cloud (Gemini & Anthropic Vertex)
    gcp_project: str | None = None
    gcp_location: str = "us-central1"

    # AWS / Bedrock
    aws_access_key: str | None = None
    aws_secret_key: str | None = None
    aws_session_token: str | None = None
    aws_region: str = "us-east-1"
    anthropic_aws_workspace_id: str | None = None

    # Microsoft Foundry (Anthropic)
    anthropic_foundry_api_key: str | None = None
    anthropic_foundry_resource: str | None = None
