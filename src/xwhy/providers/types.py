"""Shared provider types."""

from __future__ import annotations

from enum import StrEnum


class ProviderType(StrEnum):
    """Supported provider types."""

    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    ZAI = "zai"
    GROQ = "groq"
    COHERE = "cohere"
    FIREWORKS_AI = "fireworks-ai"
    GROK = "grok"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    BYTEDANCE = "bytedance"
    AZURE_OPENAI = "azure-openai"
    GCP_GEMINI = "gcp-gemini"
    ANTHROPIC_BEDROCK = "anthropic-bedrock"
    ANTHROPIC_BEDROCK_MANTLE = "anthropic-bedrock-mantle"
    ANTHROPIC_AWS = "anthropic-aws"
    ANTHROPIC_VERTEX = "anthropic-vertex"
    ANTHROPIC_FOUNDRY = "anthropic-foundry"

    @classmethod
    def from_str(cls, value: str | ProviderType) -> ProviderType:
        """Safely convert a string or enum instance to ProviderType."""
        try:
            return cls(value)
        except ValueError as err:
            valid_options = [item.value for item in cls]
            raise ValueError(
                f"'{value}' is not a valid ProviderType. "
                f"Please choose from: {valid_options}"
            ) from err
