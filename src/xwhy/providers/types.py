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

    @property
    def is_text_only(self) -> bool:
        """Return True if the provider generates only text."""
        return self in {
            self.ANTHROPIC,
            self.ZAI,
            self.GROQ,
            self.COHERE,
            self.OLLAMA,
            self.LMSTUDIO,
            self.ANTHROPIC_BEDROCK,
            self.ANTHROPIC_BEDROCK_MANTLE,
            self.ANTHROPIC_AWS,
            self.ANTHROPIC_VERTEX,
            self.ANTHROPIC_FOUNDRY,
        }

    @property
    def is_image_only(self) -> bool:
        """Return True if the provider generates only images.

        Currently returns False. Add models like Midjourney or
        StabilityAI here in the future if added to the enum.
        """
        return False

    @property
    def supports_both(self) -> bool:
        """Return True if the provider generates both text and images."""
        return self in {
            self.OPENAI,
            self.GEMINI,
            self.HUGGINGFACE,
            self.FIREWORKS_AI,
            self.GROK,
            self.OPENROUTER,
            self.BYTEDANCE,
            self.AZURE_OPENAI,
            self.GCP_GEMINI,
        }

    @classmethod
    def from_str(cls, value: str | ProviderType) -> ProviderType:
        """Convert a string or enum instance to ProviderType safely."""
        try:
            return cls(value)
        except ValueError as err:
            valid_options = [item.value for item in cls]
            raise ValueError(
                f"'{value}' is not a valid ProviderType. "
                f"Please choose from: {valid_options}"
            ) from err
