"""Shared provider types."""

from enum import StrEnum


class ProviderType(StrEnum):
    """Supported provider types.

    Only OpenAI is currently implemented.

    Todo:
        - Implement Gemini provider.
        - Implement Hugging Face provider.
        - Implement Anthropic provider.

    """

    OPENAI = "openai"

    # ------------------------------------------------------------------
    # TODO: Enable when implementations are added.
    #
    # GEMINI = "gemini"
    # HUGGINGFACE = "huggingface"
    # ANTHROPIC = "anthropic"
    # OLLAMA = "ollama"
    # ------------------------------------------------------------------
