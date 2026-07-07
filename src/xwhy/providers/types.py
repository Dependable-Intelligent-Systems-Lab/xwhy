"""Shared provider types."""

from enum import StrEnum


class ProviderType(StrEnum):
    """Supported provider types."""

    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"

    # ------------------------------------------------------------------
    # TODO: Enable when implementations are added.
    # HUGGINGFACE = "huggingface"
    # OLLAMA = "ollama"
    # ------------------------------------------------------------------
