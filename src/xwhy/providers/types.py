"""Shared provider types."""

from enum import StrEnum


class ProviderType(StrEnum):
    """Supported provider types."""

    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"

    # ------------------------------------------------------------------
    # TODO: Enable when implementations are added.
    # OLLAMA = "ollama"
    # ------------------------------------------------------------------
