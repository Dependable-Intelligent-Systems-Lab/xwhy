"""Shared provider types."""

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
