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
    AZURE_OPENAI = "azure-openai"
    GCP_GEMINI = "gcp-gemini"
    ANTHROPIC_BEDROCK = "anthropic-bedrock"
    ANTHROPIC_BEDROCK_MANTLE = "anthropic-bedrock-mantle"
    ANTHROPIC_AWS = "anthropic-aws"
    ANTHROPIC_VERTEX = "anthropic-vertex"
    ANTHROPIC_FOUNDRY = "anthropic-foundry"
