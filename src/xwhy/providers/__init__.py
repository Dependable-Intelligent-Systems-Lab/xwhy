"""Provider implementations."""

# TODO:
#   * GeminiProvider
#   * HuggingFaceProvider
#   * AnthropicProvider

from xwhy.providers.base import BaseProvider
from xwhy.providers.factory import ProviderFactory
from xwhy.providers.openai import OpenAIProvider
from xwhy.providers.types import ProviderType

ProviderFactory.register(
    provider=ProviderType.OPENAI,
    provider_cls=OpenAIProvider,
)

__all__ = [
    "BaseProvider",
    "OpenAIProvider",
    "ProviderFactory",
    "ProviderType",
]
