"""Factory for provider implementations."""

from __future__ import annotations

from typing import ClassVar

from xwhy.providers.base import BaseProvider
from xwhy.providers.types import ProviderType


class ProviderFactory:
    """Factory for provider implementations."""

    _registry: ClassVar[dict[ProviderType, type[BaseProvider]]] = {}

    @classmethod
    def register(
        cls,
        *,
        provider: ProviderType,
        provider_cls: type[BaseProvider],
    ) -> None:
        """Register a provider implementation.

        Args:
            provider:
                Provider identifier.

            provider_cls:
                Provider implementation.

        Raises:
            ValueError:
                Provider is already registered.

        """
        if provider in cls._registry:
            raise ValueError(f"Provider already registered: {provider}")

        cls._registry[provider] = provider_cls

    @classmethod
    def create(
        cls,
        *,
        provider: ProviderType,
        client: object,
    ) -> BaseProvider:
        """Create a provider instance.

        Args:
            provider:
                Provider implementation.

            client:
                Initialized provider client.

        Returns:
            Provider instance.

        Raises:
            ValueError:
                Unsupported provider.

        """
        try:
            provider_cls = cls._registry[provider]

        except KeyError as exc:
            raise ValueError(f"Unsupported provider: {provider}") from exc

        return provider_cls(client)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers."""
        cls._registry.clear()
