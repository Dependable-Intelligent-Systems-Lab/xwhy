"""Resolver for auto-instantiating providers (Hidden Factory)."""

from collections.abc import Callable
from typing import Any, ClassVar

from xwhy.providers.base import BaseProvider
from xwhy.providers.types import ProviderType


class ProviderResolver:
    """Resolver mapping provider types to default instantiation logic."""

    _builders: ClassVar[dict[ProviderType, Callable[..., BaseProvider]]] = {}

    @classmethod
    def register(
        cls, provider_type: ProviderType, builder: Callable[..., BaseProvider]
    ) -> None:
        """Register a default builder function for a provider type."""
        cls._builders[provider_type] = builder

    @classmethod
    def resolve(
        cls,
        provider: str | ProviderType | BaseProvider,
        **kwargs: Any,  # noqa: ANN401
    ) -> BaseProvider:
        """Resolve a provider identifier into an instantiated BaseProvider."""
        if isinstance(provider, BaseProvider):
            return provider

        ptype = ProviderType(provider)
        if ptype not in cls._builders:
            raise ValueError(
                f"No default builder registered for provider type: {ptype}. "
                "Please instantiate the provider manually and pass the instance."
            )

        return cls._builders[ptype](**kwargs)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered resolvers."""
        cls._builders.clear()
