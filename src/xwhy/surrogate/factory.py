"""Factory for creating surrogate models."""

from collections.abc import Callable
from typing import ClassVar

from xwhy.surrogate.base import BaseSurrogate
from xwhy.surrogate.types import SurrogateType


class SurrogateFactory:
    """Factory class for instantiating surrogate models."""

    _registry: ClassVar[dict[SurrogateType, Callable[..., BaseSurrogate]]] = {}

    @classmethod
    def register(
        cls,
        *,
        method: SurrogateType,
        builder: Callable[..., BaseSurrogate],
    ) -> None:
        """Register a new surrogate builder."""
        if method in cls._registry:
            if cls._registry[method] is builder:
                return
            raise ValueError(f"Surrogate method already registered: {method}")

        cls._registry[method] = builder

    @classmethod
    def create(
        cls,
        *,
        method: SurrogateType,
        **kwargs: object,
    ) -> BaseSurrogate:
        """Create and configure a surrogate model instance.

        Args:
            method: The type of surrogate model to create.
            **kwargs: Arguments to pass to the builder (e.g., seed, ridge_alpha).

        """
        if method not in cls._registry:
            raise ValueError(f"Unsupported surrogate method: {method}")

        builder_func = cls._registry[method]
        return builder_func(**kwargs)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered surrogate models."""
        cls._registry.clear()
