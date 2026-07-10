"""Factory for embedding implementations."""

from collections.abc import Callable
from typing import ClassVar

from xwhy.embeddings import BaseEmbedding
from xwhy.embeddings.types import EmbeddingType


class EmbeddingFactory:
    """Manage embedding model instantiation via a registry."""

    _registry: ClassVar[dict[EmbeddingType, Callable[..., BaseEmbedding]]] = {}

    @classmethod
    def register(
        cls, embedding: EmbeddingType, builder: Callable[..., BaseEmbedding]
    ) -> None:
        """Register a builder function for an embedding type.

        Args:
            embedding: The type of embedding to register.
            builder: A callable (function/lambda) that accepts keyword arguments
                and returns a BaseEmbedding instance.

        Raises:
            ValueError: If the embedding type is already registered.

        """
        if embedding in cls._registry:
            raise ValueError(f"Embedding already registered: {embedding}")
        cls._registry[embedding] = builder

    @classmethod
    def create(cls, embedding: EmbeddingType, **kwargs: object) -> BaseEmbedding:
        """Instantiate and configure an embedding model.

        Args:
            embedding: The type of embedding to create.
            **kwargs: Arbitrary keyword arguments passed to the builder function,
                such as 'settings' or 'model_name'.

        Returns:
            An instantiated BaseEmbedding object.

        Raises:
            ValueError: If the embedding type is not registered.

        """
        if embedding not in cls._registry:
            raise ValueError(f"Unsupported embedding: {embedding}")

        return cls._registry[embedding](**kwargs)

    @classmethod
    def clear(cls) -> None:
        """Reset registry to defaults."""
        cls._registry.clear()
