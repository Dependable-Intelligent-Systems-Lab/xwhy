"""Factory for embedding implementations."""

from __future__ import annotations

from typing import ClassVar

from xwhy.embeddings.base import BaseEmbedding
from xwhy.embeddings.types import EmbeddingType


class EmbeddingFactory:
    """Factory for creating embedding implementations."""

    _registry: ClassVar[dict[EmbeddingType, type[BaseEmbedding]]] = {}

    @classmethod
    def register(
        cls,
        *,
        embedding: EmbeddingType,
        embedding_cls: type[BaseEmbedding],
    ) -> None:
        """Register a new embedding implementation."""
        if embedding in cls._registry:
            raise ValueError("already registered")

        cls._registry[embedding] = embedding_cls

    @classmethod
    def create(
        cls,
        *,
        embedding: EmbeddingType,
        **kwargs: object,
    ) -> BaseEmbedding:
        """Create embedding instance."""
        try:
            embedding_cls = cls._registry[embedding]

        except KeyError as exc:
            raise ValueError(f"Unsupported embedding: {embedding}") from exc

        return embedding_cls(**kwargs)

    @classmethod
    def clear(cls) -> None:
        """Reset registry to defaults."""
        cls._registry.clear()
