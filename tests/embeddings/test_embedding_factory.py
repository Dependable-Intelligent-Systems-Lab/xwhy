"""Tests for EmbeddingFactory."""

from __future__ import annotations

import pytest

from xwhy.embeddings.factory import EmbeddingFactory
from xwhy.embeddings.types import EmbeddingType
from xwhy.embeddings.word2vec import Word2VecEmbedding


def test_register_and_create_embedding() -> None:
    """Register and create embedding successfully."""
    EmbeddingFactory.clear()

    def _builder(**kwargs: object) -> Word2VecEmbedding:
        return Word2VecEmbedding(settings=kwargs["settings"])  # type: ignore[arg-type]

    EmbeddingFactory.register(
        embedding=EmbeddingType.GLOVE,
        builder=_builder,
    )

    embedding = EmbeddingFactory.create(
        embedding=EmbeddingType.GLOVE,
        settings=object(),
    )

    assert isinstance(embedding, Word2VecEmbedding)


def test_register_duplicate_embedding() -> None:
    """Ensure duplicate registration raises error."""
    EmbeddingFactory.clear()

    EmbeddingFactory.register(
        embedding=EmbeddingType.WORD2VEC,
        builder=lambda **kwargs: Word2VecEmbedding(settings=kwargs["settings"]),
    )

    with pytest.raises(ValueError, match="already registered"):
        EmbeddingFactory.register(
            embedding=EmbeddingType.WORD2VEC,
            builder=lambda **kwargs: Word2VecEmbedding(settings=kwargs["settings"]),
        )


def test_unsupported_embedding_raises() -> None:
    """Test that unsupported embedding type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported embedding"):
        EmbeddingFactory.create(
            embedding="non_existent_embedding",  # type: ignore[arg-type]
            settings=object(),
        )
