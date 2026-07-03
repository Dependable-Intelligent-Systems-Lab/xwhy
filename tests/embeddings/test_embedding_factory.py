"""Tests for EmbeddingFactory."""

from __future__ import annotations

import pytest

from xwhy.embeddings.factory import EmbeddingFactory
from xwhy.embeddings.types import EmbeddingType
from xwhy.embeddings.word2vec import Word2VecEmbedding


def test_register_and_create_embedding() -> None:
    """Register and create embedding successfully."""
    EmbeddingFactory.register(
        embedding=EmbeddingType.GLOVE,
        embedding_cls=Word2VecEmbedding,
    )

    embedding = EmbeddingFactory.create(
        embedding=EmbeddingType.GLOVE,
        settings=object(),
    )

    assert isinstance(embedding, Word2VecEmbedding)


def test_register_duplicate_embedding() -> None:
    """Ensure duplicate registration raises error."""
    with pytest.raises(ValueError, match="already registered"):
        EmbeddingFactory.register(
            embedding=EmbeddingType.WORD2VEC,
            embedding_cls=Word2VecEmbedding,
        )


def test_unsupported_embedding_raises() -> None:
    """Test that unsupported embedding type raises ValueError."""

    class FakeType(str):
        pass

    with pytest.raises(ValueError, match="Unsupported embedding"):
        EmbeddingFactory.create(
            embedding=FakeType("unknown"),  # type: ignore[arg-type]
            settings=object(),
        )
