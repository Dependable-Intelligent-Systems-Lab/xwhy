"""Unit tests for embedding types."""

import pytest

from xwhy.embeddings.types import EmbeddingType


def test_embedding_type_from_str_success() -> None:
    """Test successful conversion from valid strings."""
    assert EmbeddingType.from_str("word2vec") == EmbeddingType.WORD2VEC
    assert EmbeddingType.from_str("glove") == EmbeddingType.GLOVE
    assert EmbeddingType.from_str("paragram_sl") == EmbeddingType.PARAGRAM_SL
    assert EmbeddingType.from_str("paragram_ws") == EmbeddingType.PARAGRAM_WS


def test_embedding_type_from_str_invalid() -> None:
    """Test that invalid input raises ValueError with a clear message."""
    invalid_input = "invalid_embedding"

    with pytest.raises(
        ValueError, match=f"'{invalid_input}' is not a valid EmbeddingType"
    ):
        EmbeddingType.from_str(invalid_input)
