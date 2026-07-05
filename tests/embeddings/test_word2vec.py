"""Tests for Word2VecEmbedding."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from xwhy.embeddings.word2vec import Word2VecEmbedding


class DummySettings:
    """Minimal settings mock."""

    embedding_cache_dir = Path("/tmp/embeddings")


class FakeEmbeddingModel:
    """A lightweight fake embedding model for testing purposes.

    This class mimics a minimal embedding model interface supporting
    dictionary-like access and membership checks.
    """

    def __init__(self, data: Mapping[str, object]) -> None:
        """Initialize the fake embedding model.

        Args:
            data: Mapping of tokens to their embedding vectors.

        """
        self._data = data

    def __contains__(self, key: str) -> bool:
        """Check whether a token exists in the embedding model.

        Args:
            key: Token to check.

        Returns:
            True if token exists, otherwise False.

        """
        return key in self._data

    def __getitem__(self, key: str) -> object:
        """Retrieve embedding vector for a token.

        Args:
            key: Token to retrieve.

        Returns:
            Embedding vector associated with the token.

        """
        return self._data[key]


def create_embedding(force_download: bool = False) -> Word2VecEmbedding:
    """Create Word2VecEmbedding instance for testing.

    Args:
        force_download: Whether to force model download.

    Returns:
        Configured Word2VecEmbedding instance.

    """
    return Word2VecEmbedding(
        settings=DummySettings(),
        force_download=force_download,
    )


# ---------------------------------------------------------------------
# Gensim load path
# ---------------------------------------------------------------------
@patch("xwhy.embeddings.word2vec.api.load")
@patch("xwhy.embeddings.word2vec.KeyedVectors.load_word2vec_format")
def test_load_from_gensim(
    mock_kv_load: MagicMock,
    mock_api_load: MagicMock,
) -> None:
    """Test loading via gensim without touching filesystem."""
    fake_model = MagicMock()
    mock_api_load.return_value = fake_model

    embedding = create_embedding(force_download=True)

    model = embedding.load()

    assert model is fake_model
    mock_api_load.assert_called_once()
    mock_kv_load.assert_not_called()


# ---------------------------------------------------------------------
# Cache path
# ---------------------------------------------------------------------
@patch("xwhy.embeddings.word2vec.KeyedVectors.load_word2vec_format")
def test_load_from_cache(
    mock_kv_load: MagicMock,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Test loading from cache path."""
    mock_kv_load.return_value = MagicMock()

    embedding = Word2VecEmbedding(
        settings=DummySettings(),
        force_download=False,
    )

    cache_dir = tmp_path
    embedding._settings.embedding_cache_dir = cache_dir

    model_path = cache_dir / "GoogleNews-vectors-negative300.bin"
    model_path.write_text("dummy")  # مهم: exists

    embedding._MODEL_FILE_MAP["word2vec-google-news-300"]["gensim"] = False

    model = embedding.load()

    assert model is not None
    mock_kv_load.assert_called_once()


# ---------------------------------------------------------------------
# Encode test
# ---------------------------------------------------------------------
def test_encode_returns_vector() -> None:
    """Test encode logic."""
    embedding = create_embedding()

    class FakeVector(list):
        def tolist(self) -> list[float]:
            return list(self)

    fake_data = {
        "hello": FakeVector([1.0, 2.0, 3.0]),
        "world": FakeVector([3.0, 2.0, 1.0]),
    }

    fake_model = FakeEmbeddingModel(fake_data)

    embedding.load = lambda: fake_model  # type: ignore

    result = embedding.encode("hello world")

    assert len(result) == 3
    assert all(isinstance(x, float) for x in result)


# ---------------------------------------------------------------------
# Empty encode fallback
# ---------------------------------------------------------------------
def test_encode_empty_text() -> None:
    """Test encoding empty result fallback."""
    embedding = create_embedding()

    embedding.load = lambda: {}  # type: ignore

    result = embedding.encode("unknown words only")

    assert result == [0.0] * 300


# ---------------------------------------------------------------------
# Download failure
# ---------------------------------------------------------------------
@patch("xwhy.embeddings.word2vec.requests.get")
def test_download_file_failure(
    mock_get: MagicMock,
) -> None:
    """Test download failure cleanup."""
    embedding = create_embedding()

    mock_get.side_effect = requests.exceptions.RequestException("network error")

    path = Path("/tmp/fake.bin")

    with pytest.raises(
        requests.exceptions.RequestException,
        match="network error",
    ):
        embedding._download_file("http://fake", path)

    assert not path.exists()
