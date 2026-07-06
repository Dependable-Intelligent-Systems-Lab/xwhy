"""Tests for Word2VecEmbedding."""

from __future__ import annotations

import gzip
import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import requests

from xwhy.config import Settings
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
        settings=cast(Settings, DummySettings()),
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
    tmp_path: Path,
) -> None:
    """Test loading from cache path."""
    mock_kv_load.return_value = MagicMock()

    embedding = Word2VecEmbedding(
        settings=cast(Settings, DummySettings()),
        force_download=False,
    )

    embedding._settings.embedding_cache_dir = tmp_path

    model_path = tmp_path / "GoogleNews-vectors-negative300.bin"
    model_path.write_text("dummy", encoding="utf-8")

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

    class FakeVector(list[float]):
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


# ---------------------------------------------------------------------
# Load logic edge cases
# ---------------------------------------------------------------------
def test_load_returns_existing_model() -> None:
    """Test load returns immediately if model is already loaded in memory."""
    embedding = create_embedding()
    fake_model = MagicMock()

    embedding._model = fake_model

    result = embedding.load()

    assert result is fake_model


def test_load_unsupported_model() -> None:
    """Test load raises ValueError for unsupported model names."""
    embedding = Word2VecEmbedding(
        settings=cast(Settings, DummySettings()),
        model_name="invalid-model-name",
    )

    with pytest.raises(ValueError, match="Unsupported model: invalid-model-name"):
        embedding.load()


@patch("xwhy.embeddings.word2vec.api.load")
def test_load_gensim_fails_fallback_to_google_news(mock_api_load: MagicMock) -> None:
    """Test fallback mechanism when gensim fails for google news model."""
    mock_api_load.side_effect = Exception("gensim failed network or timeout")

    embedding = create_embedding(force_download=True)

    embedding._download_google_news = MagicMock(return_value="fallback_model")  # type: ignore

    model = embedding.load()

    assert model == "fallback_model"
    embedding._download_google_news.assert_called_once()


@patch("xwhy.embeddings.word2vec.api.load")
def test_load_gensim_fails_no_fallback(mock_api_load: MagicMock) -> None:
    """Test RuntimeError is raised when gensim fails and model has no fallback."""
    mock_api_load.side_effect = Exception("gensim failed")

    embedding = Word2VecEmbedding(
        settings=cast(Settings, DummySettings()),
        model_name="glove.840B.300d",
        force_download=True,
    )

    with pytest.raises(
        RuntimeError, match=re.escape("Failed to load embedding model: glove.840B.300d")
    ):
        embedding.load()


# ---------------------------------------------------------------------
# GoogleNews Download and Extract logic
# ---------------------------------------------------------------------
@patch("xwhy.embeddings.word2vec.KeyedVectors.load_word2vec_format")
@patch("xwhy.embeddings.word2vec.Word2VecEmbedding._extract_gzip")
@patch("xwhy.embeddings.word2vec.Word2VecEmbedding._download_file")
def test_download_google_news_success(
    mock_download: MagicMock,
    mock_extract: MagicMock,
    mock_kv_load: MagicMock,
    tmp_path: Path,
) -> None:
    """Test full fallback download cycle for GoogleNews model."""
    embedding = create_embedding()
    fake_model = MagicMock()
    mock_kv_load.return_value = fake_model

    model_path = tmp_path / "model.bin"

    model = embedding._download_google_news(tmp_path, model_path)

    assert model is fake_model
    mock_download.assert_called_once()
    mock_extract.assert_called_once()
    mock_kv_load.assert_called_once()


@patch("xwhy.embeddings.word2vec.Word2VecEmbedding._download_file")
def test_download_google_news_failure(
    mock_download: MagicMock,
    tmp_path: Path,
) -> None:
    """Test GoogleNews download catches errors and raises RuntimeError."""
    embedding = create_embedding()
    mock_download.side_effect = OSError("Disk full or permission denied")

    model_path = tmp_path / "model.bin"

    with pytest.raises(RuntimeError, match="Failed to load GoogleNews model"):
        embedding._download_google_news(tmp_path, model_path)


@patch("xwhy.embeddings.word2vec.requests.get")
def test_download_file_success(mock_get: MagicMock, tmp_path: Path) -> None:
    """Test successful streaming download."""
    embedding = create_embedding()
    mock_response = MagicMock()

    large_chunk = b"\x00" * (100 * 1024 * 1024 + 1024)
    mock_response.iter_content.return_value = [large_chunk]
    mock_get.return_value = mock_response

    dest_path = tmp_path / "downloaded.bin"
    embedding._download_file("http://fake-url", dest_path)

    assert dest_path.exists()
    assert dest_path.stat().st_size == len(large_chunk)


@patch("xwhy.embeddings.word2vec.requests.get")
def test_download_file_too_small(mock_get: MagicMock, tmp_path: Path) -> None:
    """Test download raises OSError and cleans up if file is too small."""
    embedding = create_embedding()
    mock_response = MagicMock()

    mock_response.iter_content.return_value = [b"tiny corrupted file"]
    mock_get.return_value = mock_response

    dest_path = tmp_path / "corrupted.bin"

    with pytest.raises(OSError, match="Downloaded file too small"):
        embedding._download_file("http://fake-url", dest_path)

    assert not dest_path.exists()


def test_extract_gzip(tmp_path: Path) -> None:
    """Test gzip extraction helper."""
    embedding = create_embedding()

    src_path = tmp_path / "compressed.gz"
    dst_path = tmp_path / "extracted.txt"

    with gzip.open(src_path, "wb") as f:
        f.write(b"decompressed dummy data")

    embedding._extract_gzip(src_path, dst_path)

    assert dst_path.exists()
    assert dst_path.read_text() == "decompressed dummy data"


@patch("xwhy.embeddings.word2vec.requests.get")
def test_download_file_with_empty_chunk(mock_get: MagicMock, tmp_path: Path) -> None:
    """Test download file handles empty chunks correctly to satisfy branch coverage."""
    embedding = create_embedding()
    mock_response = MagicMock()

    large_chunk = b"\x00" * (100 * 1024 * 1024 + 1024)
    empty_chunk = b""

    mock_response.iter_content.return_value = [large_chunk, empty_chunk]
    mock_get.return_value = mock_response

    dest_path = tmp_path / "downloaded_with_empty.bin"
    embedding._download_file("http://fake-url", dest_path)

    assert dest_path.exists()
    assert dest_path.stat().st_size == len(large_chunk)
