"""Tests for Word2VecEmbedding."""

from __future__ import annotations

import gzip
import re
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
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

    dummy_model_name = "test-model-no-fallback"

    with patch.dict(
        "xwhy.embeddings.word2vec.Word2VecEmbedding._MODEL_FILE_MAP",
        {dummy_model_name: {"file": "test.txt", "gensim": True}},
    ):
        embedding = Word2VecEmbedding(
            settings=cast(Settings, DummySettings()),
            model_name=dummy_model_name,
            force_download=True,
        )

        with pytest.raises(
            RuntimeError,
            match=re.escape(f"Failed to load embedding model: {dummy_model_name}"),
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


# ---------------------------------------------------------------------
# Load logic edge cases - Fallbacks (GloVe & Paragram)
# ---------------------------------------------------------------------
@patch("xwhy.embeddings.word2vec.api.load")
def test_load_gensim_fails_fallback_to_glove(mock_api_load: MagicMock) -> None:
    """Test fallback mechanism when gensim fails for GloVe model."""
    mock_api_load.side_effect = Exception("gensim failed network or timeout")

    embedding = Word2VecEmbedding(
        settings=cast(Settings, DummySettings()),
        model_name="glove.840B.300d",
        force_download=True,
    )

    embedding._download_glove = MagicMock(return_value="fallback_glove_model")  # type: ignore

    model = embedding.load()

    assert model == "fallback_glove_model"
    embedding._download_glove.assert_called_once()


@patch("xwhy.embeddings.word2vec.api.load")
def test_load_gensim_fails_fallback_to_paragram(mock_api_load: MagicMock) -> None:
    """Test fallback mechanism when gensim fails for Paragram model."""
    mock_api_load.side_effect = Exception("gensim failed network or timeout")

    embedding = Word2VecEmbedding(
        settings=cast(Settings, DummySettings()),
        model_name="paragram_300_sl999",
        force_download=True,
    )

    embedding._download_paragram = MagicMock(return_value="fallback_paragram_model")  # type: ignore

    model = embedding.load()

    assert model == "fallback_paragram_model"
    embedding._download_paragram.assert_called_once()


# ---------------------------------------------------------------------
# GoogleNews Cache Exists bypass
# ---------------------------------------------------------------------
@patch("xwhy.embeddings.word2vec.KeyedVectors.load_word2vec_format")
@patch("xwhy.embeddings.word2vec.Word2VecEmbedding._extract_gzip")
@patch("xwhy.embeddings.word2vec.Word2VecEmbedding._download_file")
def test_download_google_news_gz_exists(
    mock_download_file: MagicMock,
    mock_extract: MagicMock,
    mock_kv_load: MagicMock,
    tmp_path: Path,
) -> None:
    """Test that download is bypassed if .gz file already exists."""
    embedding = create_embedding()
    gz_path = tmp_path / "GoogleNews-vectors-negative300.bin.gz"
    gz_path.parent.mkdir(parents=True, exist_ok=True)
    gz_path.write_bytes(b"dummy")  # Create the file so exists() returns True

    bin_path = tmp_path / "model.bin"

    embedding._download_google_news(tmp_path, bin_path)

    mock_download_file.assert_not_called()
    mock_extract.assert_called_once_with(gz_path, bin_path)


# ---------------------------------------------------------------------
# GloVe Download and Extract logic
# ---------------------------------------------------------------------
@patch("xwhy.embeddings.word2vec.KeyedVectors.load_word2vec_format")
@patch("xwhy.embeddings.word2vec.zipfile.ZipFile")
@patch("xwhy.embeddings.word2vec.Word2VecEmbedding._download_file")
def test_download_glove_success(
    mock_download: MagicMock,
    mock_zip: MagicMock,
    mock_kv_load: MagicMock,
    tmp_path: Path,
) -> None:
    """Test full fallback download, extract, and clean cycle for GloVe model."""
    embedding = Word2VecEmbedding(
        settings=cast(Settings, DummySettings()),
        model_name="glove.840B.300d",
    )
    txt_path = tmp_path / "glove.840B.300d.txt"
    bin_path = tmp_path / "model.bin"

    def extract_side_effect(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        # Simulate extraction and write duplicated lines to test deduplication
        txt_path.write_text(
            "word1 0.1 0.2\nword1 0.1 0.2\nword2 0.3 0.4\n", encoding="utf-8"
        )

    mock_zip.return_value.__enter__.return_value.extractall.side_effect = (
        extract_side_effect
    )

    fake_model = MagicMock()
    mock_kv_load.return_value = fake_model

    model = embedding._download_glove(tmp_path, bin_path, txt_path)

    assert model is fake_model
    mock_download.assert_called_once()
    mock_kv_load.assert_called_once()
    fake_model.save_word2vec_format.assert_called_once_with(str(bin_path), binary=True)

    # Assert cleanup was successful
    assert not txt_path.exists()


@patch("xwhy.embeddings.word2vec.KeyedVectors.load_word2vec_format")
@patch("xwhy.embeddings.word2vec.Word2VecEmbedding._download_file")
def test_download_glove_cache_exists(
    mock_download: MagicMock,
    mock_kv_load: MagicMock,
    tmp_path: Path,
) -> None:
    """Test GloVe download logic when cached files already exist."""
    embedding = Word2VecEmbedding(
        settings=cast(Settings, DummySettings()),
        model_name="glove.840B.300d",
    )
    txt_path = tmp_path / "glove.840B.300d.txt"
    zip_path = tmp_path / "glove.840B.300d.zip"
    clean_path = tmp_path / "glove.840B.300d.txt.cleaned"

    txt_path.write_text("dummy")
    zip_path.write_text("dummy")
    clean_path.write_text("dummy")

    embedding._download_glove(tmp_path, tmp_path / "bin", txt_path)

    mock_download.assert_not_called()  # Should skip download


@patch("xwhy.embeddings.word2vec.Word2VecEmbedding._download_file")
def test_download_glove_failure(mock_download: MagicMock, tmp_path: Path) -> None:
    """Test GloVe download catches errors and raises RuntimeError."""
    embedding = Word2VecEmbedding(
        settings=cast(Settings, DummySettings()),
        model_name="glove.840B.300d",
    )
    mock_download.side_effect = Exception("Network failure")

    with pytest.raises(RuntimeError, match="Failed to load GloVe model"):
        embedding._download_glove(tmp_path, tmp_path / "bin", tmp_path / "txt")


@patch("gdown.download")
@patch("xwhy.embeddings.word2vec.KeyedVectors.load_word2vec_format")
def test_try_gdown_download_success(
    mock_kv_load: MagicMock, mock_download: MagicMock, tmp_path: Path
) -> None:
    """Test successful fast binary download from Google Drive."""
    embedding = Word2VecEmbedding(settings=cast(Any, DummySettings()))
    bin_path = tmp_path / "model.bin"
    model_info = {"google_id": "test_gdrive_id"}

    def download_side_effect(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        output = kwargs.get("output") or args[1]
        Path(output).write_bytes(b"dummy binary data")

    mock_download.side_effect = download_side_effect

    fake_model = MagicMock()
    mock_kv_load.return_value = fake_model

    result = embedding._try_gdown_download(model_info, bin_path)

    assert result is fake_model
    assert embedding._model is fake_model
    mock_download.assert_called_once_with(
        id="test_gdrive_id", output=str(bin_path), quiet=False
    )
    mock_kv_load.assert_called_once_with(str(bin_path), binary=True, no_header=False)


def test_try_gdown_download_no_google_id(tmp_path: Path) -> None:
    """Test skip download if google_id is missing."""
    embedding = Word2VecEmbedding(settings=cast(Any, DummySettings()))
    bin_path = tmp_path / "model.bin"
    model_info = {"no_google_id_here": True}

    result = embedding._try_gdown_download(model_info, bin_path)
    assert result is None


def test_try_gdown_download_force_download(tmp_path: Path) -> None:
    """Test skip fast gdown download if force_download is True."""
    embedding = Word2VecEmbedding(
        settings=cast(Any, DummySettings()), force_download=True
    )
    bin_path = tmp_path / "model.bin"
    model_info = {"google_id": "test_gdrive_id"}

    result = embedding._try_gdown_download(model_info, bin_path)
    assert result is None


def test_try_gdown_download_exception_with_cleanup(tmp_path: Path) -> None:
    """Test exception handling during download and proper cleanup of corrupted file."""
    embedding = Word2VecEmbedding(settings=cast(Any, DummySettings()))
    bin_path = tmp_path / "model.bin"
    bin_path.write_bytes(b"corrupted partial data")  # Simulate partial download
    model_info = {"google_id": "test_gdrive_id"}

    mock_gdown = MagicMock()
    mock_gdown.download.side_effect = Exception("Network Error")

    with patch.dict("sys.modules", {"gdown": mock_gdown}):
        result = embedding._try_gdown_download(model_info, bin_path)

    assert result is None
    assert not bin_path.exists()  # Ensure cleanup logic ran


def test_try_gdown_download_exception_no_cleanup_needed(tmp_path: Path) -> None:
    """Test exception handling when file was never created (no cleanup needed)."""
    embedding = Word2VecEmbedding(settings=cast(Any, DummySettings()))
    bin_path = tmp_path / "model.bin"
    model_info = {"google_id": "test_gdrive_id"}

    mock_gdown = MagicMock()
    mock_gdown.download.side_effect = Exception("Connection Failed")

    with patch.dict("sys.modules", {"gdown": mock_gdown}):
        result = embedding._try_gdown_download(model_info, bin_path)

    assert result is None
    assert not bin_path.exists()


@patch.object(Word2VecEmbedding, "_try_gdown_download")
def test_load_gdown_shortcut_success(mock_try_gdown: MagicMock, tmp_path: Path) -> None:
    """Test load method returns immediately if gdown shortcut succeeds."""
    embedding = Word2VecEmbedding(
        settings=cast(Any, DummySettings()), model_name="glove.840B.300d"
    )

    fake_model = MagicMock()
    mock_try_gdown.return_value = fake_model

    with patch.object(embedding, "_get_cache_dir", return_value=tmp_path):
        result = embedding.load()

    assert result is fake_model
    mock_try_gdown.assert_called_once()


# =====================================================================
# Paragram Download and Extract logic
# =====================================================================
@patch("gdown.download")
@patch("xwhy.embeddings.word2vec.KeyedVectors.load_word2vec_format")
def test_download_paragram_success_zip(
    mock_kv_load: MagicMock,
    mock_download: MagicMock,
    tmp_path: Path,
) -> None:
    """Test successful Paragram download and extraction from zip."""
    embedding = Word2VecEmbedding(
        settings=cast(Any, DummySettings()),
        model_name="paragram_300_sl999",
    )
    txt_path = tmp_path / "paragram.txt"
    bin_path = tmp_path / "model.bin"

    def gdown_side_effect(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        output = kwargs.get("output") or args[1]
        with zipfile.ZipFile(output, "w") as zf:
            valid_line = "word1 " + " ".join(["0.1"] * 300) + "\n"
            zf.writestr("model.txt", valid_line)

    mock_download.side_effect = gdown_side_effect
    fake_model = MagicMock()
    mock_kv_load.return_value = fake_model

    model = embedding._download_paragram(tmp_path, bin_path, txt_path)

    assert model is fake_model
    fake_model.save_word2vec_format.assert_called_once_with(str(bin_path), binary=True)
    assert not txt_path.exists()


@patch("gdown.download")
@patch("xwhy.embeddings.word2vec.KeyedVectors.load_word2vec_format")
def test_download_paragram_success_txt_no_zip(
    mock_kv_load: MagicMock,
    mock_download: MagicMock,
    tmp_path: Path,
) -> None:
    """Test successful Paragram download when the downloaded file is a raw text."""
    embedding = Word2VecEmbedding(
        settings=cast(Any, DummySettings()),
        model_name="paragram_300_sl999",
    )
    txt_path = tmp_path / "paragram.txt"
    bin_path = tmp_path / "model.bin"

    def gdown_side_effect(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        output = kwargs.get("output") or args[1]
        valid_line = "word1 " + " ".join(["0.1"] * 300) + "\n"
        Path(output).write_text(valid_line, encoding="utf-8")

    mock_download.side_effect = gdown_side_effect
    fake_model = MagicMock()
    mock_kv_load.return_value = fake_model

    embedding._download_paragram(tmp_path, bin_path, txt_path)

    mock_kv_load.assert_called_once()
    fake_model.save_word2vec_format.assert_called_once_with(str(bin_path), binary=True)


@patch("gdown.download")
@patch("xwhy.embeddings.word2vec.shutil.copyfileobj")
@patch("xwhy.embeddings.word2vec.KeyedVectors.load_word2vec_format")
def test_download_paragram_bad_zip_handling(
    mock_kv_load: MagicMock,
    mock_copy: MagicMock,
    mock_download: MagicMock,
    tmp_path: Path,
) -> None:
    """Test Paragram extraction ignores BadZipFile (CRC error) safely."""
    embedding = Word2VecEmbedding(
        settings=cast(Any, DummySettings()),
        model_name="paragram_300_sl999",
    )

    def gdown_side_effect(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        output = kwargs.get("output") or args[1]
        with zipfile.ZipFile(output, "w") as zf:
            zf.writestr("model.txt", "dummy")

    mock_download.side_effect = gdown_side_effect
    mock_copy.side_effect = zipfile.BadZipFile("Simulated CRC Error")

    fake_model = MagicMock()
    mock_kv_load.return_value = fake_model

    embedding._download_paragram(tmp_path, tmp_path / "bin", tmp_path / "txt")

    mock_kv_load.assert_called_once()
    fake_model.save_word2vec_format.assert_called_once()


@patch("gdown.download")
def test_download_paragram_no_txt_in_zip(
    mock_download: MagicMock, tmp_path: Path
) -> None:
    """Test Paragram download raises error if zip contains no text file."""
    embedding = Word2VecEmbedding(
        settings=cast(Any, DummySettings()),
        model_name="paragram_300_sl999",
    )

    def gdown_side_effect(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        output = kwargs.get("output") or args[1]
        with zipfile.ZipFile(output, "w") as zf:
            zf.writestr("model.bin", "dummy")  # No .txt file

    mock_download.side_effect = gdown_side_effect

    with pytest.raises(RuntimeError, match="Failed to load paragram_300_sl999"):
        embedding._download_paragram(tmp_path, tmp_path / "bin", tmp_path / "txt")


@patch("xwhy.embeddings.word2vec.KeyedVectors.load_word2vec_format")
def test_download_paragram_cache_exists(
    mock_kv_load: MagicMock,
    tmp_path: Path,
) -> None:
    """Test Paragram bypasses download if txt_path already exists."""
    embedding = Word2VecEmbedding(
        settings=cast(Any, DummySettings()),
        model_name="paragram_300_sl999",
    )
    txt_path = tmp_path / "paragram.txt"
    valid_line = "word1 " + " ".join(["0.1"] * 300) + "\n"
    txt_path.write_text(valid_line)  # Create cache with correct dimension length

    mock_gdown = MagicMock()
    fake_model = MagicMock()
    mock_kv_load.return_value = fake_model

    with patch.dict("sys.modules", {"gdown": mock_gdown}):
        embedding._download_paragram(tmp_path, tmp_path / "bin", txt_path)

    mock_gdown.download.assert_not_called()
    mock_kv_load.assert_called_once()


@patch("gdown.download")
def test_download_paragram_failure(mock_download: MagicMock, tmp_path: Path) -> None:
    """Test Paragram download general exception handling."""
    embedding = Word2VecEmbedding(
        settings=cast(Any, DummySettings()),
        model_name="paragram_300_sl999",
    )

    mock_download.side_effect = Exception("gdown network failure")

    with pytest.raises(RuntimeError, match="Failed to load paragram_300_sl999"):
        embedding._download_paragram(tmp_path, tmp_path / "bin", tmp_path / "txt")


# =====================================================================
# Paragram Cleanup Coverage (True & False branches)
# =====================================================================
@patch("gdown.download")
@patch("xwhy.embeddings.word2vec.KeyedVectors.load_word2vec_format")
def test_download_paragram_cleanup_files_exist_true_branch(
    mock_kv_load: MagicMock,
    mock_download: MagicMock,
    tmp_path: Path,
) -> None:
    """Test the True branches of the cleanup block (both files exist)."""
    embedding = Word2VecEmbedding(
        settings=cast(Any, DummySettings()),
        model_name="paragram_300_sl999",
    )
    txt_path = tmp_path / "paragram.txt"
    bin_path = tmp_path / "model.bin"

    def gdown_side_effect(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        output = kwargs.get("output") or args[1]
        with zipfile.ZipFile(output, "w") as zf:
            valid_line = "word1 " + " ".join(["0.1"] * 300) + "\n"
            zf.writestr("model.txt", valid_line)

    mock_download.side_effect = gdown_side_effect

    def load_side_effect(fname: str, *args: Any, **kwargs: Any) -> MagicMock:  # noqa: ANN401
        Path(fname).write_text("dummy clean content")
        txt_path.write_text("dummy txt content")
        return MagicMock()

    mock_kv_load.side_effect = load_side_effect

    embedding._download_paragram(tmp_path, bin_path, txt_path)

    assert not txt_path.exists()

    # Ensure all .txt and .clean.txt files were wiped out
    txt_files = list(tmp_path.glob("*.txt*"))
    assert len(txt_files) == 0


@patch("gdown.download")
@patch("xwhy.embeddings.word2vec.KeyedVectors.load_word2vec_format")
def test_download_paragram_cleanup_files_missing_false_branch(
    mock_kv_load: MagicMock,
    mock_download: MagicMock,
    tmp_path: Path,
) -> None:
    """Test the False branches of the cleanup block (files missing)."""
    embedding = Word2VecEmbedding(
        settings=cast(Any, DummySettings()),
        model_name="paragram_300_sl999",
    )
    txt_path = tmp_path / "paragram.txt"
    bin_path = tmp_path / "model.bin"

    def gdown_side_effect(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        output = kwargs.get("output") or args[1]
        with zipfile.ZipFile(output, "w") as zf:
            valid_line = "word1 " + " ".join(["0.1"] * 300) + "\n"
            zf.writestr("model.txt", valid_line)

    mock_download.side_effect = gdown_side_effect

    def load_side_effect(fname: str, *args: Any, **kwargs: Any) -> MagicMock:  # noqa: ANN401
        # Simulate missing files right before cleanup logic
        if Path(fname).exists():
            Path(fname).unlink()
        if txt_path.exists():
            txt_path.unlink()
        return MagicMock()

    mock_kv_load.side_effect = load_side_effect

    embedding._download_paragram(tmp_path, bin_path, txt_path)

    assert not txt_path.exists()


@patch("gdown.download")
@patch("xwhy.embeddings.word2vec.KeyedVectors.load_word2vec_format")
def test_download_paragram_dimension_filtering(
    mock_kv_load: MagicMock,
    mock_download: MagicMock,
    tmp_path: Path,
) -> None:
    """Test Step 1 and Step 2 correctly filter lines based on expected dimensions."""
    embedding = Word2VecEmbedding(
        settings=cast(Any, DummySettings()),
        model_name="paragram_300_sl999",
    )
    txt_path = tmp_path / "paragram.txt"
    bin_path = tmp_path / "model.bin"

    def gdown_side_effect(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        output = kwargs.get("output") or args[1]

        valid_line = "word1 " + " ".join(["0.1"] * 300) + "\n"
        invalid_short = "word2 " + " ".join(["0.2"] * 299) + "\n"
        invalid_long = "word3 " + " ".join(["0.3"] * 301) + "\n"
        empty_line = "\n"

        file_content = invalid_short + valid_line + empty_line + invalid_long

        with zipfile.ZipFile(output, "w") as zf:
            zf.writestr("model.txt", file_content)

    mock_download.side_effect = gdown_side_effect

    def load_side_effect(fname: str, *args: Any, **kwargs: Any) -> MagicMock:  # noqa: ANN401
        content = Path(fname).read_text(encoding="utf-8")
        lines = content.splitlines()

        assert lines[0] == "1 300", f"Expected header '1 300', but got '{lines[0]}'"

        assert len(lines) == 2, "File should only contain the header and 1 valid line."
        assert lines[1].startswith("word1 "), (
            "Only 'word1' should have passed the filter."
        )

        return MagicMock()

    mock_kv_load.side_effect = load_side_effect

    embedding._download_paragram(tmp_path, bin_path, txt_path)

    mock_kv_load.assert_called_once()
