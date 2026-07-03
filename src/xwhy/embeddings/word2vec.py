"""Word2Vec embedding implementation."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from typing import Any, ClassVar

import gensim.downloader as api
import requests
from gensim.models import KeyedVectors

from xwhy.config import Settings
from xwhy.embeddings import BaseEmbedding
from xwhy.logger import logger


class Word2VecEmbedding(BaseEmbedding):
    """Word2Vec embedding backend."""

    _MODEL_FILE_MAP: ClassVar[dict[str, dict[str, Any]]] = {
        "word2vec-google-news-300": {
            "file": "GoogleNews-vectors-negative300.bin",
            "binary": True,
            "gensim": True,
        },
        "glove.840B.300d": {
            "file": "glove.840B.300d.txt",
            "binary": False,
            "gensim": True,
        },
        "paragram_300_sl999": {
            "file": "paragram_300_sl999.txt",
            "binary": False,
            "gensim": True,
        },
    }

    def __init__(
        self,
        *,
        settings: Settings,
        model_name: str = "word2vec-google-news-300",
        force_download: bool = False,
    ) -> None:
        """Initialize Word2Vec embedding backend."""
        self._settings = settings
        self._model_name = model_name
        self._force_download = force_download
        self._model: KeyedVectors | None = None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def load(self) -> KeyedVectors:
        """Load embedding model with caching strategy."""
        if self._model is not None:
            return self._model

        cache_dir = self._get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        if self._model_name not in self._MODEL_FILE_MAP:
            raise ValueError(f"Unsupported model: {self._model_name}")

        model_info = self._MODEL_FILE_MAP[self._model_name]
        model_path = cache_dir / model_info["file"]

        # 1. cache
        if model_path.exists() and not self._force_download:
            logger.info(f"Loading from cache: {model_path}")
            self._model = KeyedVectors.load_word2vec_format(
                str(model_path),
                binary=model_info["binary"],
            )
            return self._model

        # 2. gensim
        if model_info["gensim"]:
            logger.info(f"Loading via gensim: {self._model_name}")
            try:
                model = api.load(self._model_name)

                logger.info(f"Saving to cache: {model_path}")
                model.save_word2vec_format(
                    str(model_path),
                    binary=model_info["binary"],
                )

                self._model = model
                return model

            except Exception:
                pass

        # 3. fallback download (GoogleNews only)
        if self._model_name == "word2vec-google-news-300":
            return self._download_google_news(cache_dir, model_path)

        raise RuntimeError(f"Failed to load model: {self._model_name}")

    def encode(self, text: str) -> list[float]:
        """Encode text using averaged word vectors."""
        model = self.load()
        words = text.split()

        vectors: list[list[float]] = [
            model[word].tolist() for word in words if word in model
        ]

        if not vectors:
            return [0.0] * 300

        dim = len(vectors[0])
        result = [0.0] * dim

        for vec in vectors:
            for i, val in enumerate(vec):
                result[i] += val

        return [x / len(vectors) for x in result]

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #

    def _get_cache_dir(self) -> Path:
        return self._settings.embedding_cache_dir

    def _download_google_news(
        self,
        cache_dir: Path,
        model_path: Path,
    ) -> KeyedVectors:
        """Download GoogleNews model."""
        gz_path = model_path.with_suffix(".bin.gz")

        url = (
            "https://public.ukp.informatik.tu-darmstadt.de/"
            "reimers/wordembeddings/GoogleNews-vectors-negative300.bin.gz"
        )

        try:
            self._download_file(url, gz_path)
            self._extract_gzip(gz_path, model_path)

            self._model = KeyedVectors.load_word2vec_format(
                str(model_path),
                binary=True,
            )

            return self._model

        except Exception as error:
            raise RuntimeError("Failed to load GoogleNews model") from error

    @staticmethod
    def _download_file(url: str, path: Path) -> None:
        """Download a file using streaming requests with a size sanity check.

        Args:
            url (str): File URL.
            path (str): Destination file path.

        Raises:
            IOError: If downloaded file is unexpectedly small.
            requests.exceptions.RequestException: If request fails.

        """
        logger.info("Downloading from %s => %s", url, path)

        path.parent.mkdir(parents=True, exist_ok=True)

        min_expected_size = 100 * 1024 * 1024
        downloaded_size = 0

        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            with path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)

            if downloaded_size < min_expected_size:
                raise OSError(
                    f"Downloaded file too small: {downloaded_size / (1024**2):.2f} MB"
                )

            logger.info(
                "Download completed: %.2f MB",
                downloaded_size / (1024**2),
            )

        except (requests.exceptions.RequestException, OSError):
            if path.exists():
                path.unlink()

            raise

    @staticmethod
    def _extract_gzip(src: Path, dst: Path) -> None:
        """Extract gzip file."""
        logger.info("Extracting %s...", src)
        with gzip.open(src, "rb") as f_in, dst.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
