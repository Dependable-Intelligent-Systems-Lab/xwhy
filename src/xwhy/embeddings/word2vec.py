"""Word2Vec embedding implementation."""

from __future__ import annotations

import gzip
import shutil
import zipfile
from pathlib import Path
from typing import Any, ClassVar

import gdown
import gensim.downloader as api
import requests
from gensim.models import KeyedVectors
from tqdm.auto import tqdm

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
            "no_header": False,
            "google_id": "1vAjPzr5R1RQiuh9NOgHVFGhBmCuYWnJU",
        },
        "glove.840B.300d": {
            "file": "glove.840B.300d.txt",
            "binary": False,
            "gensim": True,
            "no_header": True,
            "google_id": "19cJAKkgrYAiT1gU-OnWTN6GaGcd7pLZI",
        },
        "paragram_300_sl999": {
            "file": "paragram_300_sl999.txt",
            "binary": False,
            "gensim": True,
            "no_header": True,
            "google_id": "1c-16FP0jvaJeyaM8JcqqKPdoVw7uqVkK",
        },
        "paragram-300-WS353": {
            "file": "paragram_300_ws353.txt",
            "binary": False,
            "gensim": True,
            "no_header": True,
            "google_id": "1bBLz6F6MJ_qx9xnSZUgI8W0QZhJYgBdu",
        },
    }

    def __init__(
        self,
        *,
        settings: Settings,
        model_name: str = "word2vec-google-news-300",
        force_download: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize Word2Vec embedding backend."""
        self._settings = settings
        self._model_name = model_name
        self._force_download = force_download
        self._model: KeyedVectors | None = None

    def load(self) -> KeyedVectors:
        """Load embedding model with caching strategy."""
        if self._model is not None:
            return self._model

        cache_dir = self._get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        if self._model_name not in self._MODEL_FILE_MAP:
            raise ValueError(f"Unsupported model: {self._model_name}")

        model_info = self._MODEL_FILE_MAP[self._model_name]
        original_model_path = cache_dir / model_info["file"]

        # 1. cache
        bin_cache_path = original_model_path.with_suffix(".bin")

        if bin_cache_path.exists() and not self._force_download:
            logger.debug(
                f"Loading embedding model from fast binary cache: {bin_cache_path}"
            )
            self._model = KeyedVectors.load_word2vec_format(
                str(bin_cache_path),
                binary=True,
            )
            return self._model

        # 2. gdown direct download option
        gdown_model = self._try_gdown_download(
            model_info=model_info,
            bin_cache_path=bin_cache_path,
        )
        if gdown_model is not None:
            return gdown_model

        # 3. gensim
        if model_info["gensim"]:
            logger.debug(f"Loading embedding model via gensim: {self._model_name}")
            try:
                model = api.load(self._model_name)
                logger.info(f"Saving to fast binary cache: {bin_cache_path}")
                model.save_word2vec_format(str(bin_cache_path), binary=True)
                self._model = model
                return model

            except Exception:
                pass

        # 4. fallback download for specific models
        if self._model_name == "word2vec-google-news-300":
            return self._download_google_news(cache_dir, bin_cache_path)
        elif self._model_name == "glove.840B.300d":
            return self._download_glove(cache_dir, bin_cache_path, original_model_path)
        elif self._model_name in ("paragram_300_sl999", "paragram-300-WS353"):
            return self._download_paragram(
                cache_dir, bin_cache_path, original_model_path
            )

        raise RuntimeError(f"Failed to load embedding model: {self._model_name}")

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

    def _get_cache_dir(self) -> Path:
        cache_dir = self._settings.embedding_cache_dir
        return cache_dir if cache_dir else Path(Path.home() / ".cache/xwhy/embeddings")

    def _try_gdown_download(
        self,
        model_info: dict[str, Any],
        bin_cache_path: Path,
    ) -> KeyedVectors | None:
        """Attempt to download pre-converted binary model from Google Drive."""
        google_id: str | None = model_info.get("google_id")

        if not google_id or self._force_download:
            return None

        logger.debug(f"Attempting binary download via gdown for ID: {google_id}")

        try:
            # Download directly to bin_cache_path since the GDrive file is already .bin
            gdown.download(id=google_id, output=str(bin_cache_path), quiet=False)

            # All files on GDrive are pre-converted binary format. Force binary=True.
            model = KeyedVectors.load_word2vec_format(
                str(bin_cache_path),
                binary=True,
                no_header=False,
            )

            self._model = model
            return self._model

        except Exception as err:
            logger.debug(f"gdown route failed for {self._model_name}: {err}")
            if bin_cache_path.exists():
                bin_cache_path.unlink()
            return None

    def _download_google_news(
        self,
        cache_dir: Path,
        bin_cache_path: Path,
    ) -> KeyedVectors:
        """Download GoogleNews model."""
        gz_path = cache_dir / "GoogleNews-vectors-negative300.bin.gz"
        url = (
            "https://public.ukp.informatik.tu-darmstadt.de/"
            "reimers/wordembeddings/GoogleNews-vectors-negative300.bin.gz"
        )

        try:
            if not gz_path.exists():
                self._download_file(url, gz_path)

            self._extract_gzip(gz_path, bin_cache_path)

            self._model = KeyedVectors.load_word2vec_format(
                str(bin_cache_path),
                binary=True,
            )

            return self._model

        except Exception as error:
            raise RuntimeError("Failed to load GoogleNews model") from error

    def _download_glove(
        self, cache_dir: Path, bin_path: Path, txt_path: Path
    ) -> KeyedVectors:
        """Download, clean, and convert GloVe model to binary."""
        zip_path = cache_dir / "glove.840B.300d.zip"
        url = "http://nlp.stanford.edu/data/glove.840B.300d.zip"

        try:
            if not zip_path.exists():
                self._download_file(url, zip_path)

            if not txt_path.exists():
                logger.info("Extracting GloVe zip file...")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(cache_dir)

            cleaned_path = txt_path.with_name(txt_path.name + ".cleaned")
            if not cleaned_path.exists():
                seen = set()
                logger.info("De-duplicating GloVe file...")
                with (
                    open(txt_path, encoding="utf-8", errors="ignore") as fin,
                    open(cleaned_path, "w", encoding="utf-8") as fout,
                ):
                    for line in tqdm(fin, desc="Cleaning GloVe"):
                        word = line.split(maxsplit=1)[0]
                        if word not in seen:
                            fout.write(line)
                            seen.add(word)

            logger.info("Loading cleaned GloVe text...")
            model = KeyedVectors.load_word2vec_format(
                str(cleaned_path), binary=False, no_header=True
            )

            logger.info("Converting GloVe to fast binary format...")
            model.save_word2vec_format(str(bin_path), binary=True)

            for path_to_remove in [cleaned_path, txt_path, zip_path]:
                if path_to_remove.exists():
                    path_to_remove.unlink()

            self._model = model
            return self._model
        except Exception as error:
            raise RuntimeError("Failed to load GloVe model") from error

    def _download_paragram(
        self, cache_dir: Path, bin_path: Path, txt_path: Path
    ) -> KeyedVectors:
        """Download and convert Paragram models to binary."""
        gdrive_ids = {
            "paragram_300_sl999": "0B9w48e1rj-MOck1fRGxaZW1LU2M",
            "paragram-300-WS353": "0B9w48e1rj-MOLVdZRzFfTlNsem8",
        }
        file_id = gdrive_ids[self._model_name]

        try:
            if not txt_path.exists():
                logger.info(f"Downloading {self._model_name} via gdown...")

                temp_path = txt_path.with_suffix(".tmp")
                gdown.download(id=file_id, output=str(temp_path), quiet=False)

                if zipfile.is_zipfile(str(temp_path)):
                    logger.info("Extracting zip archive...")
                    with zipfile.ZipFile(str(temp_path), "r") as zf:
                        # Find the text file name inside the zip archive
                        txt_filename = next(
                            (name for name in zf.namelist() if name.endswith(".txt")),
                            None,
                        )
                        if not txt_filename:
                            raise FileNotFoundError("No .txt file found inside zip.")

                        # Manually extract with Python to bypass CRC errors
                        with open(txt_path, "wb") as f_out:
                            try:
                                with zf.open(txt_filename) as f_in:
                                    shutil.copyfileobj(f_in, f_out)
                            except zipfile.BadZipFile:
                                logger.warning("Ignored CRC error.")
                    temp_path.unlink()
                else:
                    temp_path.rename(txt_path)

            logger.info("Sanitizing text file and adding header...")
            clean_txt_path = txt_path.with_suffix(".clean.txt")
            expected_dim = 300

            # Step 1: Count valid lines (lines with exactly 301
            # parts: 1 word + 300 numbers)
            valid_lines = 0
            with open(txt_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if len(line.rstrip("\n").split(" ")) == expected_dim + 1:
                        valid_lines += 1

            # Step 2: Rewrite the file with only valid lines
            # + add standard word2vec header
            with (
                open(txt_path, encoding="utf-8", errors="ignore") as f_in,
                open(clean_txt_path, "w", encoding="utf-8") as f_out,
            ):
                f_out.write(f"{valid_lines} {expected_dim}\n")
                for line in f_in:
                    if len(line.rstrip("\n").split(" ")) == expected_dim + 1:
                        f_out.write(line)

            logger.info(f"Loading {self._model_name} text into Gensim...")
            # Since we added the header to the cleaned file
            # ourselves, set no_header=False
            model = KeyedVectors.load_word2vec_format(
                str(clean_txt_path),
                binary=False,
                no_header=False,
                unicode_errors="ignore",
            )

            logger.info(f"Converting {self._model_name} to binary...")
            model.save_word2vec_format(str(bin_path), binary=True)

            # Clean up both text files (original and cleaned) to free up disk space
            if txt_path.exists():
                txt_path.unlink()
            if clean_txt_path.exists():
                clean_txt_path.unlink()

            self._model = model
            return self._model
        except Exception as error:
            raise RuntimeError(f"Failed to load {self._model_name}") from error

    @staticmethod
    def _download_file(url: str, path: Path) -> None:
        """Download a file using streaming requests."""
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.debug(f"Attempting direct download from {url}...")
            response = requests.get(url, stream=True, timeout=600)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with (
                path.open("wb") as file,
                tqdm(
                    desc=path.name,
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        size = file.write(chunk)
                        bar.update(size)

            if path.stat().st_size < 100:
                raise OSError("Downloaded file too small")

        except Exception:
            if path.exists():
                path.unlink()

            raise

    @staticmethod
    def _extract_gzip(src: Path, dst: Path) -> None:
        """Extract gzip file."""
        logger.debug("Extracting %s...", src)
        with gzip.open(src, "rb") as f_in, dst.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
