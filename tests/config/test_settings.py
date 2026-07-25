"""Tests for xwhy.config.settings."""

from pathlib import Path

import pytest

from xwhy.config.settings import Settings


def test_default_embedding_cache_dir() -> None:
    """Default embedding cache directory should be configured."""
    settings = Settings()

    expected = Path("~/.cache/xwhy/embeddings").expanduser()

    assert settings.embedding_cache_dir == expected


def test_load_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should load OPENAI_API_KEY from environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    settings = Settings()

    assert settings.openai_api_key == "test-openai-key"


def test_load_gemini_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should load GEMINI_API_KEY from environment."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    settings = Settings()

    assert settings.gemini_api_key == "test-gemini-key"


def test_default_optional_keys_are_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional provider keys should be None if not configured."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    settings = Settings()

    assert settings.openai_api_key is None
    assert settings.gemini_api_key is None
    assert settings.huggingface_token is None


def test_huggingface_token_exported_to_hf_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Export huggingface_token to HF_TOKEN when token is present."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf-test-token")

    settings = Settings()

    assert settings.huggingface_token == "hf-test-token"
    assert settings.model_config  # touch config to satisfy linters if needed
    import os

    assert os.environ.get("HF_TOKEN") == "hf-test-token"


def test_huggingface_token_none_skips_hf_token_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not set HF_TOKEN when huggingface_token is None."""
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    settings = Settings()

    assert settings.huggingface_token is None
    import os

    assert "HF_TOKEN" not in os.environ
