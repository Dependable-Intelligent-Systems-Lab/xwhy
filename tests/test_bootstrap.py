"""Unit tests for bootstrap module."""

from unittest.mock import MagicMock, patch

import pytest

from xwhy.bootstrap import (
    _build_gemini_provider,
    _build_glove,
    _build_openai_provider,
    _build_paragram,
    _build_word2vec,
)


# ---------------------------------------------------------------------
# OpenAI Provider Test
# ---------------------------------------------------------------------
@patch("xwhy.bootstrap.ProviderFactory.create")
@patch("openai.OpenAI")
def test_build_openai_provider(
    mock_openai_class: MagicMock,
    mock_factory_create: MagicMock,
) -> None:
    """Verify that OpenAI provider builder instantiates client and factory correctly."""
    mock_client_instance = MagicMock()
    mock_openai_class.return_value = mock_client_instance

    mock_provider_instance = MagicMock()
    mock_factory_create.return_value = mock_provider_instance

    result = _build_openai_provider()

    mock_openai_class.assert_called_once()
    mock_factory_create.assert_called_once_with(
        provider=pytest.importorskip("xwhy.providers.types").ProviderType.OPENAI,
        client=mock_client_instance,
    )
    assert result is mock_provider_instance


# ---------------------------------------------------------------------
# Gemini Provider Test
# ---------------------------------------------------------------------
@patch("xwhy.bootstrap.ProviderFactory.create")
@patch("google.genai.Client")
def test_build_gemini_provider(
    mmock_gemini_class: MagicMock,
    mock_factory_create: MagicMock,
) -> None:
    """Verify that Google provider builder instantiates client and factory correctly."""
    mock_client_instance = MagicMock()
    mmock_gemini_class.return_value = mock_client_instance

    mock_provider_instance = MagicMock()
    mock_factory_create.return_value = mock_provider_instance

    result = _build_gemini_provider()

    mmock_gemini_class.assert_called_once()
    mock_factory_create.assert_called_once_with(
        provider=pytest.importorskip("xwhy.providers.types").ProviderType.GEMINI,
        client=mock_client_instance,
    )
    assert result is mock_provider_instance


# ---------------------------------------------------------------------
# Anthropic Provider Test
# ---------------------------------------------------------------------
@patch("xwhy.bootstrap.ProviderFactory.create")
@patch("anthropic.Anthropic")
def test_build_anthropic_provider(
    mock_anthropic_class: MagicMock,
    mock_factory_create: MagicMock,
) -> None:
    """Verify that Anthropic provider builder instantiates client correctly."""
    mock_client_instance = MagicMock()
    mock_anthropic_class.return_value = mock_client_instance

    mock_provider_instance = MagicMock()
    mock_factory_create.return_value = mock_provider_instance

    from xwhy.bootstrap import _build_anthropic_provider

    result = _build_anthropic_provider()

    mock_anthropic_class.assert_called_once()
    mock_factory_create.assert_called_once_with(
        provider=pytest.importorskip("xwhy.providers.types").ProviderType.ANTHROPIC,
        client=mock_client_instance,
    )
    assert result is mock_provider_instance


# ---------------------------------------------------------------------
# HuggingFace Provider Test
# ---------------------------------------------------------------------
@patch("xwhy.bootstrap.ProviderFactory.create")
@patch("huggingface_hub.InferenceClient")
def test_build_huggingface_provider(
    mock_huggingface_class: MagicMock,
    mock_factory_create: MagicMock,
) -> None:
    """Verify HuggingFace provider initialization."""
    mock_client_instance = MagicMock()
    mock_huggingface_class.return_value = mock_client_instance

    mock_provider_instance = MagicMock()
    mock_factory_create.return_value = mock_provider_instance

    from xwhy.bootstrap import _build_huggingface_provider

    result = _build_huggingface_provider()

    mock_huggingface_class.assert_called_once()
    mock_factory_create.assert_called_once_with(
        provider=pytest.importorskip("xwhy.providers.types").ProviderType.HUGGINGFACE,
        client=mock_client_instance,
    )
    assert result is mock_provider_instance


# ---------------------------------------------------------------------
# Word2Vec/Embeddings Builders Tests
# ---------------------------------------------------------------------
@patch("xwhy.bootstrap.Word2VecEmbedding")
def test_build_word2vec(mock_word2vec_embedding: MagicMock) -> None:
    """Verify Word2Vec builder passes correct model_name and kwargs."""
    mock_instance = MagicMock()
    mock_word2vec_embedding.return_value = mock_instance

    result = _build_word2vec(force_download=True, custom_param="test")

    mock_word2vec_embedding.assert_called_once_with(
        model_name="word2vec-google-news-300",
        settings=pytest.importorskip("xwhy.config").settings,
        force_download=True,
        custom_param="test",
    )
    assert result is mock_instance


@patch("xwhy.bootstrap.Word2VecEmbedding")
def test_build_glove(mock_word2vec_embedding: MagicMock) -> None:
    """Verify GloVe builder passes correct model_name and kwargs."""
    mock_instance = MagicMock()
    mock_word2vec_embedding.return_value = mock_instance

    result = _build_glove(force_download=False)

    mock_word2vec_embedding.assert_called_once_with(
        model_name="glove.840B.300d",
        settings=pytest.importorskip("xwhy.config").settings,
        force_download=False,
    )
    assert result is mock_instance


@patch("xwhy.bootstrap.Word2VecEmbedding")
def test_build_paragram(mock_word2vec_embedding: MagicMock) -> None:
    """Verify Paragram builder passes correct model_name and kwargs."""
    mock_instance = MagicMock()
    mock_word2vec_embedding.return_value = mock_instance

    result = _build_paragram()

    mock_word2vec_embedding.assert_called_once_with(
        model_name="paragram_300_sl999",
        settings=pytest.importorskip("xwhy.config").settings,
    )
    assert result is mock_instance
