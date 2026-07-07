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
# Z.ai Provider Test
# ---------------------------------------------------------------------
@patch("xwhy.bootstrap.ProviderFactory.create")
@patch("openai.OpenAI")
@patch("xwhy.bootstrap.settings")
def test_build_zai_provider(
    mock_settings: MagicMock,
    mock_openai_class: MagicMock,
    mock_factory_create: MagicMock,
) -> None:
    """Verify Z.ai provider initialization with correct base_url."""
    mock_settings.zai_api_key = "dummy_zai_key"
    mock_client_instance = MagicMock()
    mock_openai_class.return_value = mock_client_instance

    mock_provider_instance = MagicMock()
    mock_factory_create.return_value = mock_provider_instance

    from xwhy.bootstrap import _build_zai_provider

    result = _build_zai_provider()

    mock_openai_class.assert_called_once_with(
        api_key="dummy_zai_key", base_url="https://api.z.ai/api/paas/v4/"
    )
    mock_factory_create.assert_called_once_with(
        provider=pytest.importorskip("xwhy.providers.types").ProviderType.ZAI,
        client=mock_client_instance,
    )
    assert result is mock_provider_instance


# ---------------------------------------------------------------------
# Groq Provider Test
# ---------------------------------------------------------------------
@patch("xwhy.bootstrap.ProviderFactory.create")
@patch("openai.OpenAI")
@patch("xwhy.bootstrap.settings")
def test_build_groq_provider(
    mock_settings: MagicMock,
    mock_openai_class: MagicMock,
    mock_factory_create: MagicMock,
) -> None:
    """Verify Groq provider initialization with correct base_url."""
    mock_settings.groq_api_key = "dummy_groq_key"
    mock_client_instance = MagicMock()
    mock_openai_class.return_value = mock_client_instance

    mock_provider_instance = MagicMock()
    mock_factory_create.return_value = mock_provider_instance

    from xwhy.bootstrap import _build_groq_provider

    result = _build_groq_provider()

    mock_openai_class.assert_called_once_with(
        api_key="dummy_groq_key", base_url="https://api.groq.com/openai/v1"
    )
    mock_factory_create.assert_called_once_with(
        provider=pytest.importorskip("xwhy.providers.types").ProviderType.GROQ,
        client=mock_client_instance,
    )
    assert result is mock_provider_instance


# ---------------------------------------------------------------------
# Cohere Provider Test
# ---------------------------------------------------------------------
@patch("xwhy.bootstrap.ProviderFactory.create")
@patch("openai.OpenAI")
@patch("xwhy.bootstrap.settings")
def test_build_cohere_provider(
    mock_settings: MagicMock,
    mock_openai_class: MagicMock,
    mock_factory_create: MagicMock,
) -> None:
    """Verify Cohere provider initialization with correct base_url."""
    mock_settings.cohere_api_key = "dummy_cohere_key"
    mock_client_instance = MagicMock()
    mock_openai_class.return_value = mock_client_instance

    mock_provider_instance = MagicMock()
    mock_factory_create.return_value = mock_provider_instance

    from xwhy.bootstrap import _build_cohere_provider

    result = _build_cohere_provider()

    mock_openai_class.assert_called_once_with(
        api_key="dummy_cohere_key", base_url="https://api.cohere.ai/compatibility/v1"
    )
    mock_factory_create.assert_called_once_with(
        provider=pytest.importorskip("xwhy.providers.types").ProviderType.COHERE,
        client=mock_client_instance,
    )
    assert result is mock_provider_instance


# ---------------------------------------------------------------------
# Fireworks AI Provider Test
# ---------------------------------------------------------------------
@patch("xwhy.bootstrap.ProviderFactory.create")
@patch("openai.OpenAI")
@patch("xwhy.bootstrap.settings")
def test_build_fireworks_provider(
    mock_settings: MagicMock,
    mock_openai_class: MagicMock,
    mock_factory_create: MagicMock,
) -> None:
    """Verify Fireworks AI provider initialization with correct base_url."""
    mock_settings.fireworks_api_key = "dummy_fireworks_key"
    mock_client_instance = MagicMock()
    mock_openai_class.return_value = mock_client_instance

    mock_provider_instance = MagicMock()
    mock_factory_create.return_value = mock_provider_instance

    from xwhy.bootstrap import _build_fireworks_provider

    result = _build_fireworks_provider()

    mock_openai_class.assert_called_once_with(
        api_key="dummy_fireworks_key", base_url="https://api.fireworks.ai/inference/v1"
    )
    mock_factory_create.assert_called_once_with(
        provider=pytest.importorskip("xwhy.providers.types").ProviderType.FIREWORKS_AI,
        client=mock_client_instance,
    )
    assert result is mock_provider_instance


# ---------------------------------------------------------------------
# Grok Provider Test
# ---------------------------------------------------------------------
@patch("xwhy.bootstrap.ProviderFactory.create")
@patch("openai.OpenAI")
@patch("xwhy.bootstrap.settings")
def test_build_grok_provider(
    mock_settings: MagicMock,
    mock_openai_class: MagicMock,
    mock_factory_create: MagicMock,
) -> None:
    """Verify Grok provider initialization with correct base_url."""
    mock_settings.grok_api_key = "dummy_grok_key"
    mock_client_instance = MagicMock()
    mock_openai_class.return_value = mock_client_instance
    mock_factory_create.return_value = MagicMock()

    from xwhy.bootstrap import _build_grok_provider

    _build_grok_provider()

    mock_openai_class.assert_called_once_with(
        api_key="dummy_grok_key", base_url="https://api.x.ai/v1"
    )
    mock_factory_create.assert_called_once_with(
        provider=pytest.importorskip("xwhy.providers.types").ProviderType.GROK,
        client=mock_client_instance,
    )


# ---------------------------------------------------------------------
# OpenRouter Provider Test
# ---------------------------------------------------------------------
@patch("xwhy.bootstrap.ProviderFactory.create")
@patch("openai.OpenAI")
@patch("xwhy.bootstrap.settings")
def test_build_openrouter_provider(
    mock_settings: MagicMock,
    mock_openai_class: MagicMock,
    mock_factory_create: MagicMock,
) -> None:
    """Verify OpenRouter provider initialization with correct base_url."""
    mock_settings.openrouter_api_key = "dummy_openrouter_key"
    mock_client_instance = MagicMock()
    mock_openai_class.return_value = mock_client_instance
    mock_factory_create.return_value = MagicMock()

    from xwhy.bootstrap import _build_openrouter_provider

    _build_openrouter_provider()

    mock_openai_class.assert_called_once_with(
        api_key="dummy_openrouter_key", base_url="https://openrouter.ai/api/v1"
    )
    mock_factory_create.assert_called_once_with(
        provider=pytest.importorskip("xwhy.providers.types").ProviderType.OPENROUTER,
        client=mock_client_instance,
    )


# ---------------------------------------------------------------------
# Ollama Provider Test
# ---------------------------------------------------------------------
@patch("xwhy.bootstrap.ProviderFactory.create")
@patch("openai.OpenAI")
def test_build_ollama_provider(
    mock_openai_class: MagicMock, mock_factory_create: MagicMock
) -> None:
    """Verify local Ollama provider initialization with hardcoded dummy key."""
    mock_client_instance = MagicMock()
    mock_openai_class.return_value = mock_client_instance
    mock_factory_create.return_value = MagicMock()

    from xwhy.bootstrap import _build_ollama_provider

    _build_ollama_provider()

    mock_openai_class.assert_called_once_with(
        api_key="ollama", base_url="http://localhost:11434/v1/"
    )
    mock_factory_create.assert_called_once_with(
        provider=pytest.importorskip("xwhy.providers.types").ProviderType.OLLAMA,
        client=mock_client_instance,
    )


# ---------------------------------------------------------------------
# LMStudio Provider Test
# ---------------------------------------------------------------------
@patch("xwhy.bootstrap.ProviderFactory.create")
@patch("openai.OpenAI")
@patch("xwhy.bootstrap.settings")
def test_build_lmstudio_provider(
    mock_settings: MagicMock,
    mock_openai_class: MagicMock,
    mock_factory_create: MagicMock,
) -> None:
    """Verify local LMStudio provider initialization from dynamic settings."""
    mock_settings.lmstudio_api_key = "dynamic-lm-key"
    mock_settings.lmstudio_base_url = "http://127.0.0.1:9090/v1"

    mock_client_instance = MagicMock()
    mock_openai_class.return_value = mock_client_instance
    mock_factory_create.return_value = MagicMock()

    from xwhy.bootstrap import _build_lmstudio_provider

    _build_lmstudio_provider()

    mock_openai_class.assert_called_once_with(
        api_key="dynamic-lm-key", base_url="http://127.0.0.1:9090/v1"
    )
    mock_factory_create.assert_called_once_with(
        provider=pytest.importorskip("xwhy.providers.types").ProviderType.LMSTUDIO,
        client=mock_client_instance,
    )


# ---------------------------------------------------------------------
# ByteDance Provider Test
# ---------------------------------------------------------------------
@patch("xwhy.bootstrap.ProviderFactory.create")
@patch("openai.OpenAI")
@patch("xwhy.bootstrap.settings")
def test_build_bytedance_provider(
    mock_settings: MagicMock,
    mock_openai_class: MagicMock,
    mock_factory_create: MagicMock,
) -> None:
    """Verify ByteDance provider initialization with correct base_url."""
    mock_settings.bytedance_api_key = "dummy_bytedance_key"
    mock_client_instance = MagicMock()
    mock_openai_class.return_value = mock_client_instance
    mock_factory_create.return_value = MagicMock()

    from xwhy.bootstrap import _build_bytedance_provider

    _build_bytedance_provider()

    mock_openai_class.assert_called_once_with(
        api_key="dummy_bytedance_key",
        base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
    )
    mock_factory_create.assert_called_once_with(
        provider=pytest.importorskip("xwhy.providers.types").ProviderType.BYTEDANCE,
        client=mock_client_instance,
    )


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
