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
from xwhy.providers.base import BaseProvider
from xwhy.providers.types import ProviderType


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
# Utilities & Dependency Checker Tests
# ---------------------------------------------------------------------


def test_ensure_dependency_success() -> None:
    """Verify that _ensure_dependency passes silently for installed packages."""
    from xwhy.bootstrap import _ensure_dependency

    # 'sys' is a built-in module and must always be available
    _ensure_dependency("sys", "vertex")


def test_ensure_dependency_raises_import_error() -> None:
    """Verify that _ensure_dependency raises an explicit ImportError when missing."""
    from xwhy.bootstrap import _ensure_dependency

    with pytest.raises(ImportError, match="'non_existent_pkg' is not installed"):
        _ensure_dependency("non_existent_pkg", "aws")


# ---------------------------------------------------------------------
# OpenAI & Gemini Native Cloud Provider Tests
# ---------------------------------------------------------------------


@patch("xwhy.bootstrap.settings")
@patch("openai.AzureOpenAI")
@patch("xwhy.bootstrap.ProviderFactory.create")
def test_build_azure_openai_provider_success(
    mock_factory_create: MagicMock,
    mock_azure_openai: MagicMock,
    mock_settings: MagicMock,
) -> None:
    """Verify successful initialization of Azure OpenAI provider."""
    mock_settings.azure_api_key = "test-key"
    mock_settings.azure_api_version = "2024-02-01"
    mock_settings.azure_endpoint = "https://test.azure.com"

    mock_client_instance = MagicMock()
    mock_azure_openai.return_value = mock_client_instance

    expected_provider = MagicMock(spec=BaseProvider)
    mock_factory_create.return_value = expected_provider

    from xwhy.bootstrap import _build_azure_openai_provider

    provider = _build_azure_openai_provider()

    mock_azure_openai.assert_called_once_with(
        api_key="test-key",
        api_version="2024-02-01",
        azure_endpoint="https://test.azure.com",
    )
    mock_factory_create.assert_called_once_with(
        provider=ProviderType.AZURE_OPENAI, client=mock_client_instance
    )
    assert provider == expected_provider


@patch("xwhy.bootstrap.settings")
def test_build_azure_openai_provider_missing_endpoint(mock_settings: MagicMock) -> None:
    """Verify ValueError is raised when azure_endpoint configuration is missing."""
    mock_settings.azure_endpoint = None

    from xwhy.bootstrap import _build_azure_openai_provider

    with pytest.raises(ValueError, match="azure_endpoint must be set"):
        _build_azure_openai_provider()


@patch("xwhy.bootstrap._ensure_dependency")
@patch("xwhy.bootstrap.settings")
@patch("google.genai.Client")
@patch("xwhy.bootstrap.ProviderFactory.create")
def test_build_gcp_gemini_provider_success(
    mock_factory_create: MagicMock,
    mock_genai_client: MagicMock,
    mock_settings: MagicMock,
    mock_ensure_dep: MagicMock,
) -> None:
    """Verify successful initialization of native GCP Gemini provider."""
    mock_settings.gcp_project = "gcp-project-123"
    mock_settings.gcp_location = "us-central1"

    mock_client_instance = MagicMock()
    mock_genai_client.return_value = mock_client_instance

    from xwhy.bootstrap import _build_gcp_gemini_provider

    _build_gcp_gemini_provider()

    mock_ensure_dep.assert_called_once_with("google.auth", "vertex")
    mock_genai_client.assert_called_once_with(
        vertexai=True,
        project="gcp-project-123",
        location="us-central1",
    )
    mock_factory_create.assert_called_once_with(
        provider=ProviderType.GCP_GEMINI, client=mock_client_instance
    )


# ---------------------------------------------------------------------
# Anthropic Cloud Integrations Provider Tests
# ---------------------------------------------------------------------


@patch("xwhy.bootstrap._ensure_dependency")
@patch("xwhy.bootstrap.settings")
@patch("anthropic.AnthropicBedrock")
@patch("xwhy.bootstrap.ProviderFactory.create")
def test_build_anthropic_bedrock_provider_success(
    mock_factory_create: MagicMock,
    mock_bedrock: MagicMock,
    mock_settings: MagicMock,
    mock_ensure_dep: MagicMock,
) -> None:
    """Verify successful initialization of legacy Anthropic Bedrock provider."""
    mock_settings.aws_access_key = "access"
    mock_settings.aws_secret_key = "secret"
    mock_settings.aws_session_token = "session"
    mock_settings.aws_region = "us-west-2"

    from xwhy.bootstrap import _build_anthropic_bedrock_provider

    _build_anthropic_bedrock_provider()

    mock_ensure_dep.assert_called_once_with("boto3", "bedrock")
    mock_bedrock.assert_called_once_with(
        aws_access_key="access",
        aws_secret_key="secret",
        aws_session_token="session",
        aws_region="us-west-2",
    )
    mock_factory_create.assert_called_once_with(
        provider=ProviderType.ANTHROPIC_BEDROCK, client=mock_bedrock.return_value
    )


@patch("xwhy.bootstrap._ensure_dependency")
@patch("xwhy.bootstrap.settings")
@patch("anthropic.AnthropicBedrockMantle")
@patch("xwhy.bootstrap.ProviderFactory.create")
def test_build_anthropic_bedrock_mantle_provider_success(
    mock_factory_create: MagicMock,
    mock_mantle: MagicMock,
    mock_settings: MagicMock,
    mock_ensure_dep: MagicMock,
) -> None:
    """Verify successful initialization of Anthropic Bedrock Mantle provider."""
    mock_settings.aws_region = "us-east-1"

    from xwhy.bootstrap import _build_anthropic_bedrock_mantle_provider

    _build_anthropic_bedrock_mantle_provider()

    mock_ensure_dep.assert_called_once_with("boto3", "bedrock")
    mock_mantle.assert_called_once_with(aws_region="us-east-1")
    mock_factory_create.assert_called_once_with(
        provider=ProviderType.ANTHROPIC_BEDROCK_MANTLE, client=mock_mantle.return_value
    )


@patch("xwhy.bootstrap._ensure_dependency")
@patch("xwhy.bootstrap.settings")
@patch("anthropic.AnthropicAWS")
@patch("xwhy.bootstrap.ProviderFactory.create")
def test_build_anthropic_aws_provider_success(
    mock_factory_create: MagicMock,
    mock_anthropic_aws: MagicMock,
    mock_settings: MagicMock,
    mock_ensure_dep: MagicMock,
) -> None:
    """Verify successful initialization of Anthropic AWS provider."""
    mock_settings.aws_region = "us-east-1"
    mock_settings.anthropic_aws_workspace_id = "ws-999"

    from xwhy.bootstrap import _build_anthropic_aws_provider

    _build_anthropic_aws_provider()

    mock_ensure_dep.assert_called_once_with("boto3", "aws")
    mock_anthropic_aws.assert_called_once_with(
        aws_region="us-east-1",
        workspace_id="ws-999",
    )
    mock_factory_create.assert_called_once_with(
        provider=ProviderType.ANTHROPIC_AWS, client=mock_anthropic_aws.return_value
    )


@patch("xwhy.bootstrap._ensure_dependency")
@patch("xwhy.bootstrap.settings")
def test_build_anthropic_aws_provider_missing_workspace(
    mock_settings: MagicMock,
    mock_ensure_dep: MagicMock,
) -> None:
    """Verify ValueError is raised when anthropic_aws_workspace_id is missing.

    This test patches _ensure_dependency to bypass the environmental check for
    'boto3', ensuring that the function strictly validates the workspace configuration.
    """
    mock_settings.anthropic_aws_workspace_id = None

    from xwhy.bootstrap import _build_anthropic_aws_provider

    with pytest.raises(ValueError, match="anthropic_aws_workspace_id must be set"):
        _build_anthropic_aws_provider()

    mock_ensure_dep.assert_called_once_with("boto3", "aws")


@patch("xwhy.bootstrap._ensure_dependency")
@patch("xwhy.bootstrap.settings")
@patch("anthropic.AnthropicVertex")
@patch("xwhy.bootstrap.ProviderFactory.create")
def test_build_anthropic_vertex_provider_success(
    mock_factory_create: MagicMock,
    mock_anthropic_vertex: MagicMock,
    mock_settings: MagicMock,
    mock_ensure_dep: MagicMock,
) -> None:
    """Verify successful initialization of Anthropic Vertex provider."""
    mock_settings.gcp_project = "vertex-project"
    mock_settings.gcp_location = "europe-west1"

    from xwhy.bootstrap import _build_anthropic_vertex_provider

    _build_anthropic_vertex_provider()

    mock_ensure_dep.assert_called_once_with("google.auth", "vertex")
    mock_anthropic_vertex.assert_called_once_with(
        project_id="vertex-project", region="europe-west1"
    )
    mock_factory_create.assert_called_once_with(
        provider=ProviderType.ANTHROPIC_VERTEX,
        client=mock_anthropic_vertex.return_value,
    )


@patch("xwhy.bootstrap.settings")
def test_build_anthropic_vertex_provider_missing_project(
    mock_settings: MagicMock,
) -> None:
    """Verify ValueError is raised when gcp_project is missing for Vertex."""
    mock_settings.gcp_project = None

    from xwhy.bootstrap import _build_anthropic_vertex_provider

    with pytest.raises(ValueError, match="gcp_project` must be set"):
        _build_anthropic_vertex_provider()


@patch("xwhy.bootstrap.settings")
@patch("anthropic.AnthropicFoundry")
@patch("xwhy.bootstrap.ProviderFactory.create")
def test_build_anthropic_foundry_provider_success(
    mock_factory_create: MagicMock,
    mock_foundry: MagicMock,
    mock_settings: MagicMock,
) -> None:
    """Verify successful initialization of Anthropic Foundry provider."""
    mock_settings.anthropic_foundry_api_key = "foundry-key"
    mock_settings.anthropic_foundry_resource = "my-resource"

    from xwhy.bootstrap import _build_anthropic_foundry_provider

    _build_anthropic_foundry_provider()

    mock_foundry.assert_called_once_with(
        api_key="foundry-key",
        resource="my-resource",
    )
    mock_factory_create.assert_called_once_with(
        provider=ProviderType.ANTHROPIC_FOUNDRY, client=mock_foundry.return_value
    )


@patch("xwhy.bootstrap.settings")
def test_build_anthropic_foundry_provider_missing_resource(
    mock_settings: MagicMock,
) -> None:
    """Verify ValueError is raised when anthropic_foundry_resource is missing."""
    mock_settings.anthropic_foundry_resource = None

    from xwhy.bootstrap import _build_anthropic_foundry_provider

    with pytest.raises(ValueError, match="anthropic_foundry_resource must be set"):
        _build_anthropic_foundry_provider()


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
