"""Application bootstrap registry."""

from typing import Any

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
from xgboost import XGBRegressor

from xwhy.config import settings
from xwhy.embeddings.factory import EmbeddingFactory
from xwhy.embeddings.types import EmbeddingType
from xwhy.embeddings.word2vec import Word2VecEmbedding
from xwhy.providers.anthropic import AnthropicProvider
from xwhy.providers.base import BaseProvider
from xwhy.providers.factory import ProviderFactory
from xwhy.providers.gemini import GeminiProvider
from xwhy.providers.huggingface import HuggingFaceProvider
from xwhy.providers.openai import OpenAIProvider
from xwhy.providers.resolver import ProviderResolver
from xwhy.providers.types import ProviderType
from xwhy.surrogate.factory import SurrogateFactory
from xwhy.surrogate.linear import LinearRegressionSurrogate
from xwhy.surrogate.tree import TreeBasedSurrogate
from xwhy.surrogate.types import SurrogateType


def _ensure_dependency(pkg_name: str, extra_name: str) -> None:
    """Check if an optional dependency is installed.

    Args:
        pkg_name: The name of the package module to import.
        extra_name: The name of the optional dependency extra flag.

    Raises:
        ImportError: If the requested module cannot be imported.

    """
    try:
        __import__(pkg_name)
    except ImportError as err:
        raise ImportError(
            f"'{pkg_name}' is not installed. "
            f"Please install with: pip install 'xwhy[{extra_name}]'"
        ) from err


def _build_anthropic_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate an Anthropic provider using configuration settings or kwargs."""
    from anthropic import Anthropic

    api_key = kwargs.pop("api_key", settings.anthropic_api_key)
    client = Anthropic(api_key=api_key, **kwargs)
    return ProviderFactory.create(provider=ProviderType.ANTHROPIC, client=client)


def _build_huggingface_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate a HuggingFace provider using configuration settings or kwargs."""
    from huggingface_hub import InferenceClient

    token = kwargs.pop("token", settings.huggingface_token)
    client = InferenceClient(token=token, **kwargs)
    return ProviderFactory.create(provider=ProviderType.HUGGINGFACE, client=client)


def _build_openai_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate an OpenAI provider using configuration settings or kwargs."""
    from openai import OpenAI

    api_key = kwargs.pop("api_key", settings.openai_api_key)
    client = OpenAI(api_key=api_key, **kwargs)
    return ProviderFactory.create(provider=ProviderType.OPENAI, client=client)


def _build_zai_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate a Z.ai provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    api_key = kwargs.pop("api_key", settings.zai_api_key)
    base_url = kwargs.pop("base_url", "https://api.z.ai/api/paas/v4/")

    client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
    return ProviderFactory.create(provider=ProviderType.ZAI, client=client)


def _build_groq_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate a Groq provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    api_key = kwargs.pop("api_key", settings.groq_api_key)
    base_url = kwargs.pop("base_url", "https://api.groq.com/openai/v1")

    client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
    return ProviderFactory.create(provider=ProviderType.GROQ, client=client)


def _build_cohere_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate a Cohere provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    api_key = kwargs.pop("api_key", settings.cohere_api_key)
    base_url = kwargs.pop("base_url", "https://api.cohere.ai/compatibility/v1")

    client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
    return ProviderFactory.create(provider=ProviderType.COHERE, client=client)


def _build_fireworks_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate a Fireworks AI provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    api_key = kwargs.pop("api_key", settings.fireworks_api_key)
    base_url = kwargs.pop("base_url", "https://api.fireworks.ai/inference/v1")

    client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
    return ProviderFactory.create(provider=ProviderType.FIREWORKS_AI, client=client)


def _build_grok_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate a Grok (xAI) provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    api_key = kwargs.pop("api_key", settings.grok_api_key)
    base_url = kwargs.pop("base_url", "https://api.x.ai/v1")

    client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
    return ProviderFactory.create(provider=ProviderType.GROK, client=client)


def _build_openrouter_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate an OpenRouter provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    api_key = kwargs.pop("api_key", settings.openrouter_api_key)
    base_url = kwargs.pop("base_url", "https://openrouter.ai/api/v1")

    client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
    return ProviderFactory.create(provider=ProviderType.OPENROUTER, client=client)


def _build_ollama_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate a local Ollama provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    api_key = kwargs.pop(
        "api_key", "ollama"
    )  # Required by standard SDK but ignored by Ollama
    base_url = kwargs.pop("base_url", "http://localhost:11434/v1/")

    client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
    return ProviderFactory.create(provider=ProviderType.OLLAMA, client=client)


def _build_lmstudio_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate a local LMStudio provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    api_key = kwargs.pop("api_key", settings.lmstudio_api_key)
    base_url = kwargs.pop("base_url", settings.lmstudio_base_url)

    client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
    return ProviderFactory.create(provider=ProviderType.LMSTUDIO, client=client)


def _build_bytedance_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate a ByteDance provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    api_key = kwargs.pop("api_key", settings.bytedance_api_key)
    base_url = kwargs.pop("base_url", "https://ark.ap-southeast.bytepluses.com/api/v3")

    client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
    return ProviderFactory.create(provider=ProviderType.BYTEDANCE, client=client)


def _build_gemini_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate a Gemini provider using configuration settings or kwargs."""
    from google import genai

    api_key = kwargs.pop("api_key", settings.gemini_api_key)

    client = genai.Client(api_key=api_key, **kwargs)
    return ProviderFactory.create(provider=ProviderType.GEMINI, client=client)


# ---------------------------------------------------------
# OpenAI & Gemini Native Cloud
# ---------------------------------------------------------
def _build_azure_openai_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate an Azure OpenAI provider using the official SDK."""
    from openai import AzureOpenAI

    api_key = kwargs.pop("api_key", settings.azure_api_key)
    api_version = kwargs.pop("api_version", settings.azure_api_version)
    azure_endpoint = kwargs.pop("azure_endpoint", settings.azure_endpoint)

    if not azure_endpoint:
        raise ValueError(
            "azure_endpoint must be set in settings or passed via kwargs "
            "to use Azure OpenAI."
        )

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint,
        **kwargs,
    )
    return ProviderFactory.create(provider=ProviderType.AZURE_OPENAI, client=client)


def _build_gcp_gemini_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate a Gemini provider running on GCP Vertex AI."""
    from google import genai

    # Needs google-cloud-aiplatform for Vertex features if used
    _ensure_dependency("google.auth", "vertex")

    vertexai = kwargs.pop("vertexai", True)
    project = kwargs.pop("project", settings.gcp_project)
    location = kwargs.pop("location", settings.gcp_location)

    client = genai.Client(
        vertexai=vertexai,
        project=project,
        location=location,
        **kwargs,
    )
    return ProviderFactory.create(provider=ProviderType.GCP_GEMINI, client=client)


# ---------------------------------------------------------
# Anthropic Cloud Integrations
# ---------------------------------------------------------
def _build_anthropic_bedrock_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate an Anthropic provider for legacy Amazon Bedrock."""
    from anthropic import AnthropicBedrock

    _ensure_dependency("boto3", "bedrock")

    aws_access_key = kwargs.pop("aws_access_key", settings.aws_access_key)
    aws_secret_key = kwargs.pop("aws_secret_key", settings.aws_secret_key)
    aws_session_token = kwargs.pop("aws_session_token", settings.aws_session_token)
    aws_region = kwargs.pop("aws_region", settings.aws_region)

    client = AnthropicBedrock(
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
        aws_session_token=aws_session_token,
        aws_region=aws_region,
        **kwargs,
    )
    return ProviderFactory.create(
        provider=ProviderType.ANTHROPIC_BEDROCK, client=client
    )


def _build_anthropic_bedrock_mantle_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate an Anthropic provider for Amazon Bedrock Mantle."""
    from anthropic import AnthropicBedrockMantle

    _ensure_dependency("boto3", "bedrock")

    aws_region = kwargs.pop("aws_region", settings.aws_region)

    client = AnthropicBedrockMantle(aws_region=aws_region, **kwargs)
    return ProviderFactory.create(
        provider=ProviderType.ANTHROPIC_BEDROCK_MANTLE, client=client
    )


def _build_anthropic_aws_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate an Anthropic provider for Claude Platform on AWS."""
    from anthropic import AnthropicAWS

    _ensure_dependency("boto3", "aws")

    aws_region = kwargs.pop("aws_region", settings.aws_region)
    workspace_id = kwargs.pop("workspace_id", settings.anthropic_aws_workspace_id)

    if not workspace_id:
        raise ValueError(
            "`workspace_id` must be set via kwargs or `anthropic_aws_workspace_id`"
            " in settings to use Anthropic AWS."
        )

    client = AnthropicAWS(
        aws_region=aws_region,
        workspace_id=workspace_id,
        **kwargs,
    )
    return ProviderFactory.create(provider=ProviderType.ANTHROPIC_AWS, client=client)


def _build_anthropic_vertex_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate an Anthropic provider for Google Cloud Vertex AI."""
    from anthropic import AnthropicVertex

    _ensure_dependency("google.auth", "vertex")

    project_id = kwargs.pop("project_id", settings.gcp_project)
    region = kwargs.pop("region", settings.gcp_location)

    if not project_id:
        raise ValueError(
            "`project_id` must be set via kwargs or `gcp_project` in"
            " settings to use Anthropic Vertex."
        )

    client = AnthropicVertex(
        project_id=project_id,
        region=region,
        **kwargs,
    )
    return ProviderFactory.create(provider=ProviderType.ANTHROPIC_VERTEX, client=client)


def _build_anthropic_foundry_provider(**kwargs: Any) -> BaseProvider:  # noqa: ANN401
    """Instantiate an Anthropic provider for Microsoft Foundry."""
    from anthropic import AnthropicFoundry

    api_key = kwargs.pop("api_key", settings.anthropic_foundry_api_key)
    resource = kwargs.pop("resource", settings.anthropic_foundry_resource)

    if not resource:
        raise ValueError(
            "`resource` must be set via kwargs or `anthropic_foundry_resource` in"
            " settings to use Anthropic Foundry."
        )

    client = AnthropicFoundry(
        api_key=api_key,
        resource=resource,
        **kwargs,
    )
    return ProviderFactory.create(
        provider=ProviderType.ANTHROPIC_FOUNDRY, client=client
    )


def _build_word2vec(**kwargs: Any) -> Word2VecEmbedding:  # noqa: ANN401
    return Word2VecEmbedding(
        model_name="word2vec-google-news-300", settings=settings, **kwargs
    )


def _build_glove(**kwargs: Any) -> Word2VecEmbedding:  # noqa: ANN401
    return Word2VecEmbedding(model_name="glove.840B.300d", settings=settings, **kwargs)


def _build_paragram_sl(**kwargs: Any) -> Word2VecEmbedding:  # noqa: ANN401
    return Word2VecEmbedding(
        model_name="paragram_300_sl999", settings=settings, **kwargs
    )


def _build_paragram_ws(**kwargs: Any) -> Word2VecEmbedding:  # noqa: ANN401
    return Word2VecEmbedding(
        model_name="paragram-300-WS353", settings=settings, **kwargs
    )


def _build_glm_ols(**kwargs: Any) -> LinearRegressionSurrogate:  # noqa: ANN401
    return LinearRegressionSurrogate(model=LinearRegression())


def _build_ridge(**kwargs: Any) -> LinearRegressionSurrogate:  # noqa: ANN401
    seed = int(kwargs.get("seed", 1024))
    alpha = float(kwargs.get("ridge_alpha", 1.0))
    return LinearRegressionSurrogate(model=Ridge(alpha=alpha, random_state=seed))


def _build_bayesian_ridge(**kwargs: Any) -> LinearRegressionSurrogate:  # noqa: ANN401
    return LinearRegressionSurrogate(model=BayesianRidge())


def _build_random_forest(**kwargs: Any) -> TreeBasedSurrogate:  # noqa: ANN401
    seed = int(kwargs.get("seed", 1024))
    return TreeBasedSurrogate(model=RandomForestRegressor(random_state=seed))


def _build_gradient_boosting(**kwargs: Any) -> TreeBasedSurrogate:  # noqa: ANN401
    seed = int(kwargs.get("seed", 1024))
    return TreeBasedSurrogate(model=GradientBoostingRegressor(random_state=seed))


def _build_xgboost(**kwargs: Any) -> TreeBasedSurrogate:  # noqa: ANN401
    seed = int(kwargs.get("seed", 1024))
    return TreeBasedSurrogate(model=XGBRegressor(random_state=seed, verbosity=0))


def register_all() -> None:
    """Register all built-in components and hidden factory builders."""
    ProviderFactory.register(
        provider=ProviderType.OPENAI,
        provider_cls=OpenAIProvider,
    )

    ProviderFactory.register(
        provider=ProviderType.GEMINI,
        provider_cls=GeminiProvider,
    )

    ProviderFactory.register(
        provider=ProviderType.ANTHROPIC,
        provider_cls=AnthropicProvider,
    )

    ProviderFactory.register(
        provider=ProviderType.HUGGINGFACE,
        provider_cls=HuggingFaceProvider,
    )

    compatible_providers = [
        ProviderType.ZAI,
        ProviderType.GROQ,
        ProviderType.COHERE,
        ProviderType.FIREWORKS_AI,
        ProviderType.GROK,
        ProviderType.OPENROUTER,
        ProviderType.OLLAMA,
        ProviderType.LMSTUDIO,
        ProviderType.BYTEDANCE,
    ]

    for provider_type in compatible_providers:
        ProviderFactory.register(
            provider=provider_type,
            provider_cls=OpenAIProvider,
        )

    ProviderFactory.register(
        provider=ProviderType.AZURE_OPENAI,
        provider_cls=OpenAIProvider,
    )
    ProviderFactory.register(
        provider=ProviderType.GCP_GEMINI,
        provider_cls=GeminiProvider,
    )
    ProviderFactory.register(
        provider=ProviderType.ANTHROPIC_BEDROCK,
        provider_cls=AnthropicProvider,
    )
    ProviderFactory.register(
        provider=ProviderType.ANTHROPIC_BEDROCK_MANTLE,
        provider_cls=AnthropicProvider,
    )
    ProviderFactory.register(
        provider=ProviderType.ANTHROPIC_AWS,
        provider_cls=AnthropicProvider,
    )
    ProviderFactory.register(
        provider=ProviderType.ANTHROPIC_VERTEX,
        provider_cls=AnthropicProvider,
    )
    ProviderFactory.register(
        provider=ProviderType.ANTHROPIC_FOUNDRY,
        provider_cls=AnthropicProvider,
    )

    EmbeddingFactory.register(EmbeddingType.WORD2VEC, _build_word2vec)
    EmbeddingFactory.register(EmbeddingType.GLOVE, _build_glove)
    EmbeddingFactory.register(EmbeddingType.PARAGRAM_SL, _build_paragram_sl)
    EmbeddingFactory.register(EmbeddingType.PARAGRAM_WS, _build_paragram_ws)

    ProviderResolver.register(
        provider_type=ProviderType.OPENAI,
        builder=_build_openai_provider,
    )

    ProviderResolver.register(
        provider_type=ProviderType.GEMINI,
        builder=_build_gemini_provider,
    )

    ProviderResolver.register(
        provider_type=ProviderType.ANTHROPIC,
        builder=_build_anthropic_provider,
    )

    ProviderResolver.register(
        provider_type=ProviderType.HUGGINGFACE,
        builder=_build_huggingface_provider,
    )

    ProviderResolver.register(
        provider_type=ProviderType.ZAI, builder=_build_zai_provider
    )

    ProviderResolver.register(
        provider_type=ProviderType.GROQ, builder=_build_groq_provider
    )

    ProviderResolver.register(
        provider_type=ProviderType.COHERE, builder=_build_cohere_provider
    )

    ProviderResolver.register(
        provider_type=ProviderType.FIREWORKS_AI, builder=_build_fireworks_provider
    )

    ProviderResolver.register(
        provider_type=ProviderType.GROK, builder=_build_grok_provider
    )

    ProviderResolver.register(
        provider_type=ProviderType.OPENROUTER, builder=_build_openrouter_provider
    )

    ProviderResolver.register(
        provider_type=ProviderType.OLLAMA, builder=_build_ollama_provider
    )

    ProviderResolver.register(
        provider_type=ProviderType.LMSTUDIO, builder=_build_lmstudio_provider
    )

    ProviderResolver.register(
        provider_type=ProviderType.BYTEDANCE, builder=_build_bytedance_provider
    )

    ProviderResolver.register(
        provider_type=ProviderType.AZURE_OPENAI, builder=_build_azure_openai_provider
    )

    ProviderResolver.register(
        provider_type=ProviderType.GCP_GEMINI, builder=_build_gcp_gemini_provider
    )

    ProviderResolver.register(
        provider_type=ProviderType.ANTHROPIC_BEDROCK,
        builder=_build_anthropic_bedrock_provider,
    )

    ProviderResolver.register(
        provider_type=ProviderType.ANTHROPIC_BEDROCK_MANTLE,
        builder=_build_anthropic_bedrock_mantle_provider,
    )

    ProviderResolver.register(
        provider_type=ProviderType.ANTHROPIC_AWS, builder=_build_anthropic_aws_provider
    )

    ProviderResolver.register(
        provider_type=ProviderType.ANTHROPIC_VERTEX,
        builder=_build_anthropic_vertex_provider,
    )

    ProviderResolver.register(
        provider_type=ProviderType.ANTHROPIC_FOUNDRY,
        builder=_build_anthropic_foundry_provider,
    )

    SurrogateFactory.register(method=SurrogateType.GLM_OLS, builder=_build_glm_ols)
    SurrogateFactory.register(method=SurrogateType.LIME, builder=_build_glm_ols)

    SurrogateFactory.register(method=SurrogateType.GLM_RIDGE, builder=_build_ridge)
    SurrogateFactory.register(method=SurrogateType.LIME_RIDGE, builder=_build_ridge)

    SurrogateFactory.register(
        method=SurrogateType.BAYLIME, builder=_build_bayesian_ridge
    )
    SurrogateFactory.register(
        method=SurrogateType.RANDOMFOREST, builder=_build_random_forest
    )
    SurrogateFactory.register(
        method=SurrogateType.GRADIENT_BOOSTING, builder=_build_gradient_boosting
    )
    SurrogateFactory.register(method=SurrogateType.XGBOOST, builder=_build_xgboost)
