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


def _build_anthropic_provider() -> BaseProvider:
    """Instantiate an Anthropic provider using configuration settings."""
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)
    return ProviderFactory.create(provider=ProviderType.ANTHROPIC, client=client)


def _build_huggingface_provider() -> BaseProvider:
    """Instantiate a HuggingFace provider using configuration settings."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(api_key=settings.huggingface_token)
    return ProviderFactory.create(provider=ProviderType.HUGGINGFACE, client=client)


def _build_openai_provider() -> BaseProvider:
    """Instantiate an OpenAI provider using configuration settings."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    return ProviderFactory.create(provider=ProviderType.OPENAI, client=client)


def _build_zai_provider() -> BaseProvider:
    """Instantiate a Z.ai provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.zai_api_key, base_url="https://api.z.ai/api/paas/v4/"
    )
    return ProviderFactory.create(provider=ProviderType.ZAI, client=client)


def _build_groq_provider() -> BaseProvider:
    """Instantiate a Groq provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1"
    )
    return ProviderFactory.create(provider=ProviderType.GROQ, client=client)


def _build_cohere_provider() -> BaseProvider:
    """Instantiate a Cohere provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.cohere_api_key,
        base_url="https://api.cohere.ai/compatibility/v1",
    )
    return ProviderFactory.create(provider=ProviderType.COHERE, client=client)


def _build_fireworks_provider() -> BaseProvider:
    """Instantiate a Fireworks AI provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.fireworks_api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )
    return ProviderFactory.create(provider=ProviderType.FIREWORKS_AI, client=client)


def _build_grok_provider() -> BaseProvider:
    """Instantiate a Grok (xAI) provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.grok_api_key, base_url="https://api.x.ai/v1")
    return ProviderFactory.create(provider=ProviderType.GROK, client=client)


def _build_openrouter_provider() -> BaseProvider:
    """Instantiate an OpenRouter provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.openrouter_api_key, base_url="https://openrouter.ai/api/v1"
    )
    return ProviderFactory.create(provider=ProviderType.OPENROUTER, client=client)


def _build_ollama_provider() -> BaseProvider:
    """Instantiate a local Ollama provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    client = OpenAI(
        api_key="ollama",  # Required by standard SDK but ignored by Ollama
        base_url="http://localhost:11434/v1/",
    )
    return ProviderFactory.create(provider=ProviderType.OLLAMA, client=client)


def _build_lmstudio_provider() -> BaseProvider:
    """Instantiate a local LMStudio provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.lmstudio_api_key, base_url=settings.lmstudio_base_url
    )
    return ProviderFactory.create(provider=ProviderType.LMSTUDIO, client=client)


def _build_bytedance_provider() -> BaseProvider:
    """Instantiate a ByteDance provider using the OpenAI-compatible SDK."""
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.bytedance_api_key,
        base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
    )
    return ProviderFactory.create(provider=ProviderType.BYTEDANCE, client=client)


def _build_gemini_provider() -> BaseProvider:
    """Instantiate a Gemini provider using configuration settings."""
    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key)
    return ProviderFactory.create(provider=ProviderType.GEMINI, client=client)


def _build_word2vec(**kwargs: Any) -> Word2VecEmbedding:  # noqa: ANN401
    return Word2VecEmbedding(
        model_name="word2vec-google-news-300", settings=settings, **kwargs
    )


def _build_glove(**kwargs: Any) -> Word2VecEmbedding:  # noqa: ANN401
    return Word2VecEmbedding(model_name="glove.840B.300d", settings=settings, **kwargs)


def _build_paragram(**kwargs: Any) -> Word2VecEmbedding:  # noqa: ANN401
    return Word2VecEmbedding(
        model_name="paragram_300_sl999", settings=settings, **kwargs
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

    EmbeddingFactory.register(EmbeddingType.WORD2VEC, _build_word2vec)
    EmbeddingFactory.register(EmbeddingType.GLOVE, _build_glove)
    EmbeddingFactory.register(EmbeddingType.PARAGRAM, _build_paragram)

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
