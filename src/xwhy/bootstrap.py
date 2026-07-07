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
