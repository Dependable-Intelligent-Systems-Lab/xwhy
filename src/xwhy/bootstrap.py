"""Application bootstrap registry."""

from xwhy.embeddings.factory import EmbeddingFactory
from xwhy.embeddings.types import EmbeddingType
from xwhy.embeddings.word2vec import Word2VecEmbedding
from xwhy.providers.factory import ProviderFactory
from xwhy.providers.openai import OpenAIProvider
from xwhy.providers.types import ProviderType


def register_all() -> None:
    """Register all built-in components."""
    ProviderFactory.register(
        provider=ProviderType.OPENAI,
        provider_cls=OpenAIProvider,
    )

    EmbeddingFactory.register(
        embedding=EmbeddingType.WORD2VEC,
        embedding_cls=Word2VecEmbedding,
    )
