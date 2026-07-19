"""Embedding module public API.

Registers and exposes available embedding implementations.
"""

from xwhy.models.embeddings.base import BaseEmbedding
from xwhy.models.embeddings.factory import EmbeddingFactory
from xwhy.models.embeddings.types import EmbeddingType
from xwhy.models.embeddings.word2vec import Word2VecEmbedding

__all__ = [
    "BaseEmbedding",
    "EmbeddingFactory",
    "EmbeddingType",
    "Word2VecEmbedding",
]
