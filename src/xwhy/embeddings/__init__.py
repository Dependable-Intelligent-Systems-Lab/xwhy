"""Embedding module public API.

Registers and exposes available embedding implementations.
"""

from xwhy.embeddings.base import BaseEmbedding
from xwhy.embeddings.factory import EmbeddingFactory
from xwhy.embeddings.types import EmbeddingType
from xwhy.embeddings.word2vec import Word2VecEmbedding

__all__ = [
    "BaseEmbedding",
    "EmbeddingFactory",
    "EmbeddingType",
    "Word2VecEmbedding",
]
