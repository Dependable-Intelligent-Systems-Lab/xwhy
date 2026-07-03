"""Embedding type definitions."""

from enum import StrEnum


class EmbeddingType(StrEnum):
    """Supported embedding backends."""

    WORD2VEC = "word2vec"
    GLOVE = "glove"
    PARAGRAM = "paragram"

    # Future:
    # SENTENCE_TRANSFORMER = "sentence_transformer"
    # CLIP = "clip"
    # BGE = "bge"
