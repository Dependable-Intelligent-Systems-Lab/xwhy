"""Embedding type definitions."""

from __future__ import annotations

from enum import StrEnum


class EmbeddingType(StrEnum):
    """Supported embedding backends."""

    WORD2VEC = "word2vec"
    GLOVE = "glove"
    PARAGRAM_SL = "paragram_sl"
    PARAGRAM_WS = "paragram_ws"
    DINOV2 = "dinov2"

    # Future:
    # SENTENCE_TRANSFORMER = "sentence_transformer"
    # CLIP = "clip"
    # BGE = "bge"

    @classmethod
    def from_str(cls, value: str | EmbeddingType) -> EmbeddingType:
        """Safely convert a string or enum instance to EmbeddingType."""
        try:
            return cls(value)
        except ValueError as err:
            valid_options = ", ".join([item.value for item in cls])
            raise ValueError(
                f"'{value}' is not a valid EmbeddingType. "
                f"Supported options are: [{valid_options}]"
            ) from err

    @property
    def is_image_embedding(self) -> bool:
        """Check if this is an image-based embedding."""
        return self in {EmbeddingType.DINOV2}

    @property
    def is_text_embedding(self) -> bool:
        """Check if this is a text-based embedding."""
        return self in {
            EmbeddingType.WORD2VEC,
            EmbeddingType.GLOVE,
            EmbeddingType.PARAGRAM_SL,
            EmbeddingType.PARAGRAM_WS,
        }
