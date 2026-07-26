"""Base embedding abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbedding(ABC):
    """Base class for all embedding implementations."""

    @property
    @abstractmethod
    def model(self) -> Any:  # noqa: ANN401
        """Read-only property to access the underlying raw embedding model.

        Raises:
            RuntimeError: If the model has not been loaded into memory yet.

        Returns:
            The loaded raw embedding model object.

        """
        pass

    @property
    @abstractmethod
    def processor(self) -> Any:  # noqa: ANN401
        """Read-only property to access the data processor/tokenizer.

        Returns:
            The associated processor object for the embedding model.

        """
        pass

    @abstractmethod
    def __call__(self, inputs: Any) -> Any:  # noqa: ANN401
        """Execute the forward pass to extract embeddings.

        Args:
            inputs: The preprocessed inputs ready for the model.

        Returns:
            The extracted embeddings/features.

        """
        pass

    @abstractmethod
    def load(self) -> Any:  # noqa: ANN401
        """Load embedding model into memory."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[float]:
        """Encode text into vector representation."""
        raise NotImplementedError
