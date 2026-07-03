"""Base embedding abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Base class for all embedding implementations."""

    @abstractmethod
    def load(self) -> object:
        """Load embedding model into memory."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[float]:
        """Encode text into vector representation."""
        raise NotImplementedError
