"""Base embedding abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbedding(ABC):
    """Base class for all embedding implementations."""

    @abstractmethod
    def load(self) -> Any:  # noqa: ANN401
        """Load embedding model into memory."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[float]:
        """Encode text into vector representation."""
        raise NotImplementedError
