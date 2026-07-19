"""Base classification abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseClassification(ABC):
    """Base class for all classification implementations."""

    @abstractmethod
    def load(self) -> Any:  # noqa: ANN401
        """Load classification model into memory."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs: Any) -> Any:  # noqa: ANN401
        """Run inference on the given inputs and return predictions/logits."""
        raise NotImplementedError
