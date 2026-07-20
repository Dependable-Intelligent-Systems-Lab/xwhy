"""Base classification abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class BaseClassification(ABC):
    """Base class for all classification implementations."""

    @property
    @abstractmethod
    def model(self) -> Any:  # noqa: ANN401
        """Read-only property to access the underlying raw model.

        Raises:
            RuntimeError: If the model has not been loaded into memory yet.

        Returns:
            The loaded raw classification model object.

        """
        pass

    @property
    @abstractmethod
    def preprocess_fn(self) -> Callable[..., Any] | None:
        """Read-only property to access the preprocessing transform function.

        Returns:
            The callable preprocessing function, or None if not applicable.

        """
        pass

    @abstractmethod
    def __call__(self, inputs: Any) -> Any:  # noqa: ANN401
        """Execute the forward pass of the classification model.

        Args:
            inputs: The preprocessed inputs (e.g., a PyTorch tensor) ready
                for inference.

        Returns:
            The raw predictions or logits from the model.

        """
        pass

    @abstractmethod
    def load(self) -> Any:  # noqa: ANN401
        """Load classification model into memory."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs: Any) -> Any:  # noqa: ANN401
        """Run inference on the given inputs and return predictions/logits."""
        raise NotImplementedError
