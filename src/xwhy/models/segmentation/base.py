"""Base segmentation abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class BaseSegmentation(ABC):
    """Base class for all segmentation implementations."""

    @property
    @abstractmethod
    def model(self) -> Any:  # noqa: ANN401
        """Read-only property to access the underlying raw segmentation model.

        Raises:
            RuntimeError: If the model has not been loaded into memory yet.

        Returns:
            The loaded raw segmentation model object.

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
        """Execute the forward pass of the segmentation model.

        Args:
            inputs: The preprocessed inputs (e.g., a PyTorch tensor).

        Returns:
            The segmentation masks or logits output by the model.

        """
        pass

    @property
    @abstractmethod
    def class_names(self) -> list[str]:
        """Return the list of class names (categories) supported by the model."""
        raise NotImplementedError

    @abstractmethod
    def load(self) -> Any:  # noqa: ANN401
        """Load segmentation model into memory."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs: Any) -> Any:  # noqa: ANN401
        """Run inference on the given inputs and return predictions/masks."""
        raise NotImplementedError
