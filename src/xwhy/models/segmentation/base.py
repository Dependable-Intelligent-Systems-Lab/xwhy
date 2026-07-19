"""Base segmentation abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSegmentation(ABC):
    """Base class for all segmentation implementations."""

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
