"""Base abstractions for statistical distance implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDistance(ABC):
    """Abstract base class for unified distance implementations.

    A distance implementation is responsible for computing a statistical
    distance between two pieces of data (Text, Images, or Tabular data).
    """

    @abstractmethod
    def compute(
        self,
        source: Any,  # noqa: ANN401
        target: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> float:
        """Compute the distance between two inputs.

        Args:
            source: First input (e.g., str for NLP, np.ndarray for Image/Tabular).
            target: Second input (e.g., str for NLP, np.ndarray for Image/Tabular).
            **kwargs: Additional parameters required by specific metrics
                      (e.g., `model` for WMD).

        Returns:
            float: Computed distance value.

        """
        raise NotImplementedError
