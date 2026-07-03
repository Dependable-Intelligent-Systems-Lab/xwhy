"""Base abstractions for statistical distance implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseDistance(ABC):
    """Abstract base class for distance implementations.

    A distance implementation is responsible for computing a statistical
    distance between two pieces of data.

    Examples include:

    - Word Mover's Distance (WMD)
    - Wasserstein distance
    - Kolmogorov-Smirnov distance
    - Kuiper statistic

    Concrete subclasses define the required model type and the specific
    distance algorithm.
    """

    @abstractmethod
    def compute(
        self,
        *,
        model: object,
        source: str,
        target: str,
    ) -> float:
        """Compute the distance between two inputs.

        Args:
            model:
                Backend model required by the implementation.

            source:
                First input.

            target:
                Second input.

        Returns:
            Computed distance value.

        """
        raise NotImplementedError
