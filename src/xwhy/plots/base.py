"""Base interfaces for plots components."""

import abc
from collections.abc import Sequence
from typing import Any

import numpy as np


class BaseTextPlotter(abc.ABC):
    """Abstract base class for text plots."""

    @abc.abstractmethod
    def plot(
        self,
        words: Sequence[str],
        scores: np.ndarray,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Plot the text plot.

        Args:
            words: Sequence of text tokens.
            scores: Array of per-token scores.
            **kwargs: Additional backend-specific plotting arguments.

        """
