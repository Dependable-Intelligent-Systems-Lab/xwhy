"""Base interfaces for surrogate models."""

from abc import ABC, abstractmethod

import numpy as np


class BaseSurrogate(ABC):
    """Abstract base class for all surrogate models."""

    @abstractmethod
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        """Fit the surrogate model to the provided data.

        Args:
            x: Feature matrix.
            y: Target values.
            weights: Optional sample weights.

        """

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict target values for the given features.

        Args:
            x: Feature matrix.

        Returns:
            np.ndarray: Predicted values.

        """

    @abstractmethod
    def coefficients(self) -> np.ndarray:
        """Get feature contributions or coefficients.

        Returns:
            np.ndarray: Feature importances or linear coefficients.

        """
