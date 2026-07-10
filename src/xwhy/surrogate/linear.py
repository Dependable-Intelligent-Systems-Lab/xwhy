"""Linear regression surrogate implementations."""

import numpy as np

from xwhy.surrogate.base import BaseSurrogate


class LinearRegressionSurrogate(BaseSurrogate):
    """Surrogate wrapper for linear models like OLS and Ridge."""

    def __init__(self, model: object) -> None:
        """Initialize the linear surrogate.

        Args:
            model: A scikit-learn compatible linear model instance.

        """
        self._model = model

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        """Fit the linear model.

        Args:
            x: Feature matrix.
            y: Target values.
            weights: Optional sample weights for weighted regression.

        """
        kwargs: dict[str, object] = {}
        if weights is not None:
            kwargs["sample_weight"] = weights
        self._model.fit(x, y, **kwargs)  # type: ignore

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict using the linear model.

        Args:
            x: Feature matrix.

        Returns:
            np.ndarray: Predictions.

        """
        return self._model.predict(x)  # type: ignore

    def coefficients(self) -> np.ndarray:
        """Extract coefficients from the linear model.

        Returns:
            np.ndarray: The linear coefficients.

        Raises:
            AttributeError: If the underlying model lacks a 'coef_' attribute.

        """
        if hasattr(self._model, "coef_"):
            return self._model.coef_  # type: ignore
        raise AttributeError("The model does not have a 'coef_' attribute.")
