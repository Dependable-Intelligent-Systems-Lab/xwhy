"""Tree-based surrogate implementations."""

import numpy as np

from xwhy.surrogate.base import BaseSurrogate


class TreeBasedSurrogate(BaseSurrogate):
    """Surrogate wrapper for tree-based models like Random Forest and XGBoost."""

    def __init__(self, model: object) -> None:
        """Initialize the tree-based surrogate.

        Args:
            model: A scikit-learn or xgboost compatible tree model instance.

        """
        self._model = model

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        """Fit the tree model.

        Args:
            x: Feature matrix.
            y: Target values.
            weights: Optional sample weights.

        """
        kwargs: dict[str, object] = {}
        if weights is not None:
            kwargs["sample_weight"] = weights
        self._model.fit(x, y, **kwargs)  # type: ignore

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict using the tree model.

        Args:
            x: Feature matrix.

        Returns:
            np.ndarray: Predicted values.

        """
        return self._model.predict(x)  # type: ignore

    def coefficients(self) -> np.ndarray:
        """Extract feature importances from the tree model.

        Returns:
            np.ndarray: The feature importances.

        Raises:
            AttributeError: If the underlying model lacks a
                            'feature_importances_' attribute.

        """
        if hasattr(self._model, "feature_importances_"):
            return self._model.feature_importances_  # type: ignore
        raise AttributeError("The model lacks a 'feature_importances_' attribute.")
