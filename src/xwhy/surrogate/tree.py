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
            np.ndarray: The feature importances, or an array of zeros if
            NaN occurs due to zero variance or lack of tree splits.

        Raises:
            AttributeError: If the underlying model lacks a
                            'feature_importances_' attribute.

        """
        if hasattr(self._model, "feature_importances_"):
            raw_importances = self._model.feature_importances_
            if raw_importances is None:
                return np.zeros((0,), dtype=float)

            importances = np.asarray(raw_importances, dtype=float)

            # Check for NaN values or division by zero errors in importances
            if np.isnan(importances).any():
                # Return an array of zeros matching the feature shape
                return np.zeros(importances.shape, dtype=float)

            return importances

        raise AttributeError("The model lacks a 'feature_importances_' attribute.")
