"""Trainer and selection logic for surrogate models."""

import numpy as np

from xwhy.logger import logger
from xwhy.metrics.regression import RegressionMetrics
from xwhy.surrogate.base import BaseSurrogate
from xwhy.surrogate.factory import SurrogateFactory
from xwhy.surrogate.types import SurrogateType


class SurrogateTrainer:
    """Service for training and evaluating surrogate models."""

    @staticmethod
    def compute_weights(
        method: SurrogateType,
        distances: np.ndarray,
        kernel_width: float = 0.25,
        normalize_distances: bool = False,
    ) -> np.ndarray:
        """Compute sample weights based on distances and method type.

        Args:
            method: The surrogate method determining global or local weighting.
            distances: 1D array of distances between original and perturbed inputs.
            kernel_width: Kernel width for exponential weighting.
            normalize_distances: Whether to scale distances by their max
                                 value (used in images).

        Returns:
            np.ndarray: Computed sample weights.

        """
        is_global = method in (SurrogateType.GLM_OLS, SurrogateType.GLM_RIDGE)

        if is_global:
            return np.ones(len(distances))

        if normalize_distances:
            max_dist = np.max(distances)
            if max_dist > 0:
                distances = distances / max_dist

        return np.sqrt(np.exp(-(distances**2) / (kernel_width**2)))

    @classmethod
    def fit_and_evaluate(
        cls,
        *,
        method: SurrogateType,
        x: np.ndarray,
        y: np.ndarray,
        distances: np.ndarray,
        seed: int = 1024,
        kernel_width: float = 0.25,
        ridge_alpha: float = 1.0,
        normalize_distances: bool = False,
    ) -> tuple[BaseSurrogate, float]:
        """Fit a surrogate model and compute its R-squared score.

        Args:
            method: The surrogate method to use.
            x: 2D array of perturbation feature vectors (binary matrix).
            y: 1D array of target predictions or similarity scores.
            distances: 1D array of distances between original and perturbed inputs.
            seed: Random seed.
            kernel_width: Kernel width for weighting.
            ridge_alpha: Ridge regularization strength.
            normalize_distances: Whether to scale distances by their max
                                 value (used in images).

        Returns:
            tuple[BaseSurrogate, float]: Trained model and its weighted R2 score.

        """
        logger.debug(f"  Testing surrogate method: {method}")

        weights = cls.compute_weights(
            method=method,
            distances=distances,
            kernel_width=kernel_width,
            normalize_distances=normalize_distances,
        )

        surrogate = SurrogateFactory.create(
            method=method, seed=seed, ridge_alpha=ridge_alpha
        )
        surrogate.fit(x, y, weights)

        y_pred = surrogate.predict(x)
        num_features = len(surrogate.coefficients())

        metrics = RegressionMetrics.calculate(
            y_true=y,
            y_pred=y_pred,
            weights=weights,
            num_features=num_features,
        )

        return surrogate, metrics.weighted_r2

    @classmethod
    def find_best(
        cls,
        *,
        x: np.ndarray,
        y: np.ndarray,
        distances: np.ndarray,
        seed: int = 1024,
        kernel_width: float = 0.25,
        ridge_alpha: float = 1.0,
        normalize_distances: bool = False,
    ) -> tuple[SurrogateType, float]:
        """Find the best surrogate model across all available types.

        Args:
            x: 2D array of perturbation feature vectors (binary matrix).
            y: 1D array of target predictions or similarity scores.
            distances: 1D array of distances between original and perturbed inputs.
            seed: Random seed.
            kernel_width: Kernel width.
            ridge_alpha: Ridge alpha.
            normalize_distances: Whether to scale distances by their max
                                 value (used in images).

        Returns:
            tuple[SurrogateType, float]: The best surrogate method type and
            its R2 score.

        """
        best_score = -float("inf")
        best_method = SurrogateType.XGBOOST

        logger.debug("Starting search for the best surrogate model...")
        for method in SurrogateType:
            try:
                _, score = cls.fit_and_evaluate(
                    method=method,
                    x=x,
                    y=y,
                    distances=distances,
                    seed=seed,
                    kernel_width=kernel_width,
                    ridge_alpha=ridge_alpha,
                    normalize_distances=normalize_distances,
                )

                logger.debug(f"    {method} R²ω: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_method = method
            except Exception:
                continue

        return best_method, best_score
