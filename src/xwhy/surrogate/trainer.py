"""Trainer and selection logic for surrogate models."""

from collections.abc import Sequence

import numpy as np

from xwhy.metrics.regression import RegressionMetrics
from xwhy.surrogate.base import BaseSurrogate
from xwhy.surrogate.factory import SurrogateFactory
from xwhy.surrogate.types import SurrogateType


class SurrogateTrainer:
    """Service for training and evaluating surrogate models."""

    @staticmethod
    def compute_weights(
        method: SurrogateType,
        wmd_scores: Sequence[tuple[str, float]],
        kernel_width: float = 0.25,
    ) -> np.ndarray:
        """Compute sample weights based on distances and method type.

        Args:
            method: The surrogate method determining global or local weighting.
            wmd_scores: Sequence of (text, distance) tuples.
            kernel_width: Kernel width for exponential weighting.

        Returns:
            np.ndarray: Computed sample weights.

        """
        n_samples = len(wmd_scores)
        is_global = method in (SurrogateType.GLM_OLS, SurrogateType.GLM_RIDGE)

        if is_global:
            return np.ones(n_samples)

        dvals = np.array([d for _, d in wmd_scores])
        return np.sqrt(np.exp(-(dvals**2) / (kernel_width**2)))  # type: ignore

    @classmethod
    def fit_and_evaluate(
        cls,
        *,
        method: SurrogateType,
        perturbations: Sequence[np.ndarray],
        similarities: Sequence[tuple[str, float]],
        wmd_scores: Sequence[tuple[str, float]],
        seed: int = 1024,
        kernel_width: float = 0.25,
        ridge_alpha: float = 1.0,
    ) -> tuple[BaseSurrogate, float]:
        """Fit a surrogate model and compute its R-squared score.

        Args:
            method: The surrogate method to use.
            perturbations: Sequence of binary perturbation vectors.
            similarities: Sequence of (text, similarity) pairs (target variable).
            wmd_scores: Sequence of (text, distance) pairs.
            seed: Random seed.
            kernel_width: Kernel width for weighting.
            ridge_alpha: Ridge regularization strength.

        Returns:
            tuple[BaseSurrogate, float]: Trained model and its weighted R2 score.

        """
        x = np.vstack(perturbations)
        y = np.array([s for _, s in similarities])
        weights = cls.compute_weights(method, wmd_scores, kernel_width)

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
        perturbations: Sequence[np.ndarray],
        similarities: Sequence[tuple[str, float]],
        wmd_scores: Sequence[tuple[str, float]],
        seed: int = 1024,
        kernel_width: float = 0.25,
        ridge_alpha: float = 1.0,
    ) -> tuple[SurrogateType, float]:
        """Find the best surrogate model across all available types.

        Args:
            perturbations: Sequence of binary perturbation vectors.
            similarities: Sequence of (text, similarity) pairs.
            wmd_scores: Sequence of (text, distance) pairs.
            seed: Random seed.
            kernel_width: Kernel width.
            ridge_alpha: Ridge alpha.

        Returns:
            tuple[SurrogateType, float]: The best surrogate method type and
            its R2 score.

        """
        best_score = -float("inf")
        best_method = SurrogateType.XGBOOST

        for method in SurrogateType:
            try:
                _, score = cls.fit_and_evaluate(
                    method=method,
                    perturbations=perturbations,
                    similarities=similarities,
                    wmd_scores=wmd_scores,
                    seed=seed,
                    kernel_width=kernel_width,
                    ridge_alpha=ridge_alpha,
                )
                if score > best_score:
                    best_score = score
                    best_method = method
            except Exception:
                continue

        return best_method, best_score
