"""Factory for creating surrogate models."""

from typing import ClassVar

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
from xgboost import XGBRegressor

from xwhy.surrogate.base import BaseSurrogate
from xwhy.surrogate.linear import LinearRegressionSurrogate
from xwhy.surrogate.tree import TreeBasedSurrogate
from xwhy.surrogate.types import SurrogateType


class SurrogateFactory:
    """Factory class for instantiating surrogate models."""

    _registry: ClassVar[dict[SurrogateType, type[BaseSurrogate]]] = {
        SurrogateType.GLM_OLS: LinearRegressionSurrogate,
        SurrogateType.GLM_RIDGE: LinearRegressionSurrogate,
        SurrogateType.LIME: LinearRegressionSurrogate,
        SurrogateType.LIME_RIDGE: LinearRegressionSurrogate,
        SurrogateType.BAYLIME: LinearRegressionSurrogate,
        SurrogateType.RANDOMFOREST: TreeBasedSurrogate,
        SurrogateType.GRADIENT_BOOSTING: TreeBasedSurrogate,
        SurrogateType.XGBOOST: TreeBasedSurrogate,
    }

    @classmethod
    def create(
        cls,
        *,
        method: SurrogateType,
        seed: int = 1024,
        ridge_alpha: float = 1.0,
    ) -> BaseSurrogate:
        """Create and configure a surrogate model instance.

        Args:
            method: The type of surrogate model to create.
            seed: Random seed for reproducibility.
            ridge_alpha: Regularization strength for Ridge models.

        Returns:
            BaseSurrogate: An instantiated and configured surrogate wrapper.

        Raises:
            ValueError: If the method is unsupported.

        """
        if method not in cls._registry:
            raise ValueError(f"Unsupported surrogate method: {method}")

        surrogate_cls = cls._registry[method]
        model: object

        if method in (SurrogateType.GLM_OLS, SurrogateType.LIME):
            model = LinearRegression()
        elif method in (SurrogateType.GLM_RIDGE, SurrogateType.LIME_RIDGE):
            model = Ridge(alpha=ridge_alpha, random_state=seed)
        elif method == SurrogateType.BAYLIME:
            model = BayesianRidge()
        elif method == SurrogateType.RANDOMFOREST:
            model = RandomForestRegressor(random_state=seed)
        elif method == SurrogateType.GRADIENT_BOOSTING:
            model = GradientBoostingRegressor(random_state=seed)
        elif method == SurrogateType.XGBOOST:
            model = XGBRegressor(random_state=seed, verbosity=0)
        else:
            raise ValueError(f"No underlying model configured for {method}")

        return surrogate_cls(model=model)
