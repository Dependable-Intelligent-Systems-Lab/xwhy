"""Unit tests for surrogate models."""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from xwhy.surrogate.factory import SurrogateFactory
from xwhy.surrogate.linear import LinearRegressionSurrogate
from xwhy.surrogate.trainer import SurrogateTrainer
from xwhy.surrogate.types import SurrogateType


def test_linear_regression_surrogate() -> None:
    """Test LinearRegressionSurrogate functionality."""
    model = LinearRegression()
    surrogate = LinearRegressionSurrogate(model)

    x = np.array([[1, 2], [3, 4]])
    y = np.array([3, 7])
    surrogate.fit(x, y)

    preds = surrogate.predict(x)
    assert len(preds) == 2

    coefs = surrogate.coefficients()
    assert len(coefs) == 2


def test_surrogate_factory() -> None:
    """Test factory creation of surrogates."""
    surrogate = SurrogateFactory.create(method=SurrogateType.GLM_OLS)
    assert isinstance(surrogate, LinearRegressionSurrogate)

    with pytest.raises(ValueError, match="Unsupported surrogate method"):
        SurrogateFactory.create(method="invalid_method")  # type: ignore


def test_surrogate_trainer() -> None:
    """Test trainer pipeline and selection logic."""
    perturbations = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
    similarities = [("t1", 0.5), ("t2", 0.6), ("t3", 0.9)]
    wmd_scores = [("t1", 0.1), ("t2", 0.2), ("t3", 0.3)]

    weights = SurrogateTrainer.compute_weights(SurrogateType.LIME, wmd_scores)
    assert len(weights) == 3

    best_method, score = SurrogateTrainer.find_best(
        perturbations=perturbations,
        similarities=similarities,
        wmd_scores=wmd_scores,
    )

    assert isinstance(best_method, SurrogateType)
    assert isinstance(score, float)
