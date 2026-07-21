"""Unit tests for surrogate models."""

from typing import Any
from unittest.mock import MagicMock, patch

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

    x_matrix = np.vstack(perturbations)
    y_target = np.array([s for _, s in similarities], dtype=float)
    distances_array = np.array([d for _, d in wmd_scores], dtype=float)

    weights = SurrogateTrainer.compute_weights(SurrogateType.LIME, distances_array)
    assert len(weights) == 3

    best_method, score = SurrogateTrainer.find_best(
        x=x_matrix,
        y=y_target,
        distances=distances_array,
    )

    assert isinstance(best_method, SurrogateType)
    assert isinstance(score, float)


def test_compute_weights_normalize_false() -> None:
    """Test Branch 1: normalize_distances is False."""
    distances = np.array([1.0, 2.0, 4.0])

    weights = SurrogateTrainer.compute_weights(
        method=SurrogateType.XGBOOST,
        distances=distances,
        kernel_width=1.0,
        normalize_distances=False,
    )

    expected_weights = np.sqrt(np.exp(-(distances**2) / (1.0**2)))
    np.testing.assert_array_almost_equal(weights, expected_weights)


def test_compute_weights_normalize_true_max_greater_than_zero() -> None:
    """Test Branch 2: normalize_distances is True AND max_dist > 0."""
    distances = np.array([1.0, 2.0, 4.0])

    weights = SurrogateTrainer.compute_weights(
        method=SurrogateType.XGBOOST,
        distances=distances,
        kernel_width=1.0,
        normalize_distances=True,
    )

    expected_distances = np.array([0.25, 0.5, 1.0])
    expected_weights = np.sqrt(np.exp(-(expected_distances**2) / (1.0**2)))

    np.testing.assert_array_almost_equal(weights, expected_weights)


def test_compute_weights_normalize_true_max_zero() -> None:
    """Test Branch 3: normalize_distances is True AND max_dist <= 0."""
    distances = np.array([0.0, 0.0, 0.0])

    weights = SurrogateTrainer.compute_weights(
        method=SurrogateType.XGBOOST,
        distances=distances,
        kernel_width=1.0,
        normalize_distances=True,
    )

    expected_weights = np.array([1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(weights, expected_weights)


@patch.object(SurrogateTrainer, "fit_and_evaluate")
def test_find_best_handles_exception(mock_fit_and_evaluate: MagicMock) -> None:
    """Test exception handling block in find_best method."""

    def side_effect(method: SurrogateType, **kwargs: Any) -> tuple[None, float]:  # noqa: ANN401
        if method == SurrogateType.GLM_OLS:
            raise ValueError("Simulated training failure!")

        return None, 0.85

    mock_fit_and_evaluate.side_effect = side_effect

    x = np.array([[1, 0], [0, 1]])
    y = np.array([1.0, 0.0])
    distances = np.array([0.5, 0.5])

    _, best_score = SurrogateTrainer.find_best(x=x, y=y, distances=distances)

    assert best_score == 0.85
    assert mock_fit_and_evaluate.call_count > 1
