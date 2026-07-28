"""Tests for tree surrogate."""

import re
from unittest.mock import MagicMock

import numpy as np
import pytest

from xwhy.surrogate.tree import TreeBasedSurrogate


def test_tree_fit_with_and_without_weights() -> None:
    """Test fit method handles sample_weight correctly."""
    mock_model = MagicMock()
    surrogate = TreeBasedSurrogate(model=mock_model)

    x = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    weights = np.array([0.5, 0.5])

    surrogate.fit(x, y, weights=None)
    mock_model.fit.assert_called_with(x, y)

    surrogate.fit(x, y, weights=weights)
    mock_model.fit.assert_called_with(x, y, sample_weight=weights)


def test_tree_coefficients_success() -> None:
    """Test extracting importances when model has valid 'feature_importances_'."""
    mock_model = MagicMock()
    expected_importances = np.array([0.1, 0.9])
    mock_model.feature_importances_ = expected_importances

    surrogate = TreeBasedSurrogate(model=mock_model)

    result = surrogate.coefficients()

    np.testing.assert_array_equal(result, expected_importances)


def test_tree_coefficients_returns_empty_for_none() -> None:
    """Test returning empty array when 'feature_importances_' is None.

    This ensures the zero variance / untraversable tree edge case is handled
    cleanly without raising type errors.
    """
    mock_model = MagicMock()
    mock_model.feature_importances_ = None

    surrogate = TreeBasedSurrogate(model=mock_model)
    result = surrogate.coefficients()

    expected = np.zeros((0,), dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_tree_coefficients_returns_zeros_for_nan() -> None:
    """Test returning array of zeros when importances contain NaN.

    This ensures the method handles scikit-learn's zero variance bug
    by safely replacing NaN arrays with zeros of the identical shape.
    """
    mock_model = MagicMock()
    # Mocking an array that contains NaN values due to zero variance
    mock_model.feature_importances_ = np.array([0.5, np.nan, 0.2])

    surrogate = TreeBasedSurrogate(model=mock_model)
    result = surrogate.coefficients()

    # The expected shape is (3,) full of zeros
    expected = np.zeros(3, dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_tree_coefficients_raises_attribute_error() -> None:
    """Test AttributeError is raised when 'feature_importances_' is missing."""
    mock_model = MagicMock(spec=[])
    surrogate = TreeBasedSurrogate(model=mock_model)

    with pytest.raises(
        AttributeError,
        match=re.escape("The model lacks a 'feature_importances_' attribute."),
    ):
        surrogate.coefficients()
