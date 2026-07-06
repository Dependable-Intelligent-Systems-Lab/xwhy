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
    """Test extracting importances when model has 'feature_importances_'."""
    mock_model = MagicMock()
    expected_importances = np.array([0.1, 0.9])
    mock_model.feature_importances_ = expected_importances

    surrogate = TreeBasedSurrogate(model=mock_model)

    result = surrogate.coefficients()

    np.testing.assert_array_equal(result, expected_importances)


def test_tree_coefficients_raises_attribute_error() -> None:
    """Test AttributeError is raised when 'feature_importances_' is missing."""
    mock_model = MagicMock(spec=[])
    surrogate = TreeBasedSurrogate(model=mock_model)

    with pytest.raises(
        AttributeError,
        match=re.escape("The model lacks a 'feature_importances_' attribute."),
    ):
        surrogate.coefficients()
