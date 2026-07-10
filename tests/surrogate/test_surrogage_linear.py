"""Tests for linear surrogate."""

import re
from unittest.mock import MagicMock

import numpy as np
import pytest

from xwhy.surrogate.linear import LinearRegressionSurrogate


def test_coefficients_success() -> None:
    """Test extracting coefficients when model has 'coef_'."""
    mock_model = MagicMock()
    expected_coefs = np.array([0.5, -0.2, 1.0])
    mock_model.coef_ = expected_coefs

    surrogate = LinearRegressionSurrogate(model=mock_model)

    result = surrogate.coefficients()

    np.testing.assert_array_equal(result, expected_coefs)


def test_coefficients_raises_attribute_error() -> None:
    """Test that AttributeError is raised when 'coef_' is missing."""
    mock_model = MagicMock(spec=[])

    surrogate = LinearRegressionSurrogate(model=mock_model)

    with pytest.raises(
        AttributeError, match=re.escape("The model does not have a 'coef_' attribute.")
    ):
        surrogate.coefficients()
