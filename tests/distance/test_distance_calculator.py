"""Test distance calculator module."""

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xwhy.distance.calculator import calculate_distance


def test_calculate_distance_unsupported_data_type() -> None:
    """Verify TypeError when input is neither string nor ndarray (e.g., list)."""
    with pytest.raises(
        TypeError,
        match=re.escape("Source data must be either a string or a numpy array"),
    ):
        calculate_distance("cosine", [1, 2], [1, 2])


def test_calculate_distance_target_mismatch() -> None:
    """Verify TypeError when source and target types do not match."""
    with pytest.raises(
        TypeError,
        match=re.escape("Source and target must be of the exact same data type"),
    ):
        calculate_distance("cosine", np.array([1, 2]), "hello")


def test_calculate_distance_invalid_text_metric() -> None:
    """Ensure text data throws error when paired with numeric metric."""
    with pytest.raises(ValueError, match="Text data requires a text-based metric"):
        calculate_distance("cosine", "hello", "world")


def test_calculate_distance_invalid_numeric_metric() -> None:
    """Ensure numeric data throws error when paired with text metric."""
    arr = np.array([1, 2, 3])
    distance_metric = "wmd"
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Numerical data (e.g., images) cannot use text-based metrics. "
            f"Received: {distance_metric}"
        ),
    ):
        calculate_distance(distance_metric, arr, arr)


@patch("xwhy.distance.distances.CosineDistance.compute")
def test_calculate_distance_numeric_success(mock_compute: MagicMock) -> None:
    """Test successful dispatch and calculation for numerical data."""
    mock_compute.return_value = 0.85
    arr = np.array([1, 2, 3])

    result = calculate_distance("cosine", arr, arr)

    assert result == 0.85
    mock_compute.assert_called_once_with(source=arr, target=arr)


@patch("xwhy.distance.wmd.WMDDistance.compute")
def test_calculate_distance_text_success(mock_compute: MagicMock) -> None:
    """Test successful dispatch and calculation for text data."""
    mock_compute.return_value = 1.2

    result = calculate_distance("wmd", "hello", "world", model="mock_model")

    assert result == 1.2
    mock_compute.assert_called_once_with(
        source="hello", target="world", model="mock_model"
    )
