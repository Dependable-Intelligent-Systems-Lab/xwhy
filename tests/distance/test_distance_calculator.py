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
        match=re.escape("Source data must be either a string or a numpy array."),
    ):
        calculate_distance("cosine", [1, 2], [1, 2])


def test_calculate_distance_target_mismatch() -> None:
    """Verify TypeError when source and target types do not match."""
    with pytest.raises(
        TypeError,
        match=re.escape("Source and target must be of the exact same data type."),
    ):
        calculate_distance("cosine", np.array([1, 2]), "hello")


def test_calculate_distance_invalid_metric_string() -> None:
    """Ensure unknown metric strings raise ValueError from DistanceType."""
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="is not a valid DistanceType"):
        calculate_distance("fake_invalid_metric", source=arr, target=arr)


@patch("xwhy.distance.distances.CosineDistance.compute")
def test_calculate_distance_numeric_success(mock_compute: MagicMock) -> None:
    """Test successful dispatch and calculation for numerical data."""
    mock_compute.return_value = 0.85
    arr = np.array([1, 2, 3])

    result = calculate_distance("cosine", arr, arr)

    assert result == 0.85
    mock_compute.assert_called_once_with(source=arr, target=arr)


def _make_tensor_mock(array: np.ndarray) -> MagicMock:
    """Create a mock that behaves like a PyTorch tensor.

    Args:
        array: The numpy array that ``.numpy()`` should return.

    Returns:
        MagicMock: A mock with ``detach().cpu().numpy()`` chain.

    """
    tensor = MagicMock()
    tensor.detach.return_value.cpu.return_value.numpy.return_value = array
    return tensor


@patch("xwhy.distance.distances.CosineDistance.compute")
def test_calculate_distance_source_tensor_conversion(
    mock_compute: MagicMock,
) -> None:
    """Verify source PyTorch-like tensor is converted to ndarray before dispatch."""
    mock_compute.return_value = 0.42
    source_arr = np.array([1.0, 2.0, 3.0])
    target_arr = np.array([1.0, 2.0, 3.0])
    source_tensor = _make_tensor_mock(source_arr)

    result = calculate_distance("cosine", source_tensor, target_arr)

    assert result == 0.42
    mock_compute.assert_called_once_with(source=source_arr, target=target_arr)


@patch("xwhy.distance.distances.CosineDistance.compute")
def test_calculate_distance_target_tensor_conversion(
    mock_compute: MagicMock,
) -> None:
    """Verify target PyTorch-like tensor is converted to ndarray before dispatch."""
    mock_compute.return_value = 0.55
    source_arr = np.array([4.0, 5.0, 6.0])
    target_arr = np.array([4.0, 5.0, 6.0])
    target_tensor = _make_tensor_mock(target_arr)

    result = calculate_distance("cosine", source_arr, target_tensor)

    assert result == 0.55
    mock_compute.assert_called_once_with(source=source_arr, target=target_arr)


@patch("xwhy.distance.distances.CosineDistance.compute")
def test_calculate_distance_both_tensors_conversion(
    mock_compute: MagicMock,
) -> None:
    """Verify both source and target tensors are converted to ndarrays."""
    mock_compute.return_value = 0.99
    source_arr = np.array([7.0, 8.0])
    target_arr = np.array([9.0, 10.0])
    source_tensor = _make_tensor_mock(source_arr)
    target_tensor = _make_tensor_mock(target_arr)

    result = calculate_distance("cosine", source_tensor, target_tensor)

    assert result == 0.99
    mock_compute.assert_called_once_with(source=source_arr, target=target_arr)


@patch("xwhy.distance.distances.CosineDistance.compute_with_p_value")
def test_calculate_distance_return_p_value(
    mock_compute_p: MagicMock,
) -> None:
    """Dispatch to compute_with_p_value when return_p_value is True."""
    mock_compute_p.return_value = (0.03, 0.85)
    arr = np.array([1.0, 2.0, 3.0])

    result = calculate_distance(
        "cosine",
        arr,
        arr,
        return_p_value=True,
        n_bootstrap=50,
    )

    assert result == (0.03, 0.85)
    mock_compute_p.assert_called_once_with(
        source=arr,
        target=arr,
        n_bootstrap=50,
    )


@patch("xwhy.distance.distances.WassersteinDistance.compute")
def test_calculate_distance_wasserstein_dispatch(mock_compute: MagicMock) -> None:
    """Dispatch to WassersteinDistance for the wasserstein metric."""
    mock_compute.return_value = 0.33
    arr = np.array([1.0, 2.0, 3.0])

    result = calculate_distance("wasserstein", arr, arr, mode="spatial")

    assert result == 0.33
    mock_compute.assert_called_once_with(source=arr, target=arr, mode="spatial")


def test_calculate_distance_text_with_numeric_metric() -> None:
    """Raise ValueError when text data uses a non-text metric."""
    with pytest.raises(
        ValueError,
        match="Text data requires a text-based metric",
    ):
        calculate_distance("cosine", "hello", "world")


def test_calculate_distance_numeric_with_text_metric() -> None:
    """Raise ValueError when numeric data uses a text metric."""
    arr = np.array([1.0, 2.0, 3.0])
    with pytest.raises(
        ValueError,
        match="cannot use text-based metrics",
    ):
        calculate_distance("wmd", arr, arr)
