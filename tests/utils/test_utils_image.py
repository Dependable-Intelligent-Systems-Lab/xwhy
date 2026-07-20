"""Tests for image utilities."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from xwhy.utils.image import (
    denormalize_tensor,
    get_default_transform,
    load_image_as_tensor,
    numpy_image_to_tensor,
    tensor_to_numpy_image,
)


def test_get_default_transform() -> None:
    """Test fallback transforms creation."""
    transform = get_default_transform()
    assert transform is not None
    assert callable(transform)


@patch("xwhy.utils.image.Image.open")
def test_load_image_as_tensor_default(mock_open: MagicMock) -> None:
    """Test load_image_as_tensor without a specific transform pipeline."""
    mock_img = MagicMock(spec=Image.Image)
    mock_img.convert.return_value = mock_img
    mock_open.return_value = mock_img

    dummy_transform = MagicMock(return_value=torch.zeros(3, 224, 224))

    with patch("xwhy.utils.image.get_default_transform", return_value=dummy_transform):
        tensor, img = load_image_as_tensor("dummy.jpg")

    assert tensor.shape == (1, 3, 224, 224)
    assert img == mock_img
    mock_open.assert_called_once_with("dummy.jpg")
    dummy_transform.assert_called_once_with(mock_img)


@patch("xwhy.utils.image.Image.open")
def test_load_image_as_tensor_custom(mock_open: MagicMock) -> None:
    """Test load_image_as_tensor with a custom transform pipeline."""
    mock_img = MagicMock(spec=Image.Image)
    mock_img.convert.return_value = mock_img
    mock_open.return_value = mock_img

    custom_transform = MagicMock(return_value=torch.ones(3, 100, 100))

    tensor, img = load_image_as_tensor("dummy.jpg", transform_fn=custom_transform)

    assert tensor.shape == (1, 3, 100, 100)
    assert img == mock_img
    custom_transform.assert_called_once_with(mock_img)


@patch("xwhy.utils.image.Image.fromarray")
def test_numpy_image_to_tensor_uint8(mock_fromarray: MagicMock) -> None:
    """Test numpy_image_to_tensor with a uint8 array."""
    mock_img = MagicMock(spec=Image.Image)
    mock_fromarray.return_value = mock_img

    custom_transform = MagicMock(return_value=torch.ones(3, 10, 10))
    np_arr = np.zeros((10, 10, 3), dtype=np.uint8)

    tensor = numpy_image_to_tensor(np_arr, transform_fn=custom_transform)

    assert tensor.shape == (1, 3, 10, 10)
    mock_fromarray.assert_called_once()
    custom_transform.assert_called_once_with(mock_img)


@patch("xwhy.utils.image.Image.fromarray")
def test_numpy_image_to_tensor_float(mock_fromarray: MagicMock) -> None:
    """Test numpy_image_to_tensor with a float array."""
    mock_img = MagicMock(spec=Image.Image)
    mock_fromarray.return_value = mock_img

    dummy_transform = MagicMock(return_value=torch.zeros(3, 10, 10))
    np_arr = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)

    with patch("xwhy.utils.image.get_default_transform", return_value=dummy_transform):
        tensor = numpy_image_to_tensor(np_arr)

    assert tensor.shape == (1, 3, 10, 10)
    mock_fromarray.assert_called_once()

    # Ensure proper conversion to uint8
    passed_array = mock_fromarray.call_args[0][0]
    assert passed_array.dtype == np.uint8
    assert passed_array[0, 0, 0] == 127


def test_denormalize_tensor() -> None:
    """Test tensor denormalization arithmetic."""
    tensor = torch.zeros(1, 3, 2, 2)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    result = denormalize_tensor(tensor, mean, std)

    # Computation: (0 * 0.5) + 0.5 = 0.5
    assert result.shape == (1, 3, 2, 2)
    assert torch.allclose(result, torch.tensor(0.5))


def test_tensor_to_numpy_image_no_denorm() -> None:
    """Test converting tensor to numpy without denormalization."""
    tensor = torch.ones(1, 3, 2, 2) * 2.0  # Multiplying by 2.0 to check clipping

    np_img = tensor_to_numpy_image(tensor, denormalize=False)

    assert isinstance(np_img, np.ndarray)
    assert np_img.shape == (2, 2, 3)
    assert np.all(np_img == 1.0)  # Clipped safely


def test_tensor_to_numpy_image_with_denorm() -> None:
    """Test converting tensor to numpy with active denormalization."""
    tensor = torch.zeros(1, 3, 2, 2)
    mean = [0.5, 0.5, 0.5]
    std = [0.2, 0.2, 0.2]

    np_img = tensor_to_numpy_image(tensor, denormalize=True, mean=mean, std=std)

    assert np_img.shape == (2, 2, 3)
    assert np.allclose(np_img, 0.5)


def test_tensor_to_numpy_image_missing_stats() -> None:
    """Test missing stats trigger a ValueError when denormalize is requested."""
    tensor = torch.zeros(1, 3, 2, 2)

    with pytest.raises(ValueError, match="mean and std must be provided"):
        tensor_to_numpy_image(tensor, denormalize=True)
