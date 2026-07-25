"""Tests for image plotting utilities."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from xwhy.plots.image import (
    _prepare_image_for_display,
    create_image_heat_mask,
    image_heatmap,
    plot_image,
)


def test_prepare_image_for_display_numpy_denormalize() -> None:
    """Test preparation of numpy array with denormalization."""
    # (H, W, C)
    np_img = np.zeros((2, 2, 3), dtype=np.float32)
    mean = [0.5, 0.5, 0.5]
    std = [0.2, 0.2, 0.2]

    result = _prepare_image_for_display(np_img, denormalize=True, mean=mean, std=std)
    assert result.shape == (2, 2, 3)
    assert np.allclose(result, 0.5)


def test_prepare_image_for_display_numpy_denormalize_missing_stats() -> None:
    """Test missing stats trigger ValueError during numpy denormalization."""
    np_img = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="mean and std must be provided"):
        _prepare_image_for_display(np_img, denormalize=True)


def test_prepare_image_for_display_tensor_4d() -> None:
    """Test preparation of a 4D PyTorch tensor (B, C, H, W)."""
    tensor_img = torch.ones(1, 3, 2, 2) * 0.5
    result = _prepare_image_for_display(tensor_img)
    assert result.shape == (2, 2, 3)
    assert np.allclose(result, 0.5)


def test_prepare_image_for_display_tensor_3d() -> None:
    """Test preparation of a 3D PyTorch tensor (C, H, W)."""
    tensor_img = torch.ones(3, 2, 2) * 0.5
    result = _prepare_image_for_display(tensor_img)
    assert result.shape == (2, 2, 3)
    assert np.allclose(result, 0.5)


def test_prepare_image_for_display_pil() -> None:
    """Test preparation of a PIL Image."""
    np_img = np.ones((2, 2, 3), dtype=np.uint8) * 255
    pil_img = Image.fromarray(np_img)
    result = _prepare_image_for_display(pil_img)
    assert result.shape == (2, 2, 3)
    assert np.allclose(result, 1.0)


def test_prepare_image_for_display_numpy_uint8() -> None:
    """Test preparation of a uint8 Numpy array."""
    np_img = np.ones((2, 2, 3), dtype=np.uint8) * 255
    result = _prepare_image_for_display(np_img)
    assert result.shape == (2, 2, 3)
    assert np.allclose(result, 1.0)


def test_prepare_image_for_display_negative_values() -> None:
    """Test mapping of [-1, 1] range to [0, 1]."""
    np_img = np.ones((2, 2, 3), dtype=np.float32) * -1.0
    result = _prepare_image_for_display(np_img)
    assert result.shape == (2, 2, 3)
    assert np.allclose(result, 0.0)


def test_prepare_image_for_display_unsupported_type() -> None:
    """Test unsupported types raise TypeError."""
    with pytest.raises(TypeError, match="Unsupported image type"):
        _prepare_image_for_display(123)


@patch("xwhy.plots.image.plt.close")
@patch("xwhy.plots.image.plt.show")
@patch("xwhy.plots.image.plt.axis")
@patch("xwhy.plots.image.plt.title")
@patch("xwhy.plots.image.plt.imshow")
@patch("xwhy.plots.image.plt.figure")
def test_plot_image_show(
    mock_figure: MagicMock,
    mock_imshow: MagicMock,
    mock_title: MagicMock,
    mock_axis: MagicMock,
    mock_show: MagicMock,
    mock_close: MagicMock,
) -> None:
    """Test plot_image displays when save_path is None."""
    dummy_img = np.zeros((2, 2, 3), dtype=np.float32)

    plot_image(dummy_img, title="Test Plot")

    mock_figure.assert_called_once_with(figsize=(8, 6))
    mock_imshow.assert_called_once()
    mock_title.assert_called_once_with("Test Plot")
    mock_axis.assert_called_once_with("off")
    mock_show.assert_called_once()
    mock_close.assert_called_once()


@patch("xwhy.plots.image.plt.close")
@patch("xwhy.plots.image.plt.savefig")
@patch("xwhy.plots.image.plt.axis")
@patch("xwhy.plots.image.plt.imshow")
@patch("xwhy.plots.image.plt.figure")
def test_plot_image_save(
    mock_figure: MagicMock,
    mock_imshow: MagicMock,
    mock_axis: MagicMock,
    mock_savefig: MagicMock,
    mock_close: MagicMock,
) -> None:
    """Test plot_image saves when save_path is provided."""
    dummy_img = np.zeros((2, 2, 3), dtype=np.float32)

    plot_image(dummy_img, save_path="dummy_path.png")

    mock_figure.assert_called_once_with(figsize=(8, 6))
    mock_imshow.assert_called_once()
    mock_axis.assert_called_once_with("off")
    mock_savefig.assert_called_once_with("dummy_path.png", bbox_inches="tight")
    mock_close.assert_called_once()


def test_create_image_heat_mask() -> None:
    """Test mapping of coefficients to the superpixel mask."""
    superpixels = np.array([[0, 0, 1], [0, 1, 2]])
    coeffs = [0.1, 0.5, 0.9]

    heat_mask = create_image_heat_mask(superpixels, coeffs)

    expected = np.array([[0.1, 0.1, 0.5], [0.1, 0.5, 0.9]])
    np.testing.assert_array_almost_equal(heat_mask, expected)


@patch("xwhy.plots.image.plt.close")
@patch("xwhy.plots.image.plt.show")
@patch("xwhy.plots.image.plt.axis")
@patch("xwhy.plots.image.plt.title")
@patch("xwhy.plots.image.plt.colorbar")
@patch("xwhy.plots.image.plt.imshow")
@patch("xwhy.plots.image.plt.figure")
def test_image_heatmap_show(
    mock_figure: MagicMock,
    mock_imshow: MagicMock,
    mock_colorbar: MagicMock,
    mock_title: MagicMock,
    mock_axis: MagicMock,
    mock_show: MagicMock,
    mock_close: MagicMock,
) -> None:
    """Test image_heatmap displays when save_path is None."""
    mock_result = MagicMock()
    mock_result.superpixels = np.array([[0, 1], [0, 1]], dtype=int)
    mock_result.coefficients = np.array([0.2, 0.8])

    image_heatmap(mock_result, title="Test Heatmap")

    mock_figure.assert_called_once_with(figsize=(8, 6))
    mock_imshow.assert_called_once()
    mock_colorbar.assert_called_once()
    mock_title.assert_called_once_with("Test Heatmap")
    mock_axis.assert_called_once_with("off")
    mock_show.assert_called_once()
    mock_close.assert_called_once()


@patch("xwhy.plots.image.plt.close")
@patch("xwhy.plots.image.plt.savefig")
@patch("xwhy.plots.image.plt.axis")
@patch("xwhy.plots.image.plt.title")
@patch("xwhy.plots.image.plt.colorbar")
@patch("xwhy.plots.image.plt.imshow")
@patch("xwhy.plots.image.plt.figure")
def test_plot_image_heatmap_save(
    mock_figure: MagicMock,
    mock_imshow: MagicMock,
    mock_colorbar: MagicMock,
    mock_title: MagicMock,
    mock_axis: MagicMock,
    mock_savefig: MagicMock,
    mock_close: MagicMock,
) -> None:
    """Test image_heatmap saves the figure when save_path is provided."""
    mock_result = MagicMock()
    mock_result.superpixels = np.array([[0, 1], [0, 1]], dtype=int)
    mock_result.coefficients = np.array([0.2, 0.8])

    image_heatmap(mock_result, title="Saved Heatmap", save_path="heatmap_output.png")

    mock_figure.assert_called_once_with(figsize=(8, 6))
    mock_imshow.assert_called_once()
    mock_colorbar.assert_called_once()
    mock_title.assert_called_once_with("Saved Heatmap")
    mock_axis.assert_called_once_with("off")
    mock_savefig.assert_called_once_with("heatmap_output.png", bbox_inches="tight")
    mock_close.assert_called_once()


def test_prepare_image_for_display_tensor_3d_permute() -> None:
    """Ensure 3D tensors (C, H, W) are permuted correctly to (H, W, C)."""
    tensor_img = torch.ones(3, 4, 5) * 0.5

    result = _prepare_image_for_display(tensor_img)

    assert result.shape == (4, 5, 3), (
        f"Expected shape (4, 5, 3), but got {result.shape}"
    )

    assert isinstance(result, np.ndarray), "Output should be a numpy array"


def test_prepare_image_for_display_tensor_3d_no_permute() -> None:
    """Test the False branch of the permute condition in tensor preparation.

    This ensures that if a 3D tensor is passed but its first dimension is
    NOT 3 (e.g., an image already in HxWxC format or a 1-channel image),
    the permute operation is skipped and dimensions remain untouched.
    """
    # Create a tensor with shape (4, 5, 3).
    # Since ndim == 3 but shape[0] == 4 (not 3), it should bypass the IF.
    tensor_img = torch.ones(4, 5, 3) * 0.5

    result = _prepare_image_for_display(tensor_img)

    # Shape must remain exactly the same since permute was NOT called
    assert result.shape == (4, 5, 3)
    assert isinstance(result, np.ndarray)


def test_prepare_image_for_display_tensor_2d() -> None:
    """Test with a 2D tensor to bypass ndim == 3 and ndim == 4 conditions."""
    # Create a 2D grayscale tensor with shape (10, 10)
    tensor_img = torch.ones(10, 10) * 0.5

    result = _prepare_image_for_display(tensor_img)

    # Shape must remain (10, 10)
    assert result.shape == (10, 10)
    assert isinstance(result, np.ndarray)
