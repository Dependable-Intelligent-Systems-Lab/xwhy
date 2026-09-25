"""Tests for image plotting utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from xwhy.core.result import ImageClassificationXWhyResult
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


def test_prepare_image_for_display_path_and_string(tmp_path: Path) -> None:
    """Verify image loading from string path and pathlib.Path."""
    img_path = tmp_path / "test.jpg"
    dummy_pil = Image.new("RGB", (2, 2), color="red")
    dummy_pil.save(img_path)

    # Test with pathlib.Path
    result_path = _prepare_image_for_display(img_path)
    assert result_path.shape == (2, 2, 3)
    assert isinstance(result_path, np.ndarray)

    # Test with string path
    result_str = _prepare_image_for_display(str(img_path))
    assert result_str.shape == (2, 2, 3)
    assert isinstance(result_str, np.ndarray)


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


@pytest.fixture
def dummy_result() -> ImageClassificationXWhyResult:
    """Provide a dummy ImageClassificationXWhyResult."""
    result = MagicMock(spec=ImageClassificationXWhyResult)
    # 2x2 image, 2x2 segments
    result.original_image = np.ones((2, 2, 3), dtype=np.float32)
    result.superpixels = np.array([[0, 1], [2, 3]], dtype=int)
    result.coefficients = [0.5, -0.3, 0.8, -0.1]
    return result


def test_get_image_and_mask_positive_negative_conflict(
    dummy_result: ImageClassificationXWhyResult,
) -> None:
    """Test ValueError when both positive and negative only flags are true."""
    from xwhy.plots.image import get_image_and_mask

    with pytest.raises(
        ValueError,
        match="positive_only and negative_only cannot be true at the same time",
    ):
        get_image_and_mask(dummy_result, positive_only=True, negative_only=True)


def test_get_image_and_mask_resize(dummy_result: ImageClassificationXWhyResult) -> None:
    """Test resizing logic in get_image_and_mask."""
    from xwhy.plots.image import get_image_and_mask

    dummy_result.original_image = np.ones((4, 4, 3), dtype=np.float32)
    temp, mask = get_image_and_mask(dummy_result, positive_only=True, num_features=2)
    assert temp.shape == (2, 2, 3)
    assert mask.shape == (2, 2)


def test_get_image_and_mask_positive_only(
    dummy_result: ImageClassificationXWhyResult,
) -> None:
    """Test positive_only feature extraction."""
    from xwhy.plots.image import get_image_and_mask

    _temp, mask = get_image_and_mask(
        dummy_result, positive_only=True, num_features=2, min_weight=0.1
    )
    # positive coeffs: 0 (0.5), 2 (0.8)
    # They should be masked as 1
    assert mask[0, 0] == 1
    assert mask[1, 0] == 1
    assert mask[0, 1] == 0
    assert mask[1, 1] == 0


def test_get_image_and_mask_negative_only(
    dummy_result: ImageClassificationXWhyResult,
) -> None:
    """Test negative_only feature extraction."""
    from xwhy.plots.image import get_image_and_mask

    _temp, mask = get_image_and_mask(
        dummy_result,
        positive_only=False,
        negative_only=True,
        num_features=2,
        min_weight=0.0,
    )
    # negative coeffs: 1 (-0.3), 3 (-0.1)
    assert mask[0, 1] == 1
    assert mask[1, 1] == 1
    assert mask[0, 0] == 0
    assert mask[1, 0] == 0


def test_get_image_and_mask_both(dummy_result: ImageClassificationXWhyResult) -> None:
    """Test mixed features extraction with boost_channels."""
    from xwhy.plots.image import get_image_and_mask

    temp, mask = get_image_and_mask(
        dummy_result,
        positive_only=False,
        negative_only=False,
        num_features=4,
        min_weight=0.2,
        boost_channels=True,
    )
    # positive: 0, 2 -> mask 1, boost channel 1 (green)
    # negative: 1 -> mask 2, boost channel 0 (red)
    # negative: 3 (-0.1) is skipped due to min_weight=0.2
    assert mask[0, 0] == 1
    assert mask[1, 0] == 1
    assert mask[0, 1] == 2
    assert mask[1, 1] == 0

    # Check boost_channels
    assert temp[0, 0, 1] == np.max(dummy_result.original_image)
    assert temp[0, 1, 0] == np.max(dummy_result.original_image)


@patch("xwhy.plots.image.plt.close")
@patch("xwhy.plots.image.plt.show")
@patch("xwhy.plots.image.plt.axis")
@patch("xwhy.plots.image.plt.title")
@patch("xwhy.plots.image.plt.imshow")
@patch("xwhy.plots.image.plt.figure")
def test_image_boundaries_show(
    mock_figure: MagicMock,
    mock_imshow: MagicMock,
    mock_title: MagicMock,
    mock_axis: MagicMock,
    mock_show: MagicMock,
    mock_close: MagicMock,
    dummy_result: ImageClassificationXWhyResult,
) -> None:
    """Test displaying image boundaries."""
    from xwhy.plots.image import image_boundaries

    image_boundaries(dummy_result, title="Boundaries")
    mock_figure.assert_called_once()
    mock_imshow.assert_called_once()
    mock_title.assert_called_once_with("Boundaries")
    mock_show.assert_called_once()
    mock_close.assert_called_once()


@patch("xwhy.plots.image.plt.close")
@patch("xwhy.plots.image.plt.savefig")
@patch("xwhy.plots.image.plt.axis")
@patch("xwhy.plots.image.plt.imshow")
@patch("xwhy.plots.image.plt.figure")
def test_image_boundaries_save(
    mock_figure: MagicMock,
    mock_imshow: MagicMock,
    mock_axis: MagicMock,
    mock_savefig: MagicMock,
    mock_close: MagicMock,
    dummy_result: ImageClassificationXWhyResult,
) -> None:
    """Test saving image boundaries."""
    from xwhy.plots.image import image_boundaries

    image_boundaries(dummy_result, save_path="bounds.png")
    mock_savefig.assert_called_once_with("bounds.png", bbox_inches="tight")
    mock_close.assert_called_once()


@patch("xwhy.plots.image.plt.close")
@patch("xwhy.plots.image.plt.show")
@patch("xwhy.plots.image.plt.axis")
@patch("xwhy.plots.image.plt.title")
@patch("xwhy.plots.image.plt.imshow")
@patch("xwhy.plots.image.plt.figure")
def test_image_regions_show(
    mock_figure: MagicMock,
    mock_imshow: MagicMock,
    mock_title: MagicMock,
    mock_axis: MagicMock,
    mock_show: MagicMock,
    mock_close: MagicMock,
    dummy_result: ImageClassificationXWhyResult,
) -> None:
    """Test displaying image regions."""
    from xwhy.plots.image import image_regions

    image_regions(dummy_result, title="Regions")
    mock_imshow.assert_called_once()
    mock_title.assert_called_once_with("Regions")
    mock_show.assert_called_once()
    mock_close.assert_called_once()


@patch("xwhy.plots.image.plt.close")
@patch("xwhy.plots.image.plt.savefig")
@patch("xwhy.plots.image.plt.axis")
@patch("xwhy.plots.image.plt.imshow")
@patch("xwhy.plots.image.plt.figure")
def test_image_regions_save(
    mock_figure: MagicMock,
    mock_imshow: MagicMock,
    mock_axis: MagicMock,
    mock_savefig: MagicMock,
    mock_close: MagicMock,
    dummy_result: ImageClassificationXWhyResult,
) -> None:
    """Test saving image regions."""
    from xwhy.plots.image import image_regions

    image_regions(dummy_result, save_path="regions.png")
    mock_savefig.assert_called_once_with("regions.png", bbox_inches="tight")
    mock_close.assert_called_once()


@patch("xwhy.plots.image.plt.close")
@patch("xwhy.plots.image.plt.show")
@patch("xwhy.plots.image.plt.tight_layout")
@patch("xwhy.plots.image.plt.subplots")
def test_image_regions_side_by_side_show(
    mock_subplots: MagicMock,
    mock_tight_layout: MagicMock,
    mock_show: MagicMock,
    mock_close: MagicMock,
    dummy_result: ImageClassificationXWhyResult,
) -> None:
    """Test displaying image regions side by side."""
    from xwhy.plots.image import image_regions_side_by_side

    mock_fig = MagicMock()
    mock_ax1 = MagicMock()
    mock_ax2 = MagicMock()
    mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

    image_regions_side_by_side(dummy_result)

    mock_subplots.assert_called_once_with(1, 2, figsize=(8, 4))
    mock_ax1.imshow.assert_called_once()
    mock_ax2.imshow.assert_called_once()
    mock_show.assert_called_once()
    mock_close.assert_called_once()


@patch("xwhy.plots.image.plt.close")
@patch("xwhy.plots.image.plt.savefig")
@patch("xwhy.plots.image.plt.tight_layout")
@patch("xwhy.plots.image.plt.subplots")
def test_image_regions_side_by_side_save(
    mock_subplots: MagicMock,
    mock_tight_layout: MagicMock,
    mock_savefig: MagicMock,
    mock_close: MagicMock,
    dummy_result: ImageClassificationXWhyResult,
) -> None:
    """Test saving image regions side by side."""
    from xwhy.plots.image import image_regions_side_by_side

    mock_fig = MagicMock()
    mock_ax1 = MagicMock()
    mock_ax2 = MagicMock()
    mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

    image_regions_side_by_side(dummy_result, save_path="sbs.png")

    mock_savefig.assert_called_once_with("sbs.png", bbox_inches="tight")
    mock_close.assert_called_once()
