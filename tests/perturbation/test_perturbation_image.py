"""Tests for image perturbation class."""

import re
from unittest.mock import patch

import numpy as np
import pytest
import torch

from xwhy.perturbation.image import ImagePerturbation


@pytest.fixture
def perturber() -> ImagePerturbation:
    """Fixture providing a default ImagePerturbation instance."""
    return ImagePerturbation(kernel_size=5, max_dist=100, ratio=0.5, seed=42)


def test_initialization() -> None:
    """Test parameters are initialized correctly."""
    pert = ImagePerturbation(kernel_size=8, max_dist=150, ratio=0.3, seed=12)
    assert pert.kernel_size == 8
    assert pert.max_dist == 150
    assert pert.ratio == 0.3
    assert pert.seed == 12


def test_generate_superpixels_tensor_chw(perturber: ImagePerturbation) -> None:
    """Test generating superpixels from a (C, H, W) PyTorch Tensor."""
    tensor_img = torch.rand(3, 10, 10)

    with patch("skimage.segmentation.quickshift") as mock_quickshift:
        mock_quickshift.return_value = np.zeros((10, 10), dtype=int)

        superpixels, num_sp = perturber.generate_superpixels(tensor_img)

        assert superpixels.shape == (10, 10)
        assert num_sp == 1

        passed_array = mock_quickshift.call_args[0][0]
        assert passed_array.shape == (10, 10, 3)
        mock_quickshift.assert_called_with(
            passed_array, kernel_size=5, max_dist=100, ratio=0.5, rng=42
        )


def test_generate_superpixels_tensor_hwc(perturber: ImagePerturbation) -> None:
    """Test generating superpixels from a (H, W, C) PyTorch Tensor."""
    tensor_img = torch.rand(10, 10, 3)

    with patch("skimage.segmentation.quickshift") as mock_quickshift:
        mock_quickshift.return_value = np.ones((10, 10), dtype=int)

        superpixels, num_sp = perturber.generate_superpixels(tensor_img)

        assert superpixels.shape == (10, 10)
        assert num_sp == 1


def test_generate_superpixels_tensor_invalid_shape(
    perturber: ImagePerturbation,
) -> None:
    """Test ValueError is raised for invalid tensor shapes."""
    tensor_img = torch.rand(10, 10)
    with pytest.raises(ValueError, match=re.escape("Unexpected tensor shape")):
        perturber.generate_superpixels(tensor_img)


def test_generate_superpixels_ndarray(perturber: ImagePerturbation) -> None:
    """Test generating superpixels directly from a NumPy array."""
    rng = np.random.default_rng(42)
    np_img = rng.random((10, 10, 3))

    with patch("skimage.segmentation.quickshift") as mock_quickshift:
        mock_quickshift.return_value = np.array([[0, 1], [2, 0]])
        superpixels, num_sp = perturber.generate_superpixels(np_img)

        assert superpixels.shape == (2, 2)
        assert num_sp == 3


def test_generate_superpixels_invalid_type(perturber: ImagePerturbation) -> None:
    """Test TypeError is raised for non-tensor/non-numpy inputs."""
    with pytest.raises(
        TypeError, match=re.escape("image must be a torch.Tensor or np.ndarray")
    ):
        perturber.generate_superpixels("invalid_image_type")  # type: ignore


def test_generate(perturber: ImagePerturbation) -> None:
    """Test generating boolean perturbation masks."""
    masks = perturber.generate(
        num_superpixels=10, num_perturbations=50, keep_probability=0.5
    )

    assert isinstance(masks, np.ndarray)
    assert masks.shape == (50, 10)
    unique_vals = np.unique(masks)
    assert np.all(np.isin(unique_vals, [0, 1]))


def test_apply_mask_with_segments(perturber: ImagePerturbation) -> None:
    """Test apply_mask effectively masks out inactive superpixels."""
    img = np.ones((2, 2, 3), dtype=np.float32)
    segments = np.array([[0, 1], [2, 3]])
    mask = np.array([1, 0, 0, 1])

    perturbed_img = perturber.apply_mask(item=img, mask=mask, segments=segments)

    assert perturbed_img.shape == (2, 2, 3)
    assert np.allclose(perturbed_img[0, 0], [1.0, 1.0, 1.0])  # Active
    assert np.allclose(perturbed_img[1, 1], [1.0, 1.0, 1.0])  # Active
    assert np.allclose(perturbed_img[0, 1], [0.0, 0.0, 0.0])  # Inactive
    assert np.allclose(perturbed_img[1, 0], [0.0, 0.0, 0.0])  # Inactive


def test_apply_mask_missing_segments(perturber: ImagePerturbation) -> None:
    """Test ValueError when segments parameter is missing."""
    img = np.ones((2, 2, 3), dtype=np.float32)
    mask = np.array([1, 0, 0, 1])

    with pytest.raises(
        ValueError, match="segments \\(superpixel map\\) must be provided"
    ):
        perturber.apply_mask(item=img, mask=mask)


def test_apply_mask_all_inactive(perturber: ImagePerturbation) -> None:
    """Test apply_mask with no active pixels ensures loops run correctly."""
    img = np.ones((2, 2, 3), dtype=np.float32)
    segments = np.array([[0, 1], [2, 3]])
    mask = np.array([0, 0, 0, 0])

    perturbed_img = perturber.apply_mask(item=img, mask=mask, segments=segments)
    assert np.allclose(perturbed_img, 0.0)
