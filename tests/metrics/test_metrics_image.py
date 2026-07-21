"""Tests for image-specific coverage metrics."""

import numpy as np
import pytest

from xwhy.metrics.image import ImageCoverageMetrics


def test_validate_shapes_success() -> None:
    """Test shape validation with matching spatial dimensions."""
    exp = np.zeros((10, 10, 3))
    mask = np.zeros((10, 10))
    # Should not raise any exception
    ImageCoverageMetrics._validate_shapes(exp, mask)


def test_validate_shapes_mismatch() -> None:
    """Test shape validation raises ValueError on mismatch."""
    exp = np.zeros((10, 10, 3))
    mask = np.zeros((10, 12))

    with pytest.raises(ValueError, match="Spatial dimensions mismatch"):
        ImageCoverageMetrics._validate_shapes(exp, mask)


def test_calculate_coverage_zero_target() -> None:
    """Test coverage calculation when target class is absent from the mask."""
    exp = np.zeros((5, 5))
    mask = np.zeros((5, 5))  # Only background (0)
    score = ImageCoverageMetrics.calculate_coverage(exp, mask, class_of_interest=1)

    assert score == 0.0


def test_calculate_coverage_2d() -> None:
    """Test 2D coverage calculation with rewards and penalties."""
    # Mask containing class 1 (target), class 2 (other), and 0 (background)
    mask = np.array(
        [
            [1, 1, 0],
            [2, 2, 0],
            [0, 0, 0],
        ]
    )

    # Explanation active on one target pixel, one 'other' pixel, and one bg pixel
    exp = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ]
    )

    # total_target_pixels = 2
    # rewards = 1 (top-left)
    # penalties = 1 (middle-left, overlapping with class 2)
    # tot = 1 - 1 = 0 -> score = 0.0
    score = ImageCoverageMetrics.calculate_coverage(exp, mask, class_of_interest=1)
    assert score == 0.0

    # Optimal explanation covering only targets
    exp_optimal = np.array(
        [
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    score_optimal = ImageCoverageMetrics.calculate_coverage(
        exp_optimal, mask, class_of_interest=1
    )
    assert score_optimal == 1.0


def test_calculate_coverage_3d() -> None:
    """Test 3D coverage calculation on multi-channel images."""
    mask = np.array(
        [
            [1, 2],
            [0, 0],
        ]
    )

    # Shape: (2, 2, 3)
    exp = np.zeros((2, 2, 3))
    # Active on target pixel
    exp[0, 0, 0] = 1
    # Active on background pixel (does not incur penalty, only 'other' class does)
    exp[1, 0, 1] = 1

    # total_target_pixels = 1
    # rewards = 1, penalties = 0
    # score = (1 - 0) / 1 = 1.0
    score = ImageCoverageMetrics.calculate_coverage(exp, mask, class_of_interest=1)
    assert score == 1.0


def test_calculate_weighted_coverage_2d() -> None:
    """Test 2D weighted coverage with positive and negative weights."""
    mask = np.array(
        [
            [1, 2],
            [0, 0],
        ]
    )

    # Weight map applied internally:
    #  1.0  -1.0
    # -1.0  -1.0

    exp = np.array(
        [
            [0.5, 0.2],
            [0.0, 0.1],
        ]
    )

    # Expected weighted sum:
    # (0.5 * 1.0) + (0.2 * -1.0) + (0.0 * -1.0) + (0.1 * -1.0)
    # = 0.5 - 0.2 + 0.0 - 0.1 = 0.2
    # Normalization (total pixels) = 4
    # Expected score: 0.2 / 4 = 0.05
    score = ImageCoverageMetrics.calculate_weighted_coverage(
        exp, mask, class_of_interest=1
    )
    assert np.isclose(score, 0.05)


def test_calculate_weighted_coverage_3d() -> None:
    """Test 3D weighted coverage handling dimensional expansion."""
    mask = np.array(
        [
            [1, 2],
            [0, 0],
        ]
    )

    # Add a single channel dimension
    exp = np.array(
        [
            [[0.5], [0.2]],
            [[0.0], [0.1]],
        ]
    )

    score = ImageCoverageMetrics.calculate_weighted_coverage(
        exp, mask, class_of_interest=1
    )
    assert np.isclose(score, 0.05)


def test_evaluate_all() -> None:
    """Test that evaluate_all computes both metrics correctly and returns a tuple."""
    mask = np.array(
        [
            [1, 2],
            [0, 0],
        ]
    )

    exp = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    )

    # Manually expected values:
    # Standard Coverage: rewards=1, penalties=0, target=1 => 1.0
    # Weighted Coverage: sum(1*1) = 1.0, total=4 => 0.25
    cov, w_cov = ImageCoverageMetrics.evaluate_all(exp, mask, class_of_interest=1)

    assert cov == 1.0
    assert w_cov == 0.25
