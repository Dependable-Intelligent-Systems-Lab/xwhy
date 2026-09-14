"""Unit tests for PointCloudPerturbation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from xwhy.perturbation.point_cloud import PointCloudPerturbation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perturber() -> PointCloudPerturbation:
    """Return a PointCloudPerturbation with a fixed seed."""
    return PointCloudPerturbation(removal_probability=0.5, seed=0)


@pytest.fixture
def sample_cloud() -> torch.Tensor:
    """Return a small point-cloud tensor of shape (N, 3)."""
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def segments() -> np.ndarray:
    """Return cluster labels for the five sample points."""
    return np.array([0, 0, 1, 1, 2], dtype=int)


# ---------------------------------------------------------------------------
# __init__ / set_seed
# ---------------------------------------------------------------------------


def test_init_stores_parameters() -> None:
    """Store removal_probability and seed on the instance."""
    p = PointCloudPerturbation(removal_probability=0.3, seed=99)
    assert p.removal_probability == 0.3
    assert p.seed == 99
    assert p._rng is not None


def test_set_seed_updates_rng(perturber: PointCloudPerturbation) -> None:
    """Replace the internal RNG when a new seed is supplied."""
    old_rng = perturber._rng
    perturber.set_seed(123)
    assert perturber.seed == 123
    assert perturber._rng is not old_rng


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def test_generate_shape_and_dtype(perturber: PointCloudPerturbation) -> None:
    """Return a binary array of the requested shape."""
    masks = perturber.generate(num_clusters=4, num_perturbations=10)
    assert isinstance(masks, np.ndarray)
    assert masks.shape == (10, 4)
    assert set(np.unique(masks)).issubset({0, 1})


def test_generate_reproducible_with_seed() -> None:
    """Produce identical masks for the same seed."""
    p1 = PointCloudPerturbation(removal_probability=0.4, seed=7)
    p2 = PointCloudPerturbation(removal_probability=0.4, seed=7)
    m1 = p1.generate(num_clusters=3, num_perturbations=5)
    m2 = p2.generate(num_clusters=3, num_perturbations=5)
    np.testing.assert_array_equal(m1, m2)


def test_generate_accepts_extra_args(
    perturber: PointCloudPerturbation,
) -> None:
    """Ignore unused positional and keyword arguments."""
    masks = perturber.generate(
        "ignored",
        num_clusters=2,
        num_perturbations=3,
        extra_kw=True,
    )
    assert masks.shape == (3, 2)


def test_generate_respects_removal_probability() -> None:
    """Keep fewer clusters when removal_probability is high."""
    # p_keep = 1 - 0.9 = 0.1 => most entries should be 0
    p = PointCloudPerturbation(removal_probability=0.9, seed=0)
    masks = p.generate(num_clusters=20, num_perturbations=50)
    keep_rate = masks.mean()
    assert keep_rate < 0.25


# ---------------------------------------------------------------------------
# apply_mask
# ---------------------------------------------------------------------------


def test_apply_mask_with_keyword_segments(
    perturber: PointCloudPerturbation,
    sample_cloud: torch.Tensor,
    segments: np.ndarray,
) -> None:
    """Keep only points whose cluster is marked 1 in the mask."""
    # Keep clusters 0 and 2, drop cluster 1
    mask = np.array([1, 0, 1])
    result = perturber.apply_mask(item=sample_cloud, mask=mask, segments=segments)

    assert isinstance(result, torch.Tensor)
    # Points 0,1 belong to cluster 0; point 4 belongs to cluster 2
    assert result.shape[0] == 3
    expected = sample_cloud[[0, 1, 4]]
    assert torch.equal(result, expected)


def test_apply_mask_with_positional_segments(
    perturber: PointCloudPerturbation,
    sample_cloud: torch.Tensor,
    segments: np.ndarray,
) -> None:
    """Accept segments supplied as a positional argument."""
    mask = np.array([0, 1, 0])
    # segments passed positionally after mask
    result = perturber.apply_mask(sample_cloud, mask, segments)

    # Only points belonging to cluster 1 (indices 2, 3)
    assert result.shape[0] == 2
    expected = sample_cloud[[2, 3]]
    assert torch.equal(result, expected)


def test_apply_mask_missing_segments_raises(
    perturber: PointCloudPerturbation,
    sample_cloud: torch.Tensor,
) -> None:
    """Raise ValueError when segments are not provided."""
    mask = np.array([1, 0, 1])
    with pytest.raises(ValueError, match="segments \\(cluster labels\\) must be"):
        perturber.apply_mask(item=sample_cloud, mask=mask)


def test_apply_mask_all_removed_returns_empty(
    perturber: PointCloudPerturbation,
    sample_cloud: torch.Tensor,
    segments: np.ndarray,
) -> None:
    """Return an empty tensor when every cluster is masked out."""
    mask = np.array([0, 0, 0])
    result = perturber.apply_mask(item=sample_cloud, mask=mask, segments=segments)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == 0


def test_apply_mask_keeps_all_points(
    perturber: PointCloudPerturbation,
    sample_cloud: torch.Tensor,
    segments: np.ndarray,
) -> None:
    """Return the original cloud when the mask keeps every cluster."""
    mask = np.array([1, 1, 1])
    result = perturber.apply_mask(item=sample_cloud, mask=mask, segments=segments)
    assert torch.equal(result, sample_cloud)
