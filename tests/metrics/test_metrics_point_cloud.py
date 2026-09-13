"""Unit tests for point cloud evaluation metrics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from xwhy.metrics.point_cloud import (
    calculate_jaccard_stability_score,
    compute_noisy_explanations,
    generate_spherical_noise_points,
)

# ---------------------------------------------------------------------------
# generate_spherical_noise_points
# ---------------------------------------------------------------------------


def test_generate_spherical_noise_points_shape_and_seed() -> None:
    """Return points of the requested shape and honour the random seed."""
    center = np.array([0.0, 0.0, 0.0])
    points_a = generate_spherical_noise_points(
        center=center, radius=1.0, num_points=5, seed=123
    )
    points_b = generate_spherical_noise_points(
        center=center, radius=1.0, num_points=5, seed=123
    )

    assert points_a.shape == (5, 3)
    np.testing.assert_array_equal(points_a, points_b)


def test_generate_spherical_noise_points_without_seed() -> None:
    """Generate points successfully when no seed is supplied."""
    center = np.array([1.0, 2.0, 3.0])
    points = generate_spherical_noise_points(
        center=center, radius=0.5, num_points=3, seed=None
    )
    assert points.shape == (3, 3)


def test_generate_spherical_noise_points_inside_sphere() -> None:
    """Keep every generated point inside the requested sphere."""
    center = np.array([0.0, 0.0, 0.0])
    radius = 2.0
    points = generate_spherical_noise_points(
        center=center, radius=radius, num_points=50, seed=0
    )
    distances = np.linalg.norm(points - center, axis=1)
    assert np.all(distances <= radius + 1e-8)


# ---------------------------------------------------------------------------
# compute_noisy_explanations
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_cloud() -> torch.Tensor:
    """Return a minimal valid point-cloud tensor of shape (N, 3)."""
    torch.manual_seed(0)
    return torch.rand(10, 3)


@pytest.fixture
def mock_explainer_result() -> MagicMock:
    """Return a mock result exposing important_clusters."""
    result = MagicMock()
    result.important_clusters = np.array([0, 2, 5])
    return result


def test_compute_noisy_explanations_rejects_bad_shape(
    sample_cloud: torch.Tensor,
) -> None:
    """Raise ValueError when the last dimension is not 3."""
    bad = torch.rand(10, 4)
    with pytest.raises(ValueError, match="Expected last dimension to be 3"):
        compute_noisy_explanations(
            sample_input=bad,
            sample_label=0,
            model=MagicMock(),
            num_iterations=1,
        )


def test_compute_noisy_explanations_squeezes_batched_input(
    sample_cloud: torch.Tensor,
    mock_explainer_result: MagicMock,
) -> None:
    """Squeeze a leading batch dimension before processing."""
    batched = sample_cloud.unsqueeze(0)
    assert batched.ndim == 3

    with patch("xwhy.metrics.point_cloud.PointCloudExplainer") as explainer_cls:
        instance = MagicMock()
        instance.explain.return_value = mock_explainer_result
        explainer_cls.return_value = instance

        result = compute_noisy_explanations(
            sample_input=batched,
            sample_label=1,
            model=MagicMock(),
            num_iterations=1,
            num_new_points=5,
            clustering_mode="kmeans",
        )

    assert len(result) == 1
    np.testing.assert_array_equal(result[0], mock_explainer_result.important_clusters)


def test_compute_noisy_explanations_kmeans_mode(
    sample_cloud: torch.Tensor,
    mock_explainer_result: MagicMock,
) -> None:
    """Run the kmeans path and collect important clusters."""
    with patch("xwhy.metrics.point_cloud.PointCloudExplainer") as explainer_cls:
        instance = MagicMock()
        instance.explain.return_value = mock_explainer_result
        explainer_cls.return_value = instance

        result = compute_noisy_explanations(
            sample_input=sample_cloud,
            sample_label=0,
            model=MagicMock(),
            num_iterations=2,
            num_new_points=4,
            clustering_mode="kmeans",
            num_clusters=8,
        )

    assert len(result) == 2
    assert explainer_cls.call_count == 2
    for clusters in result:
        np.testing.assert_array_equal(
            clusters, mock_explainer_result.important_clusters
        )


def test_compute_noisy_explanations_precomputed_success(
    sample_cloud: torch.Tensor,
    mock_explainer_result: MagicMock,
) -> None:
    """Build combined labels when clustering_mode is precomputed."""
    labels = np.array([0, 1, 0, 1, 2, 2, 0, 1, 2, 0], dtype=int)

    with patch("xwhy.metrics.point_cloud.PointCloudExplainer") as explainer_cls:
        instance = MagicMock()
        instance.explain.return_value = mock_explainer_result
        explainer_cls.return_value = instance

        result = compute_noisy_explanations(
            sample_input=sample_cloud,
            sample_label=0,
            model=MagicMock(),
            cluster_labels=labels,
            num_iterations=1,
            num_new_points=3,
            clustering_mode="precomputed",
        )

    assert len(result) == 1
    # num_clusters passed to explainer must include the extra noise cluster
    call_kwargs = explainer_cls.call_args.kwargs
    assert call_kwargs["num_clusters"] == len(np.unique(labels)) + 1


def test_compute_noisy_explanations_precomputed_missing_labels(
    sample_cloud: torch.Tensor,
) -> None:
    """Raise ValueError when precomputed mode lacks cluster_labels."""
    with pytest.raises(ValueError, match="cluster_labels required"):
        compute_noisy_explanations(
            sample_input=sample_cloud,
            sample_label=0,
            model=MagicMock(),
            cluster_labels=None,
            num_iterations=1,
            clustering_mode="precomputed",
        )


def test_compute_noisy_explanations_unsqueezes_combined_tensor(
    sample_cloud: torch.Tensor,
    mock_explainer_result: MagicMock,
) -> None:
    """Ensure the combined tensor receives a batch dimension when needed."""
    with patch("xwhy.metrics.point_cloud.PointCloudExplainer") as explainer_cls:
        instance = MagicMock()
        instance.explain.return_value = mock_explainer_result
        explainer_cls.return_value = instance

        compute_noisy_explanations(
            sample_input=sample_cloud,
            sample_label=0,
            model=MagicMock(),
            num_iterations=1,
            num_new_points=2,
            clustering_mode="kmeans",
        )

        # The tensor passed to explain must be 3-D
        call_kwargs = instance.explain.call_args.kwargs
        passed = call_kwargs["instance"]
        assert isinstance(passed, torch.Tensor)
        assert passed.ndim == 3


def test_compute_noisy_explanations_skips_unsqueeze_when_already_batched(
    sample_cloud: torch.Tensor,
    mock_explainer_result: MagicMock,
) -> None:
    """Leave a three-dimensional combined tensor unchanged.

    Args:
        sample_cloud: Fixture providing a two-dimensional point-cloud tensor.
        mock_explainer_result: Fixture providing a mock explanation result.

    """
    with (
        patch("xwhy.metrics.point_cloud.PointCloudExplainer") as explainer_cls,
        patch("xwhy.metrics.point_cloud.torch.from_numpy") as from_numpy,
    ):
        # Force the tensor that reaches the ndim check to already be 3-D
        already_batched = torch.rand(1, 15, 3)
        from_numpy.return_value = already_batched

        instance = MagicMock()
        instance.explain.return_value = mock_explainer_result
        explainer_cls.return_value = instance

        result = compute_noisy_explanations(
            sample_input=sample_cloud,
            sample_label=0,
            model=MagicMock(),
            num_iterations=1,
            num_new_points=5,
            clustering_mode="kmeans",
        )

    assert len(result) == 1
    # The tensor passed to explain must still be the 3-D one we injected
    passed = instance.explain.call_args.kwargs["instance"]
    assert passed is already_batched
    assert passed.ndim == 3


# ---------------------------------------------------------------------------
# calculate_jaccard_stability_score
# ---------------------------------------------------------------------------


def test_jaccard_empty_list_raises() -> None:
    """Raise ValueError when the important-clusters list is empty."""
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_jaccard_stability_score([])


def test_jaccard_single_entry_returns_perfect_score() -> None:
    """Return an empty score list and mean 1.0 for a single entry."""
    scores, mean = calculate_jaccard_stability_score([np.array([0, 1, 2])])
    assert scores == []
    assert mean == 1.0


def test_jaccard_multiple_entries_computes_scores() -> None:
    """Compute pairwise Jaccard scores against the baseline entry."""
    clusters = [
        np.array([0, 1, 2]),
        np.array([0, 1, 3]),
        np.array([0, 4, 5]),
    ]
    scores, mean = calculate_jaccard_stability_score(clusters)

    assert len(scores) == 2
    # |{0,1,2} ∩ {0,1,3}| / |{0,1,2,3}| = 2/4 = 0.5
    assert scores[0] == pytest.approx(0.5)
    # |{0,1,2} ∩ {0,4,5}| / |{0,1,2,4,5}| = 1/5 = 0.2
    assert scores[1] == pytest.approx(0.2)
    assert mean == pytest.approx(0.35)


def test_jaccard_disjoint_sets_yield_zero() -> None:
    """Return zero similarity when two sets share no elements."""
    clusters = [
        np.array([0, 1]),
        np.array([2, 3]),
    ]
    scores, mean = calculate_jaccard_stability_score(clusters)
    assert scores == [0.0]
    assert mean == 0.0


def test_jaccard_identical_sets_yield_one() -> None:
    """Return perfect similarity when every set equals the baseline."""
    clusters = [
        np.array([1, 2, 3]),
        np.array([1, 2, 3]),
        np.array([3, 2, 1]),
    ]
    scores, mean = calculate_jaccard_stability_score(clusters)
    assert scores == [1.0, 1.0]
    assert mean == 1.0
