"""Unit tests for the PointCloudExplainer class."""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from xwhy.core.config import PointCloudConfig
from xwhy.core.result import PointCloudXWhyResult
from xwhy.core.states import PointCloudState
from xwhy.distance.types import DistanceType
from xwhy.explainers.point_cloud import PointCloudExplainer
from xwhy.models.point_cloud.base import BasePointCloudModel
from xwhy.surrogate.types import SurrogateType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config() -> MagicMock:
    """Return a fully-populated PointCloudConfig mock.

    Returns:
        MagicMock: Config with all attributes required by the explainer.

    """
    cfg = MagicMock(spec=PointCloudConfig)
    cfg.custom_model = None
    cfg.custom_predict_fn = None
    cfg.num_clusters = 4
    cfg.num_top_features = 2
    cfg.num_perturbations = 5
    cfg.removal_probability = 0.3
    cfg.kernel_width = 0.5
    cfg.ridge_alpha = 1.0
    cfg.epsilon = 0.0
    cfg.max_iters = 10
    cfg.min_valid_ratio = 0.5
    cfg.seed = 42
    cfg.device = "cpu"
    cfg.clustering_mode = "kmeans"
    cfg.distance_type = DistanceType.WASSERSTEIN
    cfg.distance_mode = "mask"
    cfg.surrogate_type = SurrogateType.LIME
    cfg.use_best_surrogate = False
    return cfg


@pytest.fixture
def mock_model() -> MagicMock:
    """Return a mock that satisfies BasePointCloudModel interface.

    Returns:
        MagicMock: Model with predict / get_output_probabilities stubs.

    """
    model = MagicMock(spec=BasePointCloudModel)
    # predict returns (pred_class, latent, top_classes)
    model.predict.return_value = (
        0,
        torch.tensor([[0.1, 0.2, 0.3]]),
        [0, 1],
    )
    model.get_output_probabilities.return_value = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.8, 0.2], [0.55, 0.45]]
    )
    return model


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Return a minimal valid point-cloud tensor of shape (N, 3).

    Returns:
        torch.Tensor: Random points with fixed seed for reproducibility.

    """
    torch.manual_seed(0)
    return torch.rand(20, 3)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_explainer(
    config: MagicMock,
    model: MagicMock | None = None,
    **overrides: Any,  # noqa: ANN401
) -> PointCloudExplainer:
    """Construct a PointCloudExplainer with heavy patching of side-effects.

    Args:
        config: Pre-built config mock.
        model: Optional model to inject.
        **overrides: Extra kwargs forwarded to the constructor.

    Returns:
        PointCloudExplainer: Fully initialized explainer instance.

    """
    with (
        patch.object(PointCloudExplainer, "_initialize"),
        patch("xwhy.explainers.point_cloud.PointCloudState") as state_cls,
    ):
        state = MagicMock(spec=PointCloudState)
        state.device = torch.device("cpu")
        state.model = model
        state.perturbation = MagicMock()
        state_cls.return_value = state

        explainer = PointCloudExplainer(config=config, model=model, **overrides)
        # Restore the real _initialize so later tests can exercise it
        # when needed; for most tests we keep the mock state.
        explainer.state = state
        return explainer


# ---------------------------------------------------------------------------
# __init__ branches
# ---------------------------------------------------------------------------


def test_init_rejects_non_numeric_distance() -> None:
    """ValueError is raised when a text distance metric is supplied."""
    with pytest.raises(ValueError, match="Invalid distance metric"):
        PointCloudExplainer(distance_type=DistanceType.WMD)


def test_init_with_base_point_cloud_model(mock_config: MagicMock) -> None:
    """Passing a BasePointCloudModel instance is stored directly."""
    model = MagicMock(spec=BasePointCloudModel)
    with (
        patch.object(PointCloudExplainer, "_initialize"),
        patch("xwhy.explainers.point_cloud.PointCloudState") as state_cls,
    ):
        state = MagicMock()
        state.device = torch.device("cpu")
        state_cls.return_value = state

        explainer = PointCloudExplainer(config=mock_config, model=model)
        assert explainer.state.model is model


def test_init_wraps_plain_model(mock_config: MagicMock) -> None:
    """A plain torch.nn.Module is wrapped by CustomPointCloudModel."""
    plain = MagicMock(spec=torch.nn.Module)
    with (
        patch.object(PointCloudExplainer, "_initialize"),
        patch("xwhy.explainers.point_cloud.PointCloudState") as state_cls,
        patch("xwhy.explainers.point_cloud.CustomPointCloudModel") as custom_cls,
    ):
        state = MagicMock()
        state.device = torch.device("cpu")
        state_cls.return_value = state
        custom_instance = MagicMock()
        custom_cls.return_value = custom_instance

        explainer = PointCloudExplainer(config=mock_config, model=plain)
        custom_cls.assert_called_once()
        assert explainer.state.model is custom_instance


def test_init_builds_config_when_none_supplied() -> None:
    """When config is None a PointCloudConfig is constructed from kwargs."""
    with (
        patch.object(PointCloudExplainer, "_initialize"),
        patch("xwhy.explainers.point_cloud.PointCloudState") as state_cls,
        patch("xwhy.explainers.point_cloud.PointCloudConfig") as cfg_cls,
    ):
        state = MagicMock()
        state.device = torch.device("cpu")
        state_cls.return_value = state
        cfg_instance = MagicMock()
        cfg_instance.device = "cpu"
        cfg_cls.return_value = cfg_instance

        PointCloudExplainer(
            config=None,
            num_clusters=6,
            distance_type="wasserstein",
            surrogate_type="lime_ols",
        )
        cfg_cls.assert_called_once()


# ---------------------------------------------------------------------------
# _initialize
# ---------------------------------------------------------------------------


def test_initialize_creates_model_when_missing(
    mock_config: MagicMock,
) -> None:
    """_initialize builds CustomPointCloudModel when state.model is None."""
    with (
        patch("xwhy.explainers.point_cloud.PointCloudState") as state_cls,
        patch("xwhy.explainers.point_cloud.CustomPointCloudModel") as custom_cls,
        patch("xwhy.explainers.point_cloud.PointCloudPerturbation") as pert_cls,
    ):
        state = MagicMock()
        state.device = torch.device("cpu")
        state.model = None
        state_cls.return_value = state
        custom_instance = MagicMock()
        custom_cls.return_value = custom_instance
        pert_instance = MagicMock()
        pert_cls.return_value = pert_instance

        explainer = PointCloudExplainer(config=mock_config, model=None)
        # _initialize was called by constructor
        assert explainer.state.model is custom_instance
        assert explainer.state.perturbation is pert_instance


# ---------------------------------------------------------------------------
# _cluster_points
# ---------------------------------------------------------------------------


def test_cluster_precomputed_success(
    mock_config: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """Precomputed mode returns the supplied cluster_labels."""
    mock_config.clustering_mode = "precomputed"
    explainer = _make_explainer(mock_config)
    labels = np.array([0, 1, 0, 1], dtype=int)
    result = explainer._cluster_points(sample_tensor, labels)
    np.testing.assert_array_equal(result, labels)


def test_cluster_precomputed_missing_labels(
    mock_config: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """Precomputed mode without labels raises ValueError."""
    mock_config.clustering_mode = "precomputed"
    explainer = _make_explainer(mock_config)
    with pytest.raises(ValueError, match="cluster_labels must be provided"):
        explainer._cluster_points(sample_tensor, None)


def test_cluster_kmeans_path(
    mock_config: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """Kmeans mode runs FPS initialisation and sklearn KMeans."""
    mock_config.clustering_mode = "kmeans"
    mock_config.num_clusters = 3
    mock_config.max_iters = 5
    mock_config.seed = 0
    explainer = _make_explainer(mock_config)

    fake_labels = np.array([0, 1, 2, 0, 1] * 4, dtype=int)
    with patch("xwhy.explainers.point_cloud.KMeans") as kmeans_cls:
        kmeans_instance = MagicMock()
        kmeans_instance.labels_ = fake_labels
        kmeans_cls.return_value = kmeans_instance

        result = explainer._cluster_points(sample_tensor.unsqueeze(0), None)
        kmeans_cls.assert_called_once()
        np.testing.assert_array_equal(result, fake_labels)


def test_cluster_invalid_mode(
    mock_config: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """Unknown clustering_mode raises ValueError."""
    mock_config.clustering_mode = "invalid_mode"
    explainer = _make_explainer(mock_config)
    with pytest.raises(ValueError, match="Invalid clustering_mode"):
        explainer._cluster_points(sample_tensor, None)


# ---------------------------------------------------------------------------
# explain - error paths
# ---------------------------------------------------------------------------


def test_explain_rejects_non_tensor(mock_config: MagicMock) -> None:
    """Explain raises TypeError for non-tensor input."""
    explainer = _make_explainer(mock_config)
    with pytest.raises(
        TypeError, match=re.escape("sample_input must be a torch.Tensor")
    ):
        explainer.explain(instance="not-a-tensor")  # type: ignore[arg-type]


def test_explain_raises_when_model_missing(
    mock_config: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """Explain raises RuntimeError when state.model is None."""
    explainer = _make_explainer(mock_config, model=None)
    explainer.state.model = None
    with pytest.raises(RuntimeError, match="model is not initialized"):
        explainer.explain(instance=sample_tensor)


# ---------------------------------------------------------------------------
# explain - distance_mode branches
# ---------------------------------------------------------------------------


def _prepare_explain_mocks(
    explainer: PointCloudExplainer,
    num_clusters: int = 4,
    num_perturbations: int = 5,
) -> tuple[MagicMock, np.ndarray, list[torch.Tensor]]:
    """Wire common mocks for a successful explain call.

    Args:
        explainer: Explainer under test.
        num_clusters: Number of clusters to simulate.
        num_perturbations: Number of perturbation masks.

    Returns:
        tuple: (perturbation_mock, masks, perturbed_samples)

    """
    masks = np.ones((num_perturbations, num_clusters), dtype=float)
    masks[0] = np.array([1, 0, 1, 0], dtype=float)  # at least one variation

    pert = MagicMock()
    pert.generate.return_value = masks
    pert.apply_mask.side_effect = lambda item, mask, segments: item.clone()
    explainer.state.perturbation = pert

    perturbed = [torch.rand(15, 3) for _ in range(num_perturbations)]
    return pert, masks, perturbed


@patch("xwhy.explainers.point_cloud.calculate_distance", return_value=0.5)
@patch("xwhy.explainers.point_cloud.SurrogateTrainer")
@patch("xwhy.explainers.point_cloud.SurrogateFactory")
@patch("xwhy.explainers.point_cloud.RegressionMetrics")
def test_explain_mask_mode(
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_dist: MagicMock,
    mock_config: MagicMock,
    mock_model: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """distance_mode='mask' uses reference mask and calculate_distance."""
    mock_config.distance_mode = "mask"
    mock_config.use_best_surrogate = False
    explainer = _make_explainer(mock_config, model=mock_model)
    _, masks, _ = _prepare_explain_mocks(explainer)

    # Clustering stub
    with patch.object(
        explainer, "_cluster_points", return_value=np.zeros(20, dtype=int)
    ):
        # Surrogate plumbing
        surrogate = MagicMock()
        surrogate.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        surrogate.predict.return_value = np.array([0.5] * 5)
        mock_factory.create.return_value = surrogate
        mock_trainer.compute_weights.return_value = np.ones(5)
        mock_metrics.calculate.return_value = MagicMock()

        result = explainer.explain(instance=sample_tensor)

    assert isinstance(result, PointCloudXWhyResult)
    assert mock_dist.call_count == len(masks)
    # First call should compare reference (ones) vs first mask
    first_call_kwargs = mock_dist.call_args_list[0].kwargs
    assert first_call_kwargs["mode"] == "mask"


@patch("xwhy.explainers.point_cloud.calculate_distance", return_value=0.3)
@patch("xwhy.explainers.point_cloud.SurrogateTrainer")
@patch("xwhy.explainers.point_cloud.SurrogateFactory")
@patch("xwhy.explainers.point_cloud.RegressionMetrics")
def test_explain_spatial_mode(
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_dist: MagicMock,
    mock_config: MagicMock,
    mock_model: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """distance_mode='spatial' compares original vs perturbed point clouds."""
    mock_config.distance_mode = "spatial"
    mock_config.use_best_surrogate = False
    explainer = _make_explainer(mock_config, model=mock_model)
    _prepare_explain_mocks(explainer)

    with patch.object(
        explainer, "_cluster_points", return_value=np.zeros(20, dtype=int)
    ):
        surrogate = MagicMock()
        surrogate.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        surrogate.predict.return_value = np.array([0.5] * 5)
        mock_factory.create.return_value = surrogate
        mock_trainer.compute_weights.return_value = np.ones(5)
        mock_metrics.calculate.return_value = MagicMock()

        result = explainer.explain(instance=sample_tensor)

    assert isinstance(result, PointCloudXWhyResult)
    assert mock_dist.call_count == 5
    assert mock_dist.call_args_list[0].kwargs["mode"] == "spatial"


@patch("xwhy.explainers.point_cloud.calculate_distance", return_value=0.2)
@patch("xwhy.explainers.point_cloud.SurrogateTrainer")
@patch("xwhy.explainers.point_cloud.SurrogateFactory")
@patch("xwhy.explainers.point_cloud.RegressionMetrics")
def test_explain_latent_mode(
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_dist: MagicMock,
    mock_config: MagicMock,
    mock_model: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """distance_mode='latent' extracts latent vectors via model.predict."""
    mock_config.distance_mode = "latent"
    mock_config.use_best_surrogate = False
    explainer = _make_explainer(mock_config, model=mock_model)
    _prepare_explain_mocks(explainer)

    with patch.object(
        explainer, "_cluster_points", return_value=np.zeros(20, dtype=int)
    ):
        surrogate = MagicMock()
        surrogate.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        surrogate.predict.return_value = np.array([0.5] * 5)
        mock_factory.create.return_value = surrogate
        mock_trainer.compute_weights.return_value = np.ones(5)
        mock_metrics.calculate.return_value = MagicMock()

        result = explainer.explain(instance=sample_tensor)

    assert isinstance(result, PointCloudXWhyResult)
    # One call for original latent + one per perturbation
    assert mock_model.predict.call_count >= 6
    assert mock_dist.call_count == 5
    assert mock_dist.call_args_list[0].kwargs["mode"] == "latent"


def test_explain_invalid_distance_mode(
    mock_config: MagicMock,
    mock_model: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """Unknown distance_mode raises ValueError."""
    mock_config.distance_mode = "unknown_mode"
    explainer = _make_explainer(mock_config, model=mock_model)
    _prepare_explain_mocks(explainer)

    with (
        patch.object(
            explainer, "_cluster_points", return_value=np.zeros(20, dtype=int)
        ),
        pytest.raises(ValueError, match="Invalid distance_mode"),
    ):
        explainer.explain(instance=sample_tensor)


# ---------------------------------------------------------------------------
# explain - surrogate selection & fidelity plot
# ---------------------------------------------------------------------------


@patch("xwhy.explainers.point_cloud.calculate_distance", return_value=0.4)
@patch("xwhy.explainers.point_cloud.SurrogateTrainer")
@patch("xwhy.explainers.point_cloud.SurrogateFactory")
@patch("xwhy.explainers.point_cloud.RegressionMetrics")
def test_explain_use_best_surrogate(
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_dist: MagicMock,
    mock_config: MagicMock,
    mock_model: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """use_best_surrogate=True invokes SurrogateTrainer.find_best."""
    mock_config.use_best_surrogate = True
    mock_config.distance_mode = "mask"
    explainer = _make_explainer(mock_config, model=mock_model)
    _prepare_explain_mocks(explainer)

    mock_trainer.find_best.return_value = (SurrogateType.LIME, 0.95)
    mock_trainer.compute_weights.return_value = np.ones(5)

    with patch.object(
        explainer, "_cluster_points", return_value=np.zeros(20, dtype=int)
    ):
        surrogate = MagicMock()
        surrogate.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        surrogate.predict.return_value = np.array([0.5] * 5)
        mock_factory.create.return_value = surrogate
        mock_metrics.calculate.return_value = MagicMock()

        result = explainer.explain(instance=sample_tensor)

    mock_trainer.find_best.assert_called_once()
    assert isinstance(result, PointCloudXWhyResult)


@patch("xwhy.explainers.point_cloud.calculate_distance", return_value=0.4)
@patch("xwhy.explainers.point_cloud.SurrogateTrainer")
@patch("xwhy.explainers.point_cloud.SurrogateFactory")
@patch("xwhy.explainers.point_cloud.RegressionMetrics")
def test_explain_fidelity_plot(
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_dist: MagicMock,
    mock_config: MagicMock,
    mock_model: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """fidelity_plot=True triggers result.plot(show=True)."""
    mock_config.use_best_surrogate = False
    mock_config.distance_mode = "mask"
    explainer = _make_explainer(mock_config, model=mock_model)
    _prepare_explain_mocks(explainer)

    mock_trainer.compute_weights.return_value = np.ones(5)

    with patch.object(
        explainer, "_cluster_points", return_value=np.zeros(20, dtype=int)
    ):
        surrogate = MagicMock()
        surrogate.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        surrogate.predict.return_value = np.array([0.5] * 5)
        mock_factory.create.return_value = surrogate
        mock_metrics.calculate.return_value = MagicMock()

        with patch.object(PointCloudXWhyResult, "plot") as mock_plot:
            result = explainer.explain(instance=sample_tensor, fidelity_plot=True)
            mock_plot.assert_called_once_with(show=True)

    assert isinstance(result, PointCloudXWhyResult)


# ---------------------------------------------------------------------------
# explain - ndim handling & distance validation
# ---------------------------------------------------------------------------


@patch("xwhy.explainers.point_cloud.calculate_distance")
@patch("xwhy.explainers.point_cloud.SurrogateTrainer")
@patch("xwhy.explainers.point_cloud.SurrogateFactory")
@patch("xwhy.explainers.point_cloud.RegressionMetrics")
def test_explain_filters_non_finite_distances(
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_dist: MagicMock,
    mock_config: MagicMock,
    mock_model: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """Filter non-finite distances and train only on valid rows.

    2-D input is still unsqueezed; failed perturbations are dropped before
    surrogate fitting.
    """
    mock_config.distance_mode = "mask"
    mock_config.use_best_surrogate = False
    mock_config.min_valid_ratio = 0.5
    explainer = _make_explainer(mock_config, model=mock_model)
    _prepare_explain_mocks(explainer)

    # Mix finite and infinite distances -> 3 valid of 5 (ratio 0.6 >= 0.5)
    mock_dist.side_effect = [0.1, float("inf"), 0.3, float("nan"), 0.5]
    mock_trainer.compute_weights.return_value = np.ones(3)

    with patch.object(
        explainer, "_cluster_points", return_value=np.zeros(20, dtype=int)
    ):
        surrogate = MagicMock()
        surrogate.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        surrogate.predict.return_value = np.array([0.5, 0.5, 0.5])
        mock_factory.create.return_value = surrogate
        mock_metrics.calculate.return_value = MagicMock()

        result = explainer.explain(instance=sample_tensor)

    assert isinstance(result, PointCloudXWhyResult)
    distances = result.raw_data["distances"]
    assert len(distances) == 3
    assert np.all(np.isfinite(distances))
    y_target = result.raw_data["y_target"]
    assert len(y_target) == 3


@patch("xwhy.explainers.point_cloud.calculate_distance", return_value=0.15)
@patch("xwhy.explainers.point_cloud.SurrogateTrainer")
@patch("xwhy.explainers.point_cloud.SurrogateFactory")
@patch("xwhy.explainers.point_cloud.RegressionMetrics")
def test_explain_2d_unsqueeze_explicit(
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_dist: MagicMock,
    mock_config: MagicMock,
    mock_model: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """Explicitly assert the ndim==2 => unsqueeze(0) path is taken."""
    mock_config.distance_mode = "mask"
    mock_config.use_best_surrogate = False
    explainer = _make_explainer(mock_config, model=mock_model)
    _prepare_explain_mocks(explainer)
    mock_trainer.compute_weights.return_value = np.ones(5)

    with patch.object(
        explainer, "_cluster_points", return_value=np.zeros(20, dtype=int)
    ):
        surrogate = MagicMock()
        surrogate.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        surrogate.predict.return_value = np.array([0.5] * 5)
        mock_factory.create.return_value = surrogate
        mock_metrics.calculate.return_value = MagicMock()

        # sample_tensor is (20, 3) => must be unsqueezed inside explain
        assert sample_tensor.ndim == 2
        result = explainer.explain(instance=sample_tensor)

    assert isinstance(result, PointCloudXWhyResult)

    # predict is always called with keyword arguments
    call_args = mock_model.predict.call_args
    assert call_args is not None
    passed_input = call_args.kwargs["sample_input"]
    assert passed_input.ndim == 3


def test_initialize_model_none_branch(mock_config: MagicMock) -> None:
    """Force the True branch of ``if self.state.model is None`` in _initialize.

    Ensures CustomPointCloudModel is constructed and logger is called.
    """
    with (
        patch("xwhy.explainers.point_cloud.PointCloudState") as state_cls,
        patch("xwhy.explainers.point_cloud.CustomPointCloudModel") as custom_cls,
        patch("xwhy.explainers.point_cloud.PointCloudPerturbation") as pert_cls,
        patch("xwhy.explainers.point_cloud.logger") as mock_logger,
    ):
        state = MagicMock()
        state.device = torch.device("cpu")
        state.model = None  # critical: triggers the if-body
        state_cls.return_value = state

        custom_instance = MagicMock()
        custom_cls.return_value = custom_instance
        pert_instance = MagicMock()
        pert_cls.return_value = pert_instance

        explainer = PointCloudExplainer(config=mock_config, model=None)

        mock_logger.info.assert_any_call("Initializing point cloud model")
        custom_cls.assert_called_once()
        assert explainer.state.model is custom_instance
        assert explainer.state.perturbation is pert_instance


@patch("xwhy.explainers.point_cloud.calculate_distance")
@patch("xwhy.explainers.point_cloud.SurrogateTrainer")
@patch("xwhy.explainers.point_cloud.SurrogateFactory")
@patch("xwhy.explainers.point_cloud.RegressionMetrics")
def test_explain_raises_when_all_distances_non_finite(
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_dist: MagicMock,
    mock_config: MagicMock,
    mock_model: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """Raise ValueError when every perturbation distance is non-finite."""
    mock_config.distance_mode = "mask"
    mock_config.use_best_surrogate = False
    mock_config.min_valid_ratio = 0.5
    explainer = _make_explainer(mock_config, model=mock_model)
    _prepare_explain_mocks(explainer)

    mock_dist.side_effect = [float("inf")] * 5

    with (
        patch.object(
            explainer, "_cluster_points", return_value=np.zeros(20, dtype=int)
        ),
        pytest.raises(
            ValueError,
            match=re.escape("All perturbations failed (0 valid distances)"),
        ),
    ):
        explainer.explain(instance=sample_tensor)


@patch("xwhy.explainers.point_cloud.calculate_distance")
@patch("xwhy.explainers.point_cloud.SurrogateTrainer")
@patch("xwhy.explainers.point_cloud.SurrogateFactory")
@patch("xwhy.explainers.point_cloud.RegressionMetrics")
@patch("xwhy.explainers.point_cloud.logger")
def test_explain_warns_on_low_valid_ratio(
    mock_logger: MagicMock,
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_dist: MagicMock,
    mock_config: MagicMock,
    mock_model: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """Log a warning when valid_ratio is below ``min_valid_ratio``."""
    mock_config.distance_mode = "mask"
    mock_config.use_best_surrogate = False
    mock_config.min_valid_ratio = 0.5
    explainer = _make_explainer(mock_config, model=mock_model)
    _prepare_explain_mocks(explainer)

    # 2 valid of 5 -> ratio 0.4 < 0.5
    mock_dist.side_effect = [
        0.1,
        float("inf"),
        float("nan"),
        float("inf"),
        0.5,
    ]
    mock_trainer.compute_weights.return_value = np.ones(2)

    with patch.object(
        explainer, "_cluster_points", return_value=np.zeros(20, dtype=int)
    ):
        surrogate = MagicMock()
        surrogate.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        surrogate.predict.return_value = np.array([0.5, 0.5])
        mock_factory.create.return_value = surrogate
        mock_metrics.calculate.return_value = MagicMock()

        result = explainer.explain(instance=sample_tensor)

    assert isinstance(result, PointCloudXWhyResult)
    assert len(result.raw_data["y_target"]) == 2
    assert len(result.raw_data["distances"]) == 2

    warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
    assert any("Low valid perturbation ratio" in c for c in warning_calls)


@patch("xwhy.explainers.point_cloud.calculate_distance", return_value=0.25)
@patch("xwhy.explainers.point_cloud.SurrogateTrainer")
@patch("xwhy.explainers.point_cloud.SurrogateFactory")
@patch("xwhy.explainers.point_cloud.RegressionMetrics")
def test_explain_output_probs_non_tensor(
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_dist: MagicMock,
    mock_config: MagicMock,
    mock_model: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """Cover the else branch when get_output_probabilities returns a non-Tensor."""
    mock_config.distance_mode = "mask"
    mock_config.use_best_surrogate = False
    explainer = _make_explainer(mock_config, model=mock_model)
    _prepare_explain_mocks(explainer)

    # Return a plain list instead of a Tensor
    mock_model.get_output_probabilities.return_value = [
        [0.7, 0.3],
        [0.6, 0.4],
        [0.5, 0.5],
        [0.8, 0.2],
        [0.55, 0.45],
    ]
    mock_trainer.compute_weights.return_value = np.ones(5)

    with patch.object(
        explainer, "_cluster_points", return_value=np.zeros(20, dtype=int)
    ):
        surrogate = MagicMock()
        surrogate.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        surrogate.predict.return_value = np.array([0.5] * 5)
        mock_factory.create.return_value = surrogate
        mock_metrics.calculate.return_value = MagicMock()

        result = explainer.explain(instance=sample_tensor)

    assert isinstance(result, PointCloudXWhyResult)
    # y_target must have been built from the list path
    assert "y_target" in result.raw_data
    assert len(result.raw_data["y_target"]) == 5


def test_initialize_model_is_none_true_branch(
    mock_config: MagicMock,
) -> None:
    """Hit the True branch of ``if self.state.model is None`` (lines 134-144).

    The real ``_initialize`` must run; we only mock the collaborators it calls.
    """
    with (
        patch("xwhy.explainers.point_cloud.PointCloudState") as state_cls,
        patch("xwhy.explainers.point_cloud.CustomPointCloudModel") as custom_cls,
        patch("xwhy.explainers.point_cloud.PointCloudPerturbation") as pert_cls,
        patch("xwhy.explainers.point_cloud.logger") as mock_logger,
    ):
        # PointCloudState() returns an object whose .model is None
        state = MagicMock()
        state.device = torch.device("cpu")
        state.model = None
        state_cls.return_value = state

        custom_instance = MagicMock()
        custom_cls.return_value = custom_instance
        pert_instance = MagicMock()
        pert_cls.return_value = pert_instance

        # model=None => constructor never sets state.model, so _initialize
        # enters the if-body
        explainer = PointCloudExplainer(config=mock_config, model=None)

        mock_logger.info.assert_any_call("Initializing point cloud model")
        custom_cls.assert_called_once_with(
            model=mock_config.custom_model,
            predict_fn=mock_config.custom_predict_fn,
        )
        assert explainer.state.model is custom_instance
        assert explainer.state.perturbation is pert_instance


@patch("xwhy.explainers.point_cloud.calculate_distance", return_value=0.11)
@patch("xwhy.explainers.point_cloud.SurrogateTrainer")
@patch("xwhy.explainers.point_cloud.SurrogateFactory")
@patch("xwhy.explainers.point_cloud.RegressionMetrics")
def test_explain_ndim_2_true_branch(
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_dist: MagicMock,
    mock_config: MagicMock,
    mock_model: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """Hit the True branch of ``if sample_input.ndim == 2`` (lines 254-256).

    A pure 2-D tensor must be unsqueezed before being passed to the model.
    """
    mock_config.distance_mode = "mask"
    mock_config.use_best_surrogate = False
    explainer = _make_explainer(mock_config, model=mock_model)
    _prepare_explain_mocks(explainer)
    mock_trainer.compute_weights.return_value = np.ones(5)

    with patch.object(
        explainer, "_cluster_points", return_value=np.zeros(20, dtype=int)
    ):
        surrogate = MagicMock()
        surrogate.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        surrogate.predict.return_value = np.array([0.5] * 5)
        mock_factory.create.return_value = surrogate
        mock_metrics.calculate.return_value = MagicMock()

        # Guarantee the input is 2-D
        assert sample_tensor.ndim == 2
        result = explainer.explain(instance=sample_tensor)

    assert isinstance(result, PointCloudXWhyResult)

    # The tensor that reached model.predict must be 3-D
    call_args = mock_model.predict.call_args
    assert call_args is not None
    passed = call_args.kwargs["sample_input"]
    assert isinstance(passed, torch.Tensor)
    assert passed.ndim == 3


def test_initialize_creates_model_when_absent(
    mock_config: MagicMock,
) -> None:
    """Create CustomPointCloudModel when the runtime model is absent.

    Args:
        mock_config: Fixture providing a populated PointCloudConfig mock.

    """
    with (
        patch("xwhy.explainers.point_cloud.PointCloudState") as state_cls,
        patch("xwhy.explainers.point_cloud.CustomPointCloudModel") as custom_cls,
        patch("xwhy.explainers.point_cloud.PointCloudPerturbation") as pert_cls,
        patch("xwhy.explainers.point_cloud.logger") as mock_logger,
    ):
        state = MagicMock()
        state.device = torch.device("cpu")
        state.model = None
        state_cls.return_value = state

        custom_instance = MagicMock()
        custom_cls.return_value = custom_instance
        pert_instance = MagicMock()
        pert_cls.return_value = pert_instance

        explainer = PointCloudExplainer(config=mock_config, model=None)

        mock_logger.info.assert_any_call("Initializing point cloud model")
        custom_cls.assert_called_once()
        assert explainer.state.model is custom_instance
        assert explainer.state.perturbation is pert_instance


def test_initialize_when_model_is_absent(mock_config: MagicMock) -> None:
    """Construct and log a model when none exists on the runtime state.

    Args:
        mock_config: Fixture providing a populated PointCloudConfig mock.

    """
    # Use a simple namespace so attribute access is ordinary Python,
    # not MagicMock auto-creation.
    state = type("State", (), {})()
    state.device = torch.device("cpu")
    state.model = None
    state.perturbation = None

    with (
        patch(
            "xwhy.explainers.point_cloud.PointCloudState",
            return_value=state,
        ),
        patch(
            "xwhy.explainers.point_cloud.CustomPointCloudModel",
        ) as custom_cls,
        patch(
            "xwhy.explainers.point_cloud.PointCloudPerturbation",
        ) as pert_cls,
        patch("xwhy.explainers.point_cloud.logger") as mock_logger,
    ):
        custom_instance = MagicMock()
        custom_cls.return_value = custom_instance
        pert_instance = MagicMock()
        pert_cls.return_value = pert_instance

        explainer = PointCloudExplainer(config=mock_config, model=None)

        # Proves the True branch of ``if self.state.model is None`` executed
        mock_logger.info.assert_any_call("Initializing point cloud model")
        custom_cls.assert_called_once()
        assert explainer.state.model is custom_instance
        assert explainer.state.perturbation is pert_instance


@patch("xwhy.explainers.point_cloud.calculate_distance", return_value=0.11)
@patch("xwhy.explainers.point_cloud.SurrogateTrainer")
@patch("xwhy.explainers.point_cloud.SurrogateFactory")
@patch("xwhy.explainers.point_cloud.RegressionMetrics")
def test_explain_unsqueezes_two_dimensional_input(
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_dist: MagicMock,
    mock_config: MagicMock,
    mock_model: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """Unsqueeze a two-dimensional point cloud before model inference.

    Args:
        mock_metrics: Mocked RegressionMetrics class.
        mock_factory: Mocked SurrogateFactory class.
        mock_trainer: Mocked SurrogateTrainer class.
        mock_dist: Mocked calculate_distance function.
        mock_config: Fixture providing a populated PointCloudConfig mock.
        mock_model: Fixture providing a BasePointCloudModel mock.
        sample_tensor: Fixture providing a two-dimensional point-cloud tensor.

    """
    mock_config.distance_mode = "mask"
    mock_config.use_best_surrogate = False
    explainer = _make_explainer(mock_config, model=mock_model)
    _prepare_explain_mocks(explainer)
    mock_trainer.compute_weights.return_value = np.ones(5)

    with patch.object(
        explainer, "_cluster_points", return_value=np.zeros(20, dtype=int)
    ):
        surrogate = MagicMock()
        surrogate.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        surrogate.predict.return_value = np.array([0.5] * 5)
        mock_factory.create.return_value = surrogate
        mock_metrics.calculate.return_value = MagicMock()

        assert sample_tensor.ndim == 2
        result = explainer.explain(instance=sample_tensor)

    assert isinstance(result, PointCloudXWhyResult)
    call_args = mock_model.predict.call_args
    assert call_args is not None
    passed = call_args.kwargs["sample_input"]
    assert isinstance(passed, torch.Tensor)
    assert passed.ndim == 3


@patch("xwhy.explainers.point_cloud.calculate_distance", return_value=0.11)
@patch("xwhy.explainers.point_cloud.SurrogateTrainer")
@patch("xwhy.explainers.point_cloud.SurrogateFactory")
@patch("xwhy.explainers.point_cloud.RegressionMetrics")
def test_explain_accepts_three_dimensional_input(
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_dist: MagicMock,
    mock_config: MagicMock,
    mock_model: MagicMock,
    sample_tensor: torch.Tensor,
) -> None:
    """Pass a three-dimensional point cloud through without extra unsqueeze.

    Args:
        mock_metrics: Mocked RegressionMetrics class.
        mock_factory: Mocked SurrogateFactory class.
        mock_trainer: Mocked SurrogateTrainer class.
        mock_dist: Mocked calculate_distance function.
        mock_config: Fixture providing a populated PointCloudConfig mock.
        mock_model: Fixture providing a BasePointCloudModel mock.
        sample_tensor: Fixture providing a two-dimensional point-cloud tensor.

    """
    mock_config.distance_mode = "mask"
    mock_config.use_best_surrogate = False
    explainer = _make_explainer(mock_config, model=mock_model)
    _prepare_explain_mocks(explainer)
    mock_trainer.compute_weights.return_value = np.ones(5)

    batched = sample_tensor.unsqueeze(0)
    assert batched.ndim == 3

    with patch.object(
        explainer, "_cluster_points", return_value=np.zeros(20, dtype=int)
    ):
        surrogate = MagicMock()
        surrogate.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        surrogate.predict.return_value = np.array([0.5] * 5)
        mock_factory.create.return_value = surrogate
        mock_metrics.calculate.return_value = MagicMock()

        result = explainer.explain(instance=batched)

    assert isinstance(result, PointCloudXWhyResult)
    call_args = mock_model.predict.call_args
    assert call_args is not None
    passed = call_args.kwargs["sample_input"]
    assert isinstance(passed, torch.Tensor)
    assert passed.ndim == 3


def test_initialize_skips_model_creation_when_present(
    mock_config: MagicMock,
) -> None:
    """Leave an existing model untouched and only create the perturbation.

    Args:
        mock_config: Fixture providing a populated PointCloudConfig mock.

    """
    existing_model = MagicMock(spec=BasePointCloudModel)

    # Plain namespace avoids MagicMock attribute auto-creation
    state = type("State", (), {})()
    state.device = torch.device("cpu")
    state.model = existing_model
    state.perturbation = None

    with (
        patch(
            "xwhy.explainers.point_cloud.PointCloudState",
            return_value=state,
        ),
        patch(
            "xwhy.explainers.point_cloud.CustomPointCloudModel",
        ) as custom_cls,
        patch(
            "xwhy.explainers.point_cloud.PointCloudPerturbation",
        ) as pert_cls,
        patch("xwhy.explainers.point_cloud.logger") as mock_logger,
    ):
        pert_instance = MagicMock()
        pert_cls.return_value = pert_instance

        # Pass the same model so __init__ also keeps it
        explainer = PointCloudExplainer(config=mock_config, model=existing_model)

        # The if-body must NOT have run
        mock_logger.info.assert_not_called()
        custom_cls.assert_not_called()

        # The original model is preserved and perturbation is still created
        assert explainer.state.model is existing_model
        assert explainer.state.perturbation is pert_instance
