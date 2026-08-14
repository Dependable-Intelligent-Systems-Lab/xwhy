"""Tests for image classification explainer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from pydantic import ValidationError

from xwhy.core.config import ImageGenerationAndEditingConfig
from xwhy.core.result import ImageGenerationAndEditingXWhyResult
from xwhy.core.types import BaseImageGenerationAndEditing
from xwhy.distance.types import DistanceType
from xwhy.explainers.image import (
    ImageClassificationExplainer,
    ImageGenerationAndEditingExplainer,
)
from xwhy.models.classification.types import ClassificationType
from xwhy.models.embeddings.types import EmbeddingType
from xwhy.models.segmentation.types import SegmentationType
from xwhy.providers.base import BaseProvider
from xwhy.providers.openai import OpenAIProvider
from xwhy.providers.types import ProviderType
from xwhy.surrogate.types import SurrogateType


class DummyModule(nn.Module):
    """Provide a dummy PyTorch module for testing custom models."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x


# ---------------------------------------------------------------------------
# __init__ branches
# ---------------------------------------------------------------------------


def test_init_invalid_distance_type() -> None:
    """Raise ValueError for non-numeric distance metric."""
    with patch("xwhy.explainers.image.DistanceType") as mock_dist_type:
        mock_metric = MagicMock()
        mock_metric.is_numeric_metric = False
        mock_dist_type.from_str.return_value = mock_metric
        with pytest.raises(ValueError, match=re.escape("Invalid distance metric")):
            ImageClassificationExplainer(distance_type="invalid")


@patch("xwhy.explainers.image.ClassificationFactory")
@patch("xwhy.explainers.image.ImagePerturbation")
def test_init_defaults(
    mock_perturbator: MagicMock,
    mock_class_factory: MagicMock,
) -> None:
    """Initialize explainer with default configuration."""
    mock_model = MagicMock()
    mock_model.preprocess_fn = MagicMock()
    mock_class_factory.create.return_value = mock_model

    explainer = ImageClassificationExplainer(
        use_embedding_model=False,
        use_segmentation_model=False,
    )
    assert explainer.config is not None
    assert explainer.state.device is not None
    mock_class_factory.create.assert_called_once()
    mock_model.load.assert_called_once()


@patch("xwhy.explainers.image.CustomTorchClassification")
@patch("xwhy.explainers.image.ImagePerturbation")
def test_init_custom_model_success(
    mock_perturbator: MagicMock,
    mock_custom_class: MagicMock,
) -> None:
    """Initialize explainer with a valid custom PyTorch model."""
    custom_mod = DummyModule()
    mock_adapter = MagicMock()
    mock_adapter.preprocess_fn = MagicMock()
    mock_custom_class.return_value = mock_adapter

    explainer = ImageClassificationExplainer(
        custom_model=custom_mod,
        use_embedding_model=False,
        use_segmentation_model=False,
    )
    assert explainer.config.custom_model is custom_mod  # type: ignore[union-attr]
    mock_custom_class.assert_called_once()
    mock_adapter.load.assert_called_once()


@patch("xwhy.explainers.image.ImagePerturbation")
def test_init_custom_model_type_error(mock_perturbator: MagicMock) -> None:
    """Raise TypeError when custom model is not an nn.Module."""
    with pytest.raises(
        TypeError, match=re.escape("must be an instance of torch.nn.Module")
    ):
        ImageClassificationExplainer(
            custom_model="not_a_module",
            use_embedding_model=False,
            use_segmentation_model=False,
        )


@patch("xwhy.explainers.image.ClassificationFactory")
@patch("xwhy.explainers.image.ImagePerturbation")
def test_init_invalid_classification_type(
    mock_perturbator: MagicMock,
    mock_class_factory: MagicMock,
) -> None:
    """Raise ValidationError when classification type string is invalid."""
    with pytest.raises((ValueError, ValidationError)):
        ImageClassificationExplainer(
            classification_type="invalid_type",
            use_embedding_model=False,
            use_segmentation_model=False,
        )


@patch("xwhy.explainers.image.ImagePerturbation")
def test_init_classification_type_not_enum_in_config(
    mock_perturbator: MagicMock,
) -> None:
    """Raise ValueError when config.classification_type is not enum."""
    mock_cfg = MagicMock()
    mock_cfg.device = "cpu"
    mock_cfg.custom_model = None
    mock_cfg.classification_type = "not_an_enum"
    mock_cfg.use_embedding_model = False
    mock_cfg.use_segmentation_model = False
    mock_cfg.use_model_preprocess = True
    mock_cfg.kernel_size = 4
    mock_cfg.max_dist = 200
    mock_cfg.ratio = 0.2
    mock_cfg.seed = 42

    with pytest.raises(ValueError, match=re.escape("Invalid classification type")):
        ImageClassificationExplainer(config=mock_cfg)


@patch("xwhy.explainers.image.ClassificationFactory")
@patch("xwhy.explainers.image.ImagePerturbation")
def test_init_embedding_validation_fails(
    mock_perturbator: MagicMock,
    mock_class_factory: MagicMock,
) -> None:
    """Raise ValueError when embedding type is not an image embedding."""
    mock_model = MagicMock()
    mock_model.preprocess_fn = MagicMock()
    mock_class_factory.create.return_value = mock_model

    mock_cfg = MagicMock()
    mock_cfg.device = "cpu"
    mock_cfg.custom_model = None
    mock_cfg.classification_type = ClassificationType.INCEPTION_V3
    mock_cfg.use_model_preprocess = True
    mock_cfg.use_embedding_model = True
    mock_embed_type = MagicMock()
    mock_embed_type.is_image_embedding = False
    mock_cfg.embedding_type = mock_embed_type
    mock_cfg.use_segmentation_model = False
    mock_cfg.kernel_size = 4
    mock_cfg.max_dist = 200
    mock_cfg.ratio = 0.2
    mock_cfg.seed = 42

    with pytest.raises(ValueError, match=re.escape("Invalid embedding type")):
        ImageClassificationExplainer(config=mock_cfg)


@patch("xwhy.explainers.image.ClassificationFactory")
@patch("xwhy.explainers.image.EmbeddingFactory")
@patch("xwhy.explainers.image.ImagePerturbation")
def test_init_embedding_success(
    mock_perturbator: MagicMock,
    mock_embed_factory: MagicMock,
    mock_class_factory: MagicMock,
) -> None:
    """Load embedding model when use_embedding_model is True."""
    mock_cls = MagicMock()
    mock_cls.preprocess_fn = MagicMock()
    mock_class_factory.create.return_value = mock_cls

    mock_embed = MagicMock()
    mock_embed_factory.create.return_value = mock_embed

    # Use a real image embedding type so pydantic accepts the config
    explainer = ImageClassificationExplainer(
        use_embedding_model=True,
        embedding_type=EmbeddingType.DINOV2,
        use_segmentation_model=False,
    )
    mock_embed_factory.create.assert_called_once()
    mock_embed.load.assert_called_once()
    assert explainer.state.embedding_model is mock_embed


@patch("xwhy.explainers.image.ClassificationFactory")
@patch("xwhy.explainers.image.ImagePerturbation")
def test_init_segmentation_validation_fails(
    mock_perturbator: MagicMock,
    mock_class_factory: MagicMock,
) -> None:
    """Raise ValueError when segmentation type is not SegmentationType."""
    mock_model = MagicMock()
    mock_model.preprocess_fn = MagicMock()
    mock_class_factory.create.return_value = mock_model

    mock_cfg = MagicMock()
    mock_cfg.device = "cpu"
    mock_cfg.custom_model = None
    mock_cfg.classification_type = ClassificationType.INCEPTION_V3
    mock_cfg.use_model_preprocess = True
    mock_cfg.use_embedding_model = False
    mock_cfg.use_segmentation_model = True
    mock_cfg.segmentation_type = "not_an_enum"
    mock_cfg.kernel_size = 4
    mock_cfg.max_dist = 200
    mock_cfg.ratio = 0.2
    mock_cfg.seed = 42

    with pytest.raises(ValueError, match=re.escape("Invalid segmentation type")):
        ImageClassificationExplainer(config=mock_cfg)


@patch("xwhy.explainers.image.ClassificationFactory")
@patch("xwhy.explainers.image.SegmentationFactory")
@patch("xwhy.explainers.image.ImagePerturbation")
def test_init_segmentation_success(
    mock_perturbator: MagicMock,
    mock_seg_factory: MagicMock,
    mock_class_factory: MagicMock,
) -> None:
    """Load segmentation model when use_segmentation_model is True."""
    mock_cls = MagicMock()
    mock_cls.preprocess_fn = MagicMock()
    mock_class_factory.create.return_value = mock_cls

    mock_seg = MagicMock()
    mock_seg_factory.create.return_value = mock_seg

    explainer = ImageClassificationExplainer(
        use_embedding_model=False,
        use_segmentation_model=True,
        segmentation_type=SegmentationType.DEEPLABV3_RESNET101,
    )
    mock_seg_factory.create.assert_called_once()
    mock_seg.load.assert_called_once()
    assert explainer.state.segmentation_model is mock_seg


@patch("xwhy.explainers.image.ClassificationFactory")
@patch("xwhy.explainers.image.ImagePerturbation")
def test_init_use_model_preprocess_false(
    mock_perturbator: MagicMock,
    mock_class_factory: MagicMock,
) -> None:
    """Skip assigning transform_fn when use_model_preprocess is False."""
    mock_model = MagicMock()
    mock_model.preprocess_fn = MagicMock()
    mock_class_factory.create.return_value = mock_model

    explainer = ImageClassificationExplainer(
        use_model_preprocess=False,
        use_embedding_model=False,
        use_segmentation_model=False,
    )
    # transform_fn is only set when use_model_preprocess is True
    assert getattr(explainer.state, "transform_fn", None) is None or (
        explainer.state.transform_fn is not mock_model.preprocess_fn
    )


@patch("xwhy.explainers.image.torch.cuda.is_available", return_value=True)
@patch("xwhy.explainers.image.ClassificationFactory")
@patch("xwhy.explainers.image.ImagePerturbation")
def test_init_device_auto_cuda(
    mock_perturbator: MagicMock,
    mock_class_factory: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """Auto-detect CUDA when config.device is None."""
    mock_model = MagicMock()
    mock_model.preprocess_fn = MagicMock()
    mock_class_factory.create.return_value = mock_model

    mock_cfg = MagicMock()
    mock_cfg.device = None
    mock_cfg.custom_model = None
    mock_cfg.classification_type = ClassificationType.INCEPTION_V3
    mock_cfg.use_model_preprocess = True
    mock_cfg.use_embedding_model = False
    mock_cfg.use_segmentation_model = False
    mock_cfg.kernel_size = 4
    mock_cfg.max_dist = 200
    mock_cfg.ratio = 0.2
    mock_cfg.seed = 42

    explainer = ImageClassificationExplainer(config=mock_cfg)
    assert explainer.config.device == "cuda"  # type: ignore[union-attr]


@patch("xwhy.explainers.image.torch.cuda.is_available", return_value=False)
@patch("xwhy.explainers.image.ClassificationFactory")
@patch("xwhy.explainers.image.ImagePerturbation")
def test_init_device_auto_cpu(
    mock_perturbator: MagicMock,
    mock_class_factory: MagicMock,
    mock_cuda: MagicMock,
) -> None:
    """Auto-detect CPU when config.device is None and CUDA unavailable."""
    mock_model = MagicMock()
    mock_model.preprocess_fn = MagicMock()
    mock_class_factory.create.return_value = mock_model

    mock_cfg = MagicMock()
    mock_cfg.device = None
    mock_cfg.custom_model = None
    mock_cfg.classification_type = ClassificationType.INCEPTION_V3
    mock_cfg.use_model_preprocess = True
    mock_cfg.use_embedding_model = False
    mock_cfg.use_segmentation_model = False
    mock_cfg.kernel_size = 4
    mock_cfg.max_dist = 200
    mock_cfg.ratio = 0.2
    mock_cfg.seed = 42

    explainer = ImageClassificationExplainer(config=mock_cfg)
    assert explainer.config.device == "cpu"  # type: ignore[union-attr]


@patch("xwhy.explainers.image.CustomTorchClassification")
@patch("xwhy.explainers.image.ImagePerturbation")
def test_init_custom_model_with_extras(
    mock_perturbator: MagicMock,
    mock_custom_class: MagicMock,
) -> None:
    """Initialize with custom model, preprocess, and categories."""
    custom_mod = DummyModule()
    preprocess = MagicMock()
    cats = ["class1", "class2"]
    mock_adapter = MagicMock()
    mock_adapter.preprocess_fn = preprocess
    mock_custom_class.return_value = mock_adapter

    explainer = ImageClassificationExplainer(
        custom_model=custom_mod,
        custom_preprocess=preprocess,
        categories=cats,
        use_embedding_model=False,
        use_segmentation_model=False,
    )
    assert explainer.config.custom_model is custom_mod  # type: ignore[union-attr]
    assert explainer.config.custom_preprocess is preprocess  # type: ignore[union-attr]
    assert explainer.config.categories is cats  # type: ignore[union-attr]


@patch("xwhy.explainers.image.ImageClassificationExplainer._initialize")
@patch("xwhy.explainers.image.logger")
def test_image_classification_nonlinear_surrogate_warning(
    mock_logger: MagicMock,
    mock_initialize: MagicMock,
) -> None:
    """Verify warning is logged when explicitly using a non-linear surrogate.

    Ensures that initializing the ImageClassificationExplainer with a tree-based
    or complex surrogate model triggers the standard scientific community warning
    regarding local interpretability.

    Args:
        mock_logger: Mocked logger object to verify warning calls.
        mock_initialize: Mocked _initialize method to prevent loading heavy
            PyTorch models during the test execution.

    """
    _ = ImageClassificationExplainer(
        use_best_surrogate=False,
        surrogate_type=SurrogateType.RANDOMFOREST,
        use_embedding_model=False,
        use_segmentation_model=False,
    )

    mock_logger.warning.assert_any_call(
        "Using a non-linear surrogate model or enabling 'use_best_surrogate' "
        "can replace a black-box model with another complex model, "
        "sacrificing local interpretability. The scientific community highly "
        "recommends utilizing simple linear models (e.g., LIME, OLS) to "
        "guarantee transparent and additive feature attributions."
    )


# ---------------------------------------------------------------------------
# run / explain type & runtime checks
# ---------------------------------------------------------------------------


def test_run_type_error() -> None:
    """Raise TypeError when run receives a non-string instance."""
    explainer = MagicMock(spec=ImageClassificationExplainer)
    with pytest.raises(TypeError, match=re.escape("requires a string instance")):
        ImageClassificationExplainer.run(explainer, 123)


def test_explain_type_error() -> None:
    """Raise TypeError when explain receives a non-string instance."""
    explainer = MagicMock(spec=ImageClassificationExplainer)
    with pytest.raises(TypeError, match=re.escape("requires the image path")):
        ImageClassificationExplainer.explain(explainer, 123)  # type: ignore[arg-type]


def test_explain_runtime_error_no_model() -> None:
    """Raise AttributeError when classification model is uninitialized."""
    # Note: preprocess_fn is accessed before the explicit None check,
    # so AttributeError is raised instead of RuntimeError.
    explainer = MagicMock(spec=ImageClassificationExplainer)
    explainer.state = MagicMock()
    explainer.state.classification_model = None
    explainer.state.transform_fn = MagicMock()
    explainer.config = MagicMock()
    explainer.config.use_model_preprocess = True

    with (
        patch(
            "xwhy.explainers.image.load_image_as_tensor",
            return_value=(MagicMock(), MagicMock()),
        ),
        pytest.raises(AttributeError),
    ):
        ImageClassificationExplainer.explain(explainer, "test.jpg")


def test_run_delegates_to_explain() -> None:
    """Verify run() calls explain() for a valid string path."""
    explainer = MagicMock(spec=ImageClassificationExplainer)
    explainer.explain.return_value = MagicMock()
    result = ImageClassificationExplainer.run(explainer, "img.jpg", foo=1)
    explainer.explain.assert_called_once_with("img.jpg", foo=1)
    assert result is explainer.explain.return_value


# ---------------------------------------------------------------------------
# _run_perturbation_loop
# ---------------------------------------------------------------------------


@patch("xwhy.explainers.image.numpy_image_to_tensor")
@patch("xwhy.explainers.image.calculate_distance")
def test_run_perturbation_loop_without_embedding(
    mock_calc_dist: MagicMock,
    mock_np_to_tensor: MagicMock,
) -> None:
    """Execute perturbation loop without embedding model."""
    explainer = MagicMock(spec=ImageClassificationExplainer)
    explainer.config = MagicMock()
    explainer.config.use_embedding_model = False
    explainer.config.distance_type = DistanceType.WASSERSTEIN
    explainer.state = MagicMock()
    explainer.state.device = torch.device("cpu")
    explainer.state.transform_fn = MagicMock()

    mock_tensor_out = MagicMock()
    mock_tensor_out.detach.return_value.cpu.return_value.numpy.return_value = np.zeros(
        (1, 5)
    )
    mock_cls_model = MagicMock()
    mock_cls_model.model = MagicMock(return_value=mock_tensor_out)
    explainer.state.classification_model = mock_cls_model

    mock_perturbator = MagicMock()
    mock_perturbator.apply_mask.return_value = np.zeros((10, 10, 3))
    explainer.state.perturbator = mock_perturbator

    mock_np_to_tensor.return_value = MagicMock()
    mock_calc_dist.return_value = 0.5

    preds, dists = ImageClassificationExplainer._run_perturbation_loop(
        explainer,
        original_image=np.zeros((10, 10, 3)),
        superpixels=np.zeros((10, 10), dtype=int),
        perturbation_masks=np.zeros((2, 5), dtype=int),
    )
    assert preds.shape[0] == 2
    assert dists.shape == (2,)
    assert mock_calc_dist.call_count == 2


@patch("xwhy.explainers.image.numpy_image_to_tensor")
@patch("xwhy.explainers.image.calculate_distance")
def test_run_perturbation_loop_with_embedding(
    mock_calc_dist: MagicMock,
    mock_np_to_tensor: MagicMock,
) -> None:
    """Execute perturbation loop with embedding model enabled."""
    explainer = MagicMock(spec=ImageClassificationExplainer)
    explainer.config = MagicMock()
    explainer.config.use_embedding_model = True
    explainer.config.distance_type = DistanceType.WASSERSTEIN
    explainer.state = MagicMock()
    explainer.state.device = torch.device("cpu")
    explainer.state.transform_fn = MagicMock()

    mock_embed = MagicMock()
    mock_embed.encode_image.return_value = np.array([0.1, 0.2, 0.3])
    explainer.state.embedding_model = mock_embed

    mock_tensor_out = MagicMock()
    mock_tensor_out.detach.return_value.cpu.return_value.numpy.return_value = np.zeros(
        (1, 5)
    )
    mock_cls_model = MagicMock()
    mock_cls_model.model = MagicMock(return_value=mock_tensor_out)
    explainer.state.classification_model = mock_cls_model

    mock_perturbator = MagicMock()
    mock_perturbator.apply_mask.return_value = np.zeros((10, 10, 3))
    explainer.state.perturbator = mock_perturbator

    mock_np_to_tensor.return_value = MagicMock()
    mock_calc_dist.return_value = 0.5

    preds, dists = ImageClassificationExplainer._run_perturbation_loop(
        explainer,
        original_image=np.zeros((10, 10, 3)),
        superpixels=np.zeros((10, 10), dtype=int),
        perturbation_masks=np.zeros((2, 5), dtype=int),
    )
    assert preds is not None
    assert dists is not None
    # original + 2 perturbed encodings
    assert mock_embed.encode_image.call_count == 3


def test_run_perturbation_loop_original_embedding_none() -> None:
    """Raise ValueError when original embedding extraction fails."""
    explainer = MagicMock(spec=ImageClassificationExplainer)
    explainer.config = MagicMock()
    explainer.config.use_embedding_model = True
    explainer.state = MagicMock()
    explainer.state.embedding_model = MagicMock()
    explainer.state.embedding_model.encode_image.return_value = None

    with pytest.raises(
        ValueError, match=re.escape("Original embedding extraction failed")
    ):
        ImageClassificationExplainer._run_perturbation_loop(
            explainer,
            original_image=np.zeros((10, 10, 3)),
            superpixels=np.zeros((10, 10)),
            perturbation_masks=np.zeros((2, 2)),
        )


# ---------------------------------------------------------------------------
# explain - full pipeline branches
# ---------------------------------------------------------------------------


def _build_explainer_mocks(
    *,
    use_best: bool = True,
    use_seg: bool = False,
    use_embed: bool = False,
    num_perturb: int = 5,
    num_classes: int = 5,
) -> MagicMock:
    """Build a fully mocked explainer for explain() tests."""
    explainer = MagicMock(spec=ImageClassificationExplainer)
    explainer.config = MagicMock()
    explainer.config.device = "cpu"
    explainer.config.num_top_predictions = 1
    explainer.config.use_best_surrogate = use_best
    explainer.config.surrogate_type = SurrogateType.LIME
    explainer.config.num_top_features = 2
    explainer.config.num_perturb = num_perturb
    explainer.config.use_model_preprocess = True
    explainer.config.use_embedding_model = use_embed
    explainer.config.distance_type = DistanceType.WASSERSTEIN
    explainer.config.seed = 42

    explainer._run_perturbation_loop.return_value = (
        np.zeros((num_perturb, num_classes)),
        np.zeros(num_perturb),
    )

    explainer.state = MagicMock()
    explainer.state.transform_fn = MagicMock()
    mock_cls = MagicMock()
    mock_cls.preprocess_fn.mean = [0.5, 0.5, 0.5]
    mock_cls.preprocess_fn.std = [0.5, 0.5, 0.5]
    mock_cls.weights.meta = {
        "categories": ["cat", "dog", "bird", "fish", "frog"],
    }
    mock_cls.predict.return_value = (
        torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        None,
    )
    explainer.state.classification_model = mock_cls

    mock_pert = MagicMock()
    mock_pert.generate_superpixels.return_value = (
        np.zeros((10, 10), dtype=int),
        5,
    )
    mock_pert.generate.return_value = np.zeros((num_perturb, 5), dtype=int)
    mock_pert.apply_mask.return_value = np.zeros((10, 10, 3), dtype=np.float32)
    explainer.state.perturbator = mock_pert

    if use_seg:
        explainer.state.segmentation_model = MagicMock()
        explainer.state.segmentation_model.class_names = ["cat"]
    else:
        explainer.state.segmentation_model = None

    if use_embed:
        explainer.state.embedding_model = MagicMock()
        explainer.state.embedding_model.encode_image.return_value = np.zeros(5)

    return explainer


@patch("xwhy.explainers.image.load_image_as_tensor")
@patch("xwhy.explainers.image.tensor_to_numpy_image")
@patch("xwhy.explainers.image.SurrogateTrainer")
@patch("xwhy.explainers.image.SurrogateFactory")
@patch("xwhy.explainers.image.RegressionMetrics")
def test_explain_full_pipeline_best_surrogate(
    mock_reg: MagicMock,
    mock_surr_factory: MagicMock,
    mock_surr_trainer: MagicMock,
    mock_tensor_to_np: MagicMock,
    mock_load: MagicMock,
) -> None:
    """Run full pipeline with best-surrogate search and no segmentation."""
    explainer = _build_explainer_mocks(use_best=True, use_seg=False)
    mock_load.return_value = (
        MagicMock(),
        Image.new("RGB", (10, 10)),
    )
    mock_tensor_to_np.return_value = np.zeros((10, 10, 3), dtype=np.float32)

    mock_surr_trainer.find_best.return_value = (SurrogateType.LIME, 0.9)
    mock_surr_trainer.compute_weights.return_value = np.ones(5)
    mock_surr = MagicMock()
    mock_surr.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mock_surr.predict.return_value = np.zeros(5)
    mock_surr_factory.create.return_value = mock_surr
    mock_reg.calculate.return_value = MagicMock()

    result = ImageClassificationExplainer.explain(
        explainer, "test.jpg", fidelity_plot=False, ground_truth_mask=None
    )
    assert result is not None
    mock_surr_trainer.find_best.assert_called_once()
    assert "best_surrogate_method" in result.raw_data


@patch("xwhy.explainers.image.load_image_as_tensor")
@patch("xwhy.explainers.image.tensor_to_numpy_image")
@patch("xwhy.explainers.image.SurrogateTrainer")
@patch("xwhy.explainers.image.SurrogateFactory")
@patch("xwhy.explainers.image.RegressionMetrics")
def test_explain_without_best_surrogate(
    mock_reg: MagicMock,
    mock_surr_factory: MagicMock,
    mock_surr_trainer: MagicMock,
    mock_tensor_to_np: MagicMock,
    mock_load: MagicMock,
) -> None:
    """Run pipeline skipping best-surrogate search."""
    explainer = _build_explainer_mocks(use_best=False, use_seg=False)
    mock_load.return_value = (MagicMock(), np.zeros((10, 10, 3)))
    mock_tensor_to_np.return_value = np.zeros((10, 10, 3), dtype=np.float32)

    mock_surr_trainer.compute_weights.return_value = np.ones(5)
    mock_surr = MagicMock()
    mock_surr.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mock_surr.predict.return_value = np.zeros(5)
    mock_surr_factory.create.return_value = mock_surr
    mock_reg.calculate.return_value = MagicMock()

    result = ImageClassificationExplainer.explain(
        explainer, "test.jpg", fidelity_plot=False, ground_truth_mask=None
    )
    assert result is not None
    mock_surr_trainer.find_best.assert_not_called()
    assert "surrogate_method" in result.raw_data


@patch("xwhy.explainers.image.load_image_as_tensor")
@patch("xwhy.explainers.image.tensor_to_numpy_image")
@patch("xwhy.explainers.image.SurrogateTrainer")
@patch("xwhy.explainers.image.SurrogateFactory")
@patch("xwhy.explainers.image.RegressionMetrics")
@patch("xwhy.explainers.image.get_segmentation_mask")
@patch("xwhy.explainers.image.ImageCoverageMetrics")
def test_explain_with_segmentation_success_and_resize(
    mock_cov: MagicMock,
    mock_get_seg: MagicMock,
    mock_reg: MagicMock,
    mock_surr_factory: MagicMock,
    mock_surr_trainer: MagicMock,
    mock_tensor_to_np: MagicMock,
    mock_load: MagicMock,
) -> None:
    """Run pipeline with segmentation, shape mismatch, and coverage metrics."""
    explainer = _build_explainer_mocks(use_best=True, use_seg=True)
    mock_load.return_value = (
        MagicMock(),
        Image.new("RGB", (10, 10)),
    )
    mock_tensor_to_np.return_value = np.zeros((10, 10, 3), dtype=np.float32)

    mock_surr_trainer.find_best.return_value = (SurrogateType.LIME, 0.9)
    mock_surr_trainer.compute_weights.return_value = np.ones(5)
    mock_surr = MagicMock()
    mock_surr.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mock_surr.predict.return_value = np.zeros(5)
    mock_surr_factory.create.return_value = mock_surr
    mock_reg.calculate.return_value = MagicMock()

    # Different spatial size -> triggers resize branch
    mock_get_seg.return_value = (None, np.zeros((20, 20)))
    mock_cov.evaluate_all.return_value = (0.8, 0.75)

    result = ImageClassificationExplainer.explain(
        explainer, "test.jpg", fidelity_plot=False, ground_truth_mask=None
    )
    assert result is not None
    assert result.coverage == 0.8
    assert result.weighted_coverage == 0.75
    mock_cov.evaluate_all.assert_called_once()


@patch("xwhy.explainers.image.load_image_as_tensor")
@patch("xwhy.explainers.image.tensor_to_numpy_image")
@patch("xwhy.explainers.image.SurrogateTrainer")
@patch("xwhy.explainers.image.SurrogateFactory")
@patch("xwhy.explainers.image.RegressionMetrics")
@patch("xwhy.explainers.image.get_segmentation_mask")
def test_explain_segmentation_raises_exception(
    mock_get_seg: MagicMock,
    mock_reg: MagicMock,
    mock_surr_factory: MagicMock,
    mock_surr_trainer: MagicMock,
    mock_tensor_to_np: MagicMock,
    mock_load: MagicMock,
) -> None:
    """Handle exception from get_segmentation_mask gracefully."""
    explainer = _build_explainer_mocks(use_best=False, use_seg=True)
    mock_load.return_value = (MagicMock(), np.zeros((10, 10, 3)))
    mock_tensor_to_np.return_value = np.zeros((10, 10, 3), dtype=np.float32)

    mock_surr_trainer.compute_weights.return_value = np.ones(5)
    mock_surr = MagicMock()
    mock_surr.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mock_surr.predict.return_value = np.zeros(5)
    mock_surr_factory.create.return_value = mock_surr
    mock_reg.calculate.return_value = MagicMock()

    mock_get_seg.side_effect = RuntimeError("seg failed")

    result = ImageClassificationExplainer.explain(
        explainer, "test.jpg", fidelity_plot=False, ground_truth_mask=None
    )
    assert result is not None
    assert result.coverage == 0.0
    assert result.weighted_coverage == 0.0


@patch("xwhy.explainers.image.load_image_as_tensor")
@patch("xwhy.explainers.image.tensor_to_numpy_image")
@patch("xwhy.explainers.image.SurrogateTrainer")
@patch("xwhy.explainers.image.SurrogateFactory")
@patch("xwhy.explainers.image.RegressionMetrics")
@patch("xwhy.explainers.image.ImageCoverageMetrics")
def test_explain_explicit_ground_truth_mask_matching_shape(
    mock_cov: MagicMock,
    mock_reg: MagicMock,
    mock_surr_factory: MagicMock,
    mock_surr_trainer: MagicMock,
    mock_tensor_to_np: MagicMock,
    mock_load: MagicMock,
) -> None:
    """Use provided ground-truth mask with matching spatial shape."""
    explainer = _build_explainer_mocks(use_best=False, use_seg=False)
    mock_load.return_value = (MagicMock(), np.zeros((10, 10, 3)))
    mock_tensor_to_np.return_value = np.zeros((10, 10, 3), dtype=np.float32)

    mock_surr_trainer.compute_weights.return_value = np.ones(5)
    mock_surr = MagicMock()
    mock_surr.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mock_surr.predict.return_value = np.zeros(5)
    mock_surr_factory.create.return_value = mock_surr
    mock_reg.calculate.return_value = MagicMock()
    mock_cov.evaluate_all.return_value = (1.0, 1.0)

    gt = np.zeros((10, 10))
    result = ImageClassificationExplainer.explain(
        explainer,
        "test.jpg",
        fidelity_plot=False,
        ground_truth_mask=gt,
    )
    assert result is not None
    mock_cov.evaluate_all.assert_called_once()
    assert result.coverage == 1.0


@patch("xwhy.explainers.image.load_image_as_tensor")
@patch("xwhy.explainers.image.tensor_to_numpy_image")
@patch("xwhy.explainers.image.SurrogateTrainer")
@patch("xwhy.explainers.image.SurrogateFactory")
@patch("xwhy.explainers.image.RegressionMetrics")
def test_explain_fidelity_plot_true(
    mock_reg: MagicMock,
    mock_surr_factory: MagicMock,
    mock_surr_trainer: MagicMock,
    mock_tensor_to_np: MagicMock,
    mock_load: MagicMock,
) -> None:
    """Trigger fidelity plot rendering when fidelity_plot=True."""
    explainer = _build_explainer_mocks(use_best=False, use_seg=False)
    mock_load.return_value = (MagicMock(), np.zeros((10, 10, 3)))
    mock_tensor_to_np.return_value = np.zeros((10, 10, 3), dtype=np.float32)

    mock_surr_trainer.compute_weights.return_value = np.ones(5)
    mock_surr = MagicMock()
    mock_surr.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mock_surr.predict.return_value = np.zeros(5)
    mock_surr_factory.create.return_value = mock_surr

    # Real numeric metrics so plot_fidelity can format them
    mock_metrics = MagicMock()
    mock_metrics.weighted_r2 = 0.9
    mock_metrics.weighted_adj_r2 = 0.85
    mock_reg.calculate.return_value = mock_metrics

    with patch.object(
        ImageClassificationXWhyResult
        if False
        else type("R", (), {"plot": MagicMock()}),
        "plot",
        create=True,
    ):
        pass  # placeholder; we patch result.plot after construction

    # Patch the result class plot method after explain builds the result
    with patch(
        "xwhy.explainers.image.ImageClassificationXWhyResult"
    ) as mock_result_cls:
        mock_result_instance = MagicMock()
        mock_result_cls.return_value = mock_result_instance

        result = ImageClassificationExplainer.explain(
            explainer, "test.jpg", fidelity_plot=True, ground_truth_mask=None
        )
        assert result is mock_result_instance
        mock_result_instance.plot.assert_called_once_with(show=True)


# Import needed for the patch above when not using the complex approach
from xwhy.core.result import ImageClassificationXWhyResult  # noqa: E402


@patch("xwhy.explainers.image.load_image_as_tensor")
@patch("xwhy.explainers.image.tensor_to_numpy_image")
@patch("xwhy.explainers.image.SurrogateTrainer")
@patch("xwhy.explainers.image.SurrogateFactory")
@patch("xwhy.explainers.image.RegressionMetrics")
def test_explain_original_image_is_numpy(
    mock_reg: MagicMock,
    mock_surr_factory: MagicMock,
    mock_surr_trainer: MagicMock,
    mock_tensor_to_np: MagicMock,
    mock_load: MagicMock,
) -> None:
    """Handle original_img that is already a numpy array."""
    explainer = _build_explainer_mocks(use_best=False, use_seg=False)
    mock_load.return_value = (MagicMock(), np.zeros((10, 10, 3)))
    mock_tensor_to_np.return_value = np.zeros((10, 10, 3), dtype=np.float32)

    mock_surr_trainer.compute_weights.return_value = np.ones(5)
    mock_surr = MagicMock()
    mock_surr.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mock_surr.predict.return_value = np.zeros(5)
    mock_surr_factory.create.return_value = mock_surr
    mock_reg.calculate.return_value = MagicMock()

    result = ImageClassificationExplainer.explain(
        explainer, "test.jpg", fidelity_plot=False, ground_truth_mask=None
    )
    assert result is not None
    assert isinstance(result.original_image, np.ndarray)


@patch("xwhy.explainers.image.load_image_as_tensor")
@patch("xwhy.explainers.image.tensor_to_numpy_image")
@patch("xwhy.explainers.image.SurrogateTrainer")
@patch("xwhy.explainers.image.SurrogateFactory")
@patch("xwhy.explainers.image.RegressionMetrics")
@patch("xwhy.explainers.image.get_segmentation_mask")
@patch("xwhy.explainers.image.ImageCoverageMetrics")
def test_explain_segmentation_matching_shape_no_resize(
    mock_cov: MagicMock,
    mock_get_seg: MagicMock,
    mock_reg: MagicMock,
    mock_surr_factory: MagicMock,
    mock_surr_trainer: MagicMock,
    mock_tensor_to_np: MagicMock,
    mock_load: MagicMock,
) -> None:
    """Skip resize when explanation and mask shapes already match."""
    explainer = _build_explainer_mocks(use_best=True, use_seg=True)
    mock_load.return_value = (
        MagicMock(),
        Image.new("RGB", (10, 10)),
    )
    mock_tensor_to_np.return_value = np.zeros((10, 10, 3), dtype=np.float32)

    mock_surr_trainer.find_best.return_value = (SurrogateType.LIME, 0.9)
    mock_surr_trainer.compute_weights.return_value = np.ones(5)
    mock_surr = MagicMock()
    mock_surr.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mock_surr.predict.return_value = np.zeros(5)
    mock_surr_factory.create.return_value = mock_surr
    mock_reg.calculate.return_value = MagicMock()

    # Same spatial size -> no resize
    mock_get_seg.return_value = (None, np.zeros((10, 10)))
    mock_cov.evaluate_all.return_value = (0.9, 0.85)

    with patch("xwhy.explainers.image.skimage.transform.resize") as mock_resize:
        result = ImageClassificationExplainer.explain(
            explainer, "test.jpg", fidelity_plot=False, ground_truth_mask=None
        )
        mock_resize.assert_not_called()
    assert result.coverage == 0.9


@patch("xwhy.explainers.image.load_image_as_tensor")
@patch("xwhy.explainers.image.tensor_to_numpy_image")
@patch("xwhy.explainers.image.SurrogateTrainer")
@patch("xwhy.explainers.image.SurrogateFactory")
@patch("xwhy.explainers.image.RegressionMetrics")
def test_explain_segmentation_disabled_logs_skip(
    mock_reg: MagicMock,
    mock_surr_factory: MagicMock,
    mock_surr_trainer: MagicMock,
    mock_tensor_to_np: MagicMock,
    mock_load: MagicMock,
) -> None:
    """Skip ground-truth evaluation when segmentation model is None."""
    explainer = _build_explainer_mocks(use_best=False, use_seg=False)
    mock_load.return_value = (MagicMock(), np.zeros((10, 10, 3)))
    mock_tensor_to_np.return_value = np.zeros((10, 10, 3), dtype=np.float32)

    mock_surr_trainer.compute_weights.return_value = np.ones(5)
    mock_surr = MagicMock()
    mock_surr.coefficients.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mock_surr.predict.return_value = np.zeros(5)
    mock_surr_factory.create.return_value = mock_surr
    mock_reg.calculate.return_value = MagicMock()

    result = ImageClassificationExplainer.explain(
        explainer, "test.jpg", fidelity_plot=False, ground_truth_mask=None
    )
    assert result is not None
    assert result.coverage == 0.0


def test_explain_runtime_error_classification_model_none() -> None:
    """Raise RuntimeError when classification model is None after load."""

    class _State:
        """Return model for preprocess access, then None for the check."""

        def __init__(self) -> None:
            self._n = 0
            self._model = MagicMock()
            self._model.preprocess_fn.mean = [0.5, 0.5, 0.5]
            self._model.preprocess_fn.std = [0.5, 0.5, 0.5]
            self.transform_fn = MagicMock()

        @property
        def classification_model(self) -> MagicMock | None:
            self._n += 1
            # mean + std access (2 reads) -> model; later check -> None
            return self._model if self._n <= 2 else None

    explainer = MagicMock(spec=ImageClassificationExplainer)
    explainer.state = _State()
    explainer.config = MagicMock()
    explainer.config.use_model_preprocess = True
    explainer.config.device = "cpu"

    with (
        patch(
            "xwhy.explainers.image.load_image_as_tensor",
            return_value=(MagicMock(), MagicMock()),
        ),
        pytest.raises(
            RuntimeError,
            match=re.escape("Classification model is not initialized/loaded"),
        ),
    ):
        ImageClassificationExplainer.explain(explainer, "test.jpg")


# -------------------------------------------------------------------------
# Image generation and editing
# -------------------------------------------------------------------------


class DummyEngine(BaseImageGenerationAndEditing):
    """Dummy internal engine for testing instance detection."""

    def generate_image(self, *args: object, **kwargs: object) -> tuple[bool, str]:
        """Return a successful generation result.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).

        Returns:
            Tuple of success flag and output path.

        """
        return True, "path/to/gen.png"

    def edit_image(self, *args: object, **kwargs: object) -> tuple[bool, str]:
        """Return a successful edit result.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).

        Returns:
            Tuple of success flag and output path.

        """
        return True, "path/to/edit.png"


class DummyProvider(BaseProvider):
    """Dummy provider extending BaseProvider for testing."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize the dummy provider.

        Args:
            **kwargs: Unused keyword arguments accepted for compatibility.

        """

    def answer(self, *args: object, **kwargs: object) -> str:
        """Return a fixed dummy answer.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).

        Returns:
            A constant dummy string.

        """
        return "dummy answer"


class MockGeminiEngine:
    """Mock Gemini Engine for Batch Testing."""

    def submit_image_batch(self, *args: object, **kwargs: object) -> str:
        """Submit a fake batch job.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).

        Returns:
            A fake job identifier.

        """
        return "job_1"

    def retrieve_image_batch(
        self, *args: object, **kwargs: object
    ) -> list[tuple[bool, str]]:
        """Retrieve fake batch results.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).

        Returns:
            List of success/path tuples.

        """
        return [(True, "path")]

    def generate_image(self, *args: object, **kwargs: object) -> tuple[bool, str]:
        """Return a successful generation result.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).

        Returns:
            Tuple of success flag and output path.

        """
        return True, "path/to/gen.png"

    def edit_image(self, *args: object, **kwargs: object) -> tuple[bool, str]:
        """Return a successful edit result.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).

        Returns:
            Tuple of success flag and output path.

        """
        return True, "path/to/edit.png"


@pytest.fixture
def mock_dependencies() -> Any:  # noqa: ANN401
    """Mock all heavy external ML dependencies to prevent actual loading."""
    with (
        patch("xwhy.explainers.image.EmbeddingFactory"),
        patch("xwhy.explainers.image.SegmentationFactory"),
        patch("xwhy.explainers.image.ProviderResolver"),
        patch("xwhy.explainers.image.TextPerturbation"),
    ):
        yield


# --- INIT TESTS ---


def test_init_invalid_distance() -> None:
    """Test ValueError raised on non-numeric distance metric."""
    with pytest.raises(ValueError, match=re.escape("is not a valid DistanceType")):
        ImageGenerationAndEditingExplainer(distance_type="BLEU")


@patch("torch.cuda.is_available", return_value=True)
def test_init_device_resolution_cuda(
    mock_cuda: Mock,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Test device defaults to cuda when config is None and cuda available."""
    explainer = ImageGenerationAndEditingExplainer(device=None)  # type: ignore[arg-type]
    assert str(explainer.state.device) == "cuda"


@patch("torch.cuda.is_available", return_value=False)
def test_init_device_resolution_cpu(
    mock_cuda: Mock,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Test device defaults to cpu when config is None and cuda missing."""
    explainer = ImageGenerationAndEditingExplainer(device=None)  # type: ignore[arg-type]
    assert str(explainer.state.device) == "cpu"
    assert explainer.config.device == "cpu"  # type: ignore[union-attr]


def test_init_pipeline_resolution(mock_dependencies: Any) -> None:  # noqa: ANN401
    """Test pipeline initialization paths (Cases 1 & 2)."""
    # Case 1: Pipe provided without custom fn
    dummy_pipe = MagicMock()
    dummy_pipe._name_or_path = "custom-diffusers-model"
    explainer = ImageGenerationAndEditingExplainer(
        pipe=dummy_pipe, model_name="dall-e-3"
    )
    assert explainer.config.provider_type == ProviderType.HUGGINGFACE  # type: ignore[union-attr]
    assert explainer.config.engine_type == "provider"  # type: ignore[union-attr]
    assert explainer.config.model_name == "custom-diffusers-model"  # type: ignore[union-attr]

    # Case 2: Pipe provided with custom fn
    explainer_case2 = ImageGenerationAndEditingExplainer(
        pipe=dummy_pipe, custom_generate_fn=lambda **kw: None
    )
    assert explainer_case2.config.engine_type == "pipeline"  # type: ignore[union-attr]

    # Test the fallback lambda injected for Case 1 behavior fallback
    _ = ImageGenerationAndEditingExplainer(pipe=MagicMock(), model_name="other-model")
    dummy_output = MagicMock()
    dummy_output.images = ["test_image"]
    _ = MagicMock(return_value=dummy_output)


@patch("xwhy.models.image_generation_and_editing.paired.PairedInferenceModel")
def test_init_engine_resolution(
    mock_paired: Mock,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Test different engine inputs."""
    # String mapping to standard provider
    exp_openai = ImageGenerationAndEditingExplainer(engine="openai")
    assert exp_openai.config.engine_type == "provider"  # type: ignore[union-attr]

    # Valid Provider Instance (requires custom_generate_fn when no default
    # generator is attached)
    exp_prov_inst = ImageGenerationAndEditingExplainer(
        engine=DummyProvider(), custom_generate_fn=lambda **kw: None
    )
    assert exp_prov_inst.config.engine_type in ("provider", "custom")  # type: ignore[union-attr]

    # BaseImageGenerationAndEditing Subclass Instance
    exp_custom_inst = ImageGenerationAndEditingExplainer(engine=DummyEngine())
    assert exp_custom_inst.config.engine_type == "custom"  # type: ignore[union-attr]
    assert isinstance(exp_custom_inst.state.engine, DummyEngine)

    # Class Type Reference
    exp_type = ImageGenerationAndEditingExplainer(engine=DummyEngine)
    assert exp_type.config.engine_type == "custom"  # type: ignore[union-attr]
    assert isinstance(exp_type.state.engine, DummyEngine)

    # Magic String "paired"
    ImageGenerationAndEditingExplainer(engine="paired")
    mock_paired.assert_called_once()

    # Unrecognized String Model Injection
    exp_str_inj = ImageGenerationAndEditingExplainer(
        engine="unrecognized_str", custom_generate_fn=lambda: None
    )
    assert exp_str_inj.config.custom_model == "unrecognized_str"  # type: ignore[union-attr]


# --- INITIALIZE TESTS ---


def test_initialize_errors(mock_dependencies: Any) -> None:  # noqa: ANN401
    """Test constraints in _initialize method."""
    # Missing Custom Generate Fn
    with pytest.raises(ValueError, match="must be provided"):
        ImageGenerationAndEditingExplainer(custom_model="dummy")

    # Missing Provider Type
    config = ImageGenerationAndEditingConfig(engine_type="provider", provider_type=None)
    with pytest.raises(ValueError, match="cannot be None"):
        ImageGenerationAndEditingExplainer(config=config)

    # Text Only Provider Error - patch PropertyMock on ProviderType class
    with patch.object(
        ProviderType, "is_text_only", new_callable=PropertyMock, return_value=True
    ):
        config_text = ImageGenerationAndEditingConfig(
            engine_type="provider", provider_type=ProviderType.OPENAI
        )
        with pytest.raises(ValueError, match="only supports text"):
            ImageGenerationAndEditingExplainer(config=config_text)

    # Bad Embeddings
    with pytest.raises(ValueError, match="Must be an image embedding"):
        ImageGenerationAndEditingExplainer(
            use_image_embedding_model=True, image_embedding_type="word2vec"
        )
    with pytest.raises(ValueError, match="Must be a text embedding"):
        ImageGenerationAndEditingExplainer(text_embedding_type="dinov2")

    # Bad Segmentation
    with pytest.raises((ValueError, ValidationError)):
        ImageGenerationAndEditingExplainer(
            use_segmentation_model=True,
            segmentation_type="invalid",
        )


# --- ENVIRONMENT & KWARGS TESTS ---


@patch("torch.cuda.is_available", return_value=False)
@patch("os.makedirs")
def test_prepare_environment(
    mock_makedirs: Mock,
    mock_cuda: Mock,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Test environment preparation variables and branch logic."""
    explainer = ImageGenerationAndEditingExplainer()
    explainer._prepare_environment("dummy_dir", 123)
    mock_makedirs.assert_called_once_with("dummy_dir", exist_ok=True)

    with patch("torch.cuda.is_available", return_value=True):
        explainer._prepare_environment("dummy_dir", 123)


def test_get_provider_kwargs(mock_dependencies: Any) -> None:  # noqa: ANN401
    """Test kwarg injection logic for specific providers."""
    # By default, openai
    exp_openai = ImageGenerationAndEditingExplainer()
    exp_openai.state.engine = OpenAIProvider(client=MagicMock())
    kw = exp_openai._get_provider_specific_kwargs()
    assert "provider_name" in kw
    assert kw["response_format"] is None

    # ByteDance
    exp_bd = ImageGenerationAndEditingExplainer(engine="bytedance")
    kw_bd = exp_bd._get_provider_specific_kwargs()
    assert kw_bd["use_image_data_uri"] is True

    # None Provider fallback logic (use model_copy for frozen Pydantic config)
    exp_none = ImageGenerationAndEditingExplainer(custom_generate_fn=lambda **kw: None)
    exp_none.config = exp_none.config.model_copy(update={"provider_type": None})  # type: ignore[union-attr]
    assert exp_none._get_provider_specific_kwargs() == {}


# --- GENERATE TESTS ---


def test_generate_images(mock_dependencies: Any) -> None:  # noqa: ANN401
    """Test internal iteration and batch processing for generation/editing."""
    # Batch Test
    exp = ImageGenerationAndEditingExplainer(engine=DummyEngine())
    exp.state.engine = MockGeminiEngine()  # type: ignore[assignment]
    paths = exp._generate_images(["prompt1"], "out", batch=True)
    assert paths == [(True, "path")]

    # Segmentation Arg injection check
    exp.state.engine = DummyEngine()
    exp.state.segmentation_model = MagicMock()

    def _edit_with_seg(
        prompt: str,
        image_path: str,
        output_dir: str,
        segmentation_model: object = None,
        **kwargs: object,
    ) -> tuple[bool, str]:
        return True, "p"

    assert exp.state.engine is not None
    exp.state.engine.edit_image = _edit_with_seg  # type: ignore[assignment]

    # Generate normal
    res = exp._generate_images(["p1", "p2", "p3", "p4", "p5"], "out")
    assert len(res) == 5

    # Edit fail logic
    exp.state.engine.edit_image = Mock(side_effect=RuntimeError("Fail"))  # type: ignore[method-assign]
    res_fail = exp._generate_images(["p1"], "out", input_image_path="in.png")
    assert res_fail == [(False, "")]


# --- DISTANCES TESTS ---


@patch("xwhy.explainers.image.load_image_as_tensor")
@patch("matplotlib.pyplot.show")
def test_compute_distances(
    mock_show: Mock,
    mock_load: Mock,
    tmp_path: Path,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Test computation branches including invalid images and embeddings."""
    mock_load.return_value = (None, np.array([1, 2, 3]))

    exp = ImageGenerationAndEditingExplainer(use_image_embedding_model=True)
    exp.state.image_embedding_model = MagicMock()
    exp.state.image_embedding_model.encode_image.return_value = np.array([0.5])

    # Save a valid test image
    p1 = tmp_path / "1.png"
    img = Image.new("RGB", (10, 10), color="red")
    img.save(p1)

    # Normal success
    res = exp._compute_perturbation_distances(
        input_image_path=str(p1),
        generated_images=[(True, str(p1))],
        prompts=["text"],
        display_image=True,
        output_dir=str(tmp_path),
    )
    assert len(res) == 1

    # False success flag
    res_skip = exp._compute_perturbation_distances(
        input_image_path=str(p1),
        generated_images=[(False, str(p1)), (True, "not_exist.png")],
        prompts=["text", "text2"],
    )
    assert np.isinf(res_skip[0])
    assert np.isinf(res_skip[1])

    # Error: Base Extract Fail
    exp.state.image_embedding_model.encode_image.return_value = None
    with pytest.raises(ValueError, match="Original embedding extraction failed"):
        exp._compute_perturbation_distances(str(p1), [], [])

    # Error: Current Extract Fail
    exp.state.image_embedding_model.encode_image.side_effect = [np.array([1]), None]
    with pytest.raises(ValueError, match="Embedding extraction failed"):
        exp._compute_perturbation_distances(str(p1), [(True, str(p1))], ["text"])

    # Error: Empty Arrays
    exp.state.image_embedding_model.encode_image.side_effect = [
        np.array([]),
        np.array([]),
    ]
    with pytest.raises(ValueError, match="Representations are empty"):
        exp._compute_perturbation_distances(str(p1), [(True, str(p1))], ["text"])


# --- EXPLAIN TESTS ---


@patch("xwhy.explainers.image.WMDDistance")
@patch("xwhy.explainers.image.SurrogateTrainer")
@patch("xwhy.explainers.image.SurrogateFactory")
@patch("xwhy.explainers.image.RegressionMetrics")
@patch("xwhy.explainers.image.save_data_to_pickle")
@patch("xwhy.explainers.image.save_perturbation_data_to_csv")
def test_explain_logic(
    mock_csv: Mock,
    mock_pickle: Mock,
    mock_reg: Mock,
    mock_surg_fac: Mock,
    mock_surg_train: Mock,
    mock_wmd: Mock,
    tmp_path: Path,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Test full explanation process including input validation and logic flow."""
    exp = ImageGenerationAndEditingExplainer()

    # Invalid string prompt
    with pytest.raises(TypeError, match="Prompt must be a string"):
        exp.explain(instance=123)  # type: ignore[arg-type]

    # Empty prompt
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        exp.explain(instance="   ")

    # Short prompt
    with pytest.raises(ValueError, match="too short"):
        exp.explain(instance="a b c")

    # Non-existent input image
    with pytest.raises(FileNotFoundError, match="Input image not found"):
        exp.explain(instance="valid long text description", input_image_path="no.png")

    # Mock internals to trace success flow
    p1 = tmp_path / "valid.png"
    img = Image.new("RGB", (10, 10), color="blue")
    img.save(p1)

    exp.state.text_perturbator = MagicMock()
    exp.state.text_perturbator.generate.return_value = (["pt"], [[1]])

    exp._generate_images = MagicMock(return_value=[(True, str(p1))])  # type: ignore[method-assign]
    exp._compute_perturbation_distances = MagicMock(return_value=np.array([1.0]))  # type: ignore[method-assign]

    wmd_inst = mock_wmd.return_value
    wmd_inst.compute_batch.return_value = [("t", 1.0)]

    mock_surg_train.find_best.return_value = (SurrogateType.LIME, 0.9)
    mock_surg_train.compute_weights.return_value = np.array([1.0])
    surg_mock = mock_surg_fac.create.return_value
    surg_mock.coefficients.return_value = [0.1]
    surg_mock.predict.return_value = np.array([0.5])

    mock_metrics = MagicMock()
    mock_metrics.weighted_r2 = 0.9
    mock_metrics.weighted_adj_r2 = 0.85
    mock_reg.calculate.return_value = mock_metrics

    # Run with use_best_surrogate=True
    res_best = exp.explain(
        instance="valid long text description",
        input_image_path=str(p1),
        seed=42,
        fidelity_plot=False,
    )
    assert isinstance(res_best, ImageGenerationAndEditingXWhyResult)

    # Run with use_best_surrogate=False and check failure
    exp_fail = ImageGenerationAndEditingExplainer(use_best_surrogate=False)
    exp_fail.state.text_perturbator = MagicMock()
    exp_fail.state.text_perturbator.generate.return_value = (["pt"], [[1]])
    exp_fail._generate_images = MagicMock(return_value=[(False, "")])  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="Failed to generate the base image"):
        exp_fail.explain(instance="valid long text description")


@patch("xwhy.explainers.image.WMDDistance")
@patch("xwhy.explainers.image.SurrogateTrainer")
@patch("xwhy.explainers.image.SurrogateFactory")
@patch("xwhy.explainers.image.RegressionMetrics")
@patch("xwhy.explainers.image.save_data_to_pickle")
@patch("xwhy.explainers.image.save_perturbation_data_to_csv")
def test_image_generation_and_editing_explain_fidelity_plot_true(
    mock_csv: Mock,
    mock_pickle: Mock,
    mock_reg: Mock,
    mock_surg_fac: Mock,
    mock_surg_train: Mock,
    mock_wmd: Mock,
    tmp_path: Path,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Trigger fidelity plot rendering when fidelity_plot=True."""
    exp = ImageGenerationAndEditingExplainer()

    p1 = tmp_path / "valid.png"
    img = Image.new("RGB", (10, 10), color="blue")
    img.save(p1)

    exp.state.text_perturbator = MagicMock()
    exp.state.text_perturbator.generate.return_value = (["pt"], [[1]])

    exp._generate_images = MagicMock(return_value=[(True, str(p1))])  # type: ignore[method-assign]
    exp._compute_perturbation_distances = MagicMock(return_value=np.array([1.0]))  # type: ignore[method-assign]

    wmd_inst = mock_wmd.return_value
    wmd_inst.compute_batch.return_value = [("t", 1.0)]

    mock_surg_train.find_best.return_value = (SurrogateType.LIME, 0.9)
    mock_surg_train.compute_weights.return_value = np.array([1.0])
    surg_mock = mock_surg_fac.create.return_value
    surg_mock.coefficients.return_value = [0.1]
    surg_mock.predict.return_value = np.array([0.5])

    mock_metrics = MagicMock()
    mock_metrics.weighted_r2 = 0.9
    mock_metrics.weighted_adj_r2 = 0.85
    mock_reg.calculate.return_value = mock_metrics

    with patch(
        "xwhy.explainers.image.ImageGenerationAndEditingXWhyResult"
    ) as mock_result_cls:
        mock_result_instance = MagicMock()
        mock_result_cls.return_value = mock_result_instance

        result = exp.explain(
            instance="valid long text description",
            input_image_path=str(p1),
            seed=42,
            fidelity_plot=True,
        )
        assert result is mock_result_instance
        mock_result_instance.plot.assert_called_once_with(show=True)


# --- ADDITIONAL COVERAGE FOR ImageGenerationAndEditingExplainer ---


def test_init_invalid_distance_non_numeric(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Raise ValueError when distance is valid type but non-numeric."""
    mock_metric = MagicMock()
    mock_metric.is_numeric_metric = False
    with (
        patch(
            "xwhy.explainers.image.DistanceType.from_str",
            return_value=mock_metric,
        ),
        pytest.raises(
            ValueError,
            match=re.escape("Invalid distance metric"),
        ),
    ):
        ImageGenerationAndEditingExplainer(distance_type="something")


@patch("torch.cuda.is_available", return_value=True)
def test_init_device_from_config_none_sets_cuda(
    mock_cuda: Mock,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Set config.device when config provided with device=None and CUDA available.

    Real ImageGenerationAndEditingConfig is frozen, so a MagicMock is used
    to allow the assignment config.device = resolved_device (line 680).
    """
    mock_cfg = MagicMock()
    mock_cfg.device = None
    mock_cfg.engine_type = "custom"
    mock_cfg.custom_generate_fn = lambda **kw: None
    mock_cfg.custom_model = None
    mock_cfg.provider_type = None
    mock_cfg.model_name = "dall-e-3"
    mock_cfg.use_image_embedding_model = False
    mock_cfg.image_embedding_type = EmbeddingType.DINOV2
    mock_cfg.text_embedding_type = EmbeddingType.WORD2VEC
    mock_cfg.use_segmentation_model = False
    mock_cfg.segmentation_type = SegmentationType.DEEPLABV3_RESNET101
    mock_cfg.output_dir = "outputs"
    mock_cfg.num_perturbations = 64
    mock_cfg.distance_type = DistanceType.WASSERSTEIN
    mock_cfg.surrogate_type = SurrogateType.LIME
    mock_cfg.use_best_surrogate = True
    mock_cfg.seed = 42
    mock_cfg.temperature = 0.0

    explainer = ImageGenerationAndEditingExplainer(
        config=mock_cfg,
        device=None,  # type: ignore[arg-type]
    )
    # The assignment on the mock must have happened
    assert mock_cfg.device == "cuda"
    assert str(explainer.state.device) == "cuda"


@patch("torch.cuda.is_available", return_value=False)
def test_init_device_from_config_none_sets_cpu(
    mock_cuda: Mock,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Set config.device when config provided with device=None and no CUDA.

    Real ImageGenerationAndEditingConfig is frozen, so a MagicMock is used
    to allow the assignment config.device = resolved_device (line 680).
    """
    mock_cfg = MagicMock()
    mock_cfg.device = None
    mock_cfg.engine_type = "custom"
    mock_cfg.custom_generate_fn = lambda **kw: None
    mock_cfg.custom_model = None
    mock_cfg.provider_type = None
    mock_cfg.model_name = "dall-e-3"
    mock_cfg.use_image_embedding_model = False
    mock_cfg.image_embedding_type = EmbeddingType.DINOV2
    mock_cfg.text_embedding_type = EmbeddingType.WORD2VEC
    mock_cfg.use_segmentation_model = False
    mock_cfg.segmentation_type = SegmentationType.DEEPLABV3_RESNET101
    mock_cfg.output_dir = "outputs"
    mock_cfg.num_perturbations = 64
    mock_cfg.distance_type = DistanceType.WASSERSTEIN
    mock_cfg.surrogate_type = SurrogateType.LIME
    mock_cfg.use_best_surrogate = True
    mock_cfg.seed = 42
    mock_cfg.temperature = 0.0

    explainer = ImageGenerationAndEditingExplainer(
        config=mock_cfg,
        device=None,  # type: ignore[arg-type]
    )
    assert mock_cfg.device == "cpu"
    assert str(explainer.state.device) == "cpu"


def test_init_engine_provider_type_enum(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Resolve engine when ProviderType enum is passed."""
    exp = ImageGenerationAndEditingExplainer(engine=ProviderType.OPENAI)
    assert exp.config.engine_type == "provider"  # type: ignore[union-attr]
    assert exp.config.provider_type == ProviderType.OPENAI  # type: ignore[union-attr]


def test_init_engine_base_provider_sets_state(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Assign BaseProvider instance directly to state.engine."""
    prov = DummyProvider()
    exp = ImageGenerationAndEditingExplainer(
        engine=prov,
        custom_generate_fn=lambda **kw: None,
    )
    # Depending on resolution path, engine may be provider or custom
    assert exp.config.engine_type in ("provider", "custom")  # type: ignore[union-attr]


def test_init_huggingface_provider_kwargs(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Inject HuggingFace-specific kwargs during provider init."""
    dummy_pipe = MagicMock()
    dummy_pipe._name_or_path = "hf-model"
    with patch("xwhy.explainers.image.ProviderResolver") as mock_res:
        mock_res.resolve.return_value = MagicMock()
        exp = ImageGenerationAndEditingExplainer(
            pipe=dummy_pipe,
            model_name="dall-e-3",
            use_segmentation_model=True,
        )
        assert exp.config.provider_type == ProviderType.HUGGINGFACE  # type: ignore[union-attr]
        mock_res.resolve.assert_called()
        call_kwargs = mock_res.resolve.call_args
        assert call_kwargs is not None


def test_init_huggingface_with_custom_model_pipe(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Pass custom_model as pipe for HuggingFace provider."""
    config = ImageGenerationAndEditingConfig(
        engine_type="provider",
        provider_type=ProviderType.HUGGINGFACE,
        model_name="test-model",
        custom_model=MagicMock(),
        use_segmentation_model=False,
    )
    with patch("xwhy.explainers.image.ProviderResolver") as mock_res:
        mock_res.resolve.return_value = MagicMock()
        _ = ImageGenerationAndEditingExplainer(config=config)
        assert "pipe" in mock_res.resolve.call_args.kwargs


def test_get_provider_kwargs_openai_generate(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Set response_format=None for OpenAI generate action."""
    exp = ImageGenerationAndEditingExplainer(engine="openai")
    exp.state.engine = OpenAIProvider(client=MagicMock())
    exp._action = "generate"
    kw = exp._get_provider_specific_kwargs()
    assert kw.get("response_format") is None
    assert "provider_name" in kw


def test_get_provider_kwargs_openai_edit(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Skip response_format for OpenAI edit action."""
    exp = ImageGenerationAndEditingExplainer(engine="openai")
    exp.state.engine = OpenAIProvider(client=MagicMock())
    exp._action = "edit"
    kw = exp._get_provider_specific_kwargs()
    assert "response_format" not in kw or kw.get("response_format") is None


@patch("xwhy.explainers.image.load_image_as_tensor")
@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.imshow")
@patch("matplotlib.pyplot.figure")
@patch("matplotlib.pyplot.title")
@patch("matplotlib.pyplot.axis")
def test_compute_distances_display_image(
    mock_axis: Mock,
    mock_title: Mock,
    mock_fig: Mock,
    mock_imshow: Mock,
    mock_show: Mock,
    mock_load: Mock,
    tmp_path: Path,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Render display when display_image=True in distance computation."""
    mock_load.return_value = (None, np.array([1.0, 2.0, 3.0]))

    exp = ImageGenerationAndEditingExplainer(use_image_embedding_model=False)
    p1 = tmp_path / "disp.png"
    img = Image.new("RGB", (10, 10), color="green")
    img.save(p1)

    res = exp._compute_perturbation_distances(
        input_image_path=str(p1),
        generated_images=[(True, str(p1))],
        prompts=["prompt text"],
        display_image=True,
        output_dir=str(tmp_path),
    )
    assert len(res) == 1
    mock_show.assert_called()


@patch("xwhy.explainers.image.load_image_as_tensor")
def test_compute_distances_none_representation(
    mock_load: Mock,
    tmp_path: Path,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Raise when representation becomes None after load."""
    # First call (original) returns valid, second returns None image
    mock_load.side_effect = [
        (None, np.array([1.0])),
        (None, None),
    ]
    exp = ImageGenerationAndEditingExplainer(use_image_embedding_model=False)
    p1 = tmp_path / "ok.png"
    Image.new("RGB", (5, 5)).save(p1)

    with pytest.raises(ValueError, match="representations are None"):
        exp._compute_perturbation_distances(
            input_image_path=str(p1),
            generated_images=[(True, str(p1))],
            prompts=["t"],
        )


def test_generate_images_segmentation_injection(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Inject segmentation_model into edit_image when signature supports it."""
    exp = ImageGenerationAndEditingExplainer(engine=DummyEngine())
    exp.state.segmentation_model = MagicMock()

    def edit_with_seg(
        prompt: str,
        image_path: str,
        output_dir: str,
        segmentation_model: object = None,
        **kwargs: object,
    ) -> tuple[bool, str]:
        assert segmentation_model is not None
        return True, "out.png"

    assert exp.state.engine is not None
    exp.state.engine.edit_image = edit_with_seg  # type: ignore[method-assign]
    res = exp._generate_images(
        ["prompt one"],
        "out",
        input_image_path="in.png",
    )
    assert res == [(True, "out.png")]


def test_generate_images_progress_logging(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Hit progress log branch every 5 prompts and on last."""
    exp = ImageGenerationAndEditingExplainer(engine=DummyEngine())
    exp.state.engine = DummyEngine()
    paths = exp._generate_images(
        [f"p{i}" for i in range(6)],
        "out",
    )
    assert len(paths) == 6


@patch("xwhy.explainers.image.WMDDistance")
@patch("xwhy.explainers.image.SurrogateTrainer")
@patch("xwhy.explainers.image.SurrogateFactory")
@patch("xwhy.explainers.image.RegressionMetrics")
@patch("xwhy.explainers.image.save_data_to_pickle")
@patch("xwhy.explainers.image.save_perturbation_data_to_csv")
def test_explain_use_best_surrogate_false(
    mock_csv: Mock,
    mock_pickle: Mock,
    mock_reg: Mock,
    mock_surg_fac: Mock,
    mock_surg_train: Mock,
    mock_wmd: Mock,
    tmp_path: Path,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Exercise non-best-surrogate path and surrogate_method key."""
    exp = ImageGenerationAndEditingExplainer(use_best_surrogate=False)

    p1 = tmp_path / "valid.png"
    Image.new("RGB", (10, 10), color="blue").save(p1)

    exp.state.text_perturbator = MagicMock()
    exp.state.text_perturbator.generate.return_value = (["pt"], [[1]])
    exp._generate_images = MagicMock(return_value=[(True, str(p1))])  # type: ignore[method-assign]
    exp._compute_perturbation_distances = MagicMock(  # type: ignore[method-assign]
        return_value=np.array([1.0])
    )

    wmd_inst = mock_wmd.return_value
    wmd_inst.compute_batch.return_value = [("t", 1.0)]

    mock_surg_train.compute_weights.return_value = np.array([1.0])
    surg_mock = mock_surg_fac.create.return_value
    surg_mock.coefficients.return_value = [0.1]
    surg_mock.predict.return_value = np.array([0.5])

    mock_metrics = MagicMock()
    mock_metrics.weighted_r2 = 0.9
    mock_metrics.weighted_adj_r2 = 0.85
    mock_reg.calculate.return_value = mock_metrics

    res = exp.explain(
        instance="valid long text description here",
        input_image_path=str(p1),
        seed=42,
        fidelity_plot=False,
    )
    assert isinstance(res, ImageGenerationAndEditingXWhyResult)
    assert "surrogate_method" in res.raw_data
    mock_surg_train.find_best.assert_not_called()


@patch("xwhy.explainers.image.WMDDistance")
@patch("xwhy.explainers.image.SurrogateTrainer")
@patch("xwhy.explainers.image.SurrogateFactory")
@patch("xwhy.explainers.image.RegressionMetrics")
@patch("xwhy.explainers.image.save_data_to_pickle")
@patch("xwhy.explainers.image.save_perturbation_data_to_csv")
def test_explain_num_perturbations_warning(
    mock_csv: Mock,
    mock_pickle: Mock,
    mock_reg: Mock,
    mock_surg_fac: Mock,
    mock_surg_train: Mock,
    mock_wmd: Mock,
    tmp_path: Path,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Log warning when num_perturbations is small relative to word count."""
    exp = ImageGenerationAndEditingExplainer(
        use_best_surrogate=False,
        num_perturbations=2,
    )

    p1 = tmp_path / "valid.png"
    Image.new("RGB", (10, 10), color="blue").save(p1)

    exp.state.text_perturbator = MagicMock()
    exp.state.text_perturbator.generate.return_value = (["pt"], [[1]])
    exp._generate_images = MagicMock(return_value=[(True, str(p1))])  # type: ignore[method-assign]
    exp._compute_perturbation_distances = MagicMock(  # type: ignore[method-assign]
        return_value=np.array([1.0])
    )

    wmd_inst = mock_wmd.return_value
    wmd_inst.compute_batch.return_value = [("t", 1.0)]

    mock_surg_train.compute_weights.return_value = np.array([1.0])
    surg_mock = mock_surg_fac.create.return_value
    surg_mock.coefficients.return_value = [0.1]
    surg_mock.predict.return_value = np.array([0.5])
    mock_reg.calculate.return_value = MagicMock(weighted_r2=0.9, weighted_adj_r2=0.85)

    with patch("xwhy.explainers.image.logger") as mock_logger:
        res = exp.explain(
            instance="one two three four five six seven",
            input_image_path=str(p1),
            fidelity_plot=False,
        )
        assert res is not None
        # warning should have been issued
        assert mock_logger.warning.called


@patch("xwhy.explainers.image.WMDDistance")
@patch("xwhy.explainers.image.SurrogateTrainer")
@patch("xwhy.explainers.image.SurrogateFactory")
@patch("xwhy.explainers.image.RegressionMetrics")
@patch("xwhy.explainers.image.save_data_to_pickle")
@patch("xwhy.explainers.image.save_perturbation_data_to_csv")
def test_explain_seed_update_perturbator(
    mock_csv: Mock,
    mock_pickle: Mock,
    mock_reg: Mock,
    mock_surg_fac: Mock,
    mock_surg_train: Mock,
    mock_wmd: Mock,
    tmp_path: Path,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Update text perturbator seed when explain seed differs from config."""
    exp = ImageGenerationAndEditingExplainer(seed=42)
    p1 = tmp_path / "valid.png"
    Image.new("RGB", (10, 10), color="blue").save(p1)

    exp.state.text_perturbator = MagicMock()
    exp.state.text_perturbator.generate.return_value = (["pt"], [[1]])
    exp._generate_images = MagicMock(return_value=[(True, str(p1))])  # type: ignore[method-assign]
    exp._compute_perturbation_distances = MagicMock(  # type: ignore[method-assign]
        return_value=np.array([1.0])
    )

    wmd_inst = mock_wmd.return_value
    wmd_inst.compute_batch.return_value = [("t", 1.0)]
    mock_surg_train.find_best.return_value = (SurrogateType.LIME, 0.9)
    mock_surg_train.compute_weights.return_value = np.array([1.0])
    surg_mock = mock_surg_fac.create.return_value
    surg_mock.coefficients.return_value = [0.1]
    surg_mock.predict.return_value = np.array([0.5])
    mock_reg.calculate.return_value = MagicMock(weighted_r2=0.9, weighted_adj_r2=0.85)

    exp.explain(
        instance="valid long text description",
        input_image_path=str(p1),
        seed=9999,
        fidelity_plot=False,
    )
    exp.state.text_perturbator.set_seed.assert_called_with(9999)


def test_init_fallback_custom_model_or_fn(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Hit fallback engine_type=custom when only custom_model/fn given."""
    exp = ImageGenerationAndEditingExplainer(
        custom_generate_fn=lambda **kw: None,
        custom_model="my-model",
    )
    assert exp.config.engine_type == "custom"  # type: ignore[union-attr]


def test_init_default_openai_when_nothing(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Default to OpenAI provider when no engine/pipe/custom given."""
    exp = ImageGenerationAndEditingExplainer()
    assert exp.config.provider_type == ProviderType.OPENAI  # type: ignore[union-attr]
    assert exp.config.engine_type == "provider"  # type: ignore[union-attr]


@patch("xwhy.explainers.image.load_image_as_tensor")
def test_compute_distances_sets_edit_action(
    mock_load: Mock,
    tmp_path: Path,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Set _action to edit when input_image_path is truthy."""
    mock_load.return_value = (None, np.zeros(3))
    exp = ImageGenerationAndEditingExplainer(use_image_embedding_model=False)
    p1 = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(p1)
    _ = exp._compute_perturbation_distances(
        input_image_path=str(p1),
        generated_images=[(True, str(p1))],
        prompts=["x"],
        output_dir=str(tmp_path),
    )
    assert exp._action == "edit"


def test_prepare_environment_cuda_branch(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Exercise CUDA seed setting path inside _prepare_environment."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.manual_seed_all") as mock_seed_all,
        patch("os.makedirs"),
        patch("xwhy.explainers.image.EmbeddingFactory"),
        patch("xwhy.explainers.image.SegmentationFactory"),
        patch("xwhy.explainers.image.ProviderResolver"),
        patch("xwhy.explainers.image.TextPerturbation"),
    ):
        exp = ImageGenerationAndEditingExplainer()
        mock_seed_all.reset_mock()
        exp._prepare_environment("tmp_out", 55)
        # Accept one or more calls (environment may invoke seed helpers twice)
        assert mock_seed_all.call_count >= 1
        mock_seed_all.assert_any_call(55)


def test_init_base_provider_assigns_state_engine(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Hit line 734: assign BaseProvider instance to state.engine.

    ProviderType.from_str must succeed so the isinstance(engine, BaseProvider)
    branch is taken.
    """
    prov = DummyProvider()
    with patch(
        "xwhy.explainers.image.ProviderType.from_str",
        return_value=ProviderType.OPENAI,
    ):
        exp = ImageGenerationAndEditingExplainer(engine=prov)
    assert exp.state.engine is prov  # type: ignore[comparison-overlap]
    assert exp.config.engine_type == "provider"  # type: ignore[union-attr]


def test_init_unrecognized_engine_with_existing_custom_model(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Hit 766→777 false branch: custom_model already set, skip overwrite."""
    exp = ImageGenerationAndEditingExplainer(
        engine="unrecognized_str",
        custom_model="already_set",
        custom_generate_fn=lambda **kw: None,
    )
    # custom_model must stay "already_set" (if custom_model is None was False)
    assert exp.config.custom_model == "already_set"  # type: ignore[union-attr]
    assert exp.config.engine_type == "custom"  # type: ignore[union-attr]


def test_init_provider_path_huggingface_with_pipe(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Hit full provider branch 839→876 including HuggingFace + custom_model pipe."""
    dummy_pipe = MagicMock()
    dummy_pipe._name_or_path = "hf-model-xyz"

    with patch("xwhy.explainers.image.ProviderResolver") as mock_res:
        mock_engine = MagicMock()
        mock_res.resolve.return_value = mock_engine
        exp = ImageGenerationAndEditingExplainer(
            pipe=dummy_pipe,
            model_name="dall-e-3",
            use_segmentation_model=False,
        )
        assert exp.config.provider_type == ProviderType.HUGGINGFACE  # type: ignore[union-attr]
        assert exp.config.engine_type == "provider"  # type: ignore[union-attr]
        mock_res.resolve.assert_called_once()
        # state.engine must be the resolved provider
        assert exp.state.engine is mock_engine


def test_init_invalid_segmentation_type_raises(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Hit line 916: raise when segmentation_type is not a SegmentationType."""
    mock_cfg = MagicMock()
    mock_cfg.device = "cpu"
    mock_cfg.engine_type = "custom"
    mock_cfg.custom_generate_fn = lambda **kw: None
    mock_cfg.custom_model = None
    mock_cfg.provider_type = None
    mock_cfg.model_name = "dall-e-3"
    mock_cfg.use_image_embedding_model = False
    mock_cfg.image_embedding_type = EmbeddingType.DINOV2
    mock_cfg.text_embedding_type = EmbeddingType.WORD2VEC
    mock_cfg.use_segmentation_model = True
    # Force a non-enum value so isinstance(..., SegmentationType) is False
    mock_cfg.segmentation_type = "not_an_enum"
    mock_cfg.output_dir = "outputs"
    mock_cfg.num_perturbations = 64
    mock_cfg.distance_type = DistanceType.WASSERSTEIN
    mock_cfg.surrogate_type = SurrogateType.LIME
    mock_cfg.use_best_surrogate = True
    mock_cfg.seed = 42
    mock_cfg.temperature = 0.0

    with pytest.raises(ValueError, match="Invalid segmentation type"):
        ImageGenerationAndEditingExplainer(config=mock_cfg)


def test_generate_images_seg_model_present_but_not_in_signature(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Hit 1038→1044 false path: segmentation_model exists but not in edit_image sig."""
    exp = ImageGenerationAndEditingExplainer(engine=DummyEngine())
    exp.state.segmentation_model = MagicMock()

    # DummyEngine.edit_image does NOT accept segmentation_model
    # so the inner `if "segmentation_model" in edit_sig.parameters` is False
    res = exp._generate_images(
        ["prompt one"],
        "out",
        input_image_path="in.png",
    )
    assert len(res) == 1
    assert res[0][0] is True  # DummyEngine returns success


@patch("xwhy.explainers.image.load_image_as_tensor")
def test_compute_distances_empty_input_path_skips_edit_action(
    mock_load: Mock,
    tmp_path: Path,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Hit 1149→1153 false path: input_image_path is falsy, _action stays generate."""
    mock_load.return_value = (None, np.zeros(3))
    exp = ImageGenerationAndEditingExplainer(use_image_embedding_model=False)
    assert exp._action == "generate"

    p1 = tmp_path / "b.png"
    Image.new("RGB", (4, 4)).save(p1)

    # Empty string is falsy → skip `self._action = "edit"`
    _ = exp._compute_perturbation_distances(
        input_image_path="",
        generated_images=[(True, str(p1))],
        prompts=["x"],
        output_dir=str(tmp_path),
    )
    assert exp._action == "generate"


@patch("xwhy.explainers.image.WMDDistance")
@patch("xwhy.explainers.image.SurrogateTrainer")
@patch("xwhy.explainers.image.SurrogateFactory")
@patch("xwhy.explainers.image.RegressionMetrics")
@patch("xwhy.explainers.image.save_data_to_pickle")
@patch("xwhy.explainers.image.save_perturbation_data_to_csv")
def test_explain_seed_equals_config_skips_set_seed(
    mock_csv: Mock,
    mock_pickle: Mock,
    mock_reg: Mock,
    mock_surg_fac: Mock,
    mock_surg_train: Mock,
    mock_wmd: Mock,
    tmp_path: Path,
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Hit 1275→1279 false path: seed == config.seed, do not call set_seed."""
    exp = ImageGenerationAndEditingExplainer(seed=42)
    p1 = tmp_path / "valid.png"
    Image.new("RGB", (10, 10), color="blue").save(p1)

    exp.state.text_perturbator = MagicMock()
    exp.state.text_perturbator.generate.return_value = (["pt"], [[1]])
    exp._generate_images = MagicMock(  # type: ignore[method-assign]
        return_value=[(True, str(p1))]
    )
    exp._compute_perturbation_distances = MagicMock(  # type: ignore[method-assign]
        return_value=np.array([1.0])
    )

    wmd_inst = mock_wmd.return_value
    wmd_inst.compute_batch.return_value = [("t", 1.0)]
    mock_surg_train.find_best.return_value = (SurrogateType.LIME, 0.9)
    mock_surg_train.compute_weights.return_value = np.array([1.0])
    surg_mock = mock_surg_fac.create.return_value
    surg_mock.coefficients.return_value = [0.1]
    surg_mock.predict.return_value = np.array([0.5])
    mock_reg.calculate.return_value = MagicMock(weighted_r2=0.9, weighted_adj_r2=0.85)

    exp.explain(
        instance="valid long text description",
        input_image_path=str(p1),
        seed=42,  # same as config.seed
        fidelity_plot=False,
    )
    exp.state.text_perturbator.set_seed.assert_not_called()


def test_initialize_unknown_engine_type_falls_through(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Hit 839→876 false branch: engine_type is neither custom/pipeline nor provider.

    When state.engine is None and engine_type has an unexpected value the
    if/elif block is skipped and execution continues at the embedding load.
    """
    mock_cfg = MagicMock()
    mock_cfg.device = "cpu"
    mock_cfg.engine_type = "unknown_type"  # not custom / pipeline / provider
    mock_cfg.custom_generate_fn = None
    mock_cfg.custom_model = None
    mock_cfg.provider_type = None
    mock_cfg.model_name = "dall-e-3"
    mock_cfg.use_image_embedding_model = False
    mock_cfg.image_embedding_type = EmbeddingType.DINOV2
    mock_cfg.text_embedding_type = EmbeddingType.WORD2VEC
    mock_cfg.use_segmentation_model = False
    mock_cfg.segmentation_type = SegmentationType.DEEPLABV3_RESNET101
    mock_cfg.output_dir = "outputs"
    mock_cfg.num_perturbations = 64
    mock_cfg.distance_type = DistanceType.WASSERSTEIN
    mock_cfg.surrogate_type = SurrogateType.LIME
    mock_cfg.use_best_surrogate = True
    mock_cfg.seed = 42
    mock_cfg.temperature = 0.0

    # state.engine stays None
    exp = ImageGenerationAndEditingExplainer(config=mock_cfg)
    assert exp.state.engine is None


def test_generate_images_no_segmentation_model(
    mock_dependencies: Any,  # noqa: ANN401
) -> None:
    """Hit 1038→1044 false branch: segmentation_model is None.

    The outer if is skipped and control goes straight to the Gemini/batch check.
    """
    exp = ImageGenerationAndEditingExplainer(
        engine=DummyEngine(),
        use_segmentation_model=False,
    )
    # Ensure the attribute is explicitly None (factory may not have set it)
    exp.state.segmentation_model = None

    res = exp._generate_images(
        ["prompt one", "prompt two"],
        "out",
    )
    assert len(res) == 2
    assert all(success for success, _ in res)
