"""Tests for image classification explainer."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from pydantic import ValidationError

from xwhy.distance.types import DistanceType
from xwhy.explainers.image import ImageClassificationExplainer
from xwhy.models.classification.types import ClassificationType
from xwhy.models.embeddings.types import EmbeddingType
from xwhy.models.segmentation.types import SegmentationType
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
    mock_cfg.seed = 222

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
    mock_cfg.seed = 222

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
    mock_cfg.seed = 222

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
    mock_cfg.seed = 222

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
    mock_cfg.seed = 222

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
    explainer.config.seed = 222

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
