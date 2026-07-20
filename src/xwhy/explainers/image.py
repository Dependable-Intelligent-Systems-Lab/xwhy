"""Image explainer."""

from __future__ import annotations

from typing import Any

import torch

from xwhy.core.config import ImageClassificationConfig
from xwhy.core.explainer import BaseExplainer
from xwhy.core.pipeline import ExplanationPipeline
from xwhy.core.result import BaseXWhyResult
from xwhy.core.types import ImageClassificationState
from xwhy.distance.types import DistanceType
from xwhy.logger import logger
from xwhy.models.classification.factory import ClassificationFactory
from xwhy.models.classification.types import ClassificationType
from xwhy.models.embeddings.factory import EmbeddingFactory
from xwhy.models.embeddings.types import EmbeddingType
from xwhy.models.segmentation.factory import SegmentationFactory
from xwhy.models.segmentation.types import SegmentationType
from xwhy.surrogate.types import SurrogateType


class ImageClassificationExplainer(
    ExplanationPipeline,
    BaseExplainer,
):
    """Explainer for image classification models.

    This explainer loads all required runtime resources only once and can
    explain multiple images throughout its lifetime.
    """

    def __init__(
        self,
        config: ImageClassificationConfig | None = None,
        classification_type: str | ClassificationType = ClassificationType.INCEPTION_V3,
        use_model_preprocess: bool = False,
        need_normalization: bool = False,
        use_embedding_model: bool = False,
        embedding_type: str | EmbeddingType = EmbeddingType.DINOV2,
        use_segmentation_model: bool = True,
        segmentation_type: str
        | SegmentationType = SegmentationType.DEEPLABV3_RESNET101,
        device: str = "cpu",
        seed: int = 222,
        kernel_size: int = 4,
        max_dist: int = 200,
        ratio: float = 0.2,
        num_perturb: int = 150,
        distance_metric: str | DistanceType = DistanceType.WASSERSTEIN,
        surrogate_type: str | SurrogateType = SurrogateType.LIME,
        use_best_surrogate: bool = True,
        num_top_features: int = 4,
        num_top_predictions: int = 5,
    ) -> None:
        """Initialize the Image Classification explainer.

        Args:
            config:
                Optional configuration for the explainer.
            classification_type:
                Type of the classification model to explain.
            use_model_preprocess:
                Whether to use the classification model's official
                preprocessing pipeline.
            need_normalization:
                Whether image denormalization is required before
                visualization.
            use_embedding_model:
                Whether an image embedding model should be used.
            embedding_type:
                Embedding method for Image Embedding.
            use_segmentation_model:
                Whether an image segmentation model should be used.
            segmentation_type:
                Segmentation method for extracting object masks.
            device:
                Device type name.
            seed:
                Random seed used throughout the explanation pipeline.
            kernel_size:
                Kernel size used during superpixel generation.
            max_dist:
                Maximum superpixel search distance.
            ratio:
                Sampling ratio used by the superpixel algorithm.
            num_perturb:
                Number of perturbed samples.
            distance_metric:
                Distance metric name.
            surrogate_type:
                Surrogate model name.
            use_best_surrogate:
                Find best surrogate model.
            num_top_features:
                Number of important regions to highlight.
            num_top_predictions:
                Number of predictions to explain.

        """
        distance_metric = DistanceType.from_str(distance_metric)

        if not distance_metric.is_numeric_metric:
            raise ValueError(
                f"Invalid distance metric '{distance_metric}' "
                "for ImageClassificationExplainer. Must be a numeric distance."
            )

        classification_type = ClassificationType.from_str(classification_type)
        embedding_type = EmbeddingType.from_str(embedding_type)
        segmentation_type = SegmentationType.from_str(segmentation_type)
        surrogate_type = SurrogateType.from_str(surrogate_type)

        if config is None:
            config = ImageClassificationConfig(
                classification_type=classification_type,
                use_model_preprocess=use_model_preprocess,
                need_normalization=need_normalization,
                use_embedding_model=use_embedding_model,
                embedding_type=embedding_type,
                use_segmentation_model=use_segmentation_model,
                segmentation_type=segmentation_type,
                device=device,
                seed=seed,
                kernel_size=kernel_size,
                max_dist=max_dist,
                ratio=ratio,
                num_perturb=num_perturb,
                distance_metric=distance_metric,
                surrogate_type=surrogate_type,
                use_best_surrogate=use_best_surrogate,
                num_top_features=num_top_features,
                num_top_predictions=num_top_predictions,
            )

        if config.device is None:
            config.device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(config)

        self.state = ImageClassificationState(
            device_=torch.device(
                config.device,
            ),
        )

        self._initialize()

    def _initialize(self) -> None:
        """Initialize runtime resources."""
        # 1. Load Classification Model
        logger.info(f"Loading classification model: {self.config.classification_type}")  # type: ignore

        if not isinstance(self.config.classification_type, ClassificationType):  # type: ignore
            raise ValueError(
                f"Invalid classification type '{self.config.classification_type}'."  # type: ignore
            )

        self.state.classification_model = ClassificationFactory.create(
            classification=self.config.classification_type,  # type: ignore
            device=self.state.device,
        )
        self.state.classification_model.load()

        # 2. Load Embedding Model (if enabled)
        if self.config.use_embedding_model:  # type: ignore
            if not self.config.embedding_type.is_image_embedding:  # type: ignore
                raise ValueError(
                    f"Invalid embedding type '{self.config.embedding_type}' "  # type: ignore
                    "for ImageClassificationExplainer. Must be an image embedding."
                )

            logger.info(f"Loading embedding model: {self.config.embedding_type}")  # type: ignore
            self.state.embedding_model = EmbeddingFactory.create(
                embedding=self.config.embedding_type,  # type: ignore
                device=self.state.device,
            )
            self.state.embedding_model.load()

        # 3. Load Segmentation Model (if enabled)
        if self.config.use_segmentation_model:  # type: ignore
            if not isinstance(self.config.segmentation_type, SegmentationType):  # type: ignore
                raise ValueError(
                    f"Invalid segmentation type '{self.config.segmentation_type}'."  # type: ignore
                )

            logger.info(f"Loading segmentation model: {self.config.segmentation_type}")  # type: ignore
            self.state.segmentation_model = SegmentationFactory.create(
                segmentation=self.config.segmentation_type,  # type: ignore
                device=self.state.device,
            )
            self.state.segmentation_model.load()

    def run(self, instance: Any, **kwargs: Any) -> BaseXWhyResult:  # noqa: ANN401
        """Run the full explanation pipeline (ExplanationPipeline implementation).

        Args:
            instance: The input prompt string.
            **kwargs: Additional pipeline options.

        Returns:
            BaseXWhyResult: The explanation outcome.

        Raises:
            TypeError: If the instance is not a string.

        """
        if not isinstance(instance, str):
            raise TypeError("ImageClassification requires a string instance.")
        return self.explain(instance, **kwargs)

    def explain(
        self,
        instance: str,
        **kwargs: object,
    ) -> BaseXWhyResult:
        """Generate an explanation for an input image.

        Args:
            instance:
                Path to the image that should be explained.
            **kwargs:
                Additional explainer-specific options.

        Returns:
            Structured explanation result.

        Raises:
            TypeError:
                If ``instance`` is not a string.
            NotImplementedError:
                Until the implementation is completed.

        """
        if not isinstance(instance, str):
            raise TypeError(
                "ImageClassificationExplainer requires the image path as a string.",
            )

        image_path = instance  # noqa: F841

        raise NotImplementedError(
            "ImageClassificationExplainer.explain() "
            "will be implemented in a later phase.",
        )
