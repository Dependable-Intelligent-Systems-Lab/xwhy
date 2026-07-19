"""Image explainer."""

from __future__ import annotations

from typing import Any

import torch

from xwhy.core.config import ImageClassificationConfig
from xwhy.core.explainer import BaseExplainer
from xwhy.core.pipeline import ExplanationPipeline
from xwhy.core.result import BaseXWhyResult
from xwhy.core.types import ImageClassificationState
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
        use_model_preprocess: bool = False,
        need_normalization: bool = False,
        use_embedding_model: bool = False,
        seed: int = 222,
        kernel_size: int = 4,
        max_dist: int = 200,
        ratio: float = 0.2,
        num_perturb: int = 150,
        distance_metric: str = "wasserstein",
        surrogate_type: str | SurrogateType = SurrogateType.LIME,
        use_best_surrogate: bool = True,
        num_top_features: int = 4,
        num_top_predictions: int = 5,
    ) -> None:
        """Initialize the Image Classification explainer.

        Args:
            config:
                Optional configuration for the explainer.
            use_model_preprocess:
                Whether to use the classification model's official
                preprocessing pipeline.
            need_normalization:
                Whether image denormalization is required before
                visualization.
            use_embedding_model:
                Whether an image embedding model should be used.
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
        if config is None:
            surrogate_type = SurrogateType.from_str(surrogate_type)

            config = ImageClassificationConfig(
                use_model_preprocess=use_model_preprocess,
                need_normalization=need_normalization,
                use_embedding_model=use_embedding_model,
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

        super().__init__(config)

        self.state = ImageClassificationState(
            device_=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu",
            ),
        )

        self._initialize()

    def _initialize(self) -> None:
        """Initialize runtime resources."""
        # self._load_classification_model()

        # if self.config.use_embedding_model:
        #     self._load_embedding_model()

        # self._load_segmentation_model()
        pass

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
