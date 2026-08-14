"""Image explainer."""

from __future__ import annotations

import inspect
import os
import random
import time
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from xwhy.core.config import ImageClassificationConfig, ImageGenerationAndEditingConfig
from xwhy.core.explainer import BaseExplainer
from xwhy.core.pipeline import ExplanationPipeline
from xwhy.core.result import (
    ImageClassificationXWhyResult,
    ImageGenerationAndEditingXWhyResult,
)
from xwhy.core.types import (
    BaseImageGenerationAndEditing,
    ImageClassificationState,
    ImageGenerationAndEditingState,
)
from xwhy.distance.calculator import calculate_distance
from xwhy.distance.normalization import DistanceNormalizer
from xwhy.distance.types import DistanceType
from xwhy.distance.wmd import WMDDistance
from xwhy.logger import logger
from xwhy.metrics.image import ImageCoverageMetrics
from xwhy.metrics.regression import RegressionMetrics
from xwhy.models.classification.custom_torch import CustomTorchClassification
from xwhy.models.classification.factory import ClassificationFactory
from xwhy.models.classification.types import ClassificationType
from xwhy.models.embeddings.factory import EmbeddingFactory
from xwhy.models.embeddings.types import EmbeddingType
from xwhy.models.image_generation_and_editing.custom import (
    CustomImageGenerationAndEditingModel,
)
from xwhy.models.segmentation.factory import SegmentationFactory
from xwhy.models.segmentation.types import SegmentationType
from xwhy.perturbation.image import ImagePerturbation
from xwhy.perturbation.text import TextPerturbation
from xwhy.providers.base import BaseProvider
from xwhy.providers.openai import OpenAIProvider
from xwhy.providers.resolver import ProviderResolver
from xwhy.providers.types import ProviderType
from xwhy.surrogate.factory import SurrogateFactory
from xwhy.surrogate.trainer import SurrogateTrainer
from xwhy.surrogate.types import SurrogateType
from xwhy.utils.image import (
    get_segmentation_mask,
    load_image_as_tensor,
    numpy_image_to_tensor,
    tensor_to_numpy_image,
)
from xwhy.utils.io import save_data_to_pickle, save_perturbation_data_to_csv


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
        custom_model: Any = None,  # noqa: ANN401
        custom_preprocess: Any = None,  # noqa: ANN401
        categories: Any = None,  # noqa: ANN401
        classification_type: str | ClassificationType = ClassificationType.INCEPTION_V3,
        use_model_preprocess: bool = True,
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
        distance_type: str | DistanceType = DistanceType.WASSERSTEIN,
        surrogate_type: str | SurrogateType = SurrogateType.LIME,
        use_best_surrogate: bool = True,
        num_top_features: int = 4,
        num_top_predictions: int = 5,
    ) -> None:
        """Initialize the Image Classification explainer.

        Args:
            config: Optional configuration for the explainer.
            custom_model: Optional user-defined PyTorch classification
                          model (nn.Module).
            custom_preprocess: Optional preprocessing transform pipeline for
                               the custom model.
            categories: Optional list of human-readable class names corresponding
                        to model outputs.
            classification_type: Type of the classification model to explain.
            use_model_preprocess: Whether to use the classfication model's official
                                  preprocessing.
            use_embedding_model: Whether an image embedding model should be used.
            embedding_type: Embedding method for Image Embedding.
            use_segmentation_model: Whether an image segmentation model should be used.
            segmentation_type: Segmentation method for extracting object masks.
            device: Device type name.
            seed: Random seed used throughout the explanation pipeline.
            kernel_size: Kernel size used during superpixel generation.
            max_dist: Maximum superpixel search distance.
            ratio: Sampling ratio used by the superpixel algorithm.
            num_perturb: Number of perturbed samples.
            distance_type: Distance metric name.
            surrogate_type: Surrogate model name.
            use_best_surrogate: Find best surrogate model dynamically.
            num_top_features: Number of important regions to highlight.
            num_top_predictions: Number of predictions to explain.

        """
        distance_type = DistanceType.from_str(distance_type)

        if not distance_type.is_numeric_metric:
            raise ValueError(
                f"Invalid distance metric '{distance_type}' "
                "for ImageClassificationExplainer. Must be a numeric distance."
            )

        classification_type = ClassificationType.from_str(classification_type)
        embedding_type = EmbeddingType.from_str(embedding_type)
        segmentation_type = SegmentationType.from_str(segmentation_type)
        surrogate_type = SurrogateType.from_str(surrogate_type)

        if config is None:
            config = ImageClassificationConfig(
                custom_model=custom_model,
                custom_preprocess=custom_preprocess,
                categories=categories,
                classification_type=classification_type,
                use_model_preprocess=use_model_preprocess,
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
                distance_type=distance_type,
                surrogate_type=surrogate_type,
                use_best_surrogate=use_best_surrogate,
                num_top_features=num_top_features,
                num_top_predictions=num_top_predictions,
            )

        if config.device is None:
            config.device = "cuda" if torch.cuda.is_available() else "cpu"

        if (
            getattr(config, "use_best_surrogate", True)
            or not config.surrogate_type.is_linear_model  # type: ignore[attr-defined]
        ):
            logger.warning(
                "Using a non-linear surrogate model or enabling 'use_best_surrogate' "
                "can replace a black-box model with another complex model, "
                "sacrificing local interpretability. The scientific community highly "
                "recommends utilizing simple linear models (e.g., LIME, OLS) to "
                "guarantee transparent and additive feature attributions."
            )

        super().__init__(config)

        self.state = ImageClassificationState(
            device_=torch.device(config.device),
        )

        self._initialize()

    def _initialize(self) -> None:
        """Initialize runtime resources."""
        # 1. Load Classification Model
        if getattr(self.config, "custom_model", None) is not None:
            logger.info("Loading custom classification model...")

            if not isinstance(self.config.custom_model, nn.Module):  # type: ignore
                raise TypeError(
                    "The provided 'custom_model' must be an instance "
                    "of torch.nn.Module."
                )

            self.state.classification_model = CustomTorchClassification(
                model=self.config.custom_model,  # type: ignore
                preprocess_fn=getattr(self.config, "custom_preprocess", None),
                categories=getattr(self.config, "categories", None),
                device=self.state.device,
            )
            self.state.classification_model.load()

            self.state.transform_fn = self.state.classification_model.preprocess_fn

        else:
            logger.info(
                f"Loading classification model: {self.config.classification_type}"  # type: ignore[union-attr]
            )

            if not isinstance(self.config.classification_type, ClassificationType):  # type: ignore
                raise ValueError(
                    f"Invalid classification type '{self.config.classification_type}'."  # type: ignore
                )

            self.state.classification_model = ClassificationFactory.create(
                classification=self.config.classification_type,  # type: ignore
                device=self.state.device,
            )
            self.state.classification_model.load()

            # Extract the transform function directly from the adapter
            if self.config.use_model_preprocess:  # type: ignore[union-attr]
                self.state.transform_fn = self.state.classification_model.preprocess_fn

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

        # 4. Initialize Perturbator ONCE
        self.state.perturbator = ImagePerturbation(
            kernel_size=self.config.kernel_size,  # type: ignore[union-attr]
            max_dist=self.config.max_dist,  # type: ignore[union-attr]
            ratio=self.config.ratio,  # type: ignore[union-attr]
            seed=self.config.seed,  # type: ignore[union-attr]
        )

    def _run_perturbation_loop(
        self,
        original_image: np.ndarray,
        superpixels: np.ndarray,
        perturbation_masks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Iterate perturbations, compute model predictions, and calculate distances.

        Args:
            original_image (np.ndarray): Original image of shape (H, W, C).
            superpixels (np.ndarray): Superpixel segmentation mask.
            perturbation_masks (np.ndarray): Binary masks indicating active
                superpixels for each perturbation.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - predictions: Model output probabilities for each perturbation.
                - distances: Calculated distances for each perturbation.

        """
        batch_predictions = []
        distances = []

        device = self.state.device
        use_embedding = self.config.use_embedding_model  # type: ignore[union-attr]
        dist_type = self.config.distance_type  # type: ignore[union-attr]

        # 1. Pre-calculate original representation (Optimization: do this once)
        base_representation = original_image
        if use_embedding:
            original_embedding = self.state.embedding_model.encode_image(original_image)  # type: ignore[union-attr]
            if original_embedding is None:
                raise ValueError("Original embedding extraction failed.")
            base_representation = np.asarray(original_embedding)

        classifier_model = self.state.classification_model.model  # type: ignore[union-attr]
        transform = self.state.transform_fn

        logger.info(f"Running inference on {len(perturbation_masks)} perturbations...")

        for mask in tqdm(perturbation_masks, desc="Generating Neighborhood"):
            # A. Apply Perturbation
            perturbed_img = self.state.perturbator.apply_mask(  # type: ignore[union-attr]
                item=original_image, mask=mask, segments=superpixels
            )

            # B. Preprocess for Model
            tensor_batch = numpy_image_to_tensor(
                np_array=perturbed_img, transform_fn=transform
            ).to(device)

            # C. Inference
            with torch.no_grad():
                prediction = classifier_model(tensor_batch)

            batch_predictions.append(prediction.detach().cpu().numpy())

            # D. Calculate Distance
            current_representation = perturbed_img
            if use_embedding:
                perturbed_embedding = self.state.embedding_model.encode_image(  # type: ignore[union-attr]
                    perturbed_img
                )
                current_representation = np.asarray(perturbed_embedding)

            dist = calculate_distance(
                metric=dist_type,
                source=base_representation,
                target=current_representation,
            )
            distances.append(dist)

        final_predictions = np.concatenate(batch_predictions, axis=0)

        return final_predictions, np.array(distances)

    def run(self, instance: Any, **kwargs: Any) -> ImageClassificationXWhyResult:  # noqa: ANN401
        """Run the full explanation pipeline.

        Args:
            instance: The input image path.
            **kwargs: Additional pipeline options.

        Returns:
            ImageClassificationXWhyResult: The explanation outcome.

        Raises:
            TypeError: If the instance is not a string.

        """
        if not isinstance(instance, str):
            raise TypeError("ImageClassification requires a string instance.")
        return self.explain(instance, **kwargs)

    def explain(
        self,
        instance: str,
        fidelity_plot: bool = False,
        ground_truth_mask: Any = None,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> ImageClassificationXWhyResult:
        """Generate an explanation for an input image.

        Args:
            instance: Path to the image that should be explained.
            fidelity_plot: Rendering fidelity scatter plot.
            ground_truth_mask: Provided ground-truth mask for evaluation.
            **kwargs: Additional explainer-specific options.

        Returns:
            ImageClassificationXWhyResult: Structured explanation result.

        Raises:
            TypeError: If `instance` is not a string (image path).

        """
        if not isinstance(instance, str):
            raise TypeError(
                "ImageClassificationExplainer requires the image path as a string.",
            )

        image_path = instance
        transform_fn = self.state.transform_fn
        mean = self.state.classification_model.preprocess_fn.mean  # type: ignore[union-attr]
        std = self.state.classification_model.preprocess_fn.std  # type: ignore[union-attr]

        denormalize = bool(self.config.use_model_preprocess)  # type: ignore[union-attr]

        # Load Data
        input_batch, original_img = load_image_as_tensor(
            image_path=image_path, transform_fn=transform_fn
        )

        # Get Model Prediction (to find top class)
        input_batch = input_batch.to(self.config.device)  # type: ignore[union-attr]

        if self.state.classification_model is None:
            raise RuntimeError("Classification model is not initialized/loaded.")

        with torch.no_grad():
            output = self.state.classification_model.predict(inputs=input_batch)
            probs = torch.nn.functional.softmax(output[0], dim=0)

        num_top = self.config.num_top_predictions  # type: ignore[union-attr]
        top_preds = probs.topk(k=num_top)
        class_to_explain = top_preds.indices[0].item()

        # Log top predictions
        categories = self.state.classification_model.weights.meta["categories"]
        for prob, idx in zip(top_preds.values, top_preds.indices, strict=False):
            logger.info(f"Prediction: {categories[idx]}: {prob.item():.4f}")

        logger.info(f"Explaining Class: {categories[class_to_explain]}")

        # Prepare base image for perturbation
        base_image_numpy = tensor_to_numpy_image(
            tensor_batch=input_batch,
            denormalize=denormalize,
            mean=mean,
            std=std,
        )

        # Generate Superpixels & Perturbations
        superpixels, num_superpixels = self.state.perturbator.generate_superpixels(  # type: ignore[union-attr]
            image=base_image_numpy
        )
        x_matrix = self.state.perturbator.generate(  # type: ignore[union-attr]
            num_superpixels=num_superpixels,
            num_perturbations=self.config.num_perturb,  # type: ignore[union-attr]
        )

        # Run Main SMILE Loop (Inference & Distance)
        predictions, distances = self._run_perturbation_loop(
            original_image=base_image_numpy,
            superpixels=superpixels,
            perturbation_masks=x_matrix,
        )

        # Compute Explanation (Surrogate Model Training)
        logger.info("Extracting target probabilities for the predicted class...")
        y_target = predictions[:, int(class_to_explain)]

        if self.config.use_best_surrogate:  # type: ignore[union-attr]
            logger.info("Searching for the optimal surrogate model...")
            method, score = SurrogateTrainer.find_best(
                x=x_matrix,
                y=y_target,
                distances=distances,
                seed=self.config.seed,  # type: ignore[union-attr]
                normalize_distances=True,
            )
            logger.info(
                "Optimization complete. Selected surrogate model:"
                " '%s' (Best Score: %.4f)",
                method.value,
                score,
            )
        else:
            method = self.config.surrogate_type  # type: ignore[union-attr]
            logger.info("Skipping surrogate search. Using default: '%s'", method.value)

        weights = SurrogateTrainer.compute_weights(
            method=method,
            distances=distances,
            normalize_distances=True,
        )

        logger.info(f"Training surrogate model ({method.value})...")
        surrogate = SurrogateFactory.create(
            method=method,
            seed=self.config.seed,  # type: ignore[union-attr]
        )
        surrogate.fit(x_matrix, y_target, weights)

        coeffs = surrogate.coefficients()
        y_pred = surrogate.predict(x_matrix)

        logger.info("Computing regression metrics...")
        metrics = RegressionMetrics.calculate(
            y_true=y_target,
            y_pred=y_pred,
            weights=weights,
            num_features=len(coeffs),
        )

        # Visualization: Top Features Only
        num_features = self.config.num_top_features  # type: ignore[union-attr]
        top_features_indices = np.argsort(coeffs)[-num_features:]

        top_features_mask = np.zeros(num_superpixels)
        top_features_mask[top_features_indices] = True

        explanation_image = self.state.perturbator.apply_mask(  # type: ignore[union-attr]
            item=base_image_numpy, mask=top_features_mask, segments=superpixels
        )

        sem_mask = None
        cov = 0.0
        w_cov = 0.0

        if ground_truth_mask is not None:
            logger.info("Using provided ground-truth mask for evaluation.")
            sem_mask = ground_truth_mask

        # Evaluation against Ground Truth
        if self.state.segmentation_model is not None:
            try:
                _, sem_mask = get_segmentation_mask(
                    image_path=image_path,
                    segmentation_model=self.state.segmentation_model,
                    transform_fn=transform_fn,
                    device=self.config.device,  # type: ignore[union-attr]
                    class_names=self.state.segmentation_model.class_names,
                )
            except Exception as e:
                logger.warning(f"Failed to generate ground truth mask: {e}")
        else:
            logger.info(
                "Segmentation model is disabled. Skipping ground-truth "
                "evaluation metrics."
            )

        if sem_mask is not None:
            # Resize explanation image to match original mask spatial
            # dimensions if needed
            if explanation_image.shape[:2] != sem_mask.shape[:2]:
                # Preserve channels if explanation_image is 3D (H, W, C)
                target_shape = (*sem_mask.shape[:2], *explanation_image.shape[2:])

                resized = skimage.transform.resize(
                    explanation_image,
                    target_shape,
                    order=0,  # Nearest-neighbor interpolation preserves discrete values
                    preserve_range=True,
                    anti_aliasing=False,
                )  # type: ignore[no-untyped-call]

                explanation_image = cast(np.ndarray, resized).astype(
                    explanation_image.dtype
                )

            cov, w_cov = ImageCoverageMetrics.evaluate_all(
                explanation_image=explanation_image,
                semantic_mask=sem_mask,
            )

            logger.info("--- Evaluation Metrics ---")
            logger.info(f"Coverage with True Label: {cov:.4f}")
            logger.info(f"Weighted coverage with True Label: {w_cov:.4f}")
            logger.info("--------------------------")

        # Compile final Result Object
        raw_data = {
            "predictions": predictions,
            "distances": distances,
            "weights": weights,
            "y_target": y_target,
            "y_pred": y_pred,
            "x_matrix": x_matrix,
        }

        if self.config.use_best_surrogate:  # type: ignore[union-attr]
            raw_data["best_surrogate_method"] = method
        else:
            raw_data["surrogate_method"] = method

        if isinstance(original_img, Image.Image):
            base_image_numpy = np.array(original_img)
        else:
            base_image_numpy = np.array(original_img)

        result = ImageClassificationXWhyResult(
            coefficients=coeffs,
            metrics=metrics,
            raw_data=raw_data,
            original_image=base_image_numpy,
            superpixels=superpixels,
            top_features=top_features_indices,
            coverage=cov,
            weighted_coverage=w_cov,
        )

        if fidelity_plot:
            logger.info("Rendering fidelity plot as requested...")
            result.plot(show=True)

        return result


class ImageGenerationAndEditingExplainer(BaseExplainer):
    """Explainer for image generation and editing tasks.

    This class manages the lifecycle of generating text perturbations, executing
    image generation or editing models, calculating distance metrics between base
    and perturbed images, and training a surrogate model to extract feature
    importances (explanations) for the generation process.
    """

    def __init__(
        self,
        config: ImageGenerationAndEditingConfig | None = None,
        # Base Provider & Model Settings
        engine: (
            str
            | ProviderType
            | BaseProvider
            | BaseImageGenerationAndEditing
            | type[BaseImageGenerationAndEditing]
            | None
        ) = None,
        model_name: str = "dall-e-3",
        pipe: Any | None = None,  # noqa: ANN401
        # Custom Model Injection
        custom_model: Any = None,  # noqa: ANN401
        custom_generate_fn: Callable[..., Any] | None = None,
        # Core Shared Generation Parameters
        temperature: float = 0.0,
        seed: int = 1024,
        # Explainer Components
        use_image_embedding_model: bool = False,
        image_embedding_type: EmbeddingType | str = EmbeddingType.DINOV2,
        text_embedding_type: EmbeddingType | str = EmbeddingType.WORD2VEC,
        use_segmentation_model: bool = True,
        segmentation_type: (
            str | SegmentationType
        ) = SegmentationType.DEEPLABV3_RESNET101,
        # Core Explainability Settings
        output_dir: str = "outputs",
        device: str = "cpu",  # or "cuda",
        num_perturbations: int = 64,
        distance_type: DistanceType | str = DistanceType.WASSERSTEIN,
        surrogate_type: SurrogateType | str = SurrogateType.LIME,
        use_best_surrogate: bool = True,
        **provider_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the image generation and editing explainer.

        Args:
            config: Optional pre-configured settings object.
            engine: The primary model provider, custom class, or string identifier.
            model_name: Name of the underlying model to use.
            pipe: HuggingFace pipeline or custom pipeline object.
            custom_model: Custom model instance for generation/editing.
            custom_generate_fn: Callable function for custom model generation.
            temperature: Temperature parameter for the model.
            seed: Random seed for reproducibility.
            use_image_embedding_model: Flag to enable image embedding.
            image_embedding_type: Type of image embedding to utilize.
            text_embedding_type: Type of text embedding to utilize.
            use_segmentation_model: Flag to enable image segmentation.
            segmentation_type: Type of segmentation model to utilize.
            output_dir: Directory to save intermediate and final outputs.
            device: Device to run local models on ('cpu' or 'cuda').
            num_perturbations: Number of text perturbations to generate.
            distance_type: Metric used to compute distance between images.
            surrogate_type: Type of surrogate model to train for explanation.
            use_best_surrogate: Flag to automatically find the best surrogate model.
            **provider_kwargs: Additional keyword arguments for the model provider.

        Raises:
            ValueError: If an invalid distance metric is provided.

        """
        self._action: Literal["generate", "edit"] = "generate"
        distance_type = DistanceType.from_str(distance_type)

        if not distance_type.is_numeric_metric:
            raise ValueError(
                f"Invalid distance metric '{distance_type}' "
                "for ImageClassificationExplainer. Must be a numeric distance."
            )

        self._provider_kwargs = provider_kwargs

        # Resolve device and initialize state prior to engine creation
        resolved_device = device
        if config is not None and getattr(config, "device", None) is not None:
            resolved_device = config.device
        elif resolved_device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            if config is not None:
                config.device = resolved_device

        self.state = ImageGenerationAndEditingState(
            device_=torch.device(resolved_device)
        )

        # -----------------------------------------------------------------
        # Engine and Provider Type Detection
        # -----------------------------------------------------------------
        provider_type: ProviderType | str | None = None
        engine_type: Literal["provider", "custom", "pipeline"] = "provider"

        # Case 1: Pre-loaded HuggingFace Diffusers Pipeline passed via `pipe`
        if pipe is not None and custom_generate_fn is None:
            provider_type = ProviderType.HUGGINGFACE
            engine_type = "provider"
            self._provider_kwargs["pipe"] = pipe
            if model_name == "dall-e-3" and hasattr(pipe, "_name_or_path"):
                model_name = getattr(pipe, "_name_or_path", "pipeline_model")

        # Case 2: Custom pipeline passed with an explicit custom_generate_fn
        elif pipe is not None:
            engine_type = "pipeline"
            model_name = "pipeline_model"
            custom_model = pipe  # Treat the pipeline as the custom model itself
            if custom_generate_fn is None:
                # Fallback generator assuming HuggingFace diffusers standard behavior
                def _default_pipe_generate(
                    model: Any,  # noqa: ANN401
                    prompt: str,
                    **kwargs: Any,  # noqa: ANN401
                ) -> Any:  # noqa: ANN401
                    return model(prompt, **kwargs).images[0]

                custom_generate_fn = _default_pipe_generate

        # Case 3: Engine parameter provided (Standard Provider or Custom Class/Instance)
        elif engine is not None:
            is_resolved_as_standard_provider = False

            # Check if engine is a standard provider enum/string/instance
            if isinstance(engine, (str, BaseProvider, ProviderType)):
                target_str = (
                    engine
                    if isinstance(engine, str)
                    else (
                        engine.value
                        if isinstance(engine, ProviderType)
                        else engine.__class__.__name__.lower().replace("provider", "")
                    )
                )
                try:
                    provider_type = ProviderType.from_str(target_str)
                    if isinstance(engine, BaseProvider):
                        self.state.engine = engine  # type: ignore[assignment]
                    engine_type = "provider"
                    is_resolved_as_standard_provider = True
                except ValueError:
                    # Not a standard provider, pass to Custom checking
                    pass

            # Check if engine is a Custom BaseImageGenerationAndEditing
            # Instance or Subclass
            if not is_resolved_as_standard_provider:
                engine_type = "custom"
                if hasattr(engine, "__class__") and "BaseImageGenerationAndEditing" in [
                    b.__name__
                    for b in engine.__class__.__mro__  # type: ignore[union-attr]
                ]:
                    # Pre-instantiated custom engine instance
                    self.state.engine = engine  # type: ignore[assignment]
                elif isinstance(engine, type):
                    # Class type passed; instantiate with provider kwargs
                    self.state.engine = engine(**provider_kwargs)
                elif isinstance(engine, str):
                    # Handle known specific custom engines
                    engine_lower = engine.lower()
                    if engine_lower in ("paired", "img2img-turbo"):
                        from xwhy.models.image_generation_and_editing.paired import (
                            PairedInferenceModel,
                        )

                        self.state.engine = PairedInferenceModel(model_name=model_name)
                    else:
                        # String passed but has no specific resolver (e.g.,
                        # unrecognized)
                        if custom_model is None:
                            custom_model = engine

        # Fallback if config needs to be created
        elif custom_model is not None or custom_generate_fn is not None:
            engine_type = "custom"
        else:
            # Default behavior if absolutely nothing is passed
            provider_type = ProviderType.OPENAI
            engine_type = "provider"

        image_embedding_type = EmbeddingType.from_str(image_embedding_type)
        text_embedding_type = EmbeddingType.from_str(text_embedding_type)
        segmentation_type = SegmentationType.from_str(segmentation_type)
        surrogate_type = SurrogateType.from_str(surrogate_type)

        if config is None:
            config = ImageGenerationAndEditingConfig(
                provider_type=provider_type,
                engine_type=engine_type,
                model_name=model_name,
                custom_model=custom_model,
                custom_generate_fn=custom_generate_fn,
                temperature=temperature,
                seed=seed,
                use_image_embedding_model=use_image_embedding_model,
                image_embedding_type=image_embedding_type,
                text_embedding_type=text_embedding_type,
                use_segmentation_model=use_segmentation_model,
                segmentation_type=segmentation_type,
                output_dir=output_dir,
                device=resolved_device,
                num_perturbations=num_perturbations,
                distance_type=distance_type,
                surrogate_type=surrogate_type,
                use_best_surrogate=use_best_surrogate,
            )

        super().__init__(config)
        self._initialize()

    def _initialize(self) -> None:
        """Initialize runtime resources and load required models.

        Raises:
            ValueError: If configuration constraints are violated, such as missing
                custom functions or unsupported provider modes.

        """
        if self.state.engine is None:
            engine_type = self.config.engine_type  # type: ignore[union-attr]

            if engine_type in ("custom", "pipeline"):
                _ = getattr(self.config, "custom_model", None) is not None
                has_custom_fn = (
                    getattr(self.config, "custom_generate_fn", None) is not None
                )

                if not has_custom_fn:
                    raise ValueError(
                        f"When using a {engine_type} approach, 'custom_generate_fn' "
                        "must be provided."
                    )

                logger.info(
                    "Initializing %s Model Adapter...", engine_type.capitalize()
                )
                self.state.engine = CustomImageGenerationAndEditingModel(
                    generate_fn=self.config.custom_generate_fn,  # type: ignore[union-attr]
                    model=self.config.custom_model,  # type: ignore[union-attr]
                    **self._provider_kwargs,
                )
            # Logic for standard providers (Case 1)
            elif engine_type == "provider":
                provider_type = self.config.provider_type  # type: ignore[union-attr]
                if provider_type is None:
                    raise ValueError(
                        "Provider type cannot be None when engine_type is 'provider'."
                    )

                if isinstance(provider_type, ProviderType) and getattr(
                    provider_type, "is_text_only", False
                ):
                    raise ValueError(
                        f"Provider '{provider_type.value}' only supports text. "
                        "ImageGenerationAndEditingExplainer requires a provider "
                        "that supports image generation or both."
                    )

                # Add extra args for huggingface provider
                if provider_type == ProviderType.HUGGINGFACE:
                    self._provider_kwargs.update(
                        {
                            "model_name": self.config.model_name,  # type: ignore[union-attr]
                            "use_segmentation_model": (
                                self.config.use_segmentation_model  # type: ignore[union-attr]
                            ),
                            "config": self.config,
                        }
                    )
                    if getattr(self.config, "custom_model", None) is not None:
                        self._provider_kwargs["pipe"] = self.config.custom_model  # type: ignore[union-attr]

                logger.info("Resolving provider type: %s", provider_type)
                self.state.engine = ProviderResolver.resolve(  # type: ignore[assignment]
                    provider_type,
                    **self._provider_kwargs,
                )

        # 2. Load Image Embedding Model (if enabled)
        if self.config.use_image_embedding_model:  # type: ignore[union-attr]
            if not self.config.image_embedding_type.is_image_embedding:  # type: ignore[union-attr]
                raise ValueError(
                    "Invalid embedding type '%s' "
                    "for ImageGenerationAndEditingExplainer. Must be an image "
                    "embedding.",
                    self.config.image_embedding_type,  # type: ignore[union-attr]
                )

            logger.info(
                "Loading image embedding model: %s",
                self.config.image_embedding_type,  # type: ignore[union-attr]
            )
            self.state.image_embedding_model = EmbeddingFactory.create(
                embedding=self.config.image_embedding_type,  # type: ignore[union-attr]
                device=self.state.device,
            )
            self.state.image_embedding_model.load()

        # 3. Load Text Embedding Model
        if not self.config.text_embedding_type.is_text_embedding:  # type: ignore[union-attr]
            raise ValueError(
                "Invalid text embedding type '%s' "
                "for ImageGenerationAndEditingExplainer. Must be a text embedding.",
                self.config.text_embedding_type,  # type: ignore[union-attr]
            )

        logger.info(
            "Loading text embedding model: %s",
            self.config.text_embedding_type,  # type: ignore[union-attr]
        )
        embedding_factory_result = EmbeddingFactory.create(
            embedding=self.config.text_embedding_type,  # type: ignore[union-attr]
        )
        self.state.text_embedding_model = embedding_factory_result.load()
        self.state.text_embedding_model.fill_norms(force=True)  # type: ignore[union-attr]

        # 4. Load Segmentation Model (if enabled)
        if self.config.use_segmentation_model:  # type: ignore[union-attr]
            if not isinstance(self.config.segmentation_type, SegmentationType):  # type: ignore[union-attr]
                raise ValueError(
                    "Invalid segmentation type '%s'.",
                    self.config.segmentation_type,  # type: ignore[union-attr]
                )

            logger.info(
                "Loading segmentation model: %s",
                self.config.segmentation_type,  # type: ignore[union-attr]
            )
            self.state.segmentation_model = SegmentationFactory.create(
                segmentation=self.config.segmentation_type,  # type: ignore[union-attr]
                device=self.state.device,
            )
            self.state.segmentation_model.load()

        # 5. Initialize Perturbator
        logger.info("Initializing text perturbator...")
        self.state.text_perturbator = TextPerturbation(
            seed=self.config.seed  # type: ignore[union-attr]
        )

    def _prepare_environment(self, output_dir: str, seed: int) -> None:
        """Set random seeds and ensure the output directory exists.

        Args:
            output_dir: Target directory path to create.
            seed: Random seed value for torch and numpy.

        """
        logger.debug("Setting seeds for reproducibility (seed=%d)...", seed)
        random.seed(seed)
        _ = np.random.default_rng(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            logger.debug("CUDA not available. Running on CPU.")

        logger.debug("Preparing output directory at: '%s'", output_dir)
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Environment prepared: Output Dir='%s' | Seed=%d", output_dir, seed)

    def _get_provider_specific_kwargs(self) -> dict[str, Any]:
        """Retrieve provider-specific keyword arguments.

        Dynamically configures compatibility parameters for providers that require
        specialized payload flags.

        Returns:
            A dictionary containing required extra keyword arguments.

        """
        kwargs: dict[str, Any] = {}
        provider_type = self.config.provider_type  # type: ignore[union-attr]

        # Important guard since provider_type can be None for custom/pipeline cases
        if provider_type is None:
            return kwargs

        if isinstance(self.state.engine, OpenAIProvider):
            kwargs["provider_name"] = (
                provider_type.value
                if isinstance(provider_type, ProviderType)
                else str(provider_type)
            )

        if provider_type == ProviderType.BYTEDANCE:
            kwargs.update(
                {
                    "use_image_data_uri": True,
                    "use_image_url": True,
                    "use_generate_for_edit": True,
                    "response_format": None,
                }
            )
        elif provider_type == ProviderType.OPENAI and self._action == "generate":
            kwargs["response_format"] = None

        return kwargs

    def _generate_images(
        self,
        prompts: list[str],
        output_dir: str,
        input_image_path: str | None = None,
        seed: int | None = None,
        batch: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> list[tuple[bool, str]]:
        """Process image generation or editing for a list of prompts.

        Args:
            prompts: List of text prompts.
            output_dir: Directory to save the generated images.
            input_image_path: Path to the base image for editing (if any).
            seed: Random seed for generation.
            batch: Flag to use batch processing if supported by the provider.
            **kwargs: Extra parameters (e.g., size, quality, extra_body).

        Returns:
            A list of tuples containing a boolean success flag and the image path.

        """
        total_prompts = len(prompts)
        generated_paths: list[tuple[bool, str]] = []
        engine = self.state.engine

        # Merge provider kwargs (from __init__) with current method kwargs
        provider_kwargs_extras = self._get_provider_specific_kwargs()
        generation_kwargs = {
            **self._provider_kwargs,
            **kwargs,
            **provider_kwargs_extras,
            # Added here, because we get it from init as parameter not kwargs
            "model_name": self.config.model_name,  # type: ignore[union-attr]
        }

        # The underlying Provider (HuggingFace, OpenAI, etc.) will decide
        # whether to use it.
        if getattr(self.state, "segmentation_model", None) is not None:
            edit_sig = inspect.signature(engine.edit_image)  # type: ignore[union-attr]
            if "segmentation_model" in edit_sig.parameters:
                generation_kwargs["segmentation_model"] = self.state.segmentation_model

        # Handle Gemini Batch Processing
        if batch and "gemini" in engine.__class__.__name__.lower():
            logger.debug("Using batch image generation via Gemini API.")
            job_name = engine.submit_image_batch(  # type: ignore[union-attr]
                image_path=input_image_path,
                text_list=prompts,
                seed=seed,
                **generation_kwargs,
            )

            generated_paths = engine.retrieve_image_batch(  # type: ignore[union-attr]
                job_name=job_name,
                text_list=prompts,
                output_dir=output_dir,
            )
            return generated_paths

        # Handle Standard Loop (Generation / Editing)
        logger.debug(
            "Starting image generation: %d prompts | model=%s",
            total_prompts,
            self.config.model_name,  # type: ignore[union-attr]
        )

        start_time = time.perf_counter()

        for idx, text in enumerate(prompts, start=1):
            iter_start = time.perf_counter()
            logger.debug("Prompt %d text: %s", idx, text)

            try:
                if input_image_path is not None:
                    success, path = engine.edit_image(  # type: ignore[union-attr]
                        prompt=text,
                        image_path=input_image_path,
                        output_dir=output_dir,
                        **generation_kwargs,
                    )
                else:
                    success, path = engine.generate_image(  # type: ignore[union-attr]
                        prompt=text,
                        output_dir=output_dir,
                        **generation_kwargs,
                    )
            except Exception as exc:
                logger.warning("Generation failed for prompt %d: %s", idx, exc)
                success, path = False, ""

            generated_paths.append((success, path))

            iter_duration = time.perf_counter() - iter_start
            elapsed = time.perf_counter() - start_time
            avg_time = elapsed / idx
            eta = avg_time * (total_prompts - idx)

            if idx % 5 == 0 or idx == total_prompts:
                logger.debug(
                    "Progress %d/%d | Success=%s | Iter=%.4fs | ETA=%.4fs",
                    idx,
                    total_prompts,
                    success,
                    iter_duration,
                    eta,
                )

            if not success:
                logger.debug("Failed at %d/%d (path=%s)", idx, total_prompts, path)

        total_duration = time.perf_counter() - start_time
        logger.debug(
            "Completed: %d prompts | Total=%.4fs | Avg=%.4fs",
            total_prompts,
            total_duration,
            total_duration / max(total_prompts, 1),
        )

        return generated_paths

    def _compute_perturbation_distances(
        self,
        input_image_path: str,
        generated_images: list[tuple[bool, str]],
        prompts: list[str],
        display_image: bool = False,
        output_dir: str = "outputs",
    ) -> np.ndarray:
        """Compute distances between the original image and perturbations.

        Args:
            input_image_path: Path to the original input image.
            generated_images: List of generated success flags and file paths.
            prompts: List of perturbation text prompts.
            display_image: Flag to display perturbation images and details.
            output_dir: Directory to save the distance metrics array.

        Returns:
            An array of computed distance metrics for each perturbation.

        Raises:
            ValueError: If embedding extraction fails or representations are empty.

        """
        distances = []

        use_embedding = self.config.use_image_embedding_model  # type: ignore[union-attr]
        dist_type = self.config.distance_type  # type: ignore[union-attr]
        if input_image_path:
            self._action = "edit"

        # 1. Pre-calculate original representation
        _, original_image = load_image_as_tensor(image_path=input_image_path)
        base_representation = original_image
        if use_embedding:
            original_embedding = self.state.image_embedding_model.encode_image(  # type: ignore[union-attr]
                original_image
            )
            if original_embedding is None:
                raise ValueError("Original embedding extraction failed.")
            base_representation = np.asarray(original_embedding)  # type: ignore[assignment]

        logger.debug(
            "Computing distances for %d generated images...",
            len(generated_images),
        )

        for idx, ((success, img_path), text) in enumerate(
            zip(generated_images, prompts, strict=False), start=1
        ):
            if success is not None and not success:
                distances.append(float("inf"))
                continue

            if not os.path.exists(img_path):
                logger.warning("Generated image path not found: %s", img_path)
                distances.append(float("inf"))
                continue

            _, current_image = load_image_as_tensor(image_path=img_path)
            current_representation = current_image

            if use_embedding:
                current_embedding = self.state.image_embedding_model.encode_image(  # type: ignore[union-attr]
                    current_representation
                )
                if current_embedding is None:
                    raise ValueError(
                        f"Embedding extraction failed for image: {img_path}"
                    )
                current_representation = np.asarray(current_embedding)  # type: ignore[assignment]

            if current_representation is None or base_representation is None:
                raise ValueError(
                    "One or both representations are None. Check extraction."
                )

            current_representation = np.asarray(current_representation)  # type: ignore[assignment]
            base_representation = np.asarray(base_representation)  # type: ignore[assignment]

            if current_representation.size == 0 or base_representation.size == 0:  # type: ignore[comparison-overlap]
                raise ValueError("Representations are empty. Cannot compute distance.")

            # Compute distance metric
            dist = calculate_distance(
                metric=dist_type,
                source=base_representation,
                target=current_representation,
            )
            distances.append(dist)

            if display_image:
                gen_img = Image.open(img_path)
                logger.debug("Perturbation %d:", idx)
                logger.debug("Perturbed Text: %s", text)
                logger.debug("Distance (generated vs orig): %f", dist)

                plt.figure(figsize=(8, 8))
                plt.imshow(gen_img)
                plt.title(f"Perturbed Text: {text}", fontsize=12)
                plt.axis("off")
                plt.show()

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "distances_generated_vs_orig.npy")
        distances_array = np.array(distances)
        np.save(save_path, distances_array)
        logger.debug("All generated embeddings and distances saved.")

        return distances_array

    def explain(
        self,
        instance: str,
        input_image_path: Any | None = None,  # noqa: ANN401
        output_dir: str | None = None,
        normalization_mode: Literal["linear", "inverse"] = "linear",
        seed: int | None = 42,
        fidelity_plot: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> ImageGenerationAndEditingXWhyResult:
        """Generate an explanation for the given input instance.

        Args:
            instance: Text description for image generation or editing.
            input_image_path: The input object to explain.
            output_dir: Custom directory to save outputs.
            normalization_mode: Method used to normalize text similarities.
            seed: Random seed for reproducibility.
            fidelity_plot: Rendering fidelity scatter plot.
            **kwargs: Additional generation options (e.g., batch, size, extra_body).

        Returns:
            An outcome container carrying explanation data and surrogate metrics.

        Raises:
            FileNotFoundError: If the provided image path does not exist.
            TypeError: If the prompt is not a string.
            ValueError: If the prompt is empty or too short.
            RuntimeError: If base image generation fails.

        """
        kernel_width = getattr(self.config, "kernel_width", 0.25)
        ridge_alpha = getattr(self.config, "ridge_alpha", 1.0)

        prompt = instance
        output_dir = output_dir if output_dir is not None else self.config.output_dir  # type: ignore[union-attr]
        seed = seed if seed is not None else self.config.seed  # type: ignore[union-attr]

        # Extract batch flag from kwargs if provided, defaulting to False
        batch = kwargs.pop("batch", False)

        self._prepare_environment(output_dir=output_dir, seed=seed)

        if seed != self.config.seed:  # type: ignore[union-attr]
            logger.debug("Updating perturbator RNG with new seed: %d", seed)
            self.state.text_perturbator.set_seed(seed)  # type: ignore[union-attr]

        if input_image_path is not None and not os.path.exists(input_image_path):
            raise FileNotFoundError(f"Input image not found at {input_image_path}")

        logger.debug("Validating input prompt...")
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")

        normalized_prompt = prompt.strip()

        if not normalized_prompt:
            raise ValueError("Prompt cannot be empty or whitespace only.")

        logger.info("Starting explanation process...")

        prompt_words = normalized_prompt.split()
        prompt_word_count = len(prompt_words)

        if len(normalized_prompt) < 10 and prompt_word_count >= 3:
            raise ValueError(
                "Prompt is too short for reliable image editing.\n"
                f'Provided prompt ({len(normalized_prompt)} chars): "{prompt}"\n'
                "Please use a more descriptive prompt (at least 18-20 chars)."
            )

        if self.config.num_perturbations < (2 * prompt_word_count):  # type: ignore[union-attr]
            logger.warning(
                "The 'num_perturbations' (%d) is relatively small for a prompt "
                "with %d words. This may lead to inaccurate fidelity metrics "
                "(e.g., R-squared). Consider increasing it for better stability.",
                self.config.num_perturbations,  # type: ignore[union-attr]
                prompt_word_count,
            )

        logger.info("Generating text perturbations...")
        perturbed_texts, binary_masks = self.state.text_perturbator.generate(  # type: ignore[union-attr]
            text=normalized_prompt,
            num_perturbations=self.config.num_perturbations,  # type: ignore[union-attr]
        )

        logger.info("Starting unified image generation/editing step...")

        # Execute the unified generation method (handles batch, generate, edit)
        base_generation_results = self._generate_images(
            prompts=[normalized_prompt],
            output_dir=output_dir,
            input_image_path=input_image_path,
            seed=seed,
            batch=batch,
            **kwargs,
        )
        is_base_success, base_image_path = base_generation_results[0]

        if not is_base_success:
            raise RuntimeError(
                f"Failed to generate the base image for the prompt: "
                f"'{normalized_prompt}'. The explanation process cannot proceed."
            )

        generated_images = self._generate_images(
            prompts=perturbed_texts,
            output_dir=output_dir,
            input_image_path=input_image_path,
            seed=seed,
            batch=batch,
            **kwargs,
        )

        logger.info(
            "Computing %s distances between images...",
            self.config.distance_type,  # type: ignore[union-attr]
        )
        image_distances = self._compute_perturbation_distances(
            input_image_path=base_image_path,
            generated_images=generated_images,
            prompts=perturbed_texts,
            output_dir=output_dir,
        )

        logger.info("Computing WMD scores...")
        wmd_distance = WMDDistance()
        wmd_scores = wmd_distance.compute_batch(
            model=self.state.text_embedding_model,
            original=normalized_prompt,
            perturbed_texts=perturbed_texts,
        )

        logger.info("Normalizing similarities...")
        sims = DistanceNormalizer.min_max(scores=wmd_scores)

        # masks_as_arrays: list[np.ndarray] = [
        #     np.array(m, dtype=int) for m in binary_masks
        # ]

        # ---------------------------------------------------------
        # Surrogate Model Training Inputs & Targets setup:
        # X: Matrix indicating word presence/absence in perturbations.
        # Y: The change/distance in the generated output images.
        # Weights: Derived from textual distance (WMD/sims).
        # ---------------------------------------------------------
        x_features = np.vstack([np.array(m, dtype=int) for m in binary_masks])
        y_target = image_distances
        text_distances_array = np.array([d for _, d in wmd_scores])

        if self.config.use_best_surrogate:  # type: ignore[union-attr]
            logger.info(
                "Searching for the optimal surrogate model among available "
                "candidates..."
            )
            method, score = SurrogateTrainer.find_best(
                x=x_features,
                y=y_target,
                distances=text_distances_array,
                seed=seed,
                kernel_width=kernel_width,
                ridge_alpha=ridge_alpha,
            )
            logger.info(
                "Optimization complete. Selected surrogate model: "
                "'%s' (Best Score: %.4f)",
                method.value,
                score,
            )
        else:
            method = self.config.surrogate_type  # type: ignore[union-attr]
            logger.info(
                "Skipping surrogate search. Using configured default: '%s'",
                method.value,
            )

        weights = SurrogateTrainer.compute_weights(
            method=method,
            distances=text_distances_array,
            kernel_width=kernel_width,
        )

        surrogate = SurrogateFactory.create(
            method=method,
            seed=self.config.seed,  # type: ignore[union-attr]
        )
        surrogate.fit(x_features, y_target, weights)

        coeffs = surrogate.coefficients()
        y_pred = surrogate.predict(x_features)

        logger.info("Computing regression metrics...")
        metrics = RegressionMetrics.calculate(
            y_true=y_target,
            y_pred=y_pred,
            weights=weights,
            num_features=len(coeffs),
        )

        logger.info("Save variables data to pickle file...")
        save_data_to_pickle(
            output_path=os.path.join(
                output_dir,
                f"{self.config.model_name.replace('/', '_')}.pkl",  # type: ignore[union-attr]
            ),
            responses=perturbed_texts,
            perturbations=binary_masks,
            image_distances=image_distances,
            wmd_scores=wmd_scores,
            sims=sims,
            mode=normalization_mode,
            normalized_prompt=normalized_prompt,
            num_perturb=self.config.num_perturbations,  # type: ignore[union-attr]
            seed=seed,
        )

        logger.info("Saving perturbation data to CSV...")
        csv_path = save_perturbation_data_to_csv(
            perturbations=binary_masks,  # type: ignore[arg-type]
            similarities=sims,
            wmd_scores=wmd_scores,
            output_path=os.path.join(output_dir, f"perturbation_data_{method}.csv"),
        )

        raw_data = {
            "prompt": normalized_prompt,
            "csv_output_path": csv_path,
            "perturbed_texts": perturbed_texts,
            "wmd_scores": wmd_scores,
            "similarities": sims,
            "weights": weights,
            "y_target": y_target,
            "y_pred": y_pred,
        }

        if self.config.use_best_surrogate:  # type: ignore[union-attr]
            raw_data["best_surrogate_method"] = method
        else:
            raw_data["surrogate_method"] = method

        result = ImageGenerationAndEditingXWhyResult(
            words=prompt_words,
            instance=input_image_path,
            coefficients=coeffs,
            metrics=metrics,
            raw_data=raw_data,
        )

        if fidelity_plot:
            logger.info("Rendering fidelity plot as requested...")
            result.plot(show=True)

        return result
