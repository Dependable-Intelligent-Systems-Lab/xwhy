"""Image explainer."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import skimage.transform
import torch
from PIL import Image
from tqdm import tqdm

from xwhy.core.config import ImageClassificationConfig
from xwhy.core.explainer import BaseExplainer
from xwhy.core.pipeline import ExplanationPipeline
from xwhy.core.result import ImageClassificationXWhyResult
from xwhy.core.types import ImageClassificationState
from xwhy.distance.calculator import calculate_distance
from xwhy.distance.types import DistanceType
from xwhy.logger import logger
from xwhy.metrics.image import ImageCoverageMetrics
from xwhy.metrics.regression import RegressionMetrics
from xwhy.models.classification.factory import ClassificationFactory
from xwhy.models.classification.types import ClassificationType
from xwhy.models.embeddings.factory import EmbeddingFactory
from xwhy.models.embeddings.types import EmbeddingType
from xwhy.models.segmentation.factory import SegmentationFactory
from xwhy.models.segmentation.types import SegmentationType
from xwhy.perturbation.image import ImagePerturbation
from xwhy.surrogate.factory import SurrogateFactory
from xwhy.surrogate.trainer import SurrogateTrainer
from xwhy.surrogate.types import SurrogateType
from xwhy.utils.image import (
    get_segmentation_mask,
    load_image_as_tensor,
    numpy_image_to_tensor,
    tensor_to_numpy_image,
)


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
        distance_metric: str | DistanceType = DistanceType.WASSERSTEIN,
        surrogate_type: str | SurrogateType = SurrogateType.LIME,
        use_best_surrogate: bool = True,
        num_top_features: int = 4,
        num_top_predictions: int = 5,
    ) -> None:
        """Initialize the Image Classification explainer.

        Args:
            config: Optional configuration for the explainer.
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
            distance_metric: Distance metric name.
            surrogate_type: Surrogate model name.
            use_best_surrogate: Find best surrogate model dynamically.
            num_top_features: Number of important regions to highlight.
            num_top_predictions: Number of predictions to explain.

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
            device_=torch.device(config.device),
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

        # Extract the transform function directly from the adapter
        if self.config.use_model_preprocess:  # type: ignore[union-attr]
            self.state.transform_fn = self.state.classification_model.preprocess_fn  # type: ignore[assignment]

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
        dist_metric = self.config.distance_metric  # type: ignore[union-attr]

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
                metric=dist_metric,
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
        **kwargs: Any,  # noqa: ANN401
    ) -> ImageClassificationXWhyResult:
        """Generate an explanation for an input image.

        Args:
            instance: Path to the image that should be explained.
            fidelity_plot: Rendering fidelity scatter plot.
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
        top_preds = probs.topk(num_top)
        class_to_explain = top_preds.indices[0].item()

        # Log top predictions
        categories = self.state.classification_model.weights.meta["categories"]  # type: ignore[attr-defined]
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

        if self.state.segmentation_model is None:
            raise RuntimeError("Segmentation model is required for this operation.")

        # Evaluation against Ground Truth
        _, sem_mask = get_segmentation_mask(
            image_path=image_path,
            segmentation_model=self.state.segmentation_model,
            transform_fn=transform_fn,
            device=self.config.device,  # type: ignore[union-attr]
            class_names=self.state.segmentation_model.class_names,
        )

        # Resize explanation image to match original mask spatial dimensions if needed
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
