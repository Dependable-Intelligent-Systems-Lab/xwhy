"""Point cloud explainer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from sklearn.cluster import KMeans

from xwhy.core.config import PointCloudConfig
from xwhy.core.explainer import BaseExplainer
from xwhy.core.result import PointCloudXWhyResult
from xwhy.core.states import PointCloudState
from xwhy.distance.calculator import calculate_distance
from xwhy.distance.types import DistanceType
from xwhy.logger import logger
from xwhy.metrics.regression import RegressionMetrics
from xwhy.models.point_cloud.base import BasePointCloudModel
from xwhy.models.point_cloud.custom import CustomPointCloudModel
from xwhy.perturbation.point_cloud import PointCloudPerturbation
from xwhy.surrogate.factory import SurrogateFactory
from xwhy.surrogate.trainer import SurrogateTrainer
from xwhy.surrogate.types import SurrogateType


class PointCloudExplainer(BaseExplainer):
    """Explainer for Point Cloud classification tasks."""

    def __init__(
        self,
        config: PointCloudConfig | None = None,
        model: torch.nn.Module | BasePointCloudModel | Any | None = None,  # noqa: ANN401
        custom_model: Any | None = None,  # noqa: ANN401
        custom_predict_fn: Callable[..., Any] | None = None,
        num_clusters: int = 8,
        num_top_features: int = 4,
        num_perturbations: int = 50,
        removal_probability: float = 0.3,
        seed: int = 42,
        epsilon: float = 0.0,
        kernel_width: float = 0.5,
        ridge_alpha: float = 1.0,
        max_iters: int = 50,
        min_valid_ratio: float = 0.5,
        device: str = "cpu",
        clustering_mode: Literal["kmeans", "precomputed"] = "kmeans",
        distance_type: DistanceType | str = DistanceType.WASSERSTEIN,
        distance_mode: Literal["mask", "spatial", "latent"] = "mask",
        surrogate_type: SurrogateType | str = SurrogateType.LIME,
        use_best_surrogate: bool = True,
        return_p_value: bool = False,
        n_bootstrap: int = 1000,
        **model_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the Point Cloud explainer.

        Args:
            config: Optional explainer configuration instance.
            model: PyTorch model, or BasePointCloudModel wrapper.
            custom_model: Custom model fallback if model is not provided.
            custom_predict_fn: Custom prediction function.
            num_clusters: Number of clusters for point cloud segmentation.
            num_top_features: Number of top feature clusters to extract.
            num_perturbations: Number of perturbed samples.
            removal_probability: Probability of removing a cluster.
            seed: Random seed for reproducibility.
            epsilon: Numerical stability constant.
            kernel_width: Kernel width for similarity weights.
            ridge_alpha: Ridge regularization strength.
            max_iters: Maximum iterations for clustering.
            min_valid_ratio: Minimum proportion of valid (non-NaN/non-infinite)
                perturbation evaluations required for reliable surrogate training.
            device: Computation device ("cpu" or "cuda").
            clustering_mode: "kmeans" or "precomputed".
            distance_type: Metric used to compute distance between points.
            distance_mode: "mask", "spatial", or "latent".
            surrogate_type: Type of surrogate model to train for explanation.
            use_best_surrogate: Flag to automatically find the best surrogate.
            return_p_value: Whether to compute statistical significance
                (p-values) for computed distances using bootstrap sampling.
            n_bootstrap: Number of bootstrap iterations for p-value estimation.
            **model_kwargs: Additional parameters for model wrapper.

        Raises:
            ValueError: If distance metric is not numeric.

        """
        dist_enum = DistanceType.from_str(distance_type)
        surrogate_enum = SurrogateType.from_str(surrogate_type)

        if not dist_enum.is_numeric_metric:
            raise ValueError(
                f"Invalid distance metric '{dist_enum}' "
                "for PointCloudExplainer. Must be a numeric distance."
            )

        # 2. Construct or update configuration
        if config is None:
            config = PointCloudConfig(
                custom_model=custom_model,
                custom_predict_fn=custom_predict_fn,
                num_clusters=num_clusters,
                num_top_features=num_top_features,
                num_perturbations=num_perturbations,
                removal_probability=removal_probability,
                seed=seed,
                epsilon=epsilon,
                kernel_width=kernel_width,
                ridge_alpha=ridge_alpha,
                max_iters=max_iters,
                min_valid_ratio=min_valid_ratio,
                device=device,
                clustering_mode=clustering_mode,
                distance_type=dist_enum,
                distance_mode=distance_mode,
                surrogate_type=surrogate_enum,
                use_best_surrogate=use_best_surrogate,
                return_p_value=return_p_value,
                n_bootstrap=n_bootstrap,
            )

        # 3. Bind config to base class pipeline
        super().__init__(config)

        # 4. Initialize runtime state and model wrappers
        self.state = PointCloudState(
            device_=torch.device(self.config.device)  # type: ignore[union-attr]
        )
        self._model_kwargs = model_kwargs

        if model is not None:
            if isinstance(model, BasePointCloudModel):
                self.state.model = model
            else:
                self.state.model = CustomPointCloudModel(
                    model=model,
                    predict_fn=self.config.custom_predict_fn,  # type: ignore[union-attr]
                    **self._model_kwargs,
                )

        self._initialize()

    def _initialize(self) -> None:
        """Initialize model runtime resources if not already provided."""
        if self.state.model is None:
            logger.info("Initializing point cloud model")

            self.state.model = CustomPointCloudModel(
                model=self.config.custom_model,  # type: ignore[union-attr]
                predict_fn=self.config.custom_predict_fn,  # type: ignore[union-attr]
                **self._model_kwargs,
            )

        # Initialize the perturbation strategy
        self.state.perturbation = PointCloudPerturbation(
            removal_probability=self.config.removal_probability,  # type: ignore[union-attr]
            seed=self.config.seed,  # type: ignore[union-attr]
        )

    def _cluster_points(
        self,
        sample_input: torch.Tensor,
        cluster_labels: np.ndarray | None,
    ) -> np.ndarray:
        """Perform point cloud clustering or return precomputed labels."""
        mode = self.config.clustering_mode  # type: ignore[union-attr]
        num_clusters = self.config.num_clusters  # type: ignore[union-attr]
        max_iters = self.config.max_iters  # type: ignore[union-attr]
        seed = self.config.seed  # type: ignore[union-attr]

        if mode == "precomputed":
            if cluster_labels is None:
                raise ValueError(
                    "cluster_labels must be provided when "
                    "clustering_mode='precomputed'."
                )
            return cluster_labels

        if mode == "kmeans":
            # 1. Prepare data
            points = sample_input.squeeze(0).cpu().numpy()
            num_points, dim = points.shape

            # 2. Farthest Point Sampling (FPS) for center initialization
            # Localize the legacy MT19937 generator to maintain baseline
            # fidelity metrics without mutating the global np.random state.
            rng = np.random.RandomState(seed)
            centers = np.zeros((num_clusters, dim))
            center_indices = np.zeros(num_clusters, dtype=int)

            center_indices[0] = rng.randint(num_points)
            centers[0] = points[center_indices[0]]
            distances = np.sum((points - centers[0]) ** 2, axis=1)

            for i in range(1, num_clusters):
                center_indices[i] = np.argmax(distances)
                centers[i] = points[center_indices[i]]
                distances = np.minimum(
                    distances, np.sum((points - centers[i]) ** 2, axis=1)
                )

            # 3. KMeans clustering with FPS centers
            kmeans = KMeans(
                n_clusters=num_clusters,
                init=centers,
                max_iter=max_iters,
                n_init=1,
                random_state=seed,
            )
            kmeans.fit(points)
            return np.asarray(kmeans.labels_, dtype=int)

        raise ValueError(f"Invalid clustering_mode: {mode}")

    def explain(
        self,
        instance: torch.Tensor,
        sample_label: int | None = None,
        cluster_labels: np.ndarray | None = None,
        fidelity_plot: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> PointCloudXWhyResult:
        """Generate explanations for point cloud prediction.

        Args:
            instance: Point cloud tensor of shape (N, 3) or (1, N, 3).
            sample_label: Optional target class label.
            cluster_labels: Optional precomputed point cluster labels.
            fidelity_plot: Rendering fidelity scatter plot.
            **kwargs: Extra dynamic arguments.

        Returns:
            PointCloudXWhyResult: Structured explanation result object.

        Raises:
            RuntimeError: If model is not loaded.
            TypeError: If input sample is invalid type.
            ValueError: If all perturbation evaluations fail or invalid
                distance mode is encountered.

        """
        sample_input = instance
        if not isinstance(sample_input, torch.Tensor):
            raise TypeError("sample_input must be a torch.Tensor.")

        if self.state.model is None:
            raise RuntimeError("Point cloud model is not initialized.")

        # --------------------------------------------------
        # Step 1: Prediction
        # --------------------------------------------------
        if sample_input.ndim == 2:
            sample_input = sample_input.unsqueeze(0)
        sample_input = sample_input.float().to(self.state.device)

        sample_np = sample_input.squeeze(0).cpu().numpy()

        pred, _, top_classes = self.state.model.predict(
            sample_input=sample_input,
            sample_label=sample_label,
        )

        # --------------------------------------------------
        # Step 2: Clustering
        # --------------------------------------------------
        labels = self._cluster_points(sample_input, cluster_labels)

        # --------------------------------------------------
        # Step 3: Perturbation
        # --------------------------------------------------
        point_cloud = sample_input.squeeze(0)  # (N, 3)
        assert point_cloud.ndim == 2, f"Expected (N,3), got {point_cloud.shape}"

        # Generate cluster-level masks
        cluster_masks = self.state.perturbation.generate(  # type: ignore[union-attr]
            num_clusters=self.config.num_clusters,  # type: ignore[union-attr]
            num_perturbations=self.config.num_perturbations,  # type: ignore[union-attr]
        )

        # Apply masks to generate perturbed point clouds
        perturbed_samples: list[torch.Tensor] = []
        for mask in cluster_masks:
            perturbed = self.state.perturbation.apply_mask(  # type: ignore[union-attr]
                item=point_cloud,
                mask=mask,
                segments=labels,
            )
            perturbed_samples.append(perturbed)

        # --------------------------------------------------
        # Step 4: Distance computation
        # --------------------------------------------------
        distances: list[float] = []
        p_values: list[float] = []

        return_p_val = getattr(self.config, "return_p_value", False)
        n_bootstrap = getattr(self.config, "n_bootstrap", 1000)

        if self.config.distance_mode == "mask":  # type: ignore[union-attr]
            # Baseline is a full mask of 1s (all clusters present)
            reference_mask = np.ones(self.config.num_clusters)  # type: ignore[union-attr]

            for mask in cluster_masks:
                res = calculate_distance(
                    metric=self.config.distance_type,  # type: ignore[union-attr]
                    source=reference_mask,
                    target=mask,
                    mode="mask",
                    return_p_value=return_p_val,
                    n_bootstrap=n_bootstrap,
                )
                if isinstance(res, tuple):
                    p_val, dist = res
                    p_values.append(p_val)
                    distances.append(dist)
                else:
                    distances.append(res)

        elif self.config.distance_mode == "spatial":  # type: ignore[union-attr]
            original = sample_input.squeeze(0)  # Shape: (N, 3)

            for perturbed in perturbed_samples:  # Shape: (M, 3)
                res = calculate_distance(
                    metric=self.config.distance_type,  # type: ignore[union-attr]
                    source=original,
                    target=perturbed,
                    mode="spatial",
                    return_p_value=return_p_val,
                    n_bootstrap=n_bootstrap,
                )
                if isinstance(res, tuple):
                    p_val, dist = res
                    p_values.append(p_val)
                    distances.append(dist)
                else:
                    distances.append(res)

        elif self.config.distance_mode == "latent":  # type: ignore[union-attr]
            # Ensure model receives batch dimension: (1, N, 3)
            input_batch = (
                sample_input if sample_input.ndim == 3 else sample_input.unsqueeze(0)
            )
            _, original_latent, _ = self.state.model.predict(
                sample_input=input_batch,
                sample_label=sample_label,
            )
            original_latent = original_latent.squeeze(0)

            for perturbed in perturbed_samples:
                # Add batch dimension for model forward pass: (1, M, 3)
                perturbed_batch = (
                    perturbed if perturbed.ndim == 3 else perturbed.unsqueeze(0)
                )

                _, perturbed_latent, _ = self.state.model.predict(
                    sample_input=perturbed_batch,
                    sample_label=sample_label,
                )
                perturbed_latent = perturbed_latent.squeeze(0)

                res = calculate_distance(
                    metric=self.config.distance_type,  # type: ignore[union-attr]
                    source=original_latent,
                    target=perturbed_latent,
                    mode="latent",
                    return_p_value=return_p_val,
                    n_bootstrap=n_bootstrap,
                )
                if isinstance(res, tuple):
                    p_val, dist = res
                    p_values.append(p_val)
                    distances.append(dist)
                else:
                    distances.append(res)

        else:
            dist_mode = self.config.distance_mode  # type: ignore[union-attr]
            raise ValueError(f"Invalid distance_mode: {dist_mode}")

        # --------------------------------------------------
        # Step 5: Distance Validation, Weights, & Surrogate Fitting
        # --------------------------------------------------
        cfg = self.config

        # Validate distances and filter out non-finite (inf/NaN) values
        logger.info("Validating perturbation distances...")
        distances_raw = np.array(distances, dtype=float)

        valid_mask = np.isfinite(distances_raw)
        valid_count = int(np.sum(valid_mask))
        total_count = len(distances_raw)
        valid_ratio = valid_count / total_count

        min_valid_ratio = cfg.min_valid_ratio  # type: ignore[union-attr]

        if valid_count == 0:
            error_msg = (
                "All perturbations failed (0 valid distances). Cannot fit the "
                "surrogate model with an empty dataset. Aborting explanation."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        elif valid_ratio < min_valid_ratio:
            logger.warning(
                "Low valid perturbation ratio. Only %.1f%% succeeded (%d/%d). "
                "Training surrogate model with reduced sample size, which may "
                "lead to unstable explanations.",
                valid_ratio * 100,
                valid_count,
                total_count,
            )

        distances_valid = distances_raw[valid_mask]

        # Retrieve perturbation predictions for target class
        output_probs = self.state.model.get_output_probabilities(
            samples=perturbed_samples,
            device=self.state.device,
        )
        if isinstance(output_probs, torch.Tensor):
            output_np: np.ndarray = output_probs.detach().cpu().numpy()
        else:
            output_np = np.asarray(output_probs)

        y_target_raw = output_np[:, pred]
        x_matrix_raw = cluster_masks  # Shape: (num_perturbations, num_clusters)

        # Filter features and targets cleanly using the validity mask
        x_valid = x_matrix_raw[valid_mask]
        y_valid = y_target_raw[valid_mask]

        # Surrogate selection and weight calculation
        if cfg.use_best_surrogate:  # type: ignore[union-attr]
            logger.info("Searching for optimal surrogate model...")
            method, score = SurrogateTrainer.find_best(
                x=x_valid,
                y=y_valid,
                distances=distances_valid,
                seed=cfg.seed,  # type: ignore[union-attr]
                kernel_width=cfg.kernel_width,  # type: ignore[union-attr]
                epsilon=cfg.epsilon,  # type: ignore[union-attr]
                ridge_alpha=cfg.ridge_alpha,  # type: ignore[union-attr]
                normalize_distances=False,
            )
            logger.info(
                "Optimization complete. Selected surrogate model: '%s'"
                " (Best Score: %.4f)",
                method.value if hasattr(method, "value") else method,
                score,
            )
        else:
            method = cfg.surrogate_type  # type: ignore[assignment, union-attr]
            method_name = method.value if hasattr(method, "value") else method
            logger.info("Skipping surrogate search. Using default: '%s'", method_name)

        weights = SurrogateTrainer.compute_weights(
            method=method,
            distances=distances_valid,
            kernel_width=cfg.kernel_width,  # type: ignore[union-attr]
            epsilon=cfg.epsilon,  # type: ignore[union-attr]
            normalize_distances=False,
        )

        # Fit surrogate model
        method_name = method.value if hasattr(method, "value") else method
        logger.info("Training surrogate model (%s)...", method_name)
        surrogate = SurrogateFactory.create(
            method=method,
            seed=cfg.seed,  # type: ignore[union-attr]
            ridge_alpha=cfg.ridge_alpha,  # type: ignore[union-attr]
        )
        surrogate.fit(x_valid, y_valid, weights)

        coeffs = surrogate.coefficients()
        y_pred_valid = surrogate.predict(x_valid)

        # Compute regression fidelity metrics
        metrics = RegressionMetrics.calculate(
            y_true=y_valid,
            y_pred=y_pred_valid,
            weights=weights,
            num_features=len(coeffs),
        )

        top_k = cfg.num_top_features  # type: ignore[union-attr]
        top_features = np.argsort(coeffs)[-top_k:]

        raw_data: dict[str, Any] = {
            "x_matrix": x_valid,
            "y_target": y_valid,
            "y_pred": y_pred_valid,
            "weights": weights,
            "distances": distances_valid,
            "surrogate_method": method,
            "top_classes": top_classes,
        }

        if return_p_val and p_values:
            p_values_raw = np.array(p_values, dtype=float)
            raw_data["p_values"] = p_values_raw[valid_mask]

        result = PointCloudXWhyResult(
            coefficients=coeffs,
            metrics=metrics,
            raw_data=raw_data,
            important_clusters=top_features,
            sample_points=sample_np,
            cluster_labels=labels,
        )

        if fidelity_plot:
            logger.info("Rendering fidelity plot as requested...")
            result.plot(show=True)

        return result
