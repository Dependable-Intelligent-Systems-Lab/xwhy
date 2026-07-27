"""Tabular explainer implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from xwhy.core.config import TabularConfig
from xwhy.core.explainer import BaseExplainer
from xwhy.core.result import TabularXWhyResult
from xwhy.core.types import TabularState
from xwhy.distance.calculator import calculate_distance
from xwhy.distance.types import DistanceType
from xwhy.logger import logger
from xwhy.metrics.regression import RegressionMetrics
from xwhy.surrogate.factory import SurrogateFactory
from xwhy.surrogate.trainer import SurrogateTrainer
from xwhy.surrogate.types import SurrogateType


class TabularExplainer(BaseExplainer):
    """Explainer for Tabular models utilizing the SMILE algorithm.

    This explainer preserves exact Wasserstein LIME mechanics while
    integrating with the broader framework architectures.
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        config: TabularConfig | None = None,
        mode: str = "classification",
        num_perturbations: int = 500,
        kernel_width: float = 0.2,
        num_distribution_samples: int = 100,
        local_noise: float = 0.05,
        perturbation_noise: float = 0.4,
        epsilon: float = 1.0,
        distance_type: str | DistanceType = DistanceType.WASSERSTEIN,
        surrogate_type: str | SurrogateType = SurrogateType.LIME,
        use_best_surrogate: bool = True,
        seed: int = 1024,
        validate_normalization: bool = True,
    ) -> None:
        """Initialize the Tabular explainer.

        Args:
            model: Trained black-box model with a `predict` method.
            config: Optional configuration object.
            mode: Task type ("classification" or "regression").
            num_perturbations: Number of LIME samples generated.
            kernel_width: Kernel width used for weighting.
            num_distribution_samples: Samples per feature distribution.
            local_noise: Noise scale for the local instance neighborhood.
            perturbation_noise: Noise scale for perturbation distributions.
            epsilon: Scaling factor applied to the Wasserstein distance.
            distance_type: Distance metric definition.
            surrogate_type: Default surrogate method name.
            use_best_surrogate: Automatically search for the best surrogate.
            seed: Random seed for reproducibility.
            validate_normalization: Whether to warn if the input appears not
                to be normalized.

        Raises:
            ValueError: If the distance type is not valid.

        """
        distance_type = DistanceType.from_str(distance_type)
        surrogate_type = SurrogateType.from_str(surrogate_type)

        if mode not in ("classification", "regression"):
            raise ValueError("mode must be 'classification' or 'regression'.")

        if config is None:
            config = TabularConfig(
                mode=mode,  # type: ignore[arg-type]
                num_perturbations=num_perturbations,
                kernel_width=kernel_width,
                num_distribution_samples=num_distribution_samples,
                local_noise=local_noise,
                perturbation_noise=perturbation_noise,
                epsilon=epsilon,
                distance_type=distance_type,
                surrogate_type=surrogate_type,
                use_best_surrogate=use_best_surrogate,
                seed=seed,
                validate_normalization=validate_normalization,
            )

        super().__init__(config)
        self.state = TabularState()
        self.state.model = model

        self._rng = np.random.default_rng(self.config.seed)  # type: ignore[union-attr]

    def _generate_instance_distribution(
        self, instance: np.ndarray, num_features: int, noise: float, samples: int
    ) -> np.ndarray:
        """Create local Gaussian distributions for each feature.

        Args:
            instance: Target instance to explain.
            num_features: Number of features.
            noise: Variance parameter for normal distribution.
            samples: Number of observations per distribution.

        Returns:
            np.ndarray: Matrix of the local distribution.

        """
        distribution = np.zeros((samples, num_features))
        for i in range(num_features):
            distribution[:, i] = instance[i] + self._rng.normal(0, noise, samples)
        return distribution

    def explain(
        self,
        instance: np.ndarray | Sequence[Any],
        feature_names: Sequence[str] | None = None,
        fidelity_plot: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> TabularXWhyResult:
        """Generate an explanation using the specified distance algorithm.

        Args:
            instance: Target instance array of shape [n_features].
            feature_names: Optional sequence specifying column names.
            fidelity_plot: Rendering fidelity scatter plot.
            **kwargs: Additional parameters.

        Returns:
            TabularXWhyResult: The structured outcome containing weights,
                distances, and surrogate coefficients.

        Raises:
            ValueError: If the instance contains out-of-scale values, indicating
                a lack of standardization.

        """
        cfg: TabularConfig = self.config  # type: ignore[assignment]
        instance_arr = np.asarray(instance, dtype=np.float64)

        if cfg.validate_normalization and np.abs(np.mean(instance_arr)) > 5.0:
            logger.warning(
                "Instance appears not normalized. Ensure you pass standardized data."
            )

        num_features = len(instance_arr)

        # 1. Generate base perturbation samples
        x_matrix = self._rng.normal(0, 1, size=(cfg.num_perturbations, num_features))

        # 2. Local distribution around original instance
        instance_dist = self._generate_instance_distribution(
            instance=instance_arr,
            num_features=num_features,
            noise=cfg.local_noise,
            samples=cfg.num_distribution_samples,
        )

        y_target = np.zeros((cfg.num_perturbations,))
        distances = np.zeros((cfg.num_perturbations,))

        logger.info(f"Computing distances for {cfg.num_perturbations} perturbations...")

        # 3. Main Loop
        for idx, sample in enumerate(x_matrix):
            sample_dist = self._generate_instance_distribution(
                instance=sample,
                num_features=num_features,
                noise=cfg.perturbation_noise,
                samples=cfg.num_distribution_samples,
            )

            preds = self.state.model.predict(sample_dist)  # type: ignore[union-attr]

            if cfg.mode == "classification":
                y_target[idx] = np.bincount(preds.astype(int)).argmax()
            else:
                y_target[idx] = np.mean(preds)

            # ==============================
            # Compute distance (per feature)
            # ==============================
            dist_total = 0.0
            for j in range(num_features):
                dist = calculate_distance(
                    metric=cfg.distance_type,
                    source=instance_dist[:, j],
                    target=sample_dist[:, j],
                )
                dist_total += dist

            distances[idx] = dist_total

        scaled_distances = distances * cfg.epsilon

        # 4. Surrogate Training via Framework
        if cfg.use_best_surrogate:
            logger.info("Searching for optimal surrogate model...")
            method, score = SurrogateTrainer.find_best(
                x=x_matrix,
                y=y_target,
                distances=scaled_distances,
                seed=cfg.seed,
                kernel_width=cfg.kernel_width,
                normalize_distances=False,
            )
            logger.info(
                "Optimization complete. Selected surrogate model:"
                " '%s' (Best Score: %.4f)",
                method.value,
                score,
            )
        else:
            method = cfg.surrogate_type  # type: ignore[assignment]
            logger.info("Skipping surrogate search. Using default: '%s'", method.value)

        weights = SurrogateTrainer.compute_weights(
            method=method,
            distances=scaled_distances,
            kernel_width=cfg.kernel_width,
            normalize_distances=False,
        )

        logger.info(f"Training surrogate model ({method.value})...")
        surrogate = SurrogateFactory.create(method=method, seed=cfg.seed)
        surrogate.fit(x_matrix, y_target, weights)

        coeffs = surrogate.coefficients()
        y_pred = surrogate.predict(x_matrix)

        metrics = RegressionMetrics.calculate(
            y_true=y_target,
            y_pred=y_pred,
            weights=weights,
            num_features=len(coeffs),
        )

        if cfg.mode == "classification":
            y_pred = (y_pred < 0.5).astype(int).flatten()
        else:
            y_pred = y_pred.flatten()

        raw_data = {
            "x_matrix": x_matrix,
            "y_target": y_target,
            "y_pred": y_pred,
            "weights": weights,
            "distances": distances,
            "surrogate_method": method,
        }

        result = TabularXWhyResult(
            coefficients=coeffs,
            metrics=metrics,
            raw_data=raw_data,
            instance=instance_arr,
            feature_list=feature_names or [],
            base_values=0.0,
        )

        if fidelity_plot:
            logger.info("Rendering fidelity plot as requested...")
            result.plot(show=True)

        return result
