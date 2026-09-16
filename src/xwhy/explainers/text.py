"""Text explainer."""

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np

from xwhy.core.config import ExplainerConfig, TextConfig
from xwhy.core.explainer import BaseExplainer
from xwhy.core.result import TextXWhyResult
from xwhy.core.states import TextState
from xwhy.distance.calculator import calculate_distance
from xwhy.distance.types import DistanceType
from xwhy.logger import logger
from xwhy.metrics.regression import RegressionMetrics
from xwhy.models.embeddings.factory import EmbeddingFactory
from xwhy.models.embeddings.types import EmbeddingType
from xwhy.perturbation.text import TextPerturbation
from xwhy.surrogate.factory import SurrogateFactory
from xwhy.surrogate.trainer import SurrogateTrainer
from xwhy.surrogate.types import SurrogateType


class TextExplainer(BaseExplainer):
    """Explainer for natural language processing (NLP) text classification tasks."""

    def __init__(
        self,
        model: Any = None,  # noqa: ANN401
        predict_fn: Callable[..., Any] | None = None,
        config: ExplainerConfig | None = None,
        seed: int = 42,
        epsilon: float = 0.0,
        kernel_width: float = 0.5,
        ridge_alpha: float = 1.0,
        num_perturbations: int = 64,
        embedding_type: str | EmbeddingType = EmbeddingType.WORD2VEC,
        distance_type: str | DistanceType = DistanceType.WASSERSTEIN,
        surrogate_type: str | SurrogateType = SurrogateType.LIME,
        use_best_surrogate: bool = True,
        return_p_value: bool = False,
        n_bootstrap: int = 1000,
    ) -> None:
        """Initialize the text explainer.

        Args:
            model: Black-box model instance with predict_proba, predict, or __call__.
            predict_fn: Optional direct prediction function accepting list of texts.
            config: Optional configuration object for the explainer.
            seed: Random seed for reproducibility.
            epsilon: Numerical stability constant.
            kernel_width: Kernel width for similarity weights.
            ridge_alpha: Ridge regularization strength.
            num_perturbations: Default number of perturbed text samples to generate.
            embedding_type: Embedding method to extract text representations.
            distance_type: Metric used to compute distance between texts.
            surrogate_type: Default surrogate method to use if search is disabled.
            use_best_surrogate: If True, search for the best surrogate model.
            return_p_value: Whether to compute statistical significance
                (p-values) for computed distances using bootstrap sampling.
            n_bootstrap: Number of bootstrap iterations for p-value estimation.

        Raises:
            ValueError: If the embedding type is invalid for text explanation.

        """
        embedding_type = EmbeddingType.from_str(embedding_type)
        distance_type = DistanceType.from_str(distance_type)

        if not embedding_type.is_text_embedding:
            raise ValueError(
                f"Invalid embedding type '{embedding_type}' "
                "for TextExplainer. Must be a text embedding."
            )

        surrogate_type = SurrogateType.from_str(surrogate_type)

        self.state = TextState()

        if config is None:
            config = TextConfig(
                model=model,
                predict_fn=predict_fn,
                seed=seed,
                epsilon=epsilon,
                kernel_width=kernel_width,
                ridge_alpha=ridge_alpha,
                num_perturbations=num_perturbations,
                embedding_type=embedding_type,
                distance_type=distance_type,
                surrogate_type=surrogate_type,
                use_best_surrogate=use_best_surrogate,
                return_p_value=return_p_value,
                n_bootstrap=n_bootstrap,
            )

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
        self._initialize()

    @staticmethod
    def _resolve_predict_fn(
        model: Any = None,  # noqa: ANN401
        predict_fn: Callable[..., Any] | None = None,
    ) -> Callable[[Sequence[str]], np.ndarray]:
        """Resolve prediction callable from model instance or prediction function.

        Args:
            model: Black-box model instance with predict_proba, predict, or __call__.
            predict_fn: Direct prediction function accepting text inputs.

        Returns:
            Callable[[Sequence[str]], np.ndarray]: Standardized prediction callable.

        Raises:
            TypeError: If predict_fn is provided but not callable.
            ValueError: If neither model nor predict_fn is provided, or model lacks
                expected prediction methods.

        """
        if predict_fn is not None:
            if not callable(predict_fn):
                raise TypeError("Provided 'predict_fn' must be callable.")
            return cast(Callable[[Sequence[str]], np.ndarray], predict_fn)

        if model is not None:
            if hasattr(model, "predict_proba") and callable(model.predict_proba):
                return cast(
                    Callable[[Sequence[str]], np.ndarray],
                    model.predict_proba,
                )
            if hasattr(model, "predict") and callable(model.predict):
                return cast(Callable[[Sequence[str]], np.ndarray], model.predict)
            if callable(model):
                return cast(Callable[[Sequence[str]], np.ndarray], model)
            raise ValueError(
                "Provided model must be callable or possess a 'predict_proba' "
                "or 'predict' method."
            )

        raise ValueError("Either 'model' or 'predict_fn' must be provided.")

    def _initialize(self) -> None:
        """Initialize runtime state resources including embedding models."""
        if not self.config.embedding_type.is_text_embedding:  # type: ignore[union-attr]
            raise ValueError(
                "Invalid embedding type '%s' "
                "for TextExplainer. Must be a text embedding.",
                self.config.embedding_type,  # type: ignore[union-attr]
            )

        cfg_model = getattr(self.config, "model", None)
        cfg_predict_fn = getattr(self.config, "predict_fn", None)

        if cfg_model is not None or cfg_predict_fn is not None:
            self.state.model = cfg_model
            self.state.predict_fn = self._resolve_predict_fn(
                model=cfg_model, predict_fn=cfg_predict_fn
            )

        logger.info(
            "Loading text embedding model: %s",
            self.config.embedding_type,  # type: ignore[union-attr]
        )
        embedding_factory_result = EmbeddingFactory.create(
            embedding=self.config.embedding_type,  # type: ignore[union-attr]
        )
        self.state.embedding_model = embedding_factory_result.load()
        self.state.embedding_model.fill_norms(force=True)  # type: ignore[union-attr]

        logger.info("Initializing text perturbator...")
        self.state.perturbator = TextPerturbation(
            seed=self.config.seed  # type: ignore[union-attr]
        )

    def explain(
        self,
        instance: str,
        model: Any = None,  # noqa: ANN401
        predict_fn: Callable[..., Any] | None = None,
        class_index: int = 1,
        num_perturbations: int | None = None,
        fidelity_plot: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> TextXWhyResult:
        """Generate a feature attribution explanation for a text classification model.

        Args:
            instance: Input text string to be explained.
            model: Optional black-box model instance overriding initialized model.
            predict_fn: Optional prediction function overriding initialized predict_fn.
            class_index: Target output class index to explain.
            num_perturbations: Number of perturbed samples to generate.
            fidelity_plot: If True, renders and displays a fidelity plot.
            **kwargs: Additional runtime parameters.

        Returns:
            TextXWhyResult: Container holding explanation attributions and metrics.

        Raises:
            TypeError: If instance is not a string.
            ValueError: If no prediction method or model is available.
            RuntimeError: If embedding model or perturbator state is not initialized.

        """
        if not isinstance(instance, str):
            raise TypeError("TextExplainer requires the input text as a string.")

        if model is not None or predict_fn is not None:
            active_predict_fn = self._resolve_predict_fn(
                model=model, predict_fn=predict_fn
            )
        elif self.state.predict_fn is not None:
            active_predict_fn = self.state.predict_fn
        else:
            raise ValueError(
                "No prediction model or predict_fn was provided. Pass a model or "
                "predict_fn during TextExplainer initialization or to explain()."
            )

        effective_num_perturbations = (
            num_perturbations
            if num_perturbations is not None
            else self.config.num_perturbations  # type: ignore[union-attr]
        )

        logger.info("Generating perturbations...")
        if self.state.perturbator is None:
            raise RuntimeError("TextPerturbation state is not initialized.")

        perturbed_texts, binary_masks = self.state.perturbator.generate(
            text=instance,
            num_perturbations=effective_num_perturbations,
        )

        logger.info("Querying black-box model...")
        predictions = active_predict_fn(perturbed_texts)

        # Target variable (y) is the probability of the chosen class
        predictions_arr = np.array(predictions)
        if predictions_arr.ndim == 1:
            y_target = predictions_arr
        else:
            y_target = predictions_arr[:, class_index]

        logger.info(
            "Computing %s distances in the INPUT space...",
            self.config.distance_type,  # type: ignore[union-attr]
        )

        if self.state.embedding_model is None:
            raise RuntimeError("Embedding model state is not initialized.")

        base_text_representation = self.state.embedding_model.encode(instance)
        text_distances: list[tuple[str, float]] = []
        p_values: list[float] = []

        return_p_val = getattr(self.config, "return_p_value", False)
        n_bootstrap = getattr(self.config, "n_bootstrap", 1000)

        for text in perturbed_texts:
            current_text_representation = self.state.embedding_model.encode(text)

            if (
                base_text_representation.size == 0  # type: ignore[attr-defined]
                or current_text_representation.size == 0  # type: ignore[attr-defined]
            ):
                text_distances.append((text, 1.0))
                if return_p_val:
                    p_values.append(float("nan"))
                continue

            res = calculate_distance(
                metric=self.config.distance_type,  # type: ignore[union-attr]
                source=base_text_representation,
                target=current_text_representation,
                mode="spatial",
                return_p_value=return_p_val,
                n_bootstrap=n_bootstrap,
            )

            if isinstance(res, tuple):
                p_val, dist_val = res
                p_values.append(p_val)
                text_distances.append((text, float(dist_val)))
            else:
                text_distances.append((text, float(res)))

        # ---------------------------------------------------------
        # Distance Validation & Filtering setup:
        # Convert distances to numpy array and drop non-finite (inf/NaN) values.
        # ---------------------------------------------------------
        logger.info("Validating perturbation distances...")
        distances_raw = np.array([d for _, d in text_distances], dtype=float)

        # Identify valid (non-infinite, non-NaN) distances
        valid_mask = np.isfinite(distances_raw)
        valid_count = int(np.sum(valid_mask))
        total_count = len(distances_raw)
        valid_ratio = valid_count / total_count

        # Check validity threshold and warn if insufficient
        min_valid_ratio = self.config.min_valid_ratio  # type: ignore[union-attr]

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

        # Filter arrays and lists to drop failed evaluations cleanly
        valid_indices = np.where(valid_mask)[0]
        distances_valid = distances_raw[valid_mask]
        valid_text_distances = [text_distances[i] for i in valid_indices]

        masks_as_arrays: list[np.ndarray] = [
            np.array(binary_masks[i], dtype=int) for i in valid_indices
        ]
        x_valid = np.vstack(masks_as_arrays)
        y_valid = (
            y_target[valid_mask]
            if isinstance(y_target, np.ndarray)
            else np.array(y_target)[valid_mask]
        )

        # Surrogate selection and weight calculation
        if self.config.use_best_surrogate:  # type: ignore[union-attr]
            logger.info(
                "Searching for the optimal surrogate model among available "
                "candidates..."
            )
            method, score = SurrogateTrainer.find_best(
                x=x_valid,
                y=y_valid,
                distances=distances_valid,
                seed=self.config.seed,  # type: ignore[union-attr]
                epsilon=self.config.epsilon,  # type: ignore[union-attr]
                kernel_width=self.config.kernel_width,  # type: ignore[union-attr]
                ridge_alpha=self.config.ridge_alpha,  # type: ignore[union-attr]
                normalize_distances=False,
            )
            logger.info(
                "Optimization complete. Selected surrogate model: "
                "'%s' (Best Score: %.4f)",
                method.value,
                score,
            )
        else:
            method = self.config.surrogate_type  # type: ignore[assignment, union-attr]
            logger.info(
                "Skipping surrogate search. Using configured default: '%s'",
                method.value,
            )

        weights = SurrogateTrainer.compute_weights(
            method=method,
            distances=distances_valid,
            kernel_width=self.config.kernel_width,  # type: ignore[union-attr]
            epsilon=self.config.epsilon,  # type: ignore[union-attr]
            normalize_distances=False,
        )

        # Fit surrogate model
        surrogate = SurrogateFactory.create(
            method=method,
            seed=self.config.seed,  # type: ignore[union-attr]
        )
        surrogate.fit(x_valid, y_valid, weights)

        coeffs = surrogate.coefficients()
        y_pred_valid = surrogate.predict(x_valid)

        logger.info("Computing regression metrics...")
        metrics = RegressionMetrics.calculate(
            y_true=y_valid,
            y_pred=y_pred_valid,
            weights=weights,
            num_features=len(coeffs),
        )

        raw_data: dict[str, Any] = {
            "instance": instance,
            "perturbed_texts": perturbed_texts,
            "binary_masks": binary_masks,
            "text_distances": valid_text_distances,
            "distances": distances_valid,
            "weights": weights,
            "y_target": y_valid,
            "y_pred": y_pred_valid,
            "class_index": class_index,
        }

        if return_p_val and p_values:
            p_values_raw = np.array(p_values, dtype=float)
            raw_data["p_values"] = p_values_raw[valid_mask]

        if self.config.use_best_surrogate:  # type: ignore[union-attr]
            raw_data["best_surrogate_method"] = method
        else:
            raw_data["surrogate_method"] = method

        result = TextXWhyResult(
            original_output=(str(predictions[0]) if len(predictions) > 0 else ""),
            words=instance.split(),
            coefficients=coeffs,
            metrics=metrics,
            raw_data=raw_data,
        )

        if fidelity_plot:
            logger.info("Rendering fidelity plot as requested...")
            result.plot(show=True)

        return result
