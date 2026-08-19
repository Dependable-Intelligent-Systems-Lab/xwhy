"""Text explainer."""

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np

from xwhy.core.config import ExplainerConfig, TextConfig
from xwhy.core.explainer import BaseExplainer
from xwhy.core.result import TextXWhyResult
from xwhy.core.types import TextState
from xwhy.distance.wmd import WMDDistance
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
        num_perturbations: int = 64,
        embedding_type: str | EmbeddingType = EmbeddingType.WORD2VEC,
        surrogate_type: str | SurrogateType = SurrogateType.LIME,
        use_best_surrogate: bool = True,
    ) -> None:
        """Initialize the text explainer.

        Args:
            model: Black-box model instance with predict_proba, predict, or __call__.
            predict_fn: Optional direct prediction function accepting list of texts.
            config: Optional configuration object for the explainer.
            seed: Random seed for reproducibility.
            num_perturbations: Default number of perturbed text samples to generate.
            embedding_type: Embedding method used for Word Mover's Distance.
            surrogate_type: Default surrogate method to use if search is disabled.
            use_best_surrogate: If True, search for the best surrogate model.

        Raises:
            ValueError: If the embedding type is invalid for text explanation.

        """
        embedding_type = EmbeddingType.from_str(embedding_type)

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
                num_perturbations=num_perturbations,
                embedding_type=embedding_type,
                surrogate_type=surrogate_type,
                use_best_surrogate=use_best_surrogate,
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

        logger.info("Computing WMD scores in the INPUT space...")
        wmd_distance = WMDDistance()

        if self.state.embedding_model is None:
            raise RuntimeError("Embedding model state is not initialized.")

        raw_wmd_scores = wmd_distance.compute_batch(
            model=self.state.embedding_model,
            original=instance,
            perturbed_texts=perturbed_texts,
            sanitize=True,
        )

        # ---------------------------------------------------------
        # Distance Validation & Imputation setup:
        # Convert distances to numpy array and impute non-finite (inf/NaN) values.
        # ---------------------------------------------------------
        logger.info("Validating perturbation distances...")
        distances_raw = np.array([d for _, d in raw_wmd_scores], dtype=float)

        # Filter out non-finite values to determine the maximum valid distance
        valid_distances = distances_raw[np.isfinite(distances_raw)]

        # Calculate max_penalty: max valid distance + 1000, or default 1000 if
        # all failed
        if len(valid_distances) > 0:
            max_penalty = np.max(valid_distances) + 1000.0
        else:
            max_penalty = 1000.0

        # Impute infinite/NaN values with the dynamically calculated maximum penalty
        distances_array = np.where(
            np.isfinite(distances_raw), distances_raw, max_penalty
        )

        # Reconstruct wmd_scores with imputed values for downstream consistency
        wmd_scores = [
            (text, float(dist))
            for (text, _), dist in zip(raw_wmd_scores, distances_array, strict=False)
        ]

        masks_as_arrays: list[np.ndarray] = [
            np.array(m, dtype=int) for m in binary_masks
        ]
        x_matrix = np.vstack(masks_as_arrays)

        if self.config.use_best_surrogate:  # type: ignore[union-attr]
            logger.info(
                "Searching for the optimal surrogate model among available"
                " candidates..."
            )
            method, score = SurrogateTrainer.find_best(
                x=x_matrix,
                y=y_target,
                distances=distances_array,
                seed=self.config.seed,  # type: ignore[union-attr]
            )
            logger.info(
                "Optimization complete. Selected surrogate model:"
                " '%s' (Best Score: %.4f)",
                method.value,
                score,
            )
        else:
            method = self.config.surrogate_type  # type: ignore[union-attr]
            logger.info(
                "Skipping surrogate search. Using configured default: '%s'",
                method.value,
            )

        weights = SurrogateTrainer.compute_weights(method, distances_array)

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

        raw_data: dict[str, Any] = {
            "instance": instance,
            "perturbed_texts": perturbed_texts,
            "binary_masks": binary_masks,
            "wmd_scores": wmd_scores,
            "distances": distances_array,
            "weights": weights,
            "y_target": y_target,
            "y_pred": y_pred,
            "class_index": class_index,
        }

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
