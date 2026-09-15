"""LLM explainer implementation."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from xwhy.core.config import LLMConfig
from xwhy.core.explainer import BaseExplainer
from xwhy.core.result import TextXWhyResult
from xwhy.core.states import LLMState
from xwhy.distance.normalization import DistanceNormalizer
from xwhy.distance.wmd import WMDDistance
from xwhy.logger import logger
from xwhy.metrics.regression import RegressionMetrics
from xwhy.models.embeddings.factory import EmbeddingFactory
from xwhy.models.embeddings.types import EmbeddingType
from xwhy.perturbation.text import TextPerturbation
from xwhy.providers.base import BaseProvider
from xwhy.providers.resolver import ProviderResolver
from xwhy.providers.types import ProviderType
from xwhy.surrogate.factory import SurrogateFactory
from xwhy.surrogate.trainer import SurrogateTrainer
from xwhy.surrogate.types import SurrogateType


class LLMExplainer(BaseExplainer):
    """Explainer for LLM tasks integrating the full GSMILE pipeline.

    This explainer loads all required runtime resources only once and can
    explain multiple text prompts throughout its lifetime.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        provider: str | ProviderType | BaseProvider | None = None,
        model_name: str = "gpt-3.5-turbo-instruct",
        max_tokens: int = 200,
        temperature: float = 0.0,
        max_retries: int = 7,
        delay: float | None = None,
        seed: int = 42,
        epsilon: float = 0.0,
        kernel_width: float = 0.5,
        ridge_alpha: float = 1.0,
        normalization_method: Literal["linear", "inverse"] = "linear",
        num_perturbations: int = 64,
        min_valid_ratio: float = 0.5,
        embedding_type: str | EmbeddingType = EmbeddingType.WORD2VEC,
        surrogate_type: str | SurrogateType = SurrogateType.LIME,
        use_best_surrogate: bool = True,
        sanitize_distances: bool = False,
        **provider_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the LLM explainer.

        Args:
            config: Optional configuration for the explainer.
            provider: The provider instance, an enum, or a string identifier
                (e.g., "openai"). If a string or enum is passed, the
                factory resolves it automatically.
            model_name: The LLM model name.
            max_tokens: Max tokens for generation.
            temperature: Sampling temperature.
            max_retries : Maximum number of retry attempts if the LLM/VLM request
                fails.
            delay : Seconds to wait between consecutive retries.
            seed: Random seed for reproducibility.
            epsilon: Numerical stability constant.
            kernel_width: Kernel width for similarity weights.
            ridge_alpha: Ridge regularization strength.
            normalization_method : Method used to normalize text similarities.
            num_perturbations: Number of perturbed samples to generate.
            min_valid_ratio: Minimum proportion of valid (non-NaN/non-infinite)
                perturbation evaluations required for reliable surrogate training.
            embedding_type: Embedding method for WMD.
            surrogate_type: The default surrogate method to use if search is disabled.
            use_best_surrogate: If True, search for the best surrogate model
                automatically.
            sanitize_distances: If True, applies sanitize_distances to clean non-finite
                values.
            **provider_kwargs: Additional provider-specific options.

        Raises:
            ValueError: If the embedding type is invalid for LLM explanation.

        """
        embedding_type = EmbeddingType.from_str(embedding_type)

        if not embedding_type.is_text_embedding:
            raise ValueError(
                f"Invalid embedding type '{embedding_type}' "
                "for LLMExplainer. Must be a text embedding."
            )

        surrogate_type = SurrogateType.from_str(surrogate_type)

        self._provider_kwargs = provider_kwargs
        self.state = LLMState()

        provider_type = ProviderType.OPENAI

        if provider is not None:
            if isinstance(provider, BaseProvider):
                self.state.provider = provider
                class_name = provider.__class__.__name__.lower().replace("provider", "")
                try:
                    provider_type = ProviderType.from_str(class_name)
                except ValueError:
                    logger.warning(
                        f"Custom provider class '{provider.__class__.__name__}' mapped "
                        "to default config type."
                    )
                    provider_type = ProviderType.OPENAI

            elif isinstance(provider, ProviderType):
                provider_type = provider
            else:
                provider_type = ProviderType.from_str(str(provider))

        if config is None:
            config = LLMConfig(
                provider_type=provider_type,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                max_retries=max_retries,
                delay=delay,
                seed=seed,
                epsilon=epsilon,
                kernel_width=kernel_width,
                ridge_alpha=ridge_alpha,
                normalization_method=normalization_method,
                num_perturbations=num_perturbations,
                min_valid_ratio=min_valid_ratio,
                embedding_type=embedding_type,
                surrogate_type=surrogate_type,
                use_best_surrogate=use_best_surrogate,
                sanitize_distances=sanitize_distances,
            )

        super().__init__(config)
        self._initialize()

    def _initialize(self) -> None:
        """Initialize runtime resources."""
        if self.state.provider is None:
            logger.info(
                f"Resolving provider type: {self.config.provider_type}"  # type: ignore[union-attr]
            )
            self.state.provider = ProviderResolver.resolve(
                self.config.provider_type,  # type: ignore[union-attr]
                **self._provider_kwargs,
            )

        if not self.config.embedding_type.is_text_embedding:  # type: ignore[union-attr]
            raise ValueError(
                "Invalid embedding type '%s' "
                "for ImageGenerationAndEditingExplainer. Must be a text embedding.",
                self.config.embedding_type,  # type: ignore[union-attr]
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
        normalization_method: Literal["linear", "inverse"] | None = None,
        fidelity_plot: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> TextXWhyResult:
        """Generate an explanation for the given prompt.

        Args:
            instance: The input prompt to explain.
            normalization_method : Method used to normalize text similarities.
            fidelity_plot: Rendering fidelity scatter plot.
            **kwargs: Additional explainer-specific options.

        Returns:
            TextXWhyResult: The structured explanation result object
                containing visualization methods and evaluation metrics.

        Raises:
            TypeError: If instance is not a string.
            RuntimeError: If runtime resources are not initialized.

        """
        if not isinstance(instance, str):
            raise TypeError("LLMExplainer requires the input prompt as a string.")

        kwargs["max_retries"] = (
            self.config.max_retries  # type: ignore[union-attr]
            if kwargs.get("max_retries") is None
            else kwargs["max_retries"]
        )
        kwargs["delay"] = (
            self.config.delay if kwargs.get("delay") is None else kwargs["delay"]  # type: ignore[union-attr]
        )

        normalization_method = (
            self.config.normalization_method  # type: ignore[union-attr]
            if normalization_method is None
            else normalization_method
        )

        if (
            self.state.provider is None
            or self.state.embedding_model is None
            or self.state.perturbator is None
        ):
            raise RuntimeError("LLMExplainer runtime resources are not initialized.")

        prompt = instance

        logger.info("Querying provider for original response...")
        original_output = self.state.provider.answer(
            prompt=prompt,
            model=self.config.model_name,  # type: ignore[union-attr]
            max_tokens=self.config.max_tokens,  # type: ignore[union-attr]
            temperature=self.config.temperature,  # type: ignore[union-attr]
            **kwargs,
        )

        logger.info("Generating perturbations...")
        perturbed_texts, binary_masks = self.state.perturbator.generate(
            text=prompt,
            num_perturbations=self.config.num_perturbations,  # type: ignore[union-attr]
        )

        logger.info("Computing WMD scores...")
        wmd_distance = WMDDistance()
        raw_wmd_scores = wmd_distance.compute_batch(
            model=self.state.embedding_model,
            original=original_output,
            perturbed_texts=perturbed_texts,
            sanitize=self.config.sanitize_distances,  # type: ignore[union-attr]
        )

        # ---------------------------------------------------------
        # Distance Validation & Filtering setup:
        # Convert distances to numpy array and drop non-finite (inf/NaN) values.
        # ---------------------------------------------------------
        logger.info("Validating perturbation distances...")
        distances_raw = np.array([d for _, d in raw_wmd_scores], dtype=float)

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
        # Use valid_indices to filter standard Python lists (raw_wmd_scores,
        # binary_masks)
        valid_indices = np.where(valid_mask)[0]

        valid_wmd_scores = [raw_wmd_scores[i] for i in valid_indices]
        distances_valid = distances_raw[valid_mask]

        logger.info("Normalizing similarities...")
        sims = DistanceNormalizer.min_max(
            scores=valid_wmd_scores,
            mode=normalization_method,
        )

        # Build feature matrix and target array using only valid inputs
        masks_as_arrays: list[np.ndarray] = [
            np.array(binary_masks[i], dtype=int) for i in valid_indices
        ]

        x_valid = np.vstack(masks_as_arrays)
        y_valid = np.array([s for _, s in sims])

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

        # Compute weights using ONLY the valid distances
        weights = SurrogateTrainer.compute_weights(
            method=method,
            distances=distances_valid,
            kernel_width=self.config.kernel_width,  # type: ignore[union-attr]
            epsilon=self.config.epsilon,  # type: ignore[union-attr]
            normalize_distances=False,
        )

        surrogate = SurrogateFactory.create(
            method=method,
            seed=self.config.seed,  # type: ignore[union-attr]
        )

        # Fit the surrogate using strictly valid data
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

        raw_data = {
            "perturbed_texts": perturbed_texts,
            "wmd_scores": valid_wmd_scores,
            "similarities": sims,
            "weights": weights,
            "y_target": y_valid,
            "y_pred": y_pred_valid,
        }

        if self.config.use_best_surrogate:  # type: ignore[union-attr]
            raw_data["best_surrogate_method"] = method
        else:
            raw_data["surrogate_method"] = method

        result = TextXWhyResult(
            original_output=original_output,
            words=prompt.split(),
            coefficients=coeffs,
            metrics=metrics,
            raw_data=raw_data,
        )

        if fidelity_plot:
            logger.info("Rendering fidelity plot as requested...")
            result.plot(show=True)

        return result
