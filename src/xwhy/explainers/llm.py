"""LLM explainer implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from xwhy.core.config import LLMConfig
from xwhy.core.explainer import BaseExplainer
from xwhy.core.pipeline import ExplanationPipeline
from xwhy.core.result import TextXWhyResult
from xwhy.core.types import LLMState
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


class LLMExplainer(ExplanationPipeline, BaseExplainer):
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
        seed: int = 1024,
        num_perturbations: int = 64,
        embedding_type: str | EmbeddingType = EmbeddingType.WORD2VEC,
        surrogate_type: str | SurrogateType = SurrogateType.LIME,
        use_best_surrogate: bool = True,
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
            seed: Random seed for reproducibility.
            num_perturbations: Number of perturbed samples to generate.
            embedding_type: Embedding method for WMD.
            surrogate_type: The default surrogate method to use if search is disabled.
            use_best_surrogate: If True, search for the best surrogate model
                automatically.
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
                seed=seed,
                num_perturbations=num_perturbations,
                embedding_type=embedding_type,
                surrogate_type=surrogate_type,
                use_best_surrogate=use_best_surrogate,
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

    def run(self, instance: Any, **kwargs: Any) -> TextXWhyResult:  # noqa: ANN401
        """Run the full explanation pipeline (ExplanationPipeline implementation).

        Args:
            instance: The input prompt string.
            **kwargs: Additional pipeline options.

        Returns:
            TextXWhyResult: The explanation outcome.

        Raises:
            TypeError: If the instance is not a string.

        """
        if not isinstance(instance, str):
            raise TypeError("LLMExplainer requires a string instance.")
        return self.explain(instance, **kwargs)

    def explain(
        self,
        instance: str,
        fidelity_plot: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> TextXWhyResult:
        """Generate an explanation for the given prompt.

        Args:
            instance: The input prompt to explain.
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
        )

        logger.info("Generating perturbations...")
        perturbed_texts, binary_masks = self.state.perturbator.generate(
            text=prompt,
            num_perturbations=self.config.num_perturbations,  # type: ignore[union-attr]
        )

        logger.info("Computing WMD scores...")
        wmd_distance = WMDDistance()
        wmd_scores = wmd_distance.compute_batch(
            model=self.state.embedding_model,
            original=original_output,
            perturbed_texts=perturbed_texts,
        )

        logger.info("Normalizing similarities...")
        sims = DistanceNormalizer.min_max(scores=wmd_scores)

        masks_as_arrays: list[np.ndarray] = [
            np.array(m, dtype=int) for m in binary_masks
        ]

        x_matrix = np.vstack(masks_as_arrays)
        y_target = np.array([s for _, s in sims])
        distances_array = np.array([d for _, d in wmd_scores])

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

        raw_data = {
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
