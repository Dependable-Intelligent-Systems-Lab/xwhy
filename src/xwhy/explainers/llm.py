"""LLM explainer implementation."""

import numpy as np

from xwhy.core.config import ExplainerConfig
from xwhy.core.explainer import BaseExplainer
from xwhy.core.pipeline import ExplanationPipeline
from xwhy.core.result import TextXWhyResult
from xwhy.distance.normalization import DistanceNormalizer
from xwhy.distance.wmd import WMDDistance
from xwhy.embeddings.factory import EmbeddingFactory
from xwhy.embeddings.types import EmbeddingType
from xwhy.logger import logger
from xwhy.metrics.regression import RegressionMetrics
from xwhy.perturbation.text import TextPerturbation
from xwhy.providers.base import BaseProvider
from xwhy.providers.resolver import ProviderResolver
from xwhy.providers.types import ProviderType
from xwhy.providers.utils import score_perturbations
from xwhy.surrogate.factory import SurrogateFactory
from xwhy.surrogate.trainer import SurrogateTrainer
from xwhy.surrogate.types import SurrogateType


class LLMExplainer(ExplanationPipeline, BaseExplainer):
    """Explainer for LLM tasks integrating the full GSMILE pipeline."""

    def __init__(
        self,
        provider: str | ProviderType | BaseProvider,
        config: ExplainerConfig | None = None,
        use_best_surrogate: bool = True,
        default_surrogate: str | SurrogateType = SurrogateType.LIME,
    ) -> None:
        """Initialize the LLM explainer.

        Args:
            provider: The provider instance, an enum, or a string identifier
                      (e.g., "openai"). If a string or enum is passed, the
                      hidden factory resolves it automatically.
            config: Optional configuration for the explainer.
            use_best_surrogate: If True, search for the best surrogate model
                                automatically.
            default_surrogate: The default surrogate method to use if the search is
                               disabled.

        """
        # Resolve string/enum provider into a concrete instance using Hidden Factory
        resolved_provider = ProviderResolver.resolve(provider)
        super().__init__(resolved_provider, config)

        self.provider = resolved_provider
        self.use_best_surrogate = use_best_surrogate
        self.default_surrogate = SurrogateType.from_str(default_surrogate)

    def run(self, instance: object, **kwargs: object) -> TextXWhyResult:
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
        *,
        model_name: str = "gpt-3.5-turbo-instruct",
        max_tokens: int = 200,
        temperature: float = 0.0,
        seed: int = 1024,
        num_perturbations: int = 64,
        embedding_type: str | EmbeddingType = EmbeddingType.WORD2VEC,
        **kwargs: object,
    ) -> TextXWhyResult:
        """Generate an explanation for the given prompt.

        Args:
            instance: The input prompt to explain.
            model_name: The LLM model name.
            max_tokens: Max tokens for generation.
            temperature: Sampling temperature.
            seed: Random seed for reproducibility.
            num_perturbations: Number of perturbed samples to generate.
            embedding_type: Embedding method for WMD.
            **kwargs: Additional explainer-specific options.

        Returns:
            TextXWhyResult: The structured explanation result object
                containing visualization methods and evaluation metrics.

        """
        prompt = instance
        embedding_type = EmbeddingType.from_str(embedding_type)

        logger.info("Querying provider for original response...")
        original = self.provider.answer(
            prompt=prompt,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        logger.info("Generating perturbations...")
        text_perturbation = TextPerturbation(seed=seed)
        responses, perturbations = text_perturbation.generate(
            text=prompt, num_perturbations=num_perturbations
        )

        logger.info("Querying provider for perturbed responses...")
        gpt_pairs = score_perturbations(
            provider=self.provider,
            model=model_name,
            perturbations=responses,
            max_tokens=10,
            temperature=temperature,
        )

        logger.info("Loading embedding model...")
        embedding = EmbeddingFactory.create(embedding=embedding_type)
        embedding_model = embedding.load()
        embedding_model.fill_norms(force=True)

        logger.info("Computing WMD scores...")
        wmd_distance = WMDDistance()
        perturbed_texts = [pert_text for _, pert_text in gpt_pairs]
        wmd_scores = wmd_distance.compute_batch(
            model=embedding_model,
            original=original,
            perturbed_texts=perturbed_texts,
        )

        logger.info("Normalizing similarities...")
        sims = DistanceNormalizer.min_max(scores=wmd_scores)

        if self.use_best_surrogate:
            logger.info(
                "Searching for the optimal surrogate model among available"
                " candidates..."
            )
            method, score = SurrogateTrainer.find_best(
                perturbations=perturbations,
                similarities=sims,
                wmd_scores=wmd_scores,
                seed=seed,
            )
            logger.info(
                "Optimization complete. Selected surrogate model:"
                " '%s' (Best Score: %.4f)",
                method.value,
                score,
            )
        else:
            method = self.default_surrogate
            logger.info(
                "Skipping surrogate search. Using configured default: '%s'",
                method.value,
            )

        x_matrix = np.vstack(perturbations)
        y_target = np.array([s for _, s in sims])
        weights = SurrogateTrainer.compute_weights(method, wmd_scores)

        surrogate = SurrogateFactory.create(method=method, seed=seed)
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
            "gpt_pairs": gpt_pairs,
            "wmd_scores": wmd_scores,
            "similarities": sims,
            "weights": weights,
        }

        if self.use_best_surrogate:
            raw_data["best_surrogate_method"] = method
        else:
            raw_data["surrogate_method"] = method

        return TextXWhyResult(
            original_output=original,
            words=prompt.split(),
            coefficients=coeffs,
            metrics=metrics,
            raw_data=raw_data,
        )
