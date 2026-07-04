"""Utility functions for provider operations."""

from collections.abc import Sequence

from xwhy.logger import logger
from xwhy.providers.base import BaseProvider


def score_perturbations(
    provider: BaseProvider,
    *,
    model: str,
    perturbations: Sequence[str],
    max_tokens: int = 10,
    temperature: float = 0.0,
) -> list[tuple[str, str]]:
    """Query a provider for each perturbed text and return input/output pairs.

    This function is provider-agnostic. It relies on the BaseProvider
    abstraction, allowing it to seamlessly work with OpenAI, Gemini,
    Hugging Face, or any future provider implementations.

    Args:
        provider: An instantiated provider extending BaseProvider.
        model: The identifier of the AI model to use.
        perturbations: Sequence of perturbed text inputs.
        max_tokens: Maximum number of tokens for the provider response.
        temperature: Sampling temperature for generation.

    Returns:
        list[tuple[str, str]]: List of (perturbed_input, provider_output) pairs.

    """
    outputs: list[tuple[str, str]] = []

    for text in perturbations:
        logger.debug("Querying provider with perturbed text: %s", text)

        resp = provider.score(
            prompt=text,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        outputs.append((text, resp))

    for i, (inp, out) in enumerate(outputs, 1):
        logger.debug("Perturbation %d:", i)
        logger.debug("Input: %s", inp)
        logger.debug("Output: %s", out)
        logger.debug("-" * 50)

    return outputs
