"""Anthropic provider implementation."""

import time
from typing import Any

from xwhy.logger import logger
from xwhy.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Anthropic implementation of the provider interface."""

    def __init__(self, client: object) -> None:
        """Initialize the provider.

        Args:
            client: Configured Anthropic client.

        """
        super().__init__(client)
        self._client = client

    def _generate(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,  # noqa: ANN401
    ) -> str:
        """Generate text from Anthropic with built-in retries.

        Args:
            prompt: Input prompt.
            model: Anthropic model name.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.
            **kwargs: Extra parameters (supports 'max_retries' and 'delay').

        Returns:
            Generated text string.

        Raises:
            RuntimeError: If the API returns an empty response or fails
                after all retries.

        """
        max_retries: int = kwargs.get("max_retries", 7)
        delay_override: float | None = kwargs.get("delay")

        for retry_number in range(1, max_retries + 1):
            try:
                create_kwargs = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                }
                if temperature is not None:
                    create_kwargs["temperature"] = temperature

                try:
                    response = self._client.messages.create(**create_kwargs)
                except Exception as inner_exc:
                    if (
                        "temperature" in str(inner_exc).lower()
                        and "deprecated" in str(inner_exc).lower()
                    ):
                        logger.info(
                            "Temperature is deprecated for this model. Retrying "
                            "without temperature..."
                        )
                        create_kwargs.pop("temperature", None)
                        response = self._client.messages.create(**create_kwargs)
                    else:
                        raise inner_exc

                # Anthropic returns a list of ContentBlock objects, which might include
                # ThinkingBlock objects. We extract text from all blocks that have a
                # text attribute.
                result_text = ""
                if response.content:
                    for block in response.content:
                        if hasattr(block, "text"):
                            result_text += str(block.text)
                    result_text = result_text.strip()

                if not result_text:
                    error_message = (
                        "Received an empty response from the Anthropic API. "
                        "This could be due to content moderation filters, "
                        "network filtering (anti-filter), or "
                        "provider-side anomalies."
                    )
                    logger.error(error_message)
                    raise RuntimeError(error_message)

                return result_text

            except RuntimeError:
                raise

            except Exception as exc:
                if retry_number == max_retries:
                    logger.error(
                        "Anthropic request failed after %d retries: %s",
                        max_retries,
                        exc,
                    )
                    raise RuntimeError(f"Anthropic request failed: {exc}") from exc

                delay: float = (
                    delay_override
                    if delay_override is not None
                    else min(2**retry_number, 30)
                )
                logger.warning(
                    "Retry %d/%d for Anthropic text generation (Error: %s). "
                    "Waiting %s seconds...",
                    retry_number,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError("Anthropic text generation failed after max retries.")

    def answer(
        self,
        prompt: str,
        *,
        model: str = "claude-opus-4-8",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,  # noqa: ANN401
    ) -> str:
        """Generate a natural-language answer.

        Args:
            prompt: Input prompt.
            model: Anthropic model name.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.
            **kwargs: Extra parameters (supports 'max_retries' and 'delay').

        Returns:
            Generated response text string.

        """
        return self._generate(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
