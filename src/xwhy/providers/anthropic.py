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
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )

                # Anthropic returns a list of ContentBlock objects. We extract
                # the text from the first block if it exists to avoid IndexError.
                result_text = ""
                if response.content:
                    result_text = str(response.content[0].text).strip()

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
                    "Retry %d/%d for Anthropic text generation. Waiting %s seconds...",
                    retry_number,
                    max_retries,
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
