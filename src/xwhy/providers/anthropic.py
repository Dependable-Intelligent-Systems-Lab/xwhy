"""Anthropic provider implementation."""

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
    ) -> str:
        """Generate text from Anthropic.

        Args:
            prompt: Input prompt.
            model: Anthropic model name.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            Generated text.

        Raises:
            RuntimeError: If the API returns an empty response.

        """
        try:
            response = self._client.messages.create(  # type: ignore[attr-defined]
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

            # Anthropic returns a list of ContentBlock objects. We extract the text
            # from the first block if it exists to avoid IndexError.
            result_text = ""
            if response.content:
                result_text = str(response.content[0].text).strip()

            if not result_text:
                error_message = (
                    "Received an empty response from the Anthropic API. "
                    "This could be due to content moderation filters, network"
                    " filtering (anti-filter), or provider-side anomalies."
                )
                logger.error(error_message)
                raise RuntimeError(error_message)

            return result_text

        except RuntimeError:
            raise

        except Exception as exc:
            logger.error("Anthropic request failed: %s", exc)
            raise RuntimeError(f"Anthropic request failed: {exc}") from exc

    def answer(
        self,
        prompt: str,
        *,
        model: str = "claude-opus-4-8",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Generate a natural-language answer.

        Args:
            prompt: Input prompt.
            model: Anthropic model name.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            Generated response text.

        """
        return self._generate(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
