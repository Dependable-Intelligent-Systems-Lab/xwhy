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
            # from the first block.
            return str(response.content[0].text).strip()

        except Exception as exc:
            logger.error("Anthropic request failed: %s", exc)
            return ""

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
