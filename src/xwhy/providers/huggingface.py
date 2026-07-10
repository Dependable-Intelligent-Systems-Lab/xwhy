"""HuggingFace provider implementation."""

from xwhy.logger import logger
from xwhy.providers.base import BaseProvider


class HuggingFaceProvider(BaseProvider):
    """HuggingFace implementation of the provider interface."""

    def __init__(self, client: object) -> None:
        """Initialize the provider.

        Args:
            client: Configured HuggingFace InferenceClient.

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
        """Generate text from HuggingFace.

        Args:
            prompt: Input prompt.
            model: HuggingFace model name.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            Generated text.

        """
        try:
            # InferenceClient's chat.completions API is very similar to OpenAI's
            response = self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return str(response.choices[0].message.content).strip()

        except Exception as exc:
            logger.error("HuggingFace request failed: %s", exc)
            return ""

    def answer(
        self,
        prompt: str,
        *,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """Generate a natural-language answer.

        Args:
            prompt: Input prompt.
            model: HuggingFace model name.
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
