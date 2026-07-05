"""OpenAI provider implementation."""

from openai import OpenAI

from xwhy.logger import logger
from xwhy.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI implementation of the provider interface."""

    def __init__(self, client: OpenAI) -> None:
        """Initialize the provider.

        Args:
            client: Configured OpenAI client.

        """
        self._client = client

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Return whether the model uses the Responses API.

        Args:
            model: OpenAI model name.

        Returns:
            ``True`` if the model is a reasoning model.

        """
        return model.startswith(("o1", "o3", "o4", "gpt-5"))

    def _generate(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate text from OpenAI.

        Args:
            prompt: Input prompt.
            model: OpenAI model.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            Generated text.

        Raises:
            No exception is propagated. Returns an empty string if an
            unexpected error occurs.

        """
        try:
            if self._is_reasoning_model(model):
                response = self._client.responses.create(
                    model=model,
                    input=prompt,
                    max_output_tokens=max_tokens,
                    reasoning={"effort": "low"},
                )

                return response.output_text.strip()

            response = self._client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return response.choices[0].text.strip()

        except Exception as exc:
            logger.error("OpenAI request failed: %s", exc)
            return ""

    def answer(
        self,
        prompt: str,
        *,
        model: str = "gpt-3.5-turbo-instruct",
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> str:
        """Generate a natural-language answer.

        Args:
            prompt: Input prompt.
            model: OpenAI model name.
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

    def score(
        self,
        prompt: str,
        *,
        model: str = "gpt-3.5-turbo-instruct",
        max_tokens: int = 10,
        temperature: float = 0.0,
    ) -> str:
        """Generate a numeric score.

        Args:
            prompt: Prompt requesting a score.
            model: OpenAI model name.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            Numeric score as text.

            Returns ``"0"`` if generation fails.

        """
        result = self._generate(
            prompt=(f"{prompt}\nRespond with a single number only. No explanation."),
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return result if result else "0"
