"""OpenAI provider implementation."""

import re

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
        super().__init__(client)
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

        """
        try:
            if self._is_reasoning_model(model):
                reasoning_response = self._client.responses.create(
                    model=model,
                    input=prompt,
                    max_output_tokens=max_tokens,
                    reasoning={"effort": "low"},
                    temperature=temperature,
                )
                return str(reasoning_response.output_text).strip()

            completion_response = self._client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return str(completion_response.choices[0].text).strip()

        except Exception as exc:
            error_msg = str(exc).lower()

            if "temperature" in error_msg and (
                "support" in error_msg or "value" in error_msg or "allowed" in error_msg
            ):
                logger.warning(
                    "Dynamic fix applied: temperature=%f is not supported for model "
                    "'%s'. Retrying automatically with default temperature (1.0).",
                    temperature,
                    model,
                )
                if temperature != 1.0:
                    return self._generate(
                        prompt=prompt,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=1.0,
                    )

            if (
                "max_output_tokens" in error_msg
                and "integer below minimum value" in error_msg
            ):
                match = re.search(r"expected a value >= (\d+)", error_msg)

                if match:
                    required_min = int(match.group(1))
                    logger.warning(
                        "Dynamic fix applied: max_tokens=%d is too low for model '%s'. "
                        "Retrying automatically with required minimum: %d.",
                        max_tokens,
                        model,
                        required_min,
                    )

                    return self._generate(
                        prompt=prompt,
                        model=model,
                        max_tokens=required_min,
                        temperature=temperature,
                    )

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
