"""Gemini provider implementation."""

from google.genai import types

from xwhy.logger import logger
from xwhy.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    """Gemini implementation of the provider interface."""

    def __init__(self, client: object) -> None:
        """Initialize the provider.

        Args:
            client: Configured Gemini client (typically the generativeai module).

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
        """Generate text from Gemini.

        Args:
            prompt: Input prompt.
            model: Gemini model name.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            Generated text.

        """
        try:
            response = self._client.models.generate_content(  # type: ignore[attr-defined]
                model=model,
                contents=types.Part.from_text(text=prompt),
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )

            try:
                return str(response.text).strip()
            except ValueError:
                # Gemini throws ValueError on .text access if the response was
                # blocked by safety filters.
                logger.warning(
                    "Gemini generation was blocked (likely due to safety filters) "
                    "for model '%s'. Returning empty string.",
                    model,
                )
                return ""

        except Exception as exc:
            logger.error("Gemini request failed: %s", exc)
            return ""

    def answer(
        self,
        prompt: str,
        *,
        model: str = "gemini-1.5-flash",
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> str:
        """Generate a natural-language answer.

        Args:
            prompt: Input prompt.
            model: Gemini model name.
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
