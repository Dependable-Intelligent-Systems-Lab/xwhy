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

        Raises:
            RuntimeError: If the API returns an empty response or is
                          blocked by safety filters.

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
                result_text = str(response.text).strip()
            except ValueError as val_err:
                # Gemini throws ValueError on .text access if the response was
                # blocked by safety filters.
                error_message = (
                    f"Gemini generation was blocked (likely due to safety filters) "
                    f"for model '{model}'. No content returned."
                )
                logger.error(error_message)
                raise RuntimeError(error_message) from val_err

            if not result_text:
                error_message = (
                    "Received an empty response from the Gemini API. "
                    "This could be due to network filtering (anti-filter) "
                    "or provider-side anomalies."
                )
                logger.error(error_message)
                raise RuntimeError(error_message)

            return result_text

        except RuntimeError:
            raise

        except Exception as exc:
            logger.error("Gemini request failed: %s", exc)
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

    def answer(
        self,
        prompt: str,
        *,
        model: str = "gemini-2.5-flash",
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
