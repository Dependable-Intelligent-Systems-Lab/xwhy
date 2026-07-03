"""Base abstractions for external providers."""

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Abstract interface for external AI providers.

    A provider is responsible for communicating with an external service
    (for example OpenAI, Gemini or Hugging Face) and returning generated
    text responses.

    Concrete implementations must implement both ``answer`` and ``score``.
    """

    @abstractmethod
    def answer(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate a natural-language response.

        Args:
            prompt: Input prompt.
            model: Provider model identifier.
            max_tokens: Maximum number of generated tokens.
            temperature: Sampling temperature.

        Returns:
            Generated text.

        Raises:
            NotImplementedError: Implemented by subclasses.

        """
        raise NotImplementedError

    @abstractmethod
    def score(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate a numeric score.

        Implementations should instruct the provider to return only a
        numeric value.

        Args:
            prompt: Input prompt.
            model: Provider model identifier.
            max_tokens: Maximum number of generated tokens.
            temperature: Sampling temperature.

        Returns:
            Numeric score represented as a string.

        Raises:
            NotImplementedError: Implemented by subclasses.

        """
        raise NotImplementedError
