"""Type aliases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseImageGenerationAndEditing(ABC):
    """Abstract base class for all image generation and editing engines.

    Any cloud provider (e.g., OpenAI, Gemini) or custom local model
    used for image generation/editing must inherit from this class
    and implement its methods.
    """

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        output_dir: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Generate an image from a text prompt.

        Args:
            prompt: The text prompt describing the desired image.
            output_dir: Directory to save the generated image.
            **kwargs: Additional parameters specific to the underlying model/API.

        Returns:
            A tuple containing a boolean success flag and the path to the
            generated image (or error message if failed).

        Raises:
            NotImplementedError: Implemented by subclasses.

        """
        raise NotImplementedError

    @abstractmethod
    def edit_image(
        self,
        prompt: str,
        image_path: str,
        output_dir: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Edit an existing image based on a text prompt.

        Args:
            prompt: The text prompt describing the desired edits.
            image_path: Path to the original input image.
            output_dir: Directory to save the edited image.
            **kwargs: Additional parameters specific to the underlying model/API.

        Returns:
            A tuple containing a boolean success flag and the path to the
            edited image (or error message if failed).

        Raises:
            NotImplementedError: Implemented by subclasses.

        """
        raise NotImplementedError
