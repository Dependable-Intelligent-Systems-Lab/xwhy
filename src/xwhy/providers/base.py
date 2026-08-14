# src/xwhy/providers/base.py
"""Base abstractions for external providers."""

import os
import time
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image, ImageDraw, ImageFont


class BaseProvider(ABC):
    """Abstract interface for external AI providers.

    A provider is responsible for communicating with an external service
    (for example OpenAI, Gemini or Hugging Face) and returning generated
    text or image responses.

    Concrete implementations must implement both ``answer`` and ``score``.
    """

    def __init__(self, client: Any) -> None:  # noqa: ANN401
        """Initialize the provider with a client.

        Args:
            client: The initialized provider client (e.g., OpenAI client,
                HuggingFace client).

        """
        self._client = client

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

    def _create_placeholder_image(
        self,
        prompt: str,
        output_dir: str,
        filename_prefix: str = "generated",
        save: bool = True,
    ) -> str | Image.Image:
        """Create a black placeholder image with error text using Pillow.

        Args:
            prompt: The prompt that caused the generation failure.
            output_dir: The directory to save the image to.
            filename_prefix: Prefix string for the generated file name.
            save: Flag indicating whether to save the image to disk.

        Returns:
            The file path as a string if save is True, otherwise the raw
            Pillow Image object.

        """
        img_size = (600, 600)
        black_image = Image.new("RGB", img_size, color="black")
        draw = ImageDraw.Draw(black_image)

        text = f"image wasn't generated because of\n\"{prompt}\"\ndoesn't mean"

        font: Any
        try:
            font = ImageFont.truetype("arialbd.ttf", 40)
        except OSError:
            font = ImageFont.load_default()

        # Calculate text position to center it
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (img_size[0] - text_width) / 2
        y = (img_size[1] - text_height) / 2

        draw.text((x, y), text, fill="white", font=font, align="center")

        if save:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = int(time.time() * 1000)
            filename = f"{filename_prefix}_{timestamp}.png"
            gen_path = os.path.join(output_dir, filename)
            black_image.save(gen_path)
            return gen_path

        return black_image
