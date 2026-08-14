"""Unit tests for BaseProvider helper methods in base.py."""

from typing import Any
from unittest.mock import patch

from PIL import Image, ImageFont

from xwhy.providers.base import BaseProvider


class ConcreteProvider(BaseProvider):
    """Concrete implementation of BaseProvider for testing."""

    def answer(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Return dummy answer."""
        return "test response"


def test_create_placeholder_image_save_true(tmp_path: Any) -> None:  # noqa: ANN401
    """Save placeholder image to disk and return file path."""
    provider = ConcreteProvider(client=None)
    output_dir = str(tmp_path / "outputs")

    result_path = provider._create_placeholder_image(
        prompt="Test error prompt",
        output_dir=output_dir,
        filename_prefix="test_prefix",
        save=True,
    )

    assert isinstance(result_path, str)
    assert "test_prefix_" in result_path
    assert result_path.endswith(".png")


def test_create_placeholder_image_save_false() -> None:
    """Return raw Pillow Image object when save is false."""
    provider = ConcreteProvider(client=None)

    result_img = provider._create_placeholder_image(
        prompt="Test error prompt",
        output_dir="dummy_dir",
        save=False,
    )

    assert isinstance(result_img, Image.Image)
    assert result_img.size == (600, 600)


def test_create_placeholder_image_font_os_error_fallback() -> None:
    """Fallback to default font when truetype raises OSError."""
    original_truetype = ImageFont.truetype

    def mock_truetype_side_effect(font: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if font == "arialbd.ttf":
            raise OSError("Font not found")
        return original_truetype(font, *args, **kwargs)

    provider = ConcreteProvider(client=None)
    with patch(
        "xwhy.providers.base.ImageFont.truetype", side_effect=mock_truetype_side_effect
    ):
        result_img = provider._create_placeholder_image(
            prompt="Test error prompt",
            output_dir="dummy_dir",
            save=False,
        )

    assert isinstance(result_img, Image.Image)
